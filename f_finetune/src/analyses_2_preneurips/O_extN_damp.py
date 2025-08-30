import argparse
import logging
import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# Utilities

def np32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def symm_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    p_prob = p.exp()
    q_prob = q.exp()
    kl_pq = (p_prob * (p - q)).sum(-1)
    kl_qp = (q_prob * (q - p)).sum(-1)
    return kl_pq + kl_qp

def to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device) for k, v in batch.items()}

def batched(iterable, n=32):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def format_text(instruction: str, inp: str) -> str:
    if inp is None:
        inp = ""
    inp = inp.strip()
    if len(inp) == 0:
        return instruction.strip()
    return f"{instruction.strip()}\n\nInput: {inp}"

# Model / layer plumbing

def _get_layers_list(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    else:
        raise RuntimeError("Could not locate decoder layers list on model.")

@dataclass
class LayerHandles:
    layer_module: torch.nn.Module
    attn_module: torch.nn.Module
    mlp_module: torch.nn.Module
    num_heads: int
    head_dim: int
    hidden_size: int

def resolve_layer_handles(model: AutoModelForCausalLM, layer_index: int) -> LayerHandles:
    layers = _get_layers_list(model)
    if not (0 <= layer_index < len(layers)):
        raise RuntimeError(f"Layer index {layer_index} out of range [0,{len(layers)-1}]")
    layer = layers[layer_index]
    attn = None
    for name in ["self_attn", "self_attention", "attn", "attention"]:
        if hasattr(layer, name):
            attn = getattr(layer, name)
            break
    if attn is None:
        raise RuntimeError(f"Could not locate attention module on layer {layer_index}.")
    mlp = None
    for name in ["mlp", "feed_forward", "ff", "ffn"]:
        if hasattr(layer, name):
            mlp = getattr(layer, name)
            break
    if mlp is None:
        raise RuntimeError(f"Could not locate MLP module on layer {layer_index}.")
    num_heads = getattr(model.config, "num_attention_heads", None)
    hidden_size = getattr(model.config, "hidden_size", None)
    if num_heads is None:
        for attr in ["num_heads", "n_heads", "num_attention_heads"]:
            if hasattr(attn, attr):
                num_heads = int(getattr(attn, attr)); break
    if hidden_size is None:
        hidden_size = getattr(attn, "embed_dim", None) or getattr(attn, "hidden_size", None)
    head_dim = (hidden_size // num_heads) if (num_heads and hidden_size) else 0
    return LayerHandles(layer, attn, mlp, num_heads or 0, head_dim or 0, hidden_size or 0)

# Loading / dataset utils

@dataclass
class Paraphrase:
    key: str
    text: str

@dataclass
class PromptGroup:
    prompt_count: int
    original: str
    paraphrases: List[Paraphrase]

def load_jsonl_dataset(path: str, max_groups: Optional[int]=None, max_paraphrases_per_group: Optional[int]=None) -> List[PromptGroup]:
    groups: List[PromptGroup] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pc = int(obj["prompt_count"])
            instruction = obj["instruction_original"]
            inp = obj.get("input", "") or ""
            original = format_text(instruction, inp)
            paras = [Paraphrase(key=p["key"], text=format_text(p["text"], "")) for p in obj.get("paraphrases", [])]
            if max_paraphrases_per_group is not None:
                paras = paras[:max_paraphrases_per_group]
            groups.append(PromptGroup(prompt_count=pc, original=original, paraphrases=paras))
            if max_groups is not None and len(groups) >= max_groups:
                break
    return groups

# Model/tokenizer load

def load_model_tokenizer(base_model_name_or_path: str,
                         ft_model_name_or_path: Optional[str],
                         ft_lora_path: Optional[str],
                         merge_lora: bool,
                         dtype: str,
                         device: str):
    torch_dtype = dict(auto="auto", fp16=torch.float16, fp32=torch.float32, bf16=torch.bfloat16).get(dtype, "auto")
    device_map = device if device in ["cpu", "cuda", "auto"] else "auto"

    logging.info(f"Loading BASE model: {base_model_name_or_path}")
    base_tok = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=torch_dtype, device_map=device_map)

    if ft_model_name_or_path and ft_model_name_or_path != base_model_name_or_path:
        logging.info(f"Loading FT model host: {ft_model_name_or_path}")
        ft_tok = AutoTokenizer.from_pretrained(ft_model_name_or_path, use_fast=True)
        ft = AutoModelForCausalLM.from_pretrained(ft_model_name_or_path, torch_dtype=torch_dtype, device_map=device_map)
    else:
        logging.info("FT model host == BASE host.")
        ft_tok = base_tok
        ft = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=torch_dtype, device_map=device_map)

    if ft_lora_path:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed, but ft_lora_path was provided.")
        logging.info(f"Loading LoRA adapter for FT: {ft_lora_path}")
        ft = PeftModel.from_pretrained(ft, ft_lora_path)
        if merge_lora:
            logging.info("Merging LoRA into FT weights and unloading adapters.")
            ft = ft.merge_and_unload()

    tok = ft_tok
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base.eval()
    ft.eval()
    return base, ft, tok

# Dampening helpers

def tokenize(tokenizer: AutoTokenizer, texts: List[str], max_length: int, device: torch.device):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return to_device(enc, device)

def last_indices(attn_mask: torch.Tensor, window_size: int) -> List[List[int]]:
    idxs = []
    for i in range(attn_mask.shape[0]):
        L = int(attn_mask[i].sum().item())
        w = min(window_size, L)
        idxs.append(list(range(L - w, L)))
    return idxs

# kwargs-robust pre-hook utils

def _pre_get_hidden_states(args, kwargs):
    if args and len(args) > 0 and torch.is_tensor(args[0]):
        return args[0]
    if kwargs:
        if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
            return kwargs["hidden_states"]
        if "x" in kwargs and torch.is_tensor(kwargs["x"]):
            return kwargs["x"]
    raise RuntimeError("Could not locate hidden_states/x in pre-hook inputs.")

def _pre_set_hidden_states(args, kwargs, new_hs):
    if args and len(args) > 0 and torch.is_tensor(args[0]):
        new_args = (new_hs,) + tuple(args[1:])
        return new_args, kwargs
    new_kwargs = dict(kwargs) if kwargs is not None else {}
    if "hidden_states" in new_kwargs:
        new_kwargs["hidden_states"] = new_hs
        return args, new_kwargs
    if "x" in new_kwargs:
        new_kwargs["x"] = new_hs
        return args, new_kwargs
    new_args = (new_hs,) + tuple(args or ())
    return new_args, new_kwargs

def _noise_match_mean_std(ref_vec: torch.Tensor) -> torch.Tensor:
    mu = ref_vec.mean()
    std = ref_vec.std(unbiased=False)
    if float(std) < 1e-6:
        std = torch.tensor(1e-6, device=ref_vec.device, dtype=ref_vec.dtype)
    return mu + std * torch.randn_like(ref_vec)

def _make_wrong_original_picker(length_buckets: Dict[int, List[Tuple[int,int]]], my_bucket: int, my_group_idx: int):
    def pick():
        cands = [t for t in length_buckets.get(my_bucket, []) if t[0] != my_group_idx]
        if not cands:
            all_cands = [t for b, arr in length_buckets.items() for t in arr if t[0] != my_group_idx]
            if not all_cands:
                return None
            return random.choice(all_cands)
        return random.choice(cands)
    return pick

# Capture & patch (sites: 'layer_input', 'attn_in', 'mlp_in')

@dataclass
class CapturedPerLayer:
    site: str
    per_layer: Dict[int, torch.Tensor]  # k -> (W, H)
    token_count: int

def capture_site_vectors(model, tokenizer, text: str, site: str, max_length: int, device, window_size: int,
                         attn_implementation: Optional[str]=None) -> CapturedPerLayer:
    layers = _get_layers_list(model); L = len(layers)
    enc = tokenize(tokenizer, [text], max_length, device)
    idxs = last_indices(enc["attention_mask"], window_size)[0]
    store: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook_layer_input(k):
        def hook(module, args, kwargs):
            hs = _pre_get_hidden_states(args, kwargs).detach()
            store[k] = hs[0, idxs, :].clone()
            return None
        return hook

    def make_hook_pre(k):
        def hook(module, args, kwargs):
            hs = _pre_get_hidden_states(args, kwargs).detach()
            store[k] = hs[0, idxs, :].clone()
            return None
        return hook

    for k in range(L):
        h = resolve_layer_handles(model, k)
        if site == "layer_input":
            hooks.append(h.layer_module.register_forward_pre_hook(make_hook_layer_input(k), with_kwargs=True))
        elif site == "attn_in":
            hooks.append(h.attn_module.register_forward_pre_hook(make_hook_pre(k), with_kwargs=True))
        elif site == "mlp_in":
            hooks.append(h.mlp_module.register_forward_pre_hook(make_hook_pre(k), with_kwargs=True))
        else:
            raise ValueError("site must be one of {'layer_input','attn_in','mlp_in'}")

    with torch.no_grad():
        if "attn_implementation" in model.forward.__code__.co_varnames:
            _ = model(**enc, output_attentions=False, use_cache=False, attn_implementation=attn_implementation)
        else:
            _ = model(**enc, output_attentions=False, use_cache=False)

    for hk in hooks:
        try: hk.remove()
        except Exception: pass

    H = resolve_layer_handles(model, 0).hidden_size
    for k in range(L):
        if k not in store:
            store[k] = torch.zeros((len(idxs), H), device=device, dtype=torch.float32)

    return CapturedPerLayer(site=site, per_layer={k: store[k].detach().to("cpu") for k in range(L)}, token_count=int(enc["attention_mask"].sum().item()))

def patch_and_logits(model, tokenizer, text: str, site: str, layer_k: int, patch_vecs_WxH: torch.Tensor,
                     alpha: float, max_length: int, device, window_size: int, attn_implementation: Optional[str]=None) -> torch.Tensor:
    enc = tokenize(tokenizer, [text], max_length, device)
    idxs = last_indices(enc["attention_mask"], window_size)[0]
    h = resolve_layer_handles(model, layer_k)

    def make_pre_patch(module, args, kwargs):
        hs = _pre_get_hidden_states(args, kwargs).clone()
        for i, pos in enumerate(idxs):
            vec = patch_vecs_WxH[i].to(hs.device, hs.dtype)
            hs[0, pos, :] = (1.0 - alpha) * hs[0, pos, :] + alpha * vec
        return _pre_set_hidden_states(args, kwargs, hs)

    if site == "layer_input":
        hook = h.layer_module.register_forward_pre_hook(make_pre_patch, with_kwargs=True)
    elif site == "attn_in":
        hook = h.attn_module.register_forward_pre_hook(make_pre_patch, with_kwargs=True)
    elif site == "mlp_in":
        hook = h.mlp_module.register_forward_pre_hook(make_pre_patch, with_kwargs=True)
    else:
        raise ValueError("site must be one of {'layer_input','attn_in','mlp_in'}")

    with torch.no_grad():
        if "attn_implementation" in model.forward.__code__.co_varnames:
            out = model(**enc, output_attentions=False, use_cache=False, attn_implementation=attn_implementation)
        else:
            out = model(**enc, output_attentions=False, use_cache=False)
        logits = out.logits[:, -1, :]
    try:
        hook.remove()
    except Exception:
        pass
    return logits

def logits_for_texts(model, tokenizer, texts: List[str], max_length: int, device, batch_size: int,
                     attn_implementation: Optional[str]=None) -> torch.Tensor:
    all_logits = []
    for batch in batched(texts, n=max(1, int(batch_size))):
        enc = tokenize(tokenizer, batch, max_length, device)
        with torch.no_grad():
            if "attn_implementation" in model.forward.__code__.co_varnames:
                out = model(**enc, output_attentions=False, use_cache=False, attn_implementation=attn_implementation)
            else:
                out = model(**enc, output_attentions=False, use_cache=False)
            all_logits.append(out.logits[:, -1, :].detach())
    return torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, device=device)

# Survival (non-interventional)

def capture_layer_inputs_all_layers(model, tokenizer, text: str, max_length: int, device, window_size: int,
                                    site_for_survival: str="layer_input", attn_implementation: Optional[str]=None) -> Dict[int, torch.Tensor]:
    return capture_site_vectors(model, tokenizer, text, site_for_survival, max_length, device, window_size, attn_implementation).per_layer

def survival_curve(model, tokenizer, orig_text: str, para_text: str, max_length: int, device, window_size: int,
                   anchor_layer: int, site_for_survival: str="layer_input", attn_implementation: Optional[str]=None) -> Dict[int, float]:
    per_o = capture_layer_inputs_all_layers(model, tokenizer, orig_text, max_length, device, window_size, site_for_survival, attn_implementation)
    per_p = capture_layer_inputs_all_layers(model, tokenizer, para_text, max_length, device, window_size, site_for_survival, attn_implementation)
    diffs = {}
    for k in per_o.keys():
        a = per_o[k]; b = per_p[k]
        d = (a - b).to(torch.float32)
        diffs[k] = float(torch.linalg.vector_norm(d, ord=2).item())
    denom = diffs.get(anchor_layer, None)
    if denom is None or denom < 1e-12:
        denom = 1e-12
    return {k: (v / denom) for k, v in diffs.items()}

# Main experiment

def parse_list(arg: str, cast=float):
    vals = []
    for x in arg.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(cast(x))
    return vals

def parse_sites(arg: str) -> List[str]:
    out = []
    for x in arg.split(","):
        x = x.strip()
        if x:
            if x not in {"layer_input","attn_in","mlp_in"}:
                raise ValueError("patch_sites must be a comma-separated subset of {layer_input,attn_in,mlp_in}")
            out.append(x)
    if not out:
        out = ["layer_input"]
    return out

def negative_layers_to_indices(L: int, layers: List[int]) -> List[int]:
    out = []
    for k in layers:
        kk = (L + k) if k < 0 else k
        if 0 <= kk < L:
            out.append(kk)
    return sorted(set(out))

def run_dampening(groups: List[PromptGroup],
                  base_model, ft_model, tokenizer, device,
                  outdir, max_length: int, batch_size: int,
                  window_size: int, patch_sites: List[str],
                  do_base_curve: bool, do_wrong_original: bool, do_noise_control: bool,
                  alpha_layers: List[int], alpha_values: List[float],
                  anchor_layer: int, site_for_survival: str,
                  attn_implementation: Optional[str] = None):
    ensure_dir(outdir)
    curves_rows = []
    alpha_rows = []
    survival_rows = []

    layers_ft = _get_layers_list(ft_model); L = len(layers_ft)
    layers_base = _get_layers_list(base_model); assert len(layers_base) == L, "BASE/FT must have same depth."

    alpha_layers_idx = negative_layers_to_indices(L, alpha_layers)
    if not alpha_layers_idx:
        alpha_layers_idx = [max(0, L // 2), L - 1]

    # Pre-capture: FT + BASE (reuse later)
    ft_pool: Dict[str, Dict[int, Dict[int, torch.Tensor]]] = {site: {} for site in patch_sites}
    base_pool: Dict[str, Dict[int, Dict[int, torch.Tensor]]] = {site: {} for site in patch_sites} if do_base_curve else {}
    len_buckets: Dict[int, List[Tuple[int, int]]] = {}
    ft_orig_logits_cache: Dict[int, torch.Tensor] = {}
    base_orig_logits_cache: Dict[int, torch.Tensor] = {}

    logging.info("Pre-capturing originals (FT/BASE pools) + baseline logits...")
    for gi, group in enumerate(tqdm(groups, desc="original-prep", total=len(groups))):
        ft_orig_logits_cache[gi] = logits_for_texts(ft_model, tokenizer, [group.original], max_length, device, batch_size, attn_implementation)[0:1, :]
        if do_base_curve:
            base_orig_logits_cache[gi] = logits_for_texts(base_model, tokenizer, [group.original], max_length, device, batch_size, attn_implementation)[0:1, :]
        for site in patch_sites:
            cap_ft = capture_site_vectors(ft_model, tokenizer, group.original, site, max_length, device, window_size, attn_implementation)
            ft_pool[site][gi] = cap_ft.per_layer
            tL = cap_ft.token_count
            len_buckets.setdefault(tL, []).append((gi, tL))
            if do_base_curve:
                cap_base = capture_site_vectors(base_model, tokenizer, group.original, site, max_length, device, window_size, attn_implementation)
                base_pool[site][gi] = cap_base.per_layer

    def get_wrong_vecs(gi: int, site: str, layer_k: int) -> Optional[torch.Tensor]:
        my_bucket = None
        for Lb, arr in len_buckets.items():
            if any(gidx == gi for (gidx, _t) in arr):
                my_bucket = Lb; break
        picker = _make_wrong_original_picker(len_buckets, my_bucket if my_bucket is not None else -1, gi)
        pick = picker()
        if pick is None:
            return None
        g_other, _ = pick
        return ft_pool[site][g_other][layer_k]

    # Main loop with visible progress
    for gi, group in enumerate(tqdm(groups, desc="groups", total=len(groups))):
        para_texts = [p.text for p in group.paraphrases]
        if not para_texts:
            continue

        ft_logits_para_all = logits_for_texts(ft_model, tokenizer, para_texts, max_length, device, batch_size, attn_implementation)
        base_logits_para_all = logits_for_texts(base_model, tokenizer, para_texts, max_length, device, batch_size, attn_implementation) if do_base_curve else None

        logging.info(f"[group {gi} | prompt_count={group.prompt_count}] #paras={len(group.paraphrases)}; sites={patch_sites}; L={L}")
        for pi, para in enumerate(tqdm(group.paraphrases, desc=f"paras@g{gi}", leave=False)):
            try:
                kl_base_ft = float(symm_kl(ft_orig_logits_cache[gi], ft_logits_para_all[pi:pi+1, :]).item())
                kl_base_base = float(symm_kl(base_orig_logits_cache[gi], base_logits_para_all[pi:pi+1, :]).item()) if do_base_curve else None

                for site in patch_sites:
                    ft_corr_vecs_per_layer = ft_pool[site][gi]

                    # Layer curve (FT)
                    for k in range(L):
                        vecs = ft_corr_vecs_per_layer[k]
                        logits_patched_ft = patch_and_logits(ft_model, tokenizer, para.text, site, k, vecs, alpha=1.0,
                                                             max_length=max_length, device=device, window_size=window_size,
                                                             attn_implementation=attn_implementation)
                        row = {
                            "prompt_count": group.prompt_count,
                            "group_index": gi,
                            "para_key": para.key,
                            "site": site,
                            "layer_k": k,
                            "model": "FT",
                            "kl_baseline": kl_base_ft,
                            "kl_patched_correct": float(symm_kl(ft_orig_logits_cache[gi], logits_patched_ft).item())
                        }

                        if do_wrong_original:
                            wrong_vecs = get_wrong_vecs(gi, site, k)
                            if wrong_vecs is not None:
                                logits_wrong = patch_and_logits(ft_model, tokenizer, para.text, site, k, wrong_vecs, alpha=1.0,
                                                                max_length=max_length, device=device, window_size=window_size,
                                                                attn_implementation=attn_implementation)
                                row["kl_patched_wrong"] = float(symm_kl(ft_orig_logits_cache[gi], logits_wrong).item())
                            else:
                                row["kl_patched_wrong"] = float("nan")

                        if do_noise_control:
                            with torch.no_grad():
                                vecs_noise = torch.stack([_noise_match_mean_std(vecs[i]) for i in range(vecs.shape[0])], dim=0).to(torch.float32)
                            logits_noise = patch_and_logits(ft_model, tokenizer, para.text, site, k, vecs_noise, alpha=1.0,
                                                            max_length=max_length, device=device, window_size=window_size,
                                                            attn_implementation=attn_implementation)
                            row["kl_patched_noise"] = float(symm_kl(ft_orig_logits_cache[gi], logits_noise).item())

                        curves_rows.append(row)

                        # BASE curve uses pre-captured originals (no re-capture loop)
                        if do_base_curve:
                            base_vecs = base_pool[site][gi][k]
                            logits_patched_base = patch_and_logits(base_model, tokenizer, para.text, site, k, base_vecs, alpha=1.0,
                                                                   max_length=max_length, device=device, window_size=window_size,
                                                                   attn_implementation=attn_implementation)
                            curves_rows.append({
                                "prompt_count": group.prompt_count,
                                "group_index": gi,
                                "para_key": para.key,
                                "site": site,
                                "layer_k": k,
                                "model": "BASE",
                                "kl_baseline": kl_base_base,
                                "kl_patched_correct": float(symm_kl(base_orig_logits_cache[gi], logits_patched_base).item())
                            })

                    # α-sweep (FT only) at selected layers
                    for k in alpha_layers_idx:
                        vecs = ft_corr_vecs_per_layer[k]
                        for a in alpha_values:
                            logits_alpha = patch_and_logits(ft_model, tokenizer, para.text, site, k, vecs, alpha=float(a),
                                                            max_length=max_length, device=device, window_size=window_size,
                                                            attn_implementation=attn_implementation)
                            alpha_rows.append({
                                "prompt_count": group.prompt_count,
                                "group_index": gi,
                                "para_key": para.key,
                                "site": site,
                                "layer_k": k,
                                "alpha": float(a),
                                "model": "FT",
                                "kl_alpha": float(symm_kl(ft_orig_logits_cache[gi], logits_alpha).item())
                            })

                # Survival ratio (FT & BASE)
                surv_ft = survival_curve(ft_model, tokenizer, group.original, para.text, max_length, device, window_size,
                                         anchor_layer=anchor_layer, site_for_survival=site_for_survival, attn_implementation=attn_implementation)
                for k, s in surv_ft.items():
                    survival_rows.append({
                        "prompt_count": group.prompt_count,
                        "group_index": gi,
                        "para_key": para.key,
                        "model": "FT",
                        "layer_k": k,
                        "survival": float(s)
                    })
                if do_base_curve:
                    surv_base = survival_curve(base_model, tokenizer, group.original, para.text, max_length, device, window_size,
                                               anchor_layer=anchor_layer, site_for_survival=site_for_survival, attn_implementation=attn_implementation)
                    for k, s in surv_base.items():
                        survival_rows.append({
                            "prompt_count": group.prompt_count,
                            "group_index": gi,
                            "para_key": para.key,
                            "model": "BASE",
                            "layer_k": k,
                            "survival": float(s)
                        })

            except Exception as e:
                logging.exception(f"Error in group {gi} para {para.key}: {e}")

        # occasional GC to avoid long silent CUDA allocs
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Save CSVs
    curves_df = pd.DataFrame(curves_rows)
    alphas_df = pd.DataFrame(alpha_rows)
    surv_df = pd.DataFrame(survival_rows)

    curves_path = os.path.join(outdir, "dampening_curves.csv"); curves_df.to_csv(curves_path, index=False)
    alphas_path = os.path.join(outdir, "alpha_sweep.csv"); alphas_df.to_csv(alphas_path, index=False)
    surv_path = os.path.join(outdir, "survival.csv"); surv_df.to_csv(surv_path, index=False)
    logging.info("Wrote CSVs: %s | %s | %s", curves_path, alphas_path, surv_path)

    # Composite figure
    try:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        ax = axs[0]
        if len(curves_df):
            for site in patch_sites:
                sub = curves_df[(curves_df["site"] == site)]
                base_ft = sub[sub["model"] == "FT"].groupby("layer_k")["kl_baseline"].mean()
                patched_ft = sub[sub["model"] == "FT"].groupby("layer_k")["kl_patched_correct"].mean()
                ax.plot(patched_ft.index, patched_ft.values, marker="o", label=f"FT patched ({site})")
                ax.axhline(base_ft.mean(), linestyle="--", alpha=0.5, label=f"FT baseline ({site})")

                if do_wrong_original and "kl_patched_wrong" in sub.columns:
                    wrong_ft = sub[sub["model"] == "FT"].groupby("layer_k")["kl_patched_wrong"].mean()
                    ax.plot(wrong_ft.index, wrong_ft.values, linestyle=":", marker=".", label=f"FT wrong ({site})")

                if do_noise_control and "kl_patched_noise" in sub.columns:
                    noise_ft = sub[sub["model"] == "FT"].groupby("layer_k")["kl_patched_noise"].mean()
                    ax.plot(noise_ft.index, noise_ft.values, linestyle="--", marker="x", label=f"FT noise ({site})")

                if "BASE" in sub["model"].unique():
                    patched_base = sub[sub["model"] == "BASE"].groupby("layer_k")["kl_patched_correct"].mean()
                    ax.plot(patched_base.index, patched_base.values, marker="s", label=f"BASE patched ({site})")
            ax.set_title("(i) Specificity & controls")
            ax.set_xlabel("Layer k")
            ax.set_ylabel("Mean symmetric KL")

        ax = axs[1]
        if len(alphas_df):
            for k in sorted(set(alphas_df["layer_k"].tolist())):
                subk = alphas_df[(alphas_df["layer_k"] == k) & (alphas_df["model"] == "FT")]
                if not len(subk):
                    continue
                g = subk.groupby("alpha")["kl_alpha"].mean().reset_index()
                ax.plot(g["alpha"], g["kl_alpha"], marker="o", label=f"layer {k}")
            ax.set_title("(ii) α-sweep (FT)")
            ax.set_xlabel("α"); ax.set_ylabel("Mean KL")
            ax.set_xlim(0, 1)

        ax = axs[2]
        if len(surv_df):
            for model_name in ["FT"] + (["BASE"] if do_base_curve else []):
                sub = surv_df[surv_df["model"] == model_name]
                g = sub.groupby("layer_k")["survival"].mean().reset_index()
                ax.plot(g["layer_k"], g["survival"], marker="o", label=model_name)
            ax.set_title("(iii) Activation survival")
            ax.set_xlabel("Layer k"); ax.set_ylabel("mean ||Δh_k|| / ||Δh_anchor||")
            ax.set_ylim(bottom=0)

        for ax in axs:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)

        fig.tight_layout()
        figpath = os.path.join(outdir, "DAMPENING_FIG_D1.png")
        fig.savefig(figpath, dpi=160)
        plt.close(fig)
        logging.info("Wrote figure: %s", figpath)
    except Exception as e:
        logging.exception("Failed to write the composite figure: %s", e)

# CLI / Runner (dampening only)

def main():
    parser = argparse.ArgumentParser(description="Downstream Dampening experiment (rigorized)")
    parser.add_argument("--data_jsonl", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--base_model", type=str, required=True, help="HF model path/name for BASE")
    parser.add_argument("--ft_model", type=str, default=None, help="HF model path/name for FT host (defaults to BASE)")
    parser.add_argument("--ft_lora_path", type=str, default=None, help="Optional PEFT adapter path for FT")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA into FT weights")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto","cpu","cuda"], help="Device")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto","fp32","fp16","bf16"], help="Torch dtype")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for baseline forwards")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_groups", type=int, default=None, help="Limit number of prompt groups for debugging")
    parser.add_argument("--max_paraphrases", type=int, default=16, help="Limit paraphrases per group")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--attn_implementation", type=str, default=None, help="Optional attention impl hint")

    parser.add_argument("--do_dampening", action="store_true", help="Run the downstream dampening suite")

    parser.add_argument("--patch_sites", type=str, default="layer_input",
                        help="Comma-separated from {layer_input,attn_in,mlp_in}")
    parser.add_argument("--window_size", type=int, default=1, help="Number of last tokens to patch")
    parser.add_argument("--no_base_curve", action="store_true", help="Disable BASE curve")
    parser.add_argument("--no_wrong_original", action="store_true", help="Disable wrong-original control")
    parser.add_argument("--no_noise_control", action="store_true", help="Disable matched-moment noise control")

    parser.add_argument("--alpha_layers", type=str, default="-1",
                        help="Comma-separated layer indices for α-sweep (supports negatives like -1 for last)")
    parser.add_argument("--alpha_values", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--anchor_layer", type=int, default=6, help="Anchor layer for survival normalization")
    parser.add_argument("--survival_site", type=str, default="layer_input", choices=["layer_input","attn_in","mlp_in"],
                        help="Where to measure survival (non-interventional)")

    args = parser.parse_args()

    ensure_dir(args.outdir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(args.outdir, "run.log"), mode="w")]
    )
    logging.info("Args: %s", vars(args))
    set_seed(args.seed)

    logging.info("Loading dataset from %s ...", args.data_jsonl)
    groups = load_jsonl_dataset(args.data_jsonl, max_groups=args.max_groups, max_paraphrases_per_group=args.max_paraphrases)
    logging.info("Loaded %d prompt groups.", len(groups))

    base, ft, tok = load_model_tokenizer(
        base_model_name_or_path=args.base_model,
        ft_model_name_or_path=args.ft_model or args.base_model,
        ft_lora_path=args.ft_lora_path,
        merge_lora=args.merge_lora,
        dtype=args.dtype,
        device=args.device
    )
    device = next(ft.parameters()).device

    if args.do_dampening:
        run_dampening(
            groups=groups,
            base_model=base,
            ft_model=ft,
            tokenizer=tok,
            device=device,
            outdir=os.path.join(args.outdir, "DAMPENING"),
            max_length=args.max_length,
            batch_size=args.batch_size,
            window_size=args.window_size,
            patch_sites=parse_sites(args.patch_sites),
            do_base_curve=not args.no_base_curve,
            do_wrong_original=not args.no_wrong_original,
            do_noise_control=not args.no_noise_control,
            alpha_layers=[int(x) for x in args.alpha_layers.split(",") if x.strip()!=""],
            alpha_values=parse_list(args.alpha_values, cast=float),
            anchor_layer=args.anchor_layer,
            site_for_survival=args.survival_site,
            attn_implementation=args.attn_implementation
        )
    else:
        logging.info("Nothing to do. (Pass --do_dampening)")

    logging.info("Done. Outputs in %s", args.outdir)


if __name__ == "__main__":
    main()
