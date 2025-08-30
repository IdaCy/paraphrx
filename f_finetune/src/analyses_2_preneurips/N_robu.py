#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import json
import math
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
    """Symmetric KL between softmax(p_logits) and softmax(q_logits)."""
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
    """Batch generator."""
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

@dataclass
class Layer6Handles:
    layer_module: torch.nn.Module
    attn_module: torch.nn.Module
    mlp_module: torch.nn.Module
    up_proj: Optional[torch.nn.Module]
    gate_proj: Optional[torch.nn.Module]
    down_proj: Optional[torch.nn.Module]
    num_heads: int
    head_dim: int


def _get_layers_list(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    else:
        raise RuntimeError("Could not locate decoder layers list on model.")


def resolve_layer6_handles(model: AutoModelForCausalLM) -> Layer6Handles:
    """
    (Original) Best-effort resolver for layer-6 attention & MLP components across common architectures.
    """
    layers = _get_layers_list(model)
    if len(layers) < 7:
        raise RuntimeError("Model has fewer than 7 layers; cannot select layer index 6 (0-based).")
    layer = layers[6]  # 0-based index

    # Attention module heuristic names
    attn = None
    for name in ["self_attn", "self_attention", "attn", "attention"]:
        if hasattr(layer, name):
            attn = getattr(layer, name)
            break
    if attn is None:
        raise RuntimeError("Could not locate attention module on layer 6.")

    # MLP module heuristic names
    mlp = None
    for name in ["mlp", "feed_forward", "ff", "ffn"]:
        if hasattr(layer, name):
            mlp = getattr(layer, name)
            break
    if mlp is None:
        raise RuntimeError("Could not locate MLP module on layer 6.")

    # Projections
    up = None
    gate = None
    down = None
    for cand in ["up_proj", "dense_h_to_4h", "fc_in"]:
        if hasattr(mlp, cand):
            up = getattr(mlp, cand)
            break
    for cand in ["gate_proj", "fc_gate"]:
        if hasattr(mlp, cand):
            gate = getattr(mlp, cand)
            break
    for cand in ["down_proj", "dense_4h_to_h", "fc_out"]:
        if hasattr(mlp, cand):
            down = getattr(mlp, cand)
            break

    num_heads = getattr(model.config, "num_attention_heads", None)
    hidden_size = getattr(model.config, "hidden_size", None)
    head_dim = None
    if num_heads is None:
        for attr in ["num_heads", "n_heads", "num_attention_heads"]:
            if hasattr(attn, attr):
                num_heads = int(getattr(attn, attr))
                break
    if hidden_size is None:
        hidden_size = getattr(attn, "embed_dim", None) or getattr(attn, "hidden_size", None)
    if num_heads is not None and hidden_size is not None and num_heads > 0:
        head_dim = hidden_size // num_heads
    else:
        num_heads, head_dim = 0, 0

    return Layer6Handles(layer_module=layer, attn_module=attn, mlp_module=mlp,
                         up_proj=up, gate_proj=gate, down_proj=down,
                         num_heads=num_heads, head_dim=head_dim)

# generic layer resolver and capture utilities

@dataclass
class LayerHandles:
    layer_module: torch.nn.Module
    attn_module: torch.nn.Module
    mlp_module: torch.nn.Module
    up_proj: Optional[torch.nn.Module]
    gate_proj: Optional[torch.nn.Module]
    down_proj: Optional[torch.nn.Module]
    num_heads: int
    head_dim: int

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
    up = gate = down = None
    for cand in ["up_proj", "dense_h_to_4h", "fc_in"]:
        if hasattr(mlp, cand):
            up = getattr(mlp, cand); break
    for cand in ["gate_proj", "fc_gate"]:
        if hasattr(mlp, cand):
            gate = getattr(mlp, cand); break
    for cand in ["down_proj", "dense_4h_to_h", "fc_out"]:
        if hasattr(mlp, cand):
            down = getattr(mlp, cand); break

    num_heads = getattr(model.config, "num_attention_heads", None)
    hidden_size = getattr(model.config, "hidden_size", None)
    head_dim = None
    if num_heads is None:
        for attr in ["num_heads", "n_heads", "num_attention_heads"]:
            if hasattr(attn, attr):
                num_heads = int(getattr(attn, attr)); break
    if hidden_size is None:
        hidden_size = getattr(attn, "embed_dim", None) or getattr(attn, "hidden_size", None)
    if num_heads is not None and hidden_size is not None and num_heads > 0:
        head_dim = hidden_size // num_heads
    else:
        num_heads, head_dim = 0, 0
    return LayerHandles(layer, attn, mlp, up, gate, down, num_heads, head_dim)

@dataclass
class CaptureResult:
    logits_last: torch.Tensor
    down_last: torch.Tensor
    mlp_input_last: torch.Tensor
    last_token_indices: torch.Tensor
    input_ids: torch.Tensor

def build_layer_capturer(model: AutoModelForCausalLM, layer_index: int):
    handles = resolve_layer_handles(model, layer_index)
    cache = {"mlp_in": None, "down_out": None}
    def mlp_pre_hook(module, inputs):
        hs = inputs[0]
        cache["mlp_in"] = hs.detach()
        return None
    def down_forward_hook(module, inputs, output):
        cache["down_out"] = output.detach() if torch.is_tensor(output) else output[0].detach()
        return None
    pre_handle = handles.mlp_module.register_forward_pre_hook(mlp_pre_hook, with_kwargs=False)
    if handles.down_proj is not None:
        down_handle = handles.down_proj.register_forward_hook(down_forward_hook)
    else:
        down_handle = handles.mlp_module.register_forward_hook(down_forward_hook)
    return handles, cache, (pre_handle, down_handle)

def remove_hooks(hs):
    for h in hs:
        try:
            h.remove()
        except Exception:
            pass

def forward_with_capture_at_layer(model: AutoModelForCausalLM,
                                  tokenizer: AutoTokenizer,
                                  texts: List[str],
                                  layer_index: int,
                                  max_length: int,
                                  device: torch.device,
                                  attn_implementation: Optional[str]=None) -> CaptureResult:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc = to_device(enc, device)
    handles, cache, hook_handles = build_layer_capturer(model, layer_index)
    with torch.no_grad():
        if "attn_implementation" in model.forward.__code__.co_varnames:
            out = model(**enc, output_attentions=False, use_cache=False, attn_implementation=attn_implementation)
        else:
            out = model(**enc, output_attentions=False, use_cache=False)
        logits = out.logits[:, -1, :]
    remove_hooks(hook_handles)
    attn_mask = enc["attention_mask"]
    last_idx = attn_mask.sum(dim=1) - 1
    mlp_in = cache["mlp_in"]
    down_out = cache["down_out"]
    if mlp_in is None or down_out is None:
        raise RuntimeError("Failed to capture layer MLP inputs/outputs; check hooks.")
    B = enc["input_ids"].shape[0]
    batch_index = torch.arange(B, device=last_idx.device)
    mlp_in_last = mlp_in[batch_index, last_idx, :].contiguous()
    down_last = down_out[batch_index, last_idx, :].contiguous()
    return CaptureResult(logits_last=logits, down_last=down_last, mlp_input_last=mlp_in_last,
                         last_token_indices=last_idx, input_ids=enc["input_ids"])

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

def fit_ridge_map(X: np.ndarray, Y: np.ndarray, lam: float=1e-3) -> np.ndarray:
    D = X.shape[1]
    A = X.T @ X + lam * np.eye(D, dtype=X.dtype)
    B = X.T @ Y
    W = np.linalg.solve(A, B)
    return W

def orthonormal_basis_from_cols(U: np.ndarray, eps: float=1e-8) -> np.ndarray:
    Q, R = np.linalg.qr(U)
    keep = np.where(np.abs(np.diag(R)) > eps)[0]
    return Q[:, keep]

def project_components(x: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if U.size == 0:
        return np.zeros_like(x), x.copy()
    coeff = U.T @ x
    x_sens = U @ coeff
    x_orth = x - x_sens
    return x_sens, x_orth

# A (layer-6 baseline)
def forward_with_capture(model: AutoModelForCausalLM,
                         tokenizer: AutoTokenizer,
                         texts: List[str],
                         max_length: int,
                         device: torch.device,
                         attn_implementation: Optional[str]=None) -> CaptureResult:
    return forward_with_capture_at_layer(model, tokenizer, texts, 6, max_length, device, attn_implementation)

def experiment_A_subspace_patching(groups: List[PromptGroup],
                                   base_model, ft_model, tokenizer,
                                   device, outdir,
                                   max_length: int, batch_size: int,
                                   topk_vocab: int, lam_ridge: float,
                                   max_groups: Optional[int]=None):
    ensure_dir(outdir)
    rows = []
    logging.info("[A] Subspace-patching: capturing FT original+paraphrases for all groups...")
    processed = 0
    for group in tqdm(groups, desc="[A] groups", total=len(groups)):
        if max_groups is not None and processed >= max_groups:
            break
        texts_all = [group.original] + [p.text for p in group.paraphrases]
        caps: List[CaptureResult] = []
        for batch in batched(texts_all, n=batch_size):
            cap = forward_with_capture(ft_model, tokenizer, batch, max_length=max_length, device=device)
            caps.append(cap)
        logits_last = torch.cat([c.logits_last for c in caps], dim=0)
        down_last = torch.cat([c.down_last for c in caps], dim=0)
        orig_logits = logits_last[0:1, :]
        orig_down = down_last[0:1, :]
        para_logits = logits_last[1:, :]
        para_down = down_last[1:, :]
        with torch.no_grad():
            topk = min(topk_vocab, logits_last.shape[-1])
            top_ids = torch.topk(logits_last, k=topk, dim=-1).indices
            vocab_ids = torch.unique(top_ids.flatten()).tolist()
            orig_v = np32(orig_logits[:, vocab_ids])
            para_v = np32(para_logits[:, vocab_ids])
        dlogits = (para_v - orig_v)
        dH = np32(para_down) - np32(orig_down)
        W = fit_ridge_map(dH, dlogits, lam=lam_ridge)
        try:
            U, S, VT = np.linalg.svd(W, full_matrices=False)
        except np.linalg.LinAlgError:
            logging.warning("SVD failed; skipping group %s", group.prompt_count)
            processed += 1
            continue
        k = min(8, U.shape[1])
        U_sens = orthonormal_basis_from_cols(U[:, :k])
        orig_cap = forward_with_capture(ft_model, tokenizer, [group.original], max_length=max_length, device=device)
        down_orig = np32(orig_cap.down_last[0])
        logits_orig = orig_cap.logits_last[0].unsqueeze(0)
        for para in group.paraphrases:
            cap_p = forward_with_capture(ft_model, tokenizer, [para.text], max_length=max_length, device=device)
            logits_p = cap_p.logits_last
            down_p = np32(cap_p.down_last[0])
            dH_p = down_p - down_orig
            dH_sens, dH_orth = project_components(dH_p, U_sens)
            def run_with_down_patch(new_down_vec_np: np.ndarray) -> torch.Tensor:
                new_down_vec = torch.from_numpy(new_down_vec_np).to(device=device, dtype=cap_p.down_last.dtype)
                handles = resolve_layer_handles(ft_model, 6)
                enc = tokenizer([para.text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                enc = to_device(enc, device)
                last_idx = enc["attention_mask"].sum(dim=1) - 1
                def down_hook(module, inputs, output):
                    out = output.clone() if torch.is_tensor(output) else output[0].clone()
                    out[0, last_idx.item(), :] = new_down_vec
                    if torch.is_tensor(output):
                        return out
                    else:
                        if isinstance(output, tuple):
                            return (out,) + output[1:]
                        output[0] = out
                        return output
                hook = (handles.down_proj or handles.mlp_module).register_forward_hook(down_hook)
                with torch.no_grad():
                    out = ft_model(**enc, output_attentions=False, use_cache=False)
                    logits_new = out.logits[:, -1, :]
                hook.remove()
                return logits_new
            logits_rm_sens = run_with_down_patch(down_orig + dH_orth)
            logits_rm_orth = run_with_down_patch(down_orig + dH_sens)
            with torch.no_grad():
                kl_base = symm_kl(logits_orig, logits_p).item()
                kl_rm_sens = symm_kl(logits_orig, logits_rm_sens).item()
                kl_rm_orth = symm_kl(logits_orig, logits_rm_orth).item()
            rows.append({
                "prompt_count": group.prompt_count,
                "para_key": para.key,
                "kl_ft": kl_base,
                "kl_remove_sens": kl_rm_sens,
                "kl_remove_orth": kl_rm_orth,
                "sens_dim": U_sens.shape[1],
            })
        processed += 1
    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "A_subspace_patching_results.csv")
    df.to_csv(csv_path, index=False)
    if len(df) > 0:
        means = {
            "FT baseline": df["kl_ft"].mean(),
            "Remove-sens": df["kl_remove_sens"].mean(),
            "Remove-orth": df["kl_remove_orth"].mean(),
        }
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(list(means.keys()), list(means.values()))
        ax.set_ylabel("Mean symmetric KL (orig || paraphrase)")
        ax.set_title("[A] Subspace patching at layer-6 DOWN (FT)")
        for i, v in enumerate(means.values()):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "A_subspace_patching_bars.png"), dpi=150); plt.close(fig)
    logging.info("[A] Done. CSV: %s", csv_path)

# Helper for swaps
def _get_layers_for_models(dst_model, src_model):
    layers_dst = _get_layers_list(dst_model)
    layers_src = _get_layers_list(src_model)
    return layers_dst, layers_src

def swap_layer_weights(dst_model, src_model, which: str, layer_index: int):
    layers_dst, layers_src = _get_layers_for_models(dst_model, src_model)
    if len(layers_dst) <= layer_index or len(layers_src) <= layer_index:
        raise RuntimeError("Model missing requested layer.")
    layer_dst = layers_dst[layer_index]
    layer_src = layers_src[layer_index]
    def get_attn(layer):
        for nm in ["self_attn", "self_attention", "attn", "attention"]:
            if hasattr(layer, nm): return getattr(layer, nm)
        return None
    def get_mlp(layer):
        for nm in ["mlp", "feed_forward", "ff", "ffn"]:
            if hasattr(layer, nm): return getattr(layer, nm)
        return None
    attn_dst, attn_src = get_attn(layer_dst), get_attn(layer_src)
    mlp_dst, mlp_src = get_mlp(layer_dst), get_mlp(layer_src)
    if which.upper() == "DOWN":
        def get_down(m):
            for nm in ["down_proj", "dense_4h_to_h", "fc_out"]:
                if hasattr(m, nm): return getattr(m, nm)
            return None
        d_dst, d_src = get_down(mlp_dst), get_down(mlp_src)
        if d_dst is None or d_src is None:
            raise RuntimeError("down_proj not found on one of the models.")
        with torch.no_grad():
            d_dst.weight.copy_(d_src.weight)
            if getattr(d_dst, "bias", None) is not None and getattr(d_src, "bias", None) is not None:
                d_dst.bias.copy_(d_src.bias)
    elif which.upper() == "ATTN":
        def get_proj(m, names):
            for n in names:
                if hasattr(m, n): return getattr(m, n)
            return None
        for name_list in [["q_proj"], ["k_proj"], ["v_proj"], ["o_proj"], ["qkv_proj"]]:
            dst_m = get_proj(attn_dst, name_list); src_m = get_proj(attn_src, name_list)
            if dst_m is not None and src_m is not None:
                with torch.no_grad():
                    dst_m.weight.copy_(src_m.weight)
                    if getattr(dst_m, "bias", None) is not None and getattr(src_m, "bias", None) is not None:
                        dst_m.bias.copy_(src_m.bias)
    else:
        raise ValueError("which must be one of {'DOWN','ATTN'}.")

def swap_layer6_weights(dst_model, src_model, which: str):
    return swap_layer_weights(dst_model, src_model, which, 6)

def evaluate_mean_kl(model, tokenizer, groups: List[PromptGroup], device, max_length: int, batch_size: int, max_pairs: Optional[int]=None) -> float:
    kls = []
    total_pairs = 0
    for group in tqdm(groups, desc="Eval KL groups", total=len(groups)):
        texts = [group.original] + [p.text for p in group.paraphrases]
        logits_all = []
        for batch in batched(texts, n=batch_size):
            cap = forward_with_capture(model, tokenizer, batch, max_length=max_length, device=device)
            logits_all.append(cap.logits_last)
        logits_all = torch.cat(logits_all, dim=0)
        logit_orig = logits_all[0:1, :]
        for i in range(1, logits_all.shape[0]):
            kl = symm_kl(logit_orig, logits_all[i:i+1, :]).item()
            kls.append(kl); total_pairs += 1
            if max_pairs is not None and total_pairs >= max_pairs:
                return float(np.mean(kls))
    return float(np.mean(kls)) if kls else float("nan")

def experiment_B_weight_level(groups: List[PromptGroup], base_model, ft_model, tokenizer, device, outdir, max_length, batch_size, max_pairs=None):
    ensure_dir(outdir)
    import copy
    logging.info("[B] Evaluating BASE and FT (full)...")
    base_kl = evaluate_mean_kl(base_model, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
    ft_full_kl = evaluate_mean_kl(ft_model, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
    logging.info("[B] Evaluating FT−DOWN (replace FT DOWN with BASE)...")
    ft_minus_down = copy.deepcopy(ft_model).to(device); swap_layer6_weights(ft_minus_down, base_model, which="DOWN")
    ft_minus_down_kl = evaluate_mean_kl(ft_minus_down, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
    logging.info("[B] Evaluating only-DOWN (BASE with DOWN from FT)...")
    base_plus_down = copy.deepcopy(base_model).to(device); swap_layer6_weights(base_plus_down, ft_model, which="DOWN")
    only_down_kl = evaluate_mean_kl(base_plus_down, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
    logging.info("[B] Evaluating FT−ATTN (ATTN from BASE into FT)...")
    ft_minus_attn = copy.deepcopy(ft_model).to(device); swap_layer6_weights(ft_minus_attn, base_model, which="ATTN")
    ft_minus_attn_kl = evaluate_mean_kl(ft_minus_attn, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
    logging.info("[B] Evaluating only-ATTN (BASE with ATTN from FT)...")
    base_plus_attn = copy.deepcopy(base_model).to(device); swap_layer6_weights(base_plus_attn, ft_model, which="ATTN")
    only_attn_kl = evaluate_mean_kl(base_plus_attn, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
    df = pd.DataFrame([
        {"variant": "BASE", "mean_KL": base_kl},
        {"variant": "FT", "mean_KL": ft_full_kl},
        {"variant": "FT−DOWN", "mean_KL": ft_minus_down_kl},
        {"variant": "only-DOWN", "mean_KL": only_down_kl},
        {"variant": "FT−ATTN", "mean_KL": ft_minus_attn_kl},
        {"variant": "only-ATTN", "mean_KL": only_attn_kl},
    ])
    csv_path = os.path.join(outdir, "B_weight_level_variants.csv"); df.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(df["variant"], df["mean_KL"]); ax.set_ylabel("Mean symmetric KL (orig || paraphrase)")
    ax.set_title("[B] Weight-level ablations at layer-6")
    for i, v in enumerate(df["mean_KL"]): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "B_weight_level_variants.png"), dpi=150); plt.close(fig)
    logging.info("[B] Done. CSV: %s", csv_path)

# C
def experiment_C_grad_alignment(groups: List[PromptGroup], model, tokenizer, device, outdir, max_length, batch_size, max_groups=None):
    ensure_dir(outdir)
    logging.info("[C] Gradient alignment at layer-6 DOWN (robust grad capture)...")
    rows = []; processed = 0
    def _enable_param_grads(m):
        flags = [p.requires_grad for p in m.parameters()]
        for p in m.parameters(): p.requires_grad_(True)
        return flags
    def _restore_param_grads(m, flags):
        for p, f in zip(m.parameters(), flags): p.requires_grad_(f)
    for group in tqdm(groups, desc="[C] groups", total=len(groups)):
        if max_groups is not None and processed >= max_groups: break
        enc = tokenizer([group.original], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = to_device(enc, device)
        last_idx = (enc["attention_mask"].sum(dim=1) - 1).item()
        handles = resolve_layer_handles(model, 6)
        bag = {"node": None}
        def fwd_hook(module, inputs, output):
            out2 = output[0] if isinstance(output, (tuple, list)) else output
            if not out2.requires_grad: out2 = out2.detach().requires_grad_(True)
            else: out2 = out2.clone().requires_grad_(True)
            bag["node"] = out2
            if isinstance(output, (tuple, list)):
                if isinstance(output, tuple):
                    return (out2,) + tuple(output[1:])
                else:
                    output[0] = out2
                    return output
            return out2
        hook = (handles.down_proj or handles.mlp_module).register_forward_hook(fwd_hook)
        prev_flags = _enable_param_grads(model)
        try:
            with torch.enable_grad():
                out = model(**enc, output_attentions=False, use_cache=False)
                logits = out.logits[:, -1, :]
                target_id = logits.argmax(dim=-1); target_logit = logits[0, target_id]
                g_full = torch.autograd.grad(target_logit, bag["node"], retain_graph=False, create_graph=False, allow_unused=False)[0]
            hook.remove()
        finally:
            _restore_param_grads(model, prev_flags)
        if g_full is None:
            logging.warning("No gradient captured for group %s", group.prompt_count); processed += 1; continue
        g_vec = g_full[0, last_idx, :].detach().to(torch.float32).cpu().numpy()
        g_u = g_vec / (np.linalg.norm(g_vec) + 1e-9)
        cap_orig = forward_with_capture(model, tokenizer, [group.original], max_length=max_length, device=device)
        for para in group.paraphrases:
            cap_p = forward_with_capture(model, tokenizer, [para.text], max_length=max_length, device=device)
            dH = (cap_p.down_last[0] - cap_orig.down_last[0]).detach().to(torch.float32).cpu().numpy()
            dH_u = dH / (np.linalg.norm(dH) + 1e-9)
            cos = float(np.dot(dH_u, g_u))
            rows.append({"prompt_count": group.prompt_count, "para_key": para.key, "cos": cos, "cos2": cos*cos})
        processed += 1
    df = pd.DataFrame(rows); csv_path = os.path.join(outdir, "C_grad_alignment.csv"); df.to_csv(csv_path, index=False)
    if len(df) > 0:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(df["cos2"], bins=30); ax.set_title("[C] cos^2(ΔH, ∂logit/∂H_down) histogram")
        ax.set_xlabel("cos^2"); ax.set_ylabel("count"); fig.tight_layout()
        fig.savefig(os.path.join(outdir, "C_grad_alignment_hist.png"), dpi=150); plt.close(fig)
    logging.info("[C] Done. CSV: %s", csv_path)

def experiment_D_emergence_curve(groups: List[PromptGroup], model, tokenizer, device, outdir, max_length, batch_size, max_groups=None):
    ensure_dir(outdir)
    logging.info("[D] Layer-wise emergence curve (last-token residual input patching)...")
    layers = _get_layers_list(model); L = len(layers); rows = []
    def get_last_idx(enc): return (enc["attention_mask"].sum(dim=1) - 1).item()
    for group in tqdm(groups, desc="[D] groups", total=len(groups)):
        if max_groups is not None and len(rows) >= max_groups: break
        enc_o = tokenizer([group.original], padding=True, truncation=True, max_length=max_length, return_tensors="pt"); enc_o = to_device(enc_o, device)
        last_idx_o = get_last_idx(enc_o); layer_inputs_o = [None]*L; hooks = []
        def make_pre_hook(k):
            def pre_hook(module, inputs):
                hs = inputs[0].detach(); layer_inputs_o[k] = hs[0, last_idx_o, :].clone(); return None
            return pre_hook
        for k in range(L): hooks.append(layers[k].register_forward_pre_hook(make_pre_hook(k), with_kwargs=False))
        with torch.no_grad(): out_o = model(**enc_o, output_attentions=False, use_cache=False); logits_o = out_o.logits[:, -1, :]
        for h in hooks: h.remove()
        for para in group.paraphrases:
            enc_p = tokenizer([para.text], padding=True, truncation=True, max_length=max_length, return_tensors="pt"); enc_p = to_device(enc_p, device)
            last_idx_p = get_last_idx(enc_p)
            with torch.no_grad():
                out_p = model(**enc_p, output_attentions=False, use_cache=False); logits_p = out_p.logits[:, -1, :]; kl_base = symm_kl(logits_o, logits_p).item()
            for k in range(L):
                def pre_patch(module, inputs):
                    hs = inputs[0].clone(); hs[0, last_idx_p, :] = layer_inputs_o[k].to(hs); return (hs,) + tuple(inputs[1:])
                hook = layers[k].register_forward_pre_hook(pre_patch, with_kwargs=False)
                with torch.no_grad(): out_new = model(**enc_p, output_attentions=False, use_cache=False); logits_new = out_new.logits[:, -1, :]; kl_new = symm_kl(logits_o, logits_new).item()
                hook.remove()
                rows.append({"prompt_count": group.prompt_count, "para_key": para.key, "layer_k": k, "kl_baseline": kl_base, "kl_patched": kl_new, "delta": kl_new - kl_base})
    df = pd.DataFrame(rows); csv_path = os.path.join(outdir, "D_emergence_curve.csv"); df.to_csv(csv_path, index=False)
    if len(df) > 0:
        grp = df.groupby("layer_k")["kl_patched"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(grp["layer_k"], grp["kl_patched"], marker="o", label="patched KL")
        base_mean = df.groupby("layer_k")["kl_baseline"].mean().iloc[0]
        ax.axhline(base_mean, linestyle="--", label="baseline KL (mean)")
        ax.set_xlabel("Layer index k"); ax.set_ylabel("Mean symmetric KL"); ax.set_title("[D] Emergence curve (lower is better)"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "D_emergence_curve.png"), dpi=150); plt.close(fig)
    logging.info("[D] Done. CSV: %s", csv_path)

def zero_attention_heads_inplace(model, layer_index: int, head_indices: List[int]):
    layers = _get_layers_list(model)
    if not (0 <= layer_index < len(layers)): raise RuntimeError("Invalid layer_index")
    layer = layers[layer_index]
    attn = None
    for name in ["self_attn", "self_attention", "attn", "attention"]:
        if hasattr(layer, name): attn = getattr(layer, name); break
    if attn is None: raise RuntimeError(f"No attention module for layer {layer_index}")
    num_heads = getattr(model.config, "num_attention_heads", None); hidden_size = getattr(model.config, "hidden_size", None)
    if num_heads is None or hidden_size is None or num_heads == 0: raise RuntimeError("Need num_attention_heads and hidden_size on config.")
    head_dim = hidden_size // num_heads
    def get_proj(m, names):
        for n in names:
            if hasattr(m, n): return getattr(m, n)
        return None
    q_proj = get_proj(attn, ["q_proj", "q_linear", "q"]); k_proj = get_proj(attn, ["k_proj", "k_linear", "k"]); v_proj = get_proj(attn, ["v_proj", "v_linear", "v"]); o_proj = get_proj(attn, ["o_proj", "out_proj", "o"])
    with torch.no_grad():
        for h in head_indices:
            start = h * head_dim; end = (h+1) * head_dim
            for proj in [q_proj, k_proj, v_proj]:
                if proj is not None:
                    proj.weight[start:end, :] = 0.0
                    if getattr(proj, "bias", None) is not None: proj.bias[start:end] = 0.0
            if o_proj is not None:
                o_proj.weight[:, start:end] = 0.0

def experiment_E_head_ablation(groups, base_model, ft_model, tokenizer, device, outdir, max_length, batch_size, max_pairs=None):
    ensure_dir(outdir); import copy
    logging.info("[E] Evaluating head-level ablations at layer-6 (h4, h7)...")
    ft_kl = evaluate_mean_kl(ft_model, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
    ft_h4 = copy.deepcopy(ft_model).to(device); zero_attention_heads_inplace(ft_h4, layer_index=6, head_indices=[4])
    h4_kl = evaluate_mean_kl(ft_h4, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
    ft_h7 = copy.deepcopy(ft_model).to(device); zero_attention_heads_inplace(ft_h7, layer_index=6, head_indices=[7])
    h7_kl = evaluate_mean_kl(ft_h7, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
    ft_h47 = copy.deepcopy(ft_model).to(device); zero_attention_heads_inplace(ft_h47, layer_index=6, head_indices=[4,7])
    h47_kl = evaluate_mean_kl(ft_h47, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
    df = pd.DataFrame([{"variant":"FT","mean_KL":ft_kl},{"variant":"FT (−h4)","mean_KL":h4_kl},{"variant":"FT (−h7)","mean_KL":h7_kl},{"variant":"FT (−h4,−h7)","mean_KL":h47_kl}])
    csv_path = os.path.join(outdir, "E_head_ablation.csv"); df.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(7,4)); ax.bar(df["variant"], df["mean_KL"]); ax.set_ylabel("Mean symmetric KL"); ax.set_title("[E] Head ablation at layer-6")
    for i, v in enumerate(df["mean_KL"]): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "E_head_ablation.png"), dpi=150); plt.close(fig)
    logging.info("[E] Done. CSV: %s", csv_path)

def experiment_F_singular_paths(groups, ft_model, tokenizer, device, outdir, max_length, batch_size, max_groups=None, top_m: int=16):
    ensure_dir(outdir); logging.info("[F] Singular-direction path test through DOWN (layer-6)...")
    h = resolve_layer_handles(ft_model, 6); W = np32(h.down_proj.weight)
    try: U, S, VT = np.linalg.svd(W, full_matrices=False)
    except np.linalg.LinAlgError: logging.warning("[F] SVD failed on DOWN weights; skipping."); return
    U_left = U; rows = []
    for group in tqdm(groups, desc="[F] groups", total=len(groups)):
        if max_groups is not None and len(rows) >= max_groups: break
        if not group.paraphrases: continue
        texts = [group.original, group.paraphrases[0].text]
        cap = forward_with_capture(ft_model, tokenizer, texts, max_length=max_length, device=device)
        down_orig = np32(cap.down_last[0]); down_para = np32(cap.down_last[1]); dH = down_para - down_orig
        k = min(top_m, U_left.shape[1]); U_top = U_left[:, :k]
        def proj_U(vec, Ucols): 
            if Ucols.size == 0: return np.zeros_like(vec)
            coeff = Ucols.T @ vec; return Ucols @ coeff
        dH_top = proj_U(dH, U_top); dH_comp = dH - dH_top
        def patch_and_kl(new_dH):
            new_down = down_orig + new_dH
            enc_p = tokenizer([texts[1]], padding=True, truncation=True, max_length=max_length, return_tensors="pt"); enc_p = to_device(enc_p, device)
            last_idx = (enc_p["attention_mask"].sum(dim=1) - 1).item(); handles = resolve_layer_handles(ft_model, 6)
            def down_hook(module, inputs, output):
                base = output[0] if isinstance(output, (tuple, list)) else output
                out = base.clone(); out[0, last_idx, :] = torch.from_numpy(new_down).to(out)
                if isinstance(output, (tuple, list)):
                    if isinstance(output, tuple): return (out,) + tuple(output[1:])
                    output[0] = out; return output
                return out
            hook = (handles.down_proj or handles.mlp_module).register_forward_hook(down_hook)
            with torch.no_grad(): out_new = ft_model(**enc_p, output_attentions=False, use_cache=False); logits_new = out_new.logits[:, -1, :]
            hook.remove()
            enc_o = tokenizer([texts[0]], padding=True, truncation=True, max_length=max_length, return_tensors="pt"); enc_o = to_device(enc_o, device)
            with torch.no_grad(): out_o = ft_model(**enc_o, output_attentions=False, use_cache=False)
            return symm_kl(out_o.logits[:, -1, :], logits_new).item()
        kl_top = patch_and_kl(dH_top); kl_comp = patch_and_kl(dH_comp)
        rows.append({"prompt_count": group.prompt_count, "k_top": k, "kl_top": kl_top, "kl_comp": kl_comp})
    df = pd.DataFrame(rows); csv_path = os.path.join(outdir, "F_singular_paths.csv"); df.to_csv(csv_path, index=False)
    if len(df) > 0:
        means = {"Top-m": df["kl_top"].mean(), "Complement": df["kl_comp"].mean()}
        fig, ax = plt.subplots(figsize=(6,4)); ax.bar(list(means.keys()), list(means.values())); ax.set_ylabel("Mean symmetric KL"); ax.set_title(f"[F] Singular path test through DOWN (m={top_m})")
        for i, v in enumerate(means.values()): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "F_singular_paths.png"), dpi=150); plt.close(fig)
    logging.info("[F] Done. CSV: %s", csv_path)

def experiment_G_tf_link(groups, ft_model, tokenizer, device, outdir, max_length, batch_size, lam_ridge=1e-3, topk_vocab=128, max_groups=None):
    ensure_dir(outdir); logging.info("[G] Orthogonal routing fraction vs. ΔKL proxy...]")
    rows = []; processed = 0
    for group in tqdm(groups, desc="[G] groups", total=len(groups)):
        if max_groups is not None and processed >= max_groups: break
        texts_all = [group.original] + [p.text for p in group.paraphrases]
        caps = []
        for batch in batched(texts_all, n=batch_size):
            cap = forward_with_capture(ft_model, tokenizer, batch, max_length=max_length, device=device); caps.append(cap)
        logits_last = torch.cat([c.logits_last for c in caps], dim=0); down_last = torch.cat([c.down_last for c in caps], dim=0)
        with torch.no_grad():
            topk = min(topk_vocab, logits_last.shape[-1]); top_ids = torch.topk(logits_last, k=topk, dim=-1).indices; vocab_ids = torch.unique(top_ids.flatten()).tolist()
        orig_v = np32(logits_last[0:1, vocab_ids]); para_v = np32(logits_last[1:, vocab_ids]); dlogits = (para_v - orig_v)
        dH = np32(down_last[1:, :]) - np32(down_last[0:1, :])
        W = fit_ridge_map(dH, dlogits, lam=lam_ridge)
        try: U, S, VT = np.linalg.svd(W, full_matrices=False)
        except np.linalg.LinAlgError: processed += 1; continue
        k = min(8, U.shape[1]); U_sens = orthonormal_basis_from_cols(U[:, :k])
        orig_logits = logits_last[0:1, :]
        for i, para in enumerate(group.paraphrases):
            d = dH[i]; d_sens, d_orth = project_components(d, U_sens); frac_orth = float(np.linalg.norm(d_orth) / (np.linalg.norm(d) + 1e-9))
            enc = tokenizer([para.text], padding=True, truncation=True, max_length=max_length, return_tensors="pt"); enc = to_device(enc, device)
            last_idx = enc["attention_mask"].sum(dim=1) - 1; handles = resolve_layer_handles(ft_model, 6)
            base_down_np = np32(down_last[0:1, :])[0]; new_down = (base_down_np + d_orth).astype(np.float32)
            def down_hook(module, inputs, output):
                base = output[0] if isinstance(output, (tuple, list)) else output
                out = base.clone(); out[0, last_idx.item(), :] = torch.from_numpy(new_down).to(out)
                if isinstance(output, (tuple, list)):
                    if isinstance(output, tuple): return (out,) + tuple(output[1:])
                    output[0] = out; return output
                return out
            hook = (handles.down_proj or handles.mlp_module).register_forward_hook(down_hook)
            with torch.no_grad(): out_new = ft_model(**enc, output_attentions=False, use_cache=False)
            hook.remove()
            with torch.no_grad():
                cap_p = forward_with_capture(ft_model, tokenizer, [para.text], max_length=max_length, device=device)
                kl_base = symm_kl(orig_logits, cap_p.logits_last).item(); kl_new = symm_kl(orig_logits, out_new.logits[:, -1, :]).item(); delta = kl_new - kl_base
            rows.append({"prompt_count": group.prompt_count, "para_key": para.key, "frac_orth": frac_orth, "delta_KL_remove_sens_minus_base": delta})
        processed += 1
    df = pd.DataFrame(rows); csv_path = os.path.join(outdir, "G_orth_fraction_vs_deltaKL.csv"); df.to_csv(csv_path, index=False)
    if len(df) > 0:
        fig, ax = plt.subplots(figsize=(5,4)); ax.scatter(df["frac_orth"], df["delta_KL_remove_sens_minus_base"], s=8, alpha=0.6)
        ax.set_xlabel("Fraction of ΔH at DOWN in orthogonal subspace"); ax.set_ylabel("ΔKL (remove-sens − base)"); ax.set_title("[G] Orthogonal routing vs ΔKL proxy")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "G_scatter_fracOrth_vs_deltaKL.png"), dpi=150); plt.close(fig)
        corr = df[["frac_orth", "delta_KL_remove_sens_minus_base"]].corr().iloc[0,1]
        with open(os.path.join(outdir, "G_correlation.txt"), "w") as f: f.write(f"Pearson corr(frac_orth, ΔKL_remove_sens_minus_base) = {corr:.4f}\n")
    logging.info("[G] Done. CSV: %s", csv_path)


def parse_layers_arg(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x.strip() != ""]

# A_sweep: subspace patching at arbitrary layers
def experiment_A_sweep(groups: List[PromptGroup], ft_model, tokenizer, device, outdir, max_length, batch_size, topk_vocab, lam_ridge, layers: List[int], max_groups=None):
    ensure_dir(outdir); logging.info(f"[A_sweep] Layers={layers}")
    for k in layers:
        subdir = os.path.join(outdir, f"layer_{k}"); ensure_dir(subdir)
        rows = []; processed = 0
        for group in tqdm(groups, desc=f"[A_sweep k={k}] groups", total=len(groups)):
            if max_groups is not None and processed >= max_groups: break
            texts_all = [group.original] + [p.text for p in group.paraphrases]
            caps = []
            for batch in batched(texts_all, n=batch_size):
                cap = forward_with_capture_at_layer(ft_model, tokenizer, batch, k, max_length, device); caps.append(cap)
            logits_last = torch.cat([c.logits_last for c in caps], dim=0); down_last = torch.cat([c.down_last for c in caps], dim=0)
            orig_logits = logits_last[0:1, :]; orig_down = down_last[0:1, :]; para_logits = logits_last[1:, :]; para_down = down_last[1:, :]
            with torch.no_grad():
                topk = min(topk_vocab, logits_last.shape[-1]); top_ids = torch.topk(logits_last, k=topk, dim=-1).indices; vocab_ids = torch.unique(top_ids.flatten()).tolist()
                orig_v = np32(orig_logits[:, vocab_ids]); para_v = np32(para_logits[:, vocab_ids])
            dlogits = (para_v - orig_v); dH = np32(para_down) - np32(orig_down)
            W = fit_ridge_map(dH, dlogits, lam=lam_ridge)
            try: U, S, VT = np.linalg.svd(W, full_matrices=False)
            except np.linalg.LinAlgError: processed += 1; continue
            U_sens = orthonormal_basis_from_cols(U[:, :min(8, U.shape[1])])
            orig_cap = forward_with_capture_at_layer(ft_model, tokenizer, [group.original], k, max_length, device)
            down_orig = np32(orig_cap.down_last[0]); logits_orig = orig_cap.logits_last[0].unsqueeze(0)
            for para in group.paraphrases:
                cap_p = forward_with_capture_at_layer(ft_model, tokenizer, [para.text], k, max_length, device)
                logits_p = cap_p.logits_last; down_p = np32(cap_p.down_last[0]); dH_p = down_p - down_orig
                dH_sens, dH_orth = project_components(dH_p, U_sens)
                def run_with_down_patch(new_down_vec_np: np.ndarray) -> torch.Tensor:
                    new_down_vec = torch.from_numpy(new_down_vec_np).to(device=device, dtype=cap_p.down_last.dtype)
                    handles = resolve_layer_handles(ft_model, k)
                    enc = tokenizer([para.text], padding=True, truncation=True, max_length=max_length, return_tensors="pt"); enc = to_device(enc, device)
                    last_idx = enc["attention_mask"].sum(dim=1) - 1
                    def down_hook(module, inputs, output):
                        base = output[0] if isinstance(output, (tuple, list)) else output
                        out = base.clone(); out[0, last_idx.item(), :] = new_down_vec
                        if isinstance(output, (tuple, list)):
                            if isinstance(output, tuple): return (out,) + tuple(output[1:])
                            output[0] = out; return output
                        return out
                    hook = (handles.down_proj or handles.mlp_module).register_forward_hook(down_hook)
                    with torch.no_grad(): out = ft_model(**enc, output_attentions=False, use_cache=False); logits_new = out.logits[:, -1, :]
                    hook.remove(); return logits_new
                logits_rm_sens = run_with_down_patch(down_orig + dH_orth); logits_rm_orth = run_with_down_patch(down_orig + dH_sens)
                with torch.no_grad():
                    kl_base = symm_kl(logits_orig, logits_p).item(); kl_rm_sens = symm_kl(logits_orig, logits_rm_sens).item(); kl_rm_orth = symm_kl(logits_orig, logits_rm_orth).item()
                rows.append({"prompt_count": group.prompt_count, "para_key": para.key, "layer": k, "kl_ft": kl_base, "kl_remove_sens": kl_rm_sens, "kl_remove_orth": kl_rm_orth})
            processed += 1
        df = pd.DataFrame(rows); csv = os.path.join(subdir, "A_sweep_results.csv"); df.to_csv(csv, index=False)
        if len(df) > 0:
            means = df.groupby("layer")[["kl_ft","kl_remove_sens","kl_remove_orth"]].mean().iloc[0].to_dict()
            fig, ax = plt.subplots(figsize=(6,4)); ax.bar(list(means.keys()), list(means.values()))
            ax.set_ylabel("Mean symmetric KL"); ax.set_title(f"[A_sweep] Layer {k}")
            for i, v in enumerate(means.values()): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
            fig.tight_layout(); fig.savefig(os.path.join(subdir, "A_sweep_bars.png"), dpi=150); plt.close(fig)
        logging.info("[A_sweep] Done for layer %d. CSV: %s", k, csv)

# B_sweep: weight-level ablations at arbitrary layers
def experiment_B_sweep(groups: List[PromptGroup], base_model, ft_model, tokenizer, device, outdir, max_length, batch_size, layers: List[int], max_pairs=None):
    ensure_dir(outdir); import copy
    for k in layers:
        subdir = os.path.join(outdir, f"layer_{k}"); ensure_dir(subdir)
        logging.info(f"[B_sweep] layer {k}: variants eval...")
        base_kl = evaluate_mean_kl(base_model, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
        ft_kl = evaluate_mean_kl(ft_model, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
        ft_minus_down = copy.deepcopy(ft_model).to(device); swap_layer_weights(ft_minus_down, base_model, "DOWN", k)
        ft_minus_down_kl = evaluate_mean_kl(ft_minus_down, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
        base_plus_down = copy.deepcopy(base_model).to(device); swap_layer_weights(base_plus_down, ft_model, "DOWN", k)
        only_down_kl = evaluate_mean_kl(base_plus_down, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
        ft_minus_attn = copy.deepcopy(ft_model).to(device); swap_layer_weights(ft_minus_attn, base_model, "ATTN", k)
        ft_minus_attn_kl = evaluate_mean_kl(ft_minus_attn, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
        base_plus_attn = copy.deepcopy(base_model).to(device); swap_layer_weights(base_plus_attn, ft_model, "ATTN", k)
        only_attn_kl = evaluate_mean_kl(base_plus_attn, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
        df = pd.DataFrame([
            {"variant":"BASE","mean_KL":base_kl},
            {"variant":"FT","mean_KL":ft_kl},
            {"variant":"FT−DOWN","mean_KL":ft_minus_down_kl},
            {"variant":"only-DOWN","mean_KL":only_down_kl},
            {"variant":"FT−ATTN","mean_KL":ft_minus_attn_kl},
            {"variant":"only-ATTN","mean_KL":only_attn_kl},
        ])
        csv = os.path.join(subdir, "B_sweep_variants.csv"); df.to_csv(csv, index=False)
        fig, ax = plt.subplots(figsize=(8,4)); ax.bar(df["variant"], df["mean_KL"]); ax.set_ylabel("Mean symmetric KL"); ax.set_title(f"[B_sweep] layer {k}")
        for i, v in enumerate(df["mean_KL"]): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
        fig.tight_layout(); fig.savefig(os.path.join(subdir, "B_sweep_variants.png"), dpi=150); plt.close(fig)
        logging.info("[B_sweep] Done for layer %d. CSV: %s", k, csv)

# C_sweep: gradient alignment at arbitrary layers
def experiment_C_sweep(groups: List[PromptGroup], model, tokenizer, device, outdir, max_length, batch_size, layers: List[int], max_groups=None):
    ensure_dir(outdir)
    for k in layers:
        subdir = os.path.join(outdir, f"layer_{k}"); ensure_dir(subdir)
        logging.info(f"[C_sweep] Gradient alignment at layer {k}...")
        rows = []; processed = 0
        def _enable_param_grads(m): flags = [p.requires_grad for p in m.parameters()]; [p.requires_grad_(True) for p in m.parameters()]; return flags
        def _restore_param_grads(m, flags): [p.requires_grad_(f) for p, f in zip(m.parameters(), flags)]
        for group in tqdm(groups, desc=f"[C_sweep k={k}] groups", total=len(groups)):
            if max_groups is not None and processed >= max_groups: break
            enc = tokenizer([group.original], padding=True, truncation=True, max_length=max_length, return_tensors="pt"); enc = to_device(enc, device)
            last_idx = (enc["attention_mask"].sum(dim=1) - 1).item()
            handles = resolve_layer_handles(model, k); bag = {"node": None}
            def fwd_hook(module, inputs, output):
                base = output[0] if isinstance(output, (tuple, list)) else output
                out2 = base.detach().requires_grad_(True)
                bag["node"] = out2
                if isinstance(output, (tuple, list)):
                    if isinstance(output, tuple): return (out2,) + tuple(output[1:])
                    output[0] = out2; return output
                return out2
            hook = (handles.down_proj or handles.mlp_module).register_forward_hook(fwd_hook)
            prev_flags = _enable_param_grads(model)
            try:
                with torch.enable_grad():
                    out = model(**enc, output_attentions=False, use_cache=False); logits = out.logits[:, -1, :]
                    target_id = logits.argmax(dim=-1); target_logit = logits[0, target_id]
                    g_full = torch.autograd.grad(target_logit, bag["node"], retain_graph=False, create_graph=False, allow_unused=False)[0]
                hook.remove()
            finally:
                _restore_param_grads(model, prev_flags)
            if g_full is None: processed += 1; continue
            g_vec = g_full[0, last_idx, :].detach().to(torch.float32).cpu().numpy(); g_u = g_vec / (np.linalg.norm(g_vec)+1e-9)
            cap_orig = forward_with_capture_at_layer(model, tokenizer, [group.original], k, max_length, device)
            for para in group.paraphrases:
                cap_p = forward_with_capture_at_layer(model, tokenizer, [para.text], k, max_length, device)
                dH = (cap_p.down_last[0] - cap_orig.down_last[0]).detach().to(torch.float32).cpu().numpy()
                dH_u = dH / (np.linalg.norm(dH)+1e-9); cos = float(np.dot(dH_u, g_u))
                rows.append({"prompt_count": group.prompt_count, "para_key": para.key, "layer": k, "cos": cos, "cos2": cos*cos})
            processed += 1
        df = pd.DataFrame(rows); csv = os.path.join(subdir, "C_sweep_grad_alignment.csv"); df.to_csv(csv, index=False)
        if len(df)>0:
            fig, ax = plt.subplots(figsize=(6,4)); ax.hist(df["cos2"], bins=30); ax.set_title(f"[C_sweep] cos^2 @ layer {k}")
            ax.set_xlabel("cos^2"); ax.set_ylabel("count"); fig.tight_layout(); fig.savefig(os.path.join(subdir, "C_sweep_hist.png"), dpi=150); plt.close(fig)
        logging.info("[C_sweep] Done for layer %d. CSV: %s", k, csv)

# H_rmsnorm: ablate RMSNorm at chosen layers by bypassing normalization (identity)
def _collect_norm_modules(layer: nn.Module) -> List[nn.Module]:
    norms = []
    for name, mod in layer.named_modules():
        cls = mod.__class__.__name__.lower()
        if "norm" in cls:
            norms.append(mod)
    return norms

def _bypass_forward_factory():
    def _forward_identity(x, *args, **kwargs):
        return x
    return _forward_identity

def experiment_H_rmsnorm(groups: List[PromptGroup], model, tokenizer, device, outdir, max_length, batch_size, layers: List[int], max_pairs=None):
    ensure_dir(outdir); import copy
    for k in layers:
        subdir = os.path.join(outdir, f"layer_{k}"); ensure_dir(subdir)
        m = copy.deepcopy(model).to(device)
        # Replace all *norm* modules inside the chosen layer with identity forwards
        lay = _get_layers_list(m)[k]
        norms = _collect_norm_modules(lay)
        originals = []
        for mod in norms:
            originals.append((mod, mod.forward))
            mod.forward = _bypass_forward_factory()
        logging.info(f"[H_rmsnorm] Layer {k}: bypassed {len(norms)} norm modules.")
        kl = evaluate_mean_kl(m, tokenizer, groups, device, max_length, batch_size, max_pairs=max_pairs)
        # restore (not strictly needed after deepcopy usage)
        for mod, f in originals:
            mod.forward = f
        with open(os.path.join(subdir, "H_rmsnorm.txt"), "w") as f:
            f.write(f"Layer {k}: mean KL after bypassing norms = {kl:.6f}\n")
        logging.info("[H_rmsnorm] Done for layer %d. KL=%.4f", k, kl)

# I_attn: patch attention output last-token with original's vector
def _replace_first_in_structure(orig_output, new_tensor):
    """Return a structure of the same type as orig_output, with the first tensor replaced by new_tensor."""
    if torch.is_tensor(orig_output):
        return new_tensor
    if isinstance(orig_output, tuple):
        return (new_tensor,) + tuple(orig_output[1:])
    if isinstance(orig_output, list):
        out = list(orig_output)
        out[0] = new_tensor
        return out
    # Fallback: just return new_tensor (shouldn't happen for HF attention modules)
    return new_tensor

def _first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output

def experiment_I_attn_patch(groups: List[PromptGroup], model, tokenizer, device, outdir, max_length, batch_size, layers: List[int], max_groups=None):
    ensure_dir(outdir)
    for k in layers:
        subdir = os.path.join(outdir, f"layer_{k}"); ensure_dir(subdir)
        logging.info(f"[I_attn] Attention-output patching at layer {k}...")
        rows = []; processed = 0
        for group in tqdm(groups, desc=f"[I_attn k={k}] groups", total=len(groups)):
            if max_groups is not None and processed >= max_groups: break
            # Capture original attn OUT last-token
            enc_o = tokenizer([group.original], padding=True, truncation=True, max_length=max_length, return_tensors="pt"); enc_o = to_device(enc_o, device)
            last_idx_o = (enc_o["attention_mask"].sum(dim=1) - 1).item()
            handles = resolve_layer_handles(model, k)
            store = {"attn_out": None}
            def attn_hook_capture(module, inputs, output):
                base = _first_tensor(output)
                store["attn_out"] = base.detach()[0, last_idx_o, :].clone()
                return output
            hook_cap = handles.attn_module.register_forward_hook(attn_hook_capture)
            with torch.no_grad(): out_o = model(**enc_o, output_attentions=False, use_cache=False); logits_o = out_o.logits[:, -1, :]
            hook_cap.remove()

            for para in group.paraphrases:
                enc_p = tokenizer([para.text], padding=True, truncation=True, max_length=max_length, return_tensors="pt"); enc_p = to_device(enc_p, device)
                last_idx_p = (enc_p["attention_mask"].sum(dim=1) - 1).item()
                def attn_hook_patch(module, inputs, output):
                    base = _first_tensor(output).clone()
                    base[0, last_idx_p, :] = store["attn_out"].to(base)
                    return _replace_first_in_structure(output, base)
                hook = handles.attn_module.register_forward_hook(attn_hook_patch)
                with torch.no_grad(): out_new = model(**enc_p, output_attentions=False, use_cache=False); logits_new = out_new.logits[:, -1, :]
                hook.remove()
                with torch.no_grad(): out_p = model(**enc_p, output_attentions=False, use_cache=False); logits_p = out_p.logits[:, -1, :]
                rows.append({
                    "prompt_count": group.prompt_count,
                    "para_key": para.key,
                    "layer": k,
                    "kl_base": symm_kl(logits_o, logits_p).item(),
                    "kl_patched": symm_kl(logits_o, logits_new).item()
                })
            processed += 1
        df = pd.DataFrame(rows); csv = os.path.join(subdir, "I_attn_patch.csv"); df.to_csv(csv, index=False)
        if len(df)>0:
            grp = df.groupby("layer")[["kl_base","kl_patched"]].mean().iloc[0].to_dict()
            fig, ax = plt.subplots(figsize=(6,4)); ax.bar(list(grp.keys()), list(grp.values()))
            ax.set_ylabel("Mean symmetric KL"); ax.set_title(f"[I_attn] layer {k}")
            for i, v in enumerate(grp.values()): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
            fig.tight_layout(); fig.savefig(os.path.join(subdir, "I_attn_patch.png"), dpi=150); plt.close(fig)
        logging.info("[I_attn] Done for layer %d. CSV: %s", k, csv)

# Markdown summary generator
def _mean_ci(series: pd.Series):
    s = series.dropna().astype(float)
    if len(s) == 0:
        return (float("nan"), float("nan"), 0)
    m = float(s.mean())
    if len(s) == 1:
        return (m, float("nan"), 1)
    # 95% CI via normal approximation
    sem = float(s.std(ddof=1)) / (len(s) ** 0.5)
    ci = 1.96 * sem
    return (m, ci, len(s))

def _fmt_mean_ci(m, ci, n, digits=4):
    if (m != m):  # NaN check
        return "n/a"
    if (ci != ci):  # NaN
        return f"{m:.{digits}f} (n={n})"
    return f"{m:.{digits}f} ± {ci:.{digits}f} (n={n})"

def _read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _glob_layer_subdirs(root):
    subs = []
    if not os.path.isdir(root):
        return subs
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.startswith("layer_"):
            try:
                k = int(name.split("_", 1)[1])
            except Exception:
                continue
            subs.append((k, p))
    subs.sort()
    return subs

def write_markdown_summary(outdir: str):
    """
    Collates all experiment outputs that exist and writes <outdir>/SUMMARY.md
    with compact tables of the key numbers so plots are optional.
    """
    lines = []
    add = lines.append
    add(f"# Robustness Proof Experiments – Statistical Summary\n")
    add(f"_Output directory: `{outdir}`_\n")

    # A] Subspace patching
    a_dir = os.path.join(outdir, "A_subspace_patching")
    a_csv = os.path.join(a_dir, "A_subspace_patching_results.csv")
    dfA = _read_csv_safe(a_csv)
    add("## A. Subspace patching (layer 6)\n")
    if dfA is not None and len(dfA):
        for col in ["kl_ft", "kl_remove_sens", "kl_remove_orth"]:
            m, ci, n = _mean_ci(dfA[col])
            add(f"- **{col}**: {_fmt_mean_ci(m, ci, n)}")
        add("")
    else:
        add("_No A results found._\n")

    # B] Weight-level ablations (layer 6)
    b_dir = os.path.join(outdir, "B_weight_level")
    b_csv = os.path.join(b_dir, "B_weight_level_variants.csv")
    dfB = _read_csv_safe(b_csv)
    add("## B. Weight-level ablations (layer 6)\n")
    if dfB is not None and len(dfB):
        add("| Variant | Mean KL |")
        add("|---|---:|")
        for _, r in dfB.iterrows():
            add(f"| {r['variant']} | {r['mean_KL']:.4f} |")
        add("")
    else:
        add("_No B results found._\n")

    # C] Grad alignment (layer 6)
    c_dir = os.path.join(outdir, "C_grad_alignment")
    c_csv = os.path.join(c_dir, "C_grad_alignment.csv")
    dfC = _read_csv_safe(c_csv)
    add("## C. Gradient alignment (layer 6)\n")
    if dfC is not None and len(dfC):
        for col in ["cos", "cos2"]:
            m, ci, n = _mean_ci(dfC[col])
            add(f"- **mean {col}**: {_fmt_mean_ci(m, ci, n)}")
        add("")
    else:
        add("_No C results found._\n")

    # D] Emergence curve (per-layer deltas)
    d_dir = os.path.join(outdir, "D_emergence_curve")
    d_csv = os.path.join(d_dir, "D_emergence_curve.csv")
    dfD = _read_csv_safe(d_csv)
    add("## D. Emergence curve\n")
    if dfD is not None and len(dfD):
        grp = dfD.groupby("layer_k").agg(
            kl_patched_mean=("kl_patched", "mean"),
            kl_baseline_mean=("kl_baseline", "mean"),
            delta_mean=("delta", "mean"),
            n=("delta", "size"),
        ).reset_index()
        add("| Layer k | Patched KL (mean) | Baseline KL (mean) | Δ (mean) | n |")
        add("|---:|---:|---:|---:|---:|")
        for _, r in grp.iterrows():
            add(f"| {int(r['layer_k'])} | {r['kl_patched_mean']:.4f} | {r['kl_baseline_mean']:.4f} | {r['delta_mean']:.4f} | {int(r['n'])} |")
        # Where is delta minimal?
        best = grp.loc[grp["kl_patched_mean"].idxmin()]
        add(f"\n- **Lowest mean patched KL at layer k={int(best['layer_k'])}** (patched={best['kl_patched_mean']:.4f}, baseline={best['kl_baseline_mean']:.4f}).\n")
    else:
        add("_No D results found._\n")

    # E] Head ablation (layer 6)
    e_dir = os.path.join(outdir, "E_head_ablation")
    e_csv = os.path.join(e_dir, "E_head_ablation.csv")
    dfE = _read_csv_safe(e_csv)
    add("## E. Head ablation (layer 6)\n")
    if dfE is not None and len(dfE):
        add("| Variant | Mean KL |")
        add("|---|---:|")
        for _, r in dfE.iterrows():
            add(f"| {r['variant']} | {r['mean_KL']:.4f} |")
        add("")
    else:
        add("_No E results found._\n")

    # F] Singular paths
    f_dir = os.path.join(outdir, "F_singular_paths")
    f_csv = os.path.join(f_dir, "F_singular_paths.csv")
    dfF = _read_csv_safe(f_csv)
    add("## F. Singular-direction path test (layer 6)\n")
    if dfF is not None and len(dfF):
        m_top, ci_top, n_top = _mean_ci(dfF["kl_top"])
        m_comp, ci_comp, n_comp = _mean_ci(dfF["kl_comp"])
        add(f"- **KL (Top-m)**: {_fmt_mean_ci(m_top, ci_top, n_top)}")
        add(f"- **KL (Complement)**: {_fmt_mean_ci(m_comp, ci_comp, n_comp)}\n")
    else:
        add("_No F results found._\n")

    # G] Orthogonal fraction vs ΔKL
    g_dir = os.path.join(outdir, "G_tf_link")
    g_csv = os.path.join(g_dir, "G_orth_fraction_vs_deltaKL.csv")
    dfG = _read_csv_safe(g_csv)
    add("## G. Orthogonal fraction vs ΔKL\n")
    if dfG is not None and len(dfG):
        try:
            corr = float(dfG[["frac_orth", "delta_KL_remove_sens_minus_base"]].corr().iloc[0,1])
        except Exception:
            corr = float("nan")
        m, ci, n = _mean_ci(dfG["delta_KL_remove_sens_minus_base"])
        add(f"- **Pearson corr(frac_orth, ΔKL_remove_sens−base)**: {corr:.4f}")
        add(f"- **ΔKL_remove_sens−base (mean)**: {_fmt_mean_ci(m, ci, n)}\n")
    else:
        add("_No G results found._\n")

    # A_sweep] per-layer summary
    a_sw_root = os.path.join(outdir, "A_sweep")
    add("## A_sweep. Subspace patching across layers\n")
    if os.path.isdir(a_sw_root):
        rows = []
        for k, p in _glob_layer_subdirs(a_sw_root):
            csv = os.path.join(p, "A_sweep_results.csv")
            df = _read_csv_safe(csv)
            if df is None or not len(df): continue
            m_ft, ci_ft, n = _mean_ci(df["kl_ft"])
            m_rs, ci_rs, _ = _mean_ci(df["kl_remove_sens"])
            m_ro, ci_ro, _ = _mean_ci(df["kl_remove_orth"])
            rows.append((k, _fmt_mean_ci(m_ft, ci_ft, n), _fmt_mean_ci(m_rs, ci_rs, n), _fmt_mean_ci(m_ro, ci_ro, n)))
        if rows:
            add("| Layer | KL_ft | KL_remove_sens | KL_remove_orth |")
            add("|---:|---:|---:|---:|")
            for k, s1, s2, s3 in rows:
                add(f"| {k} | {s1} | {s2} | {s3} |")
            add("")
        else:
            add("_No A_sweep results found._\n")
    else:
        add("_No A_sweep directory._\n")

    # B_sweep] per-layer summary
    b_sw_root = os.path.join(outdir, "B_sweep")
    add("## B_sweep. Weight-level ablations across layers\n")
    if os.path.isdir(b_sw_root):
        header_done = False
        for k, p in _glob_layer_subdirs(b_sw_root):
            csv = os.path.join(p, "B_sweep_variants.csv")
            df = _read_csv_safe(csv)
            if df is None or not len(df): continue
            if not header_done:
                add("| Layer | Variant | Mean KL |")
                add("|---:|---|---:|")
                header_done = True
            for _, r in df.iterrows():
                add(f"| {k} | {r['variant']} | {r['mean_KL']:.4f} |")
        if header_done:
            add("")
        else:
            add("_No B_sweep results found._\n")
    else:
        add("_No B_sweep directory._\n")

    # C_sweep] per-layer summary
    c_sw_root = os.path.join(outdir, "C_sweep")
    add("## C_sweep. Gradient alignment across layers\n")
    if os.path.isdir(c_sw_root):
        rows = []
        for k, p in _glob_layer_subdirs(c_sw_root):
            csv = os.path.join(p, "C_sweep_grad_alignment.csv")
            df = _read_csv_safe(csv)
            if df is None or not len(df): continue
            m1, ci1, n1 = _mean_ci(df["cos"])
            m2, ci2, n2 = _mean_ci(df["cos2"])
            rows.append((k, _fmt_mean_ci(m1, ci1, n1), _fmt_mean_ci(m2, ci2, n2)))
        if rows:
            add("| Layer | mean cos | mean cos^2 |")
            add("|---:|---:|---:|")
            for k, s1, s2 in rows:
                add(f"| {k} | {s1} | {s2} |")
            add("")
        else:
            add("_No C_sweep results found._\n")
    else:
        add("_No C_sweep directory._\n")

    # H_rmsnorm] per-layer summary (reads txt files)
    h_root = os.path.join(outdir, "H_rmsnorm")
    add("## H_rmsnorm. RMSNorm bypass across layers\n")
    if os.path.isdir(h_root):
        rows = []
        for k, p in _glob_layer_subdirs(h_root):
            txt = os.path.join(p, "H_rmsnorm.txt")
            if os.path.exists(txt):
                try:
                    with open(txt, "r") as f:
                        line = f.read().strip()
                except Exception:
                    line = ""
                rows.append((k, line))
        if rows:
            add("| Layer | Result |")
            add("|---:|---|")
            for k, t in rows:
                add(f"| {k} | {t} |")
            add("")
        else:
            add("_No H_rmsnorm results found._\n")
    else:
        add("_No H_rmsnorm directory._\n")

    # I_attn_patch] per-layer summary
    i_root = os.path.join(outdir, "I_attn_patch")
    add("## I_attn_patch. Attention-output patching across layers\n")
    if os.path.isdir(i_root):
        rows = []
        for k, p in _glob_layer_subdirs(i_root):
            csv = os.path.join(p, "I_attn_patch.csv")
            df = _read_csv_safe(csv)
            if df is None or not len(df): continue
            # average over rows
            m_b, ci_b, n_b = _mean_ci(df["kl_base"])
            m_p, ci_p, n_p = _mean_ci(df["kl_patched"])
            rows.append((k, _fmt_mean_ci(m_b, ci_b, n_b), _fmt_mean_ci(m_p, ci_p, n_p)))
        if rows:
            add("| Layer | KL_base (mean±CI) | KL_patched (mean±CI) |")
            add("|---:|---:|---:|")
            for k, sb, sp in rows:
                add(f"| {k} | {sb} | {sp} |")
            add("")
        else:
            add("_No I_attn_patch results found._\n")
    else:
        add("_No I_attn_patch directory._\n")

    # Write file
    md_path = os.path.join(outdir, "SUMMARY.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logging.info("Wrote Markdown summary: %s", md_path)

# CLI / Runner

def main():
    parser = argparse.ArgumentParser(description="Layer-6 invariance causal proof suite (+sweeps & new analyses)")
    parser.add_argument("--data_jsonl", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--base_model", type=str, required=True, help="HF model path/name for BASE")
    parser.add_argument("--ft_model", type=str, default=None, help="HF model path/name for FT host (defaults to BASE)")
    parser.add_argument("--ft_lora_path", type=str, default=None, help="Optional PEFT adapter path for FT")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA into FT weights")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto","cpu","cuda"], help="Device")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto","fp32","fp16","bf16"], help="Torch dtype")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_groups", type=int, default=None, help="Limit number of prompt groups for debugging")
    parser.add_argument("--max_paraphrases", type=int, default=16, help="Limit paraphrases per group")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--do_A", action="store_true")
    parser.add_argument("--do_B", action="store_true")
    parser.add_argument("--do_C", action="store_true")
    parser.add_argument("--do_D", action="store_true")
    parser.add_argument("--do_E", action="store_true")
    parser.add_argument("--do_F", action="store_true")
    parser.add_argument("--do_G", action="store_true")
    parser.add_argument("--topk_vocab", type=int, default=128)
    parser.add_argument("--ridge_lam", type=float, default=1e-3)
    parser.add_argument("--singular_top_m", type=int, default=16)
    parser.add_argument("--max_pairs_eval", type=int, default=None, help="Optional cap on #pairs for fast eval in B/E")
    parser.add_argument("--do_A_sweep", action="store_true", help="Run subspace-patching at arbitrary layers")
    parser.add_argument("--A_layers", type=str, default="6", help="Comma-separated layer indices for A_sweep")
    parser.add_argument("--do_B_sweep", action="store_true", help="Run weight-level ablations at arbitrary layers")
    parser.add_argument("--B_layers", type=str, default="6", help="Comma-separated layer indices for B_sweep")
    parser.add_argument("--do_C_sweep", action="store_true", help="Run grad-alignment at arbitrary layers")
    parser.add_argument("--C_layers", type=str, default="6", help="Comma-separated layer indices for C_sweep")
    parser.add_argument("--do_H_rmsnorm", action="store_true", help="Bypass RMSNorms at chosen layers and eval KL")
    parser.add_argument("--H_layers", type=str, default="6", help="Comma-separated layer indices for H_rmsnorm")
    parser.add_argument("--do_I_attn_patch", action="store_true", help="Patch attn OUT with original at chosen layers")
    parser.add_argument("--I_layers", type=str, default="6", help="Comma-separated layer indices for I_attn_patch")
    parser.add_argument("--no_md_summary", action="store_true",
                        help="If set, do not write the consolidated Markdown SUMMARY.md")

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

    # Load data
    logging.info("Loading dataset from %s ...", args.data_jsonl)
    groups = load_jsonl_dataset(args.data_jsonl, max_groups=args.max_groups, max_paraphrases_per_group=args.max_paraphrases)
    logging.info("Loaded %d prompt groups.", len(groups))

    # Load models
    base, ft, tok = load_model_tokenizer(
        base_model_name_or_path=args.base_model,
        ft_model_name_or_path=args.ft_model or args.base_model,
        ft_lora_path=args.ft_lora_path,
        merge_lora=args.merge_lora,
        dtype=args.dtype,
        device=args.device
    )
    device = next(ft.parameters()).device

    # Original experiments
    if args.do_A:
        experiment_A_subspace_patching(groups, base, ft, tok, device, os.path.join(args.outdir, "A_subspace_patching"),
                                       max_length=args.max_length, batch_size=args.batch_size,
                                       topk_vocab=args.topk_vocab, lam_ridge=args.ridge_lam, max_groups=args.max_groups)
    if args.do_B:
        experiment_B_weight_level(groups, base, ft, tok, device, os.path.join(args.outdir, "B_weight_level"),
                                  max_length=args.max_length, batch_size=args.batch_size, max_pairs=args.max_pairs_eval)
    if args.do_C:
        experiment_C_grad_alignment(groups, ft, tok, device, os.path.join(args.outdir, "C_grad_alignment"),
                                    max_length=args.max_length, batch_size=args.batch_size, max_groups=args.max_groups)
    if args.do_D:
        experiment_D_emergence_curve(groups, ft, tok, device, os.path.join(args.outdir, "D_emergence_curve"),
                                     max_length=args.max_length, batch_size=args.batch_size, max_groups=args.max_groups)
    if args.do_E:
        experiment_E_head_ablation(groups, base, ft, tok, device, os.path.join(args.outdir, "E_head_ablation"),
                                   max_length=args.max_length, batch_size=args.batch_size, max_pairs=args.max_pairs_eval)
    if args.do_F:
        experiment_F_singular_paths(groups, ft, tok, device, os.path.join(args.outdir, "F_singular_paths"),
                                    max_length=args.max_length, batch_size=args.batch_size, max_groups=args.max_groups, top_m=args.singular_top_m)
    if args.do_G:
        experiment_G_tf_link(groups, ft, tok, device, os.path.join(args.outdir, "G_tf_link"),
                             max_length=args.max_length, batch_size=args.batch_size, lam_ridge=args.ridge_lam, topk_vocab=args.topk_vocab, max_groups=args.max_groups)

    # New analyses
    if args.do_A_sweep:
        experiment_A_sweep(groups, ft, tok, device, os.path.join(args.outdir, "A_sweep"),
                           max_length=args.max_length, batch_size=args.batch_size, topk_vocab=args.topk_vocab,
                           lam_ridge=args.ridge_lam, layers=parse_layers_arg(args.A_layers), max_groups=args.max_groups)
    if args.do_B_sweep:
        experiment_B_sweep(groups, base, ft, tok, device, os.path.join(args.outdir, "B_sweep"),
                           max_length=args.max_length, batch_size=args.batch_size, layers=parse_layers_arg(args.B_layers),
                           max_pairs=args.max_pairs_eval)
    if args.do_C_sweep:
        experiment_C_sweep(groups, ft, tok, device, os.path.join(args.outdir, "C_sweep"),
                           max_length=args.max_length, batch_size=args.batch_size, layers=parse_layers_arg(args.C_layers),
                           max_groups=args.max_groups)
    if args.do_H_rmsnorm:
        experiment_H_rmsnorm(groups, ft, tok, device, os.path.join(args.outdir, "H_rmsnorm"),
                             max_length=args.max_length, batch_size=args.batch_size, layers=parse_layers_arg(args.H_layers),
                             max_pairs=args.max_pairs_eval)
    if args.do_I_attn_patch:
        experiment_I_attn_patch(groups, ft, tok, device, os.path.join(args.outdir, "I_attn_patch"),
                                max_length=args.max_length, batch_size=args.batch_size, layers=parse_layers_arg(args.I_layers),
                                max_groups=args.max_groups)

    if not args.no_md_summary:
        try:
            write_markdown_summary(args.outdir)
        except Exception as e:
            logging.exception("Failed to write SUMMARY.md: %s", e)

    logging.info("All requested experiments completed. Outputs in %s", args.outdir)


if __name__ == "__main__":
    main()
