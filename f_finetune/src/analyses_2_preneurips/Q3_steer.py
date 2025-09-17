#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply analysis-style activation steering to generation with α-sweep,
driven by an explicit (prompt_count, paraphrase_key) pairs file
"""

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict  # for overall mean/CI plots

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional PEFT
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# Chat template

def build_chat_prompt(tokenizer, instruction: str, inp: str = "") -> str:
    meta = "Answer concisely and directly. Focus on task semantics; ignore stylistic tone cues. End after the answer."
    user_core = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    user_msg = f"{meta}\n\n{user_core}"
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# Robust layer/site plumbing

def _layers_list(model: AutoModelForCausalLM):
    for path in ["model.model.layers", "model.layers", "transformer.h"]:
        cur = model
        ok = True
        for p in path.split("."):
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok:
            return cur
    raise RuntimeError("Could not locate decoder layers on model.")

@dataclass
class LayerHandles:
    layer_module: torch.nn.Module
    attn_module: torch.nn.Module
    mlp_module: torch.nn.Module

def resolve_layer_handles(model: AutoModelForCausalLM, layer_index: int) -> LayerHandles:
    layers = _layers_list(model)
    if not (0 <= layer_index < len(layers)):
        raise RuntimeError(f"Layer index {layer_index} out of range [0,{len(layers)-1}]")
    layer = layers[layer_index]
    attn = None
    for nm in ["self_attn", "self_attention", "attn", "attention"]:
        if hasattr(layer, nm):
            attn = getattr(layer, nm); break
    if attn is None:
        raise RuntimeError(f"Could not find attention submodule at layer {layer_index}.")
    mlp = None
    for nm in ["mlp", "ffn", "feed_forward", "ff"]:
        if hasattr(layer, nm):
            mlp = getattr(layer, nm); break
    if mlp is None:
        raise RuntimeError(f"Could not find MLP/FFN submodule at layer {layer_index}.")
    return LayerHandles(layer_module=layer, attn_module=attn, mlp_module=mlp)

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


# Tokenization / positions (anchor-aware)

def _maybe_get_eot_ids(tok: AutoTokenizer) -> List[int]:
    # Gemma-style chat templates usually register these as added tokens
    candidates = ["</end_of_turn>", "<end_of_turn>"]
    ids = []
    for s in candidates:
        tid = tok.convert_tokens_to_ids(s)
        if tid is not None and tid != tok.unk_token_id:
            ids.append(int(tid))
    return ids

def compute_positions_for_anchor(tokenizer: AutoTokenizer, text: str, window_size: int, anchor: str) -> List[int]:
    # IMPORTANT: use the same tokenization pathway as generation (add_special_tokens=True)
    enc = tokenizer(text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
    ids = enc["input_ids"][0].tolist()
    T = len(ids)
    if anchor == "end_of_user":
        eots = _maybe_get_eot_ids(tokenizer)
        if eots:
            idxs = [i for i, t in enumerate(ids) if t in eots]
            if idxs:
                e = idxs[-1]  # last user </end_of_turn>
                start = max(0, e - window_size)
                return list(range(start, e))  # patch up to (but not including) the EOT
    # fallback to end-of-prompt
    start = max(0, T - window_size)
    return list(range(start, T))

def tokenize(tokenizer, texts: List[str], device: torch.device, max_length: int = 2048):
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}

@torch.no_grad()
def logits_last_token(model, tokenizer, text: str, device: torch.device) -> torch.Tensor:
    enc = tokenize(tokenizer, [text], device)
    out = model(**enc, output_attentions=False, use_cache=False)
    return out.logits[:, -1, :].detach()  # [1, V]


# Capture vectors at site/layer for explicit positions

@dataclass
class Capture:
    per_pos: torch.Tensor  # [W, H]

@torch.no_grad()
def capture_site_vecs_at_positions(model, tokenizer, text: str, device: torch.device,
                                   site: str, layer_idx: int, positions: List[int]) -> Capture:
    """Capture [W,H] from the chosen site/layer at exact token positions."""
    enc = tokenize(tokenizer, [text], device)
    pos = positions[:]  # copy
    h = resolve_layer_handles(model, layer_idx)
    store: Dict[str, torch.Tensor] = {}

    def _grab_from_tensor(t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 3 or t.shape[0] != 1:
            return None
        T = t.shape[1]
        valid = [p for p in pos if 0 <= p < T]
        if not valid:
            return None
        return t[0, valid, :].detach().clone()

    # PRE SITES
    def pre_layer_input(module, args, kwargs):
        hs = _pre_get_hidden_states(args, kwargs)
        store["x"] = _grab_from_tensor(hs)
        return None

    def pre_generic(module, args, kwargs):
        hs = _pre_get_hidden_states(args, kwargs)
        store["x"] = _grab_from_tensor(hs)
        return None

    # OUT SITES
    def fwd_out(module, args, output):
        y = output[0] if isinstance(output, tuple) else output
        got = _grab_from_tensor(y)
        if got is not None:
            store["y"] = got
        return None

    hook = None
    try:
        if site == "layer_input":
            hook = h.layer_module.register_forward_pre_hook(pre_layer_input, with_kwargs=True)
        elif site == "attn_in":
            hook = h.attn_module.register_forward_pre_hook(pre_generic, with_kwargs=True)
        elif site == "mlp_in":
            hook = h.mlp_module.register_forward_pre_hook(pre_generic, with_kwargs=True)
        elif site == "attn_out":
            hook = h.attn_module.register_forward_hook(fwd_out, with_kwargs=False)
        elif site == "mlp_out":
            hook = h.mlp_module.register_forward_hook(fwd_out, with_kwargs=False)
        elif site == "layer_out":
            hook = h.layer_module.register_forward_hook(fwd_out, with_kwargs=False)
        else:
            raise ValueError("site must be one of {'layer_input','attn_in','mlp_in','attn_out','mlp_out','layer_out'}")

        _ = model(**enc, output_attentions=False, use_cache=False)
    finally:
        if hook is not None:
            try: hook.remove()
            except Exception: pass

    key = "x" if site.endswith("_in") or site == "layer_input" else "y"
    vecs = store.get(key)
    if vecs is None:
        raise RuntimeError(f"Capture failed at site={site}; positions may be out of range.")
    return Capture(per_pos=vecs.to(torch.float32).cpu())


# Patching BASE during paraphrase inference (pre/out sites)

class PatchHook:
    def __init__(self, model, site: str, layer_idx: int,
                 mode: str, alpha: float,
                 ft_orig: Optional[torch.Tensor],   # [W,H] or None
                 base_orig: Optional[torch.Tensor], # [W,H] or None
                 positions: List[int],
                 single_use: bool = True,
                 decode_vec: Optional[torch.Tensor] = None):  # persistent decode vector
        self.model = model
        self.site = site
        self.layer_idx = layer_idx
        self.mode = mode
        self.alpha = float(alpha)
        self.ft_orig = ft_orig  # CPU float32 or None
        self.base_orig = base_orig
        self.positions = positions[:]
        self.single_use = single_use
        self.decode_vec = decode_vec  # [1,H] or None
        self.hook = None
        self._used = False

    def __enter__(self):
        if self.alpha == 0.0:
            class Noop:
                def remove(self): pass
            self.hook = Noop()
            return self

        h = resolve_layer_handles(self.model, self.layer_idx)

        def _apply_into(hs: torch.Tensor) -> torch.Tensor:
            if hs.dim() != 3 or hs.shape[0] != 1:
                return hs
            B, T, H = hs.shape
            pos = []
            for p in self.positions:
                if p < 0:
                    pos.append(T + p)  # allow negative index like -1 for current position
                elif 0 <= p < T:
                    pos.append(p)
            if not pos:
                return hs
            hs2 = hs.clone()

            if self.mode == "ft_abs":
                for i, p in enumerate(pos):
                    if self.ft_orig is not None and i < len(self.ft_orig):
                        v = self.ft_orig[i].to(hs2.device, hs2.dtype)
                    elif self.decode_vec is not None:
                        v = self.decode_vec[0].to(hs2.device, hs2.dtype)
                    else:
                        continue
                    hs2[0, p, :] = (1.0 - self.alpha) * hs2[0, p, :] + self.alpha * v

            elif self.mode == "ft_delta":
                for i, p in enumerate(pos):
                    if (self.ft_orig is not None) and (self.base_orig is not None) and (i < len(self.ft_orig)) and (i < len(self.base_orig)):
                        delta = (self.ft_orig[i] - self.base_orig[i]).to(hs2.device, hs2.dtype)
                    elif self.decode_vec is not None:
                        delta = self.decode_vec[0].to(hs2.device, hs2.dtype)
                    else:
                        continue
                    hs2[0, p, :] = hs2[0, p, :] + self.alpha * delta
            else:
                raise ValueError("mode must be one of {'ft_delta','ft_abs'}")
            return hs2

        def pre_hook(module, args, kwargs):
            if self.single_use and self._used:
                return None
            hs = _pre_get_hidden_states(args, kwargs)
            hs2 = _apply_into(hs)
            self._used = True
            return _pre_set_hidden_states(args, kwargs, hs2)

        def fwd_hook(module, args, output):
            if self.single_use and self._used:
                return None
            y = output[0] if isinstance(output, tuple) else output
            if y.dim() != 3:
                return None
            y2 = _apply_into(y)
            self._used = True
            if isinstance(output, tuple):
                return (y2,) + tuple(output[1:])
            return y2

        if self.site == "layer_input":
            self.hook = h.layer_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        elif self.site == "attn_in":
            self.hook = h.attn_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        elif self.site == "mlp_in":
            self.hook = h.mlp_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        elif self.site == "attn_out":
            self.hook = h.attn_module.register_forward_hook(fwd_hook, with_kwargs=False)
        elif self.site == "mlp_out":
            self.hook = h.mlp_module.register_forward_hook(fwd_hook, with_kwargs=False)
        elif self.site == "layer_out":
            self.hook = h.layer_module.register_forward_hook(fwd_hook, with_kwargs=False)
        else:
            raise ValueError("site must be one of {'layer_input','attn_in','mlp_in','attn_out','mlp_out','layer_out'}")

        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.hook:
                self.hook.remove()
        except Exception:
            pass


# Generation helpers

@torch.no_grad()
def generate_text(model, tokenizer, prompt_text: str, device: torch.device, max_new_tokens: int = 256,
                  do_sample: bool = False, temperature: float = 0.7, top_p: float = 0.9, seed: Optional[int] = None) -> str:
    if seed is not None:
        torch.manual_seed(seed)
    enc = tokenize(tokenizer, [prompt_text], device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,                # sampling support
        temperature=temperature,            #         top_p=top_p,                        #         eos_token_id=model.config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        output_scores=False,
        return_dict_in_generate=True,
        use_cache=True,
    )
    seq = out.sequences[0]
    in_len = int(enc["attention_mask"].sum(dim=1).item())
    ans_ids = seq[in_len:]
    txt = tokenizer.decode(ans_ids, skip_special_tokens=True).strip()

    s = prompt_text
    original_user_message = ""
    if "</start_of_turn>" in s:
        try:
            original_user_message = s.split("</start_of_turn>user")[-1].split("</end_of_turn>")[0].strip()
        except Exception:
            original_user_message = ""
    if original_user_message and txt.strip() == original_user_message.strip():
        txt = ""
    else:
        marker = "model\n"
        mp = txt.find(marker)
        if mp != -1:
            txt = txt[mp + len(marker):].lstrip()

    return txt


# Data models + loading

@dataclass
class Item:
    prompt_count: int
    instruction_original: str
    paraphrase_key: str
    paraphrase_text: str
    inp: str
    ground_truth_output: Optional[str] = None

def load_pairs(path: str) -> List[Tuple[int, str]]:
    pairs: List[Tuple[int, str]] = []
    p = Path(path)
    if p.suffix.lower() in [".json", ".json5"]:
        obj = json.load(open(p, "r", encoding="utf-8"))
        if isinstance(obj, dict) and "pairs" in obj:
            obj = obj["pairs"]
        if not isinstance(obj, list):
            raise SystemExit("--pairs_file JSON must be a list or a dict with 'pairs': [...]")
        for row in obj:
            pairs.append((int(row["prompt_count"]), str(row["paraphrase_key"])))
    elif p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                pairs.append((int(row["prompt_count"]), str(row["paraphrase_key"])))
    elif p.suffix.lower() == ".csv":
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((int(row["prompt_count"]), str(row["paraphrase_key"])))
    else:
        try:
            obj = json.loads(open(p, "r", encoding="utf-8").read())
            if isinstance(obj, dict) and "pairs" in obj:
                obj = obj["pairs"]
            for row in obj:
                pairs.append((int(row["prompt_count"]), str(row["paraphrase_key"])))
        except Exception as e:
            raise SystemExit(f"Unrecognized pairs file format: {e}")
    return pairs

def load_items_from_pairs(prompts_json_path: str,
                          pairs: List[Tuple[int, str]]) -> List[Item]:
    data = json.load(open(prompts_json_path, "r", encoding="utf-8"))
    by_id = {int(x["prompt_count"]): x for x in data}
    out: List[Item] = []
    for pc, key in pairs:
        row = by_id.get(int(pc))
        if not row:
            logging.warning("prompt_count=%s not found in prompts JSON.", pc); continue
        if key not in row:
            logging.warning("paraphrase key %s not found for prompt_count=%s.", key, pc); continue
        instr = row["instruction_original"]
        inp = (row.get("input", "") or "")
        para_text = row[key]
        gt = row.get("output", None)
        out.append(Item(prompt_count=int(pc),
                        instruction_original=instr,
                        paraphrase_key=key,
                        paraphrase_text=para_text,
                        inp=inp,
                        ground_truth_output=gt))
    return out


# KL helper

def symm_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    p_prob = p.exp(); q_prob = q.exp()
    return (p_prob * (p - q)).sum(-1) + (q_prob * (q - p)).sum(-1)


# Model loading

def load_models_and_tokenizer(base_model: str,
                              ft_model: Optional[str],
                              ft_lora_adapter: Optional[str],
                              device: torch.device,
                              dtype: str = "bf16"):
    dt = (dtype or "bf16").lower()
    if dt == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif dt in ("fp16", "float16", "half"):
        torch_dtype = torch.float16
    elif dt in ("fp32", "float32"):
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    tok = AutoTokenizer.from_pretrained(ft_model or base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch_dtype).to(device).eval()

    if ft_lora_adapter is not None:
        host = AutoModelForCausalLM.from_pretrained(ft_model or base_model, torch_dtype=torch_dtype).to(device).eval()
        if not _HAS_PEFT:
            raise RuntimeError("peft is not installed but --ft_lora_adapter was provided.")
        ft = PeftModel.from_pretrained(host, ft_lora_adapter).to(device).eval()
    else:
        ft_path = ft_model or base_model
        ft = AutoModelForCausalLM.from_pretrained(ft_path, torch_dtype=torch_dtype).to(device).eval()

    return base, ft, tok


# Global vector creation

@dataclass
class GlobalVector:
    kind: str              # 'ft_abs_mean' or 'ft_delta_mean'
    vec: torch.Tensor      # [W,H] (CPU float32)

def build_global_vector(vector_items: List[Item],
                        tokenizer: AutoTokenizer,
                        base_model, ft_model,
                        device: torch.device,
                        site: str, layer_idx: int,
                        window_size: int, anchor: str,
                        mode: str) -> GlobalVector:
    assert mode in ("ft_abs_mean","ft_delta_mean")
    vecs = []
    for it in vector_items:
        orig_text = build_chat_prompt(tokenizer, it.instruction_original, it.inp)
        pos = compute_positions_for_anchor(tokenizer, orig_text, window_size, anchor)
        cap_ft = capture_site_vecs_at_positions(ft_model, tokenizer, orig_text, device, site, layer_idx, pos).per_pos
        if mode == "ft_abs_mean":
            vecs.append(cap_ft)
        else:
            cap_bo = capture_site_vecs_at_positions(base_model, tokenizer, orig_text, device, site, layer_idx, pos).per_pos
            vecs.append(cap_ft - cap_bo)
    if not vecs:
        raise SystemExit("Global vector creation: no vectors captured.")
    W = min(v.shape[0] for v in vecs)
    vecs_trim = [v[-W:, :] for v in vecs]
    mean_vec = torch.stack(vecs_trim, dim=0).mean(dim=0).to(torch.float32).cpu()
    logging.info(f"Built global vector ({mode}) with shape {tuple(mean_vec.shape)}; ||vec||={mean_vec.norm().item():.6f}")
    return GlobalVector(kind=mode, vec=mean_vec)


# Orchestrator

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(Path(args.out_dir) / "run.log", mode="w", encoding="utf-8")]
    )
    logging.info("Args: %s", vars(args))

    if not args.pairs_file:
        raise SystemExit("Please provide --pairs_file with (prompt_count, paraphrase_key) rows.")
    if not args.prompts_json:
        raise SystemExit("Please provide --prompts_json.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    base, ft, tok = load_models_and_tokenizer(args.base_model, args.ft_model, args.ft_lora_adapter, device, args.dtype)

    # Evaluation pairs
    eval_pairs = load_pairs(args.pairs_file)
    eval_items: List[Item] = load_items_from_pairs(args.prompts_json, eval_pairs)
    if not eval_items:
        raise SystemExit("No eval pairs resolved against prompts JSON.")

    # Optional global steering vector
    global_vec: Optional[GlobalVector] = None
    if args.vector_creation_pairs:
        v_pairs = load_pairs(args.vector_creation_pairs)
        v_items = load_items_from_pairs(args.prompts_json, v_pairs)
        if not v_items:
            raise SystemExit("vector_creation_pairs provided, but resolved to 0 items.")
        mode = "ft_delta_mean" if args.global_vector_mode == "ft_delta_mean" else "ft_abs_mean"
        global_vec = build_global_vector(
            vector_items=v_items,
            tokenizer=tok,
            base_model=base, ft_model=ft,
            device=device,
            site=args.site,
            layer_idx=(args.layer_idx if args.zero_indexed_layer else args.layer_idx - 1),
            window_size=args.window_size,
            anchor=args.anchor,
            mode=mode
        )
        np.save(Path(args.out_dir) / "global_vector.npy", global_vec.vec.numpy())
        with open(Path(args.out_dir) / "global_vector.meta.json", "w", encoding="utf-8") as fh:
            json.dump({"mode": global_vec.kind, "site": args.site, "layer_idx": args.layer_idx,
                       "zero_indexed_layer": args.zero_indexed_layer,
                       "window_size": args.window_size, "anchor": args.anchor}, fh, indent=2)

    # α list / layer / site
    alphas = [float(x) for x in args.alphas.split(",") if x.strip() != ""]
    layer_idx = args.layer_idx if args.zero_indexed_layer else (args.layer_idx - 1)
    site = args.site
    window_size = args.window_size
    anchor = args.anchor

    # Output text file for answers
    txt_path = Path(args.out_dir) / "results.txt"
    tf = open(txt_path, "w", encoding="utf-8")

    # aggregate KL for overall plots
    agg_kl: Dict[float, List[float]] = defaultdict(list)

    for it in eval_items:
        logging.info(f"Pair: prompt_count={it.prompt_count} | key={it.paraphrase_key}")

        orig_prompt = build_chat_prompt(tok, it.instruction_original, it.inp)
        para_prompt = build_chat_prompt(tok, it.paraphrase_text, it.inp)

        # Anchor positions
        if args.delta_context == "original":
            pos_src_prompt = orig_prompt
        else:
            pos_src_prompt = para_prompt
        pos_orig = compute_positions_for_anchor(tok, pos_src_prompt, window_size, anchor)
        pos_para = compute_positions_for_anchor(tok, para_prompt, window_size, anchor)
        if not pos_orig or not pos_para:
            logging.warning("Empty positions for this pair; skipping.")
            continue
        W = min(len(pos_orig), len(pos_para))
        pos_orig = pos_orig[-W:]
        pos_para = pos_para[-W:]

        # Baselines
        base_ans_orig = generate_text(base, tok, orig_prompt, device, args.max_new_tokens,
                                      do_sample=args.do_sample, temperature=args.temperature,
                                      top_p=args.top_p, seed=args.seed)
        base_ans_para = generate_text(base, tok, para_prompt, device, args.max_new_tokens,
                                      do_sample=args.do_sample, temperature=args.temperature,
                                      top_p=args.top_p, seed=args.seed)
        ft_ans_para   = generate_text(ft,   tok, para_prompt, device, args.max_new_tokens,
                                      do_sample=args.do_sample, temperature=args.temperature,
                                      top_p=args.top_p, seed=args.seed)

        # Target logits for α→KL
        target_name = args.target_logits.lower()
        if target_name == "base_original":
            target_logits = logits_last_token(base, tok, orig_prompt, device)
        elif target_name == "ft_original":
            target_logits = logits_last_token(ft, tok, orig_prompt, device)
        else:
            raise ValueError("--target_logits must be one of {'base_original','ft_original'}")

        # Per-example vs global vector
        if args.use_global_vector and global_vec is not None:
            if global_vec.kind == "ft_abs_mean":
                ft_mat = global_vec.vec[-W:, :]
                bo_mat = None
                logging.info(f"[GLOBAL {global_vec.kind}] ||vec||={ft_mat.norm().item():.6f} (W={W})")
            else:
                delta = global_vec.vec[-W:, :]
                ft_mat = delta + 0.0
                bo_mat = torch.zeros_like(delta)
                logging.info(f"[GLOBAL {global_vec.kind}] ||Δ||={delta.norm().item():.6f} (W={W})")
        else:
            # capture from chosen site/layer at the selected source prompt (original or paraphrase)
            cap_ft_src  = capture_site_vecs_at_positions(ft,   tok, pos_src_prompt, device, site, layer_idx, pos_orig).per_pos
            if args.steer_mode == "ft_delta":
                cap_bo_src  = capture_site_vecs_at_positions(base, tok, pos_src_prompt, device, site, layer_idx, pos_orig).per_pos
                delta_norm = (cap_ft_src - cap_bo_src).norm().item()
                logging.info(f"[PER-EX] ||Δ|| at {site}/layer {layer_idx} (W={W}): {delta_norm:.6f}")
                ft_mat = cap_ft_src
                bo_mat = cap_bo_src
            else:
                ft_norm = cap_ft_src.norm().item()
                logging.info(f"[PER-EX] ||FT_src|| at {site}/layer {layer_idx} (W={W}): {ft_norm:.6f}")
                ft_mat = cap_ft_src
                bo_mat = None

        # α sweep
        kl_points: List[Tuple[float, float]] = []
        answers_by_alpha: Dict[float, str] = {}

        para_logits_unsteered = logits_last_token(base, tok, para_prompt, device)
        kl0 = float(symm_kl(target_logits, para_logits_unsteered).item())
        kl_points.append((0.0, kl0))
        agg_kl[0.0].append(kl0)  # aggregate
        answers_by_alpha[0.0] = base_ans_para

        for a in alphas:
            if a == 0.0:
                continue
            # Prefill steering hook (single use)
            with PatchHook(base, site, layer_idx, args.steer_mode, a,
                           ft_orig=ft_mat[-W:, :], base_orig=(bo_mat[-W:, :] if bo_mat is not None else None),
                           positions=pos_para[-W:], single_use=True):

                # optional decode-time persistent steering at last position
                if args.enable_decode_time_steer:
                    if args.steer_mode == "ft_delta":
                        if bo_mat is not None:
                            decode_vec = (ft_mat - bo_mat).mean(dim=0, keepdim=True)
                        else:
                            decode_vec = ft_mat.mean(dim=0, keepdim=True)
                    else:
                        decode_vec = ft_mat.mean(dim=0, keepdim=True)

                    with PatchHook(base, site, layer_idx, args.steer_mode, a,
                                   ft_orig=None, base_orig=None,
                                   positions=[-1], single_use=False,
                                   decode_vec=decode_vec):
                        lgs = logits_last_token(base, tok, para_prompt, device)
                        klv = float(symm_kl(target_logits, lgs).item())
                        kl_points.append((a, klv))
                        agg_kl[a].append(klv)  # aggregate
                        ans = generate_text(base, tok, para_prompt, device, args.max_new_tokens,
                                            do_sample=args.do_sample, temperature=args.temperature,
                                            top_p=args.top_p, seed=args.seed)
                        answers_by_alpha[a] = ans
                else:
                    lgs = logits_last_token(base, tok, para_prompt, device)
                    klv = float(symm_kl(target_logits, lgs).item())
                    kl_points.append((a, klv))
                    agg_kl[a].append(klv)  # aggregate
                    ans = generate_text(base, tok, para_prompt, device, args.max_new_tokens,
                                        do_sample=args.do_sample, temperature=args.temperature,
                                        top_p=args.top_p, seed=args.seed)
                    answers_by_alpha[a] = ans

        # Write per-pair section
        tf.write(f"\n=== prompt_count={it.prompt_count} | key={it.paraphrase_key} ===\n")
        tf.write(f"Instruction (original):\n{it.instruction_original}\n")
        if it.inp:
            tf.write(f"Input:\n{it.inp}\n")
        tf.write(f"Paraphrase ({it.paraphrase_key}):\n{it.paraphrase_text}\n")

        if it.ground_truth_output is not None:
            tf.write("\nGround truth (prompts_json['output']):\n")
            tf.write(it.ground_truth_output.strip() + "\n")

        tf.write("\nFresh baselines (this run):\n")
        tf.write("BASE answer (original):\n" + base_ans_orig + "\n")
        tf.write("BASE answer (paraphrase):\n" + base_ans_para + "\n")
        tf.write("FT   answer (paraphrase):\n" + ft_ans_para + "\n")
        tf.write(f"Target logits for KL: {args.target_logits}\n")

        if args.use_global_vector and global_vec is not None:
            tf.write(f"Global vector mode: {global_vec.kind}\n")

        tf.write("\nSteered answers:\n")
        for a in sorted(answers_by_alpha.keys()):
            tf.write(f"[alpha={a:.3f}] {args.steer_mode} @ {site}/layer {layer_idx}:\n{answers_by_alpha[a]}\n")

        tf.write("\nKL vs alpha (paraphrase vs {}):\n".format(args.target_logits))
        tf.write(", ".join([f"({a:.3f},{k:.4f})" for (a, k) in sorted(kl_points, key=lambda x: x[0])]) + "\n")

        # Plot (ii) α-sweep
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [a for (a, k) in sorted(kl_points, key=lambda x: x[0])]
        ys = [k for (a, k) in sorted(kl_points, key=lambda x: x[0])]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("α")
        plt.ylabel(f"Symmetric KL (paraphrase vs {args.target_logits.replace('_',' ')})")
        plt.title(f"(ii) α-sweep | pc={it.prompt_count} | {it.paraphrase_key}\nmode={args.steer_mode} site={site} layer={layer_idx} anchor={anchor}")
        plt.grid(alpha=0.2)
        out_png = Path(args.out_dir) / f"alpha_sweep_pc{it.prompt_count}_{it.paraphrase_key}.png"
        plt.savefig(out_png, dpi=160)
        plt.close()
        logging.info(f"Wrote plot: {out_png}")

    tf.close()
    logging.info(f"Wrote answers: {txt_path}")

        # Overall plots: mean-only and mean with 95% CI over pairs
        if agg_kl:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = sorted(agg_kl.keys(), key=lambda a: a)
        means = []
        cis_low = []
        cis_high = []
        for a in xs:
            arr = np.array(agg_kl[a], dtype=np.float64)
            m = float(arr.mean())
            means.append(m)
            n = len(arr)
            if n > 1:
                se = float(arr.std(ddof=1) / np.sqrt(n))
                ci = 1.96 * se
            else:
                ci = 0.0
            cis_low.append(m - ci)
            cis_high.append(m + ci)

        # Mean-only
        plt.figure()
        plt.plot(xs, means, marker="o")
        plt.xlabel("α")
        plt.ylabel(f"Mean Symmetric KL (paraphrase vs {args.target_logits.replace('_',' ')})")
        plt.title("Overall α-sweep (mean over pairs)")
        plt.grid(alpha=0.2)
        out_png_mean = Path(args.out_dir) / "alpha_sweep_overall_mean.png"
        plt.savefig(out_png_mean, dpi=160)
        plt.close()
        logging.info(f"Wrote plot: {out_png_mean}")

        # Mean + 95% CI
        plt.figure()
        plt.plot(xs, means, marker="o")
        plt.fill_between(xs, cis_low, cis_high, alpha=0.2, label="95% CI")
        plt.xlabel("α")
        plt.ylabel(f"Mean Symmetric KL (paraphrase vs {args.target_logits.replace('_',' ')})")
        plt.title("Overall α-sweep (mean ± 95% CI over pairs)")
        plt.legend()
        plt.grid(alpha=0.2)
        out_png_ci = Path(args.out_dir) / "alpha_sweep_overall_mean_ci.png"
        plt.savefig(out_png_ci, dpi=160)
        plt.close()
        logging.info(f"Wrote plot: {out_png_ci}")

    logging.info("Done.")


# CLI

def main():
    ap = argparse.ArgumentParser("Steer specific (prompt_count, paraphrase_key) pairs with α-sweep.")
    # Models
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--ft_model", type=str, default=None, help="Path/ID of FT host (if different from base)")
    ap.add_argument("--ft_lora_adapter", type=str, default=None, help="Directory with LoRA adapter (e.g., layer-12 FT).")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--cpu", action="store_true")

    # Data
    ap.add_argument("--prompts_json", type=str, required=True, help="Prompts JSON with instruct_* keys + 'output'")
    ap.add_argument("--pairs_file", type=str, required=True, help="JSON/JSONL/CSV mapping of {prompt_count, paraphrase_key} rows")

    # Steering eval
    ap.add_argument("--layer_idx", type=int, default=12, help="Layer index (1-based by default)")
    ap.add_argument("--zero_indexed_layer", action="store_true")
    ap.add_argument("--site", type=str, default="attn_in",
                    choices=["layer_input","attn_in","mlp_in","attn_out","mlp_out","layer_out"])
    ap.add_argument("--anchor", type=str, default="end_of_user", choices=["end_of_user","end_of_prompt"],
                    help="Where to anchor the last W tokens for capture/patch")
    ap.add_argument("--window_size", type=int, default=8, help="Number of tokens at anchor to capture/patch (W)")
    ap.add_argument("--steer_mode", type=str, default="ft_delta", choices=["ft_delta","ft_abs"],
                    help="ft_delta: add (FT_orig - BASE_orig); ft_abs: mix toward FT_orig")
    ap.add_argument("--alphas", type=str, default="0,0.5,1.0,2.0,3.0")
    ap.add_argument("--target_logits", type=str, default="ft_original", choices=["base_original","ft_original"],
                    help="Which ORIGINAL to compare against in the KL plot")

    # Global vector creation
    ap.add_argument("--vector_creation_pairs", type=str, default=None,
                    help="Pairs file to whitelist items for building a single global vector")
    ap.add_argument("--global_vector_mode", type=str, default="ft_delta_mean",
                    choices=["ft_delta_mean","ft_abs_mean"],
                    help="How to build the global vector if vector_creation_pairs is provided")
    ap.add_argument("--use_global_vector", action="store_true",
                    help="If set and a global vector is available, use it for all eval pairs instead of per-example capture")

    # Output
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default="steer_apply_out_pairs")

    # Decode-time persistent steering toggle
    ap.add_argument("--enable_decode_time_steer", action="store_true",
                    help="If set, also steer the last position persistently during decode (positions=[-1]).")

    # Sampling controls (default off)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling during generation.")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (when --do_sample).")
    ap.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling (when --do_sample).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")

    # Choose whether steering vectors are captured on the original or paraphrase prompt
    ap.add_argument("--delta_context", type=str, default="original", choices=["original","paraphrase"],
                    help="Capture FT/BASE vectors on 'original' or 'paraphrase' prompt for steering.")

    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
