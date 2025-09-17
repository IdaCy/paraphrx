#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply analysis-style activation steering to generation with α-sweep,
driven by explicit (prompt_count, paraphrase_key) pairs — now with support
for injecting an external rank-1 fine-tune vector for steering
"""

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

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

# Safetensors for reading LoRA adapter weights
try:
    from safetensors.torch import load_file as safe_load
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False


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


# Tokenisation / positions (anchor-aware)

def _maybe_get_eot_ids(tok: AutoTokenizer) -> List[int]:
    candidates = ["</end_of_turn>", "<end_of_turn>"]
    ids = []
    for s in candidates:
        tid = tok.convert_tokens_to_ids(s)
        if tid is not None and tid != tok.unk_token_id:
            ids.append(int(tid))
    return ids

def compute_positions_for_anchor(tokenizer: AutoTokenizer, text: str, window_size: int, anchor: str) -> List[int]:
    enc = tokenizer(text, return_tensors="pt", padding=False, truncation=False, add_special_tokens=True)
    ids = enc["input_ids"][0].tolist()
    T = len(ids)
    if anchor == "end_of_user":
        eots = _maybe_get_eot_ids(tokenizer)
        if eots:
            idxs = [i for i, t in enumerate(ids) if t in eots]
            if idxs:
                e = idxs[-1]
                start = max(0, e - window_size)
                return list(range(start, e))
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
    return out.logits[:, -1, :].detach()


# Capture vectors at site/layer for explicit positions

@dataclass
class Capture:
    per_pos: torch.Tensor  # [W, H]

@torch.no_grad()
def capture_site_vecs_at_positions(model, tokenizer, text: str, device: torch.device,
                                   site: str, layer_idx: int, positions: List[int]) -> Capture:
    enc = tokenize(tokenizer, [text], device)
    pos = positions[:]
    h = resolve_layer_handles(model, layer_idx)
    store: Dict[str, torch.Tensor] = {}

    def _grab_from_tensor(t: torch.Tensor) -> Optional[torch.Tensor]:
        if t.dim() != 3 or t.shape[0] != 1:
            return None
        T = t.shape[1]
        valid = [p for p in pos if 0 <= p < T]
        if not valid:
            return None
        return t[0, valid, :].detach().clone()

    def pre_layer_input(module, args, kwargs):
        hs = _pre_get_hidden_states(args, kwargs)
        store["x"] = _grab_from_tensor(hs)
        return None

    def pre_generic(module, args, kwargs):
        hs = _pre_get_hidden_states(args, kwargs)
        store["x"] = _grab_from_tensor(hs)
        return None

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
                 decode_vec: Optional[torch.Tensor] = None):
        self.model = model
        self.site = site
        self.layer_idx = layer_idx
        self.mode = mode
        self.alpha = float(alpha)
        self.ft_orig = ft_orig
        self.base_orig = base_orig
        self.positions = positions[:]
        self.single_use = single_use
        self.decode_vec = decode_vec
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
                    pos.append(T + p)
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


# Rank-1 LoRA steering hook (Δy = α * a * (b·x))

def resolve_linear_module(model: AutoModelForCausalLM, layer_index: int, target: str) -> torch.nn.Module:
    """
    target in {"mlp.down_proj","mlp.up_proj","mlp.gate_proj",
               "attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"}
    """
    h = resolve_layer_handles(model, layer_index)
    fam, proj = target.split(".")
    if fam == "mlp":
        sub = getattr(h.mlp_module, proj, None)
    elif fam == "attn":
        # gemma uses q_proj/k_proj/v_proj/o_proj under attention module
        sub = getattr(h.attn_module, proj, None)
    else:
        raise ValueError(f"Unsupported target family: {target}")
    # relax type requirement; accept Linear-like modules (quantized/wrapped)
    if sub is None or not hasattr(sub, "weight"):
        raise RuntimeError(f"Could not find target module with a 'weight' for target={target} at layer={layer_index}.")
    return sub

def _find_key_contains(d: Dict[str, torch.Tensor], must: List[str]) -> Optional[str]:
    for k in d.keys():
        if k.endswith(".weight") and all(s in k for s in must):
            return k
    return None

def _any_of(*candidates: str) -> List[str]:
    return list(candidates)

def load_rank1_from_adapter(adapter_dir: str, layer_index0: int, target: str) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns (a_vec[out_dim], b_vec[in_dim], lora_scale) from a PEFT LoRA adapter (rank=1).
    Applies lora_scale = lora_alpha / r to a_vec so α is an extra user dial.
    """
    if not _HAS_SAFETENSORS:
        raise RuntimeError("safetensors not installed; cannot read adapter_model.safetensors.")
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    st_path  = Path(adapter_dir) / "adapter_model.safetensors"
    if not cfg_path.exists() or not st_path.exists():
        raise RuntimeError(f"Adapter files not found in {adapter_dir} (need adapter_config.json and adapter_model.safetensors).")

    cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
    st  = safe_load(str(st_path))

    r = int(cfg.get("r", 1))
    lora_alpha = float(cfg.get("lora_alpha", 1.0))
    lora_scale = lora_alpha / max(1, r)

    # We search keys robustly across PEFT variants
    # Typical patterns include:
    #  "base_model.model.model.layers.{i}.mlp.down_proj.lora_A.weight"
    #  "model.layers.{i}.self_attn.q_proj.lora_B.default.weight"
    layer_token_opts = _any_of(f".layers.{layer_index0}.", f"layers.{layer_index0}.")
    fam, proj = target.split(".")
    fam_opts = _any_of(fam, "self_attn" if fam=="attn" else fam, "attention" if fam=="attn" else fam)

    A_key = None
    B_key = None
    for lt in layer_token_opts:
        for famk in fam_opts:
            # allow presence/absence of "self_" prefixes and ".default."
            candidates_A = [
                _find_key_contains(st, [lt, famk, proj, "lora_A", ".default."]),
                _find_key_contains(st, [lt, famk, proj, "lora_A"]),
            ]
            candidates_B = [
                _find_key_contains(st, [lt, famk, proj, "lora_B", ".default."]),
                _find_key_contains(st, [lt, famk, proj, "lora_B"]),
            ]
            A_key = next((c for c in candidates_A if c is not None), None)
            B_key = next((c for c in candidates_B if c is not None), None)
            if A_key and B_key:
                break
        if A_key and B_key:
            break

    if not A_key or not B_key:
        # Last-ditch: just find by layer id and proj token anywhere
        A_key = A_key or _find_key_contains(st, [f".layers.{layer_index0}.", proj, "lora_A"])
        B_key = B_key or _find_key_contains(st, [f".layers.{layer_index0}.", proj, "lora_B"])

    if not A_key or not B_key:
        raise KeyError(f"Could not locate lora_A/lora_B weights for layer {layer_index0} target={target} in adapter.")

    A = st[A_key]  # [r, in_features]
    B = st[B_key]  # [out_features, r]

    # add explicit logging of the exact keys and shapes found (no behavior change)
    logging.info(f"[rank1 adapter] Found keys: A_key='{A_key}', B_key='{B_key}' with shapes A{tuple(A.shape)} B{tuple(B.shape)}; r={r}, lora_alpha={lora_alpha}")

    if not (A.dim()==2 and B.dim()==2):
        raise RuntimeError(f"Unexpected LoRA tensor ranks: A{tuple(A.shape)} B{tuple(B.shape)}")
    if A.shape[0] != 1 or B.shape[1] != 1:
        raise AssertionError(f"This code expects rank=1. Got r={A.shape[0]} and {B.shape[1]}.")

    b_vec = A[0, :].contiguous().float()                # in_features
    a_vec = (B[:, 0].contiguous().float()) * lora_scale # out_features (includes LoRA scaling)
    return a_vec, b_vec, lora_scale

class Rank1LoRAHook:
    """
    Applies y <- y + alpha * a * (b · x) for a chosen Linear module.
    Restricts to token positions `positions` (supports -1 for current decode token).
    """
    def __init__(self, model, layer_idx: int, target: str,
                 a_vec: torch.Tensor, b_vec: torch.Tensor, alpha: float,
                 positions: List[int], single_use: bool = False):
        self.model = model
        self.layer_idx = layer_idx
        self.target = target
        self.alpha = float(alpha)
        self.positions = positions[:]
        self.single_use = single_use
        self._used = False
        self._pre = None
        self._post = None
        self._cache = {}
        self._a_cpu = a_vec.detach().cpu().float()
        self._b_cpu = b_vec.detach().cpu().float()

    def __enter__(self):
        if self.alpha == 0.0:
            class Noop:
                def remove(self): pass
            self._pre = self._post = Noop()
            return self

        lin = resolve_linear_module(self.model, self.layer_idx, self.target)
        # Move to module device/dtype
        self._a = self._a_cpu.to(device=lin.weight.device, dtype=lin.weight.dtype)
        self._b = self._b_cpu.to(device=lin.weight.device, dtype=lin.weight.dtype)

        def pre_hook(module, args, kwargs):
            if self.single_use and self._used:
                return None
            # Linear forward signature: y = x @ W.T + b
            # We read x (args[0])
            x = args[0] if (args and torch.is_tensor(args[0])) else kwargs.get("input", None)
            if x is None:
                return None
            if x.dim() == 2:
                # [BT, In]
                s = x @ self._b  # [BT]
                self._cache["s"] = s
                self._cache["shape2d"] = True
            elif x.dim() == 3:
                # [B, T, In]
                s = torch.einsum("bti,i->bt", x, self._b)  # [B,T]
                self._cache["s"] = s
                self._cache["shape2d"] = False
            else:
                self._cache["s"] = None
            return None

        def post_hook(module, args, output):
            if self.single_use and self._used:
                return None
            y = output
            s = self._cache.get("s", None)
            if s is None:
                self._used = True
                return y

            a = self._a
            if y.dim() == 2:
                # [BT, Out]
                if self._cache.get("shape2d", False):
                    y = y + (self.alpha * s.unsqueeze(-1)) * a.unsqueeze(0)
                else:
                    y = y + (self.alpha * s.reshape(-1).unsqueeze(-1)) * a.unsqueeze(0)
                self._used = True
                return y

            # [B, T, Out] — apply only at selected token positions
            B, T, H = y.shape
            pos = []
            for p in self.positions:
                p = (T + p) if p < 0 else p
                if 0 <= p < T: pos.append(p)
            if not pos:
                self._used = True
                return y

            a_row = a.view(1, 1, H)
            for p in pos:
                y[:, p, :] = y[:, p, :] + (self.alpha * s[:, p].unsqueeze(-1)) * a_row.squeeze(0)
            self._used = True
            return y

        self._pre  = lin.register_forward_pre_hook(pre_hook, with_kwargs=True)
        self._post = lin.register_forward_hook(post_hook, with_kwargs=False)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._pre:  self._pre.remove()
            if self._post: self._post.remove()
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
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=model.config.eos_token_id,
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
            obj = json.load(open(p, "r", encoding="utf-8").read())
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


# External steering vector loader

def _np_from_any(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, list):
        return np.array(obj, dtype=np.float32)
    if isinstance(obj, (float, int)):
        return np.array([obj], dtype=np.float32)
    raise ValueError("Unsupported JSON content for vector.")

def load_external_vector(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Steer vector not found: {p}")
    suf = p.suffix.lower()
    if suf == ".npy":
        arr = np.load(p)
    elif suf == ".npz":
        npz = np.load(p)
        if "vec" in npz:
            arr = npz["vec"]
        else:
            keys = list(npz.keys())
            if not keys:
                raise SystemExit("NPZ file contains no arrays.")
            arr = npz[keys[0]]
    elif suf == ".pt":
        t = torch.load(p, map_location="cpu")
        if isinstance(t, torch.Tensor):
            arr = t.detach().cpu().float().numpy()
        elif isinstance(t, dict) and "vec" in t and isinstance(t["vec"], torch.Tensor):
            arr = t["vec"].detach().cpu().float().numpy()
        else:
            raise SystemExit("Unsupported .pt content: expected Tensor or dict with 'vec'.")
    elif suf in (".json", ".json5"):
        obj = json.load(open(p, "r", encoding="utf-8"))
        arr = _np_from_any(obj)
    elif suf == ".txt":
        vals = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                parts = [w for w in line.strip().replace(",", " ").split() if w]
                vals.extend([float(x) for x in parts])
        arr = np.array(vals, dtype=np.float32)
    else:
        try:
            obj = json.load(open(p, "r", encoding="utf-8"))
            arr = _np_from_any(obj)
        except Exception as e:
            raise SystemExit(f"Unsupported steer vector format: {e}")

    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim > 2:
        raise SystemExit(f"Steer vector has unsupported ndim={arr.ndim}. Provide [H] or [W,H].")
    return arr


def prepare_vector_matrix(ext_vec: np.ndarray, W: int, H_expect: Optional[int] = None) -> torch.Tensor:
    if ext_vec.ndim == 1:
        H = int(ext_vec.shape[0])
        if (H_expect is not None) and (H != H_expect):
            logging.warning("External vector H=%d differs from model hidden size H=%d. Proceeding anyway.", H, H_expect)
        mat = np.tile(ext_vec[None, :], (W, 1))
    elif ext_vec.ndim == 2:
        Wv, Hv = int(ext_vec.shape[0]), int(ext_vec.shape[1])
        if (H_expect is not None) and (Hv != H_expect):
            logging.warning("External matrix H=%d differs from model hidden size H=%d. Proceeding anyway.", Hv, H_expect)
        if Wv == W:
            mat = ext_vec
        elif Wv > W:
            mat = ext_vec[-W:, :]
        else:
            reps = int(np.ceil(W / Wv))
            mat = np.vstack([ext_vec] * reps)[-W:, :]
    else:
        raise SystemExit(f"Unsupported external vector shape {ext_vec.shape}; need [H] or [W,H].")
    return torch.from_numpy(mat.astype(np.float32)).cpu()


def maybe_normalize(mat: torch.Tensor, target_norm: Optional[float], unit_norm: bool) -> torch.Tensor:
    if not unit_norm and (target_norm is None):
        return mat
    eps = 1e-8
    norms = torch.linalg.norm(mat, dim=1, keepdim=True).clamp_min(eps)
    if unit_norm:
        mat = mat / norms
    if target_norm is not None:
        mat = mat * float(target_norm)
    return mat


# Global vector creation (optional)

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

    # α list / layer / site
    alphas = [float(x) for x in args.alphas.split(",") if x.strip() != ""]
    layer_idx_cfg = args.layer_idx if args.zero_indexed_layer else (args.layer_idx - 1)
    site = args.site
    window_size = args.window_size
    anchor = args.anchor

    # Load external steering vector if provided
    external_vec_mat: Optional[torch.Tensor] = None
    if args.steer_vector_path:
        np_vec = load_external_vector(args.steer_vector_path)
        dummy = "Test"
        pos_dummy = compute_positions_for_anchor(tok, build_chat_prompt(tok, dummy), window_size, anchor)
        if not pos_dummy:
            pos_dummy = list(range(window_size))
        cap = capture_site_vecs_at_positions(base, tok, build_chat_prompt(tok, dummy), device, site, layer_idx_cfg, [pos_dummy[-1]]).per_pos
        H_model = int(cap.shape[1])
        external_vec_mat = prepare_vector_matrix(np_vec, W=window_size, H_expect=H_model)
        external_vec_mat = maybe_normalize(external_vec_mat,
                                           target_norm=args.steer_vector_target_norm,
                                           unit_norm=args.steer_vector_normalize)
        logging.info("Loaded external steer vector: shape=%s | per-row ||v||=%.6f (mean)",
                     tuple(external_vec_mat.shape),
                     float(torch.linalg.norm(external_vec_mat, dim=1).mean().item()))

    # Optional global steering vector (from whitelist pairs)
    global_vec: Optional[GlobalVector] = None
    if args.vector_creation_pairs and (external_vec_mat is None) and (not args.rank1_adapter_dir):
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
            site=site,
            layer_idx=layer_idx_cfg,
            window_size=window_size,
            anchor=anchor,
            mode=mode
        )
        np.save(Path(args.out_dir) / "global_vector.npy", global_vec.vec.numpy())
        with open(Path(args.out_dir) / "global_vector.meta.json", "w", encoding="utf-8") as fh:
            json.dump({"mode": global_vec.kind, "site": site, "layer_idx": args.layer_idx,
                       "zero_indexed_layer": args.zero_indexed_layer,
                       "window_size": window_size, "anchor": anchor}, fh, indent=2)

    # If using rank-1 LoRA steering, load a/b once
    rank1_setup = None
    if args.rank1_adapter_dir:
        layer_for_rank1 = (args.rank1_layer if args.rank1_layer is not None else args.layer_idx)
        layer0 = (layer_for_rank1 if args.zero_indexed_layer else (layer_for_rank1 - 1))
        a_vec, b_vec, lora_scale = load_rank1_from_adapter(args.rank1_adapter_dir, layer0, args.rank1_target)
        rank1_setup = {
            "layer0": layer0,
            "target": args.rank1_target,
            "a_vec": a_vec,  # already includes LoRA coefficient (lora_alpha/r)
            "b_vec": b_vec,
            "lora_scale": lora_scale
        }
        logging.info(f"Loaded rank-1 adapter vectors for layer={layer0}, target={args.rank1_target}; "
                     f"||a||={a_vec.norm().item():.6f}, ||b||={b_vec.norm().item():.6f}, lora_scale={lora_scale:.6f}")

    # Output text file for answers
    txt_path = Path(args.out_dir) / "results.txt"
    tf = open(txt_path, "w", encoding="utf-8")

    # Aggregate KL for overall plots
    agg_kl: Dict[float, List[float]] = defaultdict(list)

    for it in eval_items:
        logging.info(f"Pair: prompt_count={it.prompt_count} | key={it.paraphrase_key}")

        orig_prompt = build_chat_prompt(tok, it.instruction_original, it.inp)
        para_prompt = build_chat_prompt(tok, it.paraphrase_text, it.inp)

        # Anchor positions
        pos_src_prompt = orig_prompt if args.delta_context == "original" else para_prompt
        pos_orig = compute_positions_for_anchor(tok, pos_src_prompt, window_size, anchor)
        pos_para = compute_positions_for_anchor(tok, para_prompt, window_size, anchor)
        if not pos_orig or not pos_para:
            logging.warning("Empty positions for this pair; skipping.")
            continue
        W = min(len(pos_orig), len(pos_para), window_size)
        pos_orig = pos_orig[-W:]
        # allow decode-time persistent steering by adding -1
        pos_para_for_rank1 = pos_para[-W:] + ([-1] if args.enable_decode_time_steer else [])
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

        # Determine steering matrix for non-rank1 paths
        use_global = args.use_global_vector and (global_vec is not None) and (external_vec_mat is None) and (not args.rank1_adapter_dir)

        if args.rank1_adapter_dir:
            # Rank-1 path uses a/b; no ft_mat/bo_mat needed
            pass
        elif external_vec_mat is not None:
            ft_mat = external_vec_mat[-W:, :].clone()
            if args.steer_vector_mode == "delta":
                bo_mat = torch.zeros_like(ft_mat)
                logging.info(f"[EXTERNAL delta] ||v|| per-row mean={float(torch.linalg.norm(ft_mat,dim=1).mean().item()):.6f} (W={W})")
            else:
                bo_mat = None
                logging.info(f"[EXTERNAL abs] target per-row ||v|| mean={float(torch.linalg.norm(ft_mat,dim=1).mean().item()):.6f} (W={W})")
        elif use_global:
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
            cap_ft_src  = capture_site_vecs_at_positions(ft,   tok, pos_src_prompt, device, site, layer_idx_cfg, pos_orig).per_pos
            if args.steer_mode == "ft_delta":
                cap_bo_src  = capture_site_vecs_at_positions(base, tok, pos_src_prompt, device, site, layer_idx_cfg, pos_orig).per_pos
                delta_norm = (cap_ft_src - cap_bo_src).norm().item()
                logging.info(f"[PER-EX] ||Δ|| at {site}/layer {layer_idx_cfg} (W={W}): {delta_norm:.6f}")
                ft_mat = cap_ft_src[-W:, :]
                bo_mat = cap_bo_src[-W:, :]
            else:
                ft_norm = cap_ft_src.norm().item()
                logging.info(f"[PER-EX] ||FT_src|| at {site}/layer {layer_idx_cfg} (W={W}): {ft_norm:.6f}")
                ft_mat = cap_ft_src[-W:, :]
                bo_mat = None

        # If user forced a steer_vector_mode, override steer_mode for application (non-rank1 paths)
        apply_mode = args.steer_mode
        if external_vec_mat is not None:
            apply_mode = "ft_delta" if args.steer_vector_mode == "delta" else "ft_abs"

        # α sweep
        kl_points: List[Tuple[float, float]] = []
        answers_by_alpha: Dict[float, str] = {}

        para_logits_unsteered = logits_last_token(base, tok, para_prompt, device)
        kl0 = float(symm_kl(target_logits, para_logits_unsteered).item())
        kl_points.append((0.0, kl0))
        agg_kl[0.0].append(kl0)
        answers_by_alpha[0.0] = base_ans_para

        for a in alphas:
            if a == 0.0:
                continue

            if args.rank1_adapter_dir:
                # Keep the rank-1 hook active for the whole forward/generation
                with Rank1LoRAHook(base,
                                   layer_idx=rank1_setup["layer0"],
                                   target=rank1_setup["target"],
                                   a_vec=rank1_setup["a_vec"],
                                   b_vec=rank1_setup["b_vec"],
                                   alpha=a,
                                   positions=pos_para_for_rank1,
                                   single_use=False):
                    lgs = logits_last_token(base, tok, para_prompt, device)
                    klv = float(symm_kl(target_logits, lgs).item())
                    kl_points.append((a, klv))
                    agg_kl[a].append(klv)
                    ans = generate_text(base, tok, para_prompt, device, args.max_new_tokens,
                                        do_sample=args.do_sample, temperature=args.temperature,
                                        top_p=args.top_p, seed=args.seed)
                    answers_by_alpha[a] = ans

            else:
                # Non-rank1 existing paths
                if apply_mode == "ft_delta" and (bo_mat is None):
                    bo_arg = torch.zeros_like(ft_mat[-W:, :])
                else:
                    bo_arg = (bo_mat[-W:, :] if bo_mat is not None else None)

                with PatchHook(base, site, layer_idx_cfg, apply_mode, a,
                               ft_orig=ft_mat[-W:, :], base_orig=bo_arg,
                               positions=pos_para[-W:], single_use=True):

                    if args.enable_decode_time_steer:
                        if apply_mode == "ft_delta":
                            decode_vec = (ft_mat - (bo_mat if bo_mat is not None else 0.0)).mean(dim=0, keepdim=True)
                        else:
                            decode_vec = ft_mat.mean(dim=0, keepdim=True)

                        with PatchHook(base, site, layer_idx_cfg, apply_mode, a,
                                       ft_orig=None, base_orig=None,
                                       positions=[-1], single_use=False,
                                       decode_vec=decode_vec):
                            lgs = logits_last_token(base, tok, para_prompt, device)
                            klv = float(symm_kl(target_logits, lgs).item())
                            kl_points.append((a, klv))
                            agg_kl[a].append(klv)
                            ans = generate_text(base, tok, para_prompt, device, args.max_new_tokens,
                                                do_sample=args.do_sample, temperature=args.temperature,
                                                top_p=args.top_p, seed=args.seed)
                            answers_by_alpha[a] = ans
                    else:
                        lgs = logits_last_token(base, tok, para_prompt, device)
                        klv = float(symm_kl(target_logits, lgs).item())
                        kl_points.append((a, klv))
                        agg_kl[a].append(klv)
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

        if args.rank1_adapter_dir:
            tf.write(f"Rank-1 steering: layer={rank1_setup['layer0']} target={rank1_setup['target']} (LoRA coeff included)\n")
        elif external_vec_mat is not None:
            tf.write(f"External steer vector: mode={args.steer_vector_mode} | shape={tuple(external_vec_mat.shape)}\n")
        elif global_vec is not None and args.use_global_vector:
            tf.write(f"Global vector mode: {global_vec.kind}\n")

        tf.write("\nSteered answers:\n")
        for a in sorted(answers_by_alpha.keys()):
            mode_label = "rank1_lora" if args.rank1_adapter_dir else apply_mode
            where = f"{args.rank1_target}@layer {rank1_setup['layer0']}" if args.rank1_adapter_dir else f"{site}/layer {layer_idx_cfg}"
            tf.write(f"[alpha={a:.3f}] {mode_label} @ {where}:\n{answers_by_alpha[a]}\n")

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
        title_where = f"{args.rank1_target}@L{rank1_setup['layer0']}" if args.rank1_adapter_dir else f"{site}/L{layer_idx_cfg}"
        plt.title(f"(ii) α-sweep | pc={it.prompt_count} | {it.paraphrase_key}\nmode={'rank1_lora' if args.rank1_adapter_dir else apply_mode} site={title_where} anchor={anchor}")
        plt.grid(alpha=0.2)
        out_png = Path(args.out_dir) / f"alpha_sweep_pc{it.prompt_count}_{it.paraphrase_key}.png"
        plt.savefig(out_png, dpi=160)
        plt.close()
        logging.info(f"Wrote plot: {out_png}")

    tf.close()
    logging.info(f"Wrote answers: {txt_path}")

        # Overall plots: mean-only and mean ± 95% CI over pairs
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
                    help="Default steering mode when not using --steer_vector_path.")
    ap.add_argument("--alphas", type=str, default="0,0.5,1.0,2.0,3.0")
    ap.add_argument("--target_logits", type=str, default="ft_original", choices=["base_original","ft_original"],
                    help="Which ORIGINAL to compare against in the KL plot")

    # Global vector creation (optional, unchanged)
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
                    help="Capture FT/BASE vectors on 'original' or 'paraphrase' prompt for steering (non-external modes).")

    # External steering vector options
    ap.add_argument("--steer_vector_path", type=str, default=None,
                    help="Path to external steering vector (.npy/.npz/.pt/.json/.txt). Shape [H] or [W,H].")
    ap.add_argument("--steer_vector_mode", type=str, default="delta", choices=["delta","abs"],
                    help="If using --steer_vector_path: treat vector as a delta direction ('delta') or absolute target ('abs').")
    ap.add_argument("--steer_vector_normalize", action="store_true",
                    help="Normalize each row of the external vector to unit norm before scaling by α.")
    ap.add_argument("--steer_vector_target_norm", type=float, default=None,
                    help="If set, scale each row of the external vector to this L2 norm before α (applied after --steer_vector_normalize).")
    ap.add_argument("--auto_flip_sign", action="store_true",
                    help="Delta mode only: probe ±v at --sign_probe_alpha and pick the sign that reduces KL to target the most.")
    ap.add_argument("--sign_probe_alpha", type=float, default=0.5,
                    help="Alpha used to probe sign when --auto_flip_sign.")

    # Rank-1 LoRA steering options
    ap.add_argument("--rank1_adapter_dir", type=str, default=None,
                    help="Directory with LoRA adapter_model.safetensors & adapter_config.json (rank=1).")
    ap.add_argument("--rank1_target", type=str, default="mlp.down_proj",
                    choices=["mlp.down_proj","mlp.up_proj","mlp.gate_proj",
                             "attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"],
                    help="Which Linear to emulate with the rank-1 LoRA.")
    ap.add_argument("--rank1_layer", type=int, default=None,
                    help="1-based layer index for the target Linear. If omitted, uses --layer_idx.")

    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
