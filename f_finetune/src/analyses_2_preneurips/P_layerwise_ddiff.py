#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layerwise diff-of-diffs (ΔΔ) across MLP & Attention micro-stations with token-site control,
to-original vs paraphrase-only dispersions, jumps and Jacobians
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import os, tempfile
# writable MPL cache dir before importing matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse: PEFT guarded import
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# Reused utils

def _sanitize_attn_mask(mask: torch.Tensor) -> torch.Tensor:
    # Make mask a float tensor (0/1) to avoid dtype promotion issues in model internals.
    if mask is None:
        return None
    if mask.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return mask
    if mask.dtype == torch.bool:
        return mask.to(torch.float32)
    # int64 or anything else -> float32
    return mask.to(torch.float32)

def _choose_plot_layers(all_layers: List[int],
                        stride: int = 1,
                        max_layer: Optional[int] = None,
                        force_layers_csv: str = "") -> List[int]:
    force = {int(x) for x in force_layers_csv.split(",") if x.strip()}
    Ls = [L for L in all_layers if (max_layer is None or L <= max_layer)]
    if stride > 1:
        Ls = Ls[::stride]
    # ensure forced ones (e.g., 12,20) are present if they exist in the run
    Ls = sorted(set(Ls).union(force).intersection(all_layers))
    return Ls

# Re-imported helpers (verbatim compatible with your script)
def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def format_eta(done: int, total: int, start_time: float) -> str:
    if done == 0: return "ETA --:--"
    elapsed = time.time() - start_time
    rate = done / max(elapsed, 1e-6)
    remain = (total - done) / max(rate, 1e-6)
    return f"ETA {int(remain//60):02d}:{int(remain%60):02d} (elapsed {int(elapsed//60):02d}:{int(elapsed%60):02d})"

@dataclass
class PromptSet:
    prompt_count: int
    instruction_original: str
    input_text: str
    paraphrases: List[Tuple[str,str]]  # (key, text)

def load_selection_jsonl(path: str | Path, max_prompts: Optional[int] = None) -> List[PromptSet]:
    items = []
    n = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            paraphrases = [(p["key"], p["text"]) for p in obj.get("paraphrases", []) if isinstance(p, dict)]
            items.append(PromptSet(
                prompt_count=int(obj["prompt_count"]),
                instruction_original=obj.get("instruction_original",""),
                input_text=obj.get("input","") or "",
                paraphrases=paraphrases
            ))
            n += 1
            if max_prompts and n >= max_prompts:
                break
    return items

def encode(tokenizer, text: str, device: torch.device, max_len: int):
    out = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=max_len)
    return out["input_ids"].to(device), out["attention_mask"].to(device)

def mean_pool_tokens(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    s = (x * m.unsqueeze(-1)).sum(1)  # [1,D]
    c = m.sum(1).clamp(min=1.0)       # [1]
    return (s / c.unsqueeze(-1)).squeeze(0).to(torch.float32)  # [D]

def mean_pool_tokens_for_slices(x: torch.Tensor, mask: torch.Tensor, n_slices: int) -> List[Optional[np.ndarray]]:
    assert x.dim() == 3 and mask.dim() == 2 and x.size(0) == 1 and mask.size(0) == 1
    with torch.no_grad():
        pos = (mask[0] > 0).nonzero(as_tuple=False).squeeze(1).cpu().numpy()
        n = int(pos.size)
        if n == 0 or n_slices <= 0:
            return [None] * max(n_slices, 0)
        cuts = np.linspace(0, n, n_slices + 1).astype(int)
        out: List[Optional[np.ndarray]] = []
        for i in range(n_slices):
            sel = pos[cuts[i]:cuts[i + 1]]
            if sel.size == 0:
                out.append(None)
            else:
                m = torch.zeros_like(mask[0], dtype=x.dtype)
                m[torch.from_numpy(sel).to(m.device)] = 1
                s = (x[0] * m.unsqueeze(-1)).sum(dim=0)  # [D]
                c = m.sum().clamp(min=1.0)
                out.append( (s / c).to(torch.float32).cpu().numpy() )
        return out

# MLP accessor & residual hook (reused)
class MLPAccessor:
    def __init__(self, model, layer: int):
        self.model = model
        self.layer = layer
        self.mlp = self._get_mlp_module(model, layer)
        self.kind = self._detect_kind(self.mlp)
        p = next(self.mlp.parameters())
        self.device = p.device
        self.dtype = p.dtype
        self._act_fn = getattr(self.mlp, "act_fn", None) or getattr(self.mlp, "activation_fn", None)
        if self._act_fn is None:
            import torch.nn.functional as F
            self._act_fn = F.silu if self.kind == "swiglu" else F.gelu

    def _get_mlp_module(self, model, i: int):
        for stem, leaf in [("model.model.layers","mlp"), ("model.layers","mlp"), ("transformer.h","mlp")]:
            try:
                base = model
                for part in stem.split("."): base = getattr(base, part)
                return getattr(base[i], leaf)
            except Exception: pass
        raise RuntimeError(f"Could not locate MLP for layer {i}")

    def _detect_kind(self, mlp) -> str:
        if hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"): return "swiglu"
        if any(hasattr(mlp, n) for n in ["wi","fc_in","dense_h_to_4h"]): return "gelu"
        raise RuntimeError("Unknown MLP kind")

    @torch.no_grad()
    def UP(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu": return self.mlp.up_proj(h)
        if hasattr(self.mlp,"wi"): return self.mlp.wi(h)
        if hasattr(self.mlp,"fc_in"): return self.mlp.fc_in(h)
        if hasattr(self.mlp,"dense_h_to_4h"): return self.mlp.dense_h_to_4h(h)
        raise RuntimeError("UP unavailable")

    @torch.no_grad()
    def GATE_PRE(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu": return self.mlp.gate_proj(h)
        if hasattr(self.mlp,"wi"): return self.mlp.wi(h)
        if hasattr(self.mlp,"fc_in"): return self.mlp.fc_in(h)
        if hasattr(self.mlp,"dense_h_to_4h"): return self.mlp.dense_h_to_4h(h)
        raise RuntimeError("GATE_PRE unavailable")

    @torch.no_grad()
    def GATE_ACT(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            return self._act_fn(self.mlp.gate_proj(h))
        import torch.nn.functional as F
        if hasattr(self.mlp,"wi"):            pre = self.mlp.wi(h)
        elif hasattr(self.mlp,"fc_in"):       pre = self.mlp.fc_in(h)
        elif hasattr(self.mlp,"dense_h_to_4h"): pre = self.mlp.dense_h_to_4h(h)
        else: raise RuntimeError("GATE_ACT unavailable")
        return F.gelu(pre)

    @torch.no_grad()
    def POST(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            up = self.mlp.up_proj(h); gate = self._act_fn(self.mlp.gate_proj(h))
            return up * gate
        import torch.nn.functional as F
        if hasattr(self.mlp,"wi"): return F.gelu(self.mlp.wi(h))
        if hasattr(self.mlp,"fc_in"): return F.gelu(self.mlp.fc_in(h))
        if hasattr(self.mlp,"dense_h_to_4h"): return F.gelu(self.mlp.dense_h_to_4h(h))
        raise RuntimeError("POST unavailable")

    @torch.no_grad()
    def DOWN(self, post: torch.Tensor) -> torch.Tensor:
        post = post.to(self.device, self.dtype)
        for name in ["down_proj","wo","fc_out","dense_4h_to_h"]:
            if hasattr(self.mlp, name): return getattr(self.mlp, name)(post)
        raise RuntimeError("DOWN unavailable")

class ResidualHook:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.buffer = None
        self.mlp = self._get_mlp(model, layer)
        self.hook = self.mlp.register_forward_pre_hook(self._hook)

    def _get_mlp(self, model, i):
        for stem, leaf in [("model.model.layers","mlp"), ("model.layers","mlp"), ("transformer.h","mlp")]:
            try:
                base = model
                for part in stem.split("."): base = getattr(base, part)
                return getattr(base[i], leaf)
            except Exception: pass
        raise RuntimeError("Could not find mlp for residual hook")

    def _hook(self, module, inputs):
        self.buffer = inputs[0].detach()  # [B,T,D]

    def close(self):
        try: self.hook.remove()
        except Exception: pass

# Metrics (reused + new)
def cosine_to_original(mat: np.ndarray) -> float:
    if mat.shape[0] < 2: return float("nan")
    x0 = mat[0]; X = mat[1:]
    x0n = x0 / (np.linalg.norm(x0) + 1e-12)
    Xn  = X  / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = (Xn @ x0n)
    return float(cos.mean())

def avg_l2_to_original(mat: np.ndarray) -> float:
    if mat.shape[0] < 2: return float("nan")
    x0 = mat[0]
    diffs = mat[1:] - x0[None, :]
    d = np.sqrt((diffs**2).sum(axis=1))
    return float(d.mean())

def avg_pairwise_l2_without_first(mat: np.ndarray) -> float:
    """Average pairwise L2 among rows 1..N-1 (paraphrase-only)."""
    if mat.shape[0] < 3:  # need at least 2 paraphrases
        return float("nan")
    X = mat[1:]
    n = X.shape[0]
    # upper triangle average
    s = 0.0; k = 0
    for i in range(n):
        d = X[i+1:] - X[i]
        if d.size == 0: continue
        dist = np.linalg.norm(d, axis=1)
        s += float(dist.sum()); k += dist.size
    return s / max(k, 1)

# Small stats helpers (reused)
def mean_ci(vals: np.ndarray) -> Tuple[float,float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: return 0.0, 0.0
    m = float(vals.mean())
    s = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    ci = 1.96 * s / max(1.0, math.sqrt(vals.size))
    return m, ci

def summarize_vec(v: np.ndarray) -> Dict[str, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(mean=float("nan"), median=float("nan"), std=float("nan"), min=float("nan"), max=float("nan"), n=0)
    return dict(mean=float(np.mean(v)), median=float(np.median(v)),
                std=float(np.std(v, ddof=1) if v.size>1 else 0.0),
                min=float(np.min(v)), max=float(np.max(v)), n=int(v.size))

# NEW: Attention tap

class AttnTaps:
    """
    Generic hooks to capture:
      - ATTN_IN: residual stream right BEFORE attention RMSNorm (the norm input)
      - ATTN_OUT: output of attention projection (o_proj/out_proj) BEFORE residual add
    Works across LLaMA/Gemma-like HF layers.
    """
    def __init__(self, model, layer_index: int):
        self.model = model
        self.layer_index = layer_index
        self._layer = self._get_layer(model, layer_index)

        self.buf_in  = None
        self.buf_out = None

        self.hook_in  = None
        self.hook_out = None

        self._install()

    def _get_layer(self, model, i: int):
        for stem in ["model.model.layers", "model.layers", "transformer.h"]:
            try:
                base = model
                for part in stem.split("."): base = getattr(base, part)
                return base[i]
            except Exception: pass
        raise RuntimeError(f"Cannot reach layer {i}")

    def _find_modules(self):
        # Common names
        ln_in  = getattr(self._layer, "input_layernorm", None) or getattr(self._layer, "ln1", None) or getattr(self._layer, "attention_norm", None)
        attn   = getattr(self._layer, "self_attn", None) or getattr(self._layer, "attention", None)
        oproj  = None
        if attn is not None:
            for name in ["o_proj", "out_proj", "dense", "wo", "proj_out"]:
                if hasattr(attn, name):
                    oproj = getattr(attn, name); break
        if ln_in is None or oproj is None:
            raise RuntimeError("Could not find attention norm/proj modules for hooks.")
        return ln_in, oproj

    def _hook_in_fn(self, module, inputs):
        # Inputs to LN: (x,), where x is residual stream pre-attn
        self.buf_in = inputs[0].detach()

    def _hook_out_fn(self, module, inputs, output):
        # output can be a Tensor or a tuple (Tensor, ...)
        if isinstance(output, tuple):
            output = output[0]
        self.buf_out = output.detach()

    def _install(self):
        ln_in, oproj = self._find_modules()
        self.hook_in  = ln_in.register_forward_pre_hook(self._hook_in_fn)
        self.hook_out = oproj.register_forward_hook(self._hook_out_fn)

    def close(self):
        try:
            if self.hook_in:  self.hook_in.remove()
            if self.hook_out: self.hook_out.remove()
        except Exception:
            pass

# NEW: capture whole layer output (residual post)
class LayerOutTap:
    """Capture the layer's forward() output (after MLP residual add) = resid_post."""
    def __init__(self, model, layer_index: int):
        self._layer = self._get_layer(model, layer_index)
        self.buf = None
        self.h = self._layer.register_forward_hook(self._hook)

    def _get_layer(self, model, i: int):
        for stem in ["model.model.layers", "model.layers", "transformer.h"]:
            try:
                base = model
                for part in stem.split("."): base = getattr(base, part)
                return base[i]
            except Exception: pass
        raise RuntimeError(f"Cannot reach layer {i}")

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        self.buf = output.detach()

    def close(self):
        try:
            if self.h: self.h.remove()
        except Exception:
            pass

# Batching & token-site ops

def build_batch(tokenizer, texts: List[str], device: torch.device, max_len: int = 4096):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)

def select_token_vectors(seq_tensor: torch.Tensor,
                         attn_mask: torch.Tensor,
                         site: str,
                         tokenizer,
                         input_ids: torch.Tensor | None = None) -> torch.Tensor:
    """
    site ∈ {"last_prompt_token", "prompt_mean", "first_assistant_token"}.
    Returns [B, D] pooled vectors from [B, T, D] seq_tensor.
    For "first_assistant_token" we assume the caller ran a forward with one-token
    extension (assistant_start token), so we simply pick the last position (T-1).
    """
    B, T, D = seq_tensor.shape
    out = torch.zeros(B, D, dtype=seq_tensor.dtype, device=seq_tensor.device)
    if site == "prompt_mean":
        # mean over valid prompt tokens (mask==1)
        m = attn_mask.to(seq_tensor.dtype)  # [B,T]
        s = (seq_tensor * m.unsqueeze(-1)).sum(1)        # [B,D]
        c = m.sum(1).clamp(min=1.0).unsqueeze(-1)        # [B,1]
        out = (s / c).to(torch.float32)
    elif site == "last_prompt_token":
        idxs = attn_mask.sum(1).to(torch.long) - 1       # [B]
        for b in range(B):
            t = int(idxs[b].item())
            out[b] = seq_tensor[b, t]
        out = out.to(torch.float32)
    elif site == "first_assistant_token":
        # assume caller appended a single assistant start token — pick last token
        out = seq_tensor[:, -1, :].to(torch.float32)
    else:
        raise ValueError(f"Unknown token site {site}")
    return out

# NEW: greedy-generate first K tokens and mean-pool last-position vectors across steps
@torch.no_grad()
def _mean_over_first_k_generated(
    model, tokenizer, layer_idx: int,
    input_ids: torch.Tensor, attention_mask: torch.Tensor,
    k: int,  # configurable via --gen_k
    assistant_start_id: int,
) -> Dict[str, np.ndarray]:
    """
    Greedy-generate k tokens and average station vectors over those k steps at the last position.
    """
    device = input_ids.device
    B = input_ids.size(0)

    # 1) append assistant-start
    ids = torch.cat([input_ids, torch.full((B,1), assistant_start_id, dtype=input_ids.dtype, device=device)], dim=1)
    mask = torch.cat([attention_mask, torch.ones((B,1), dtype=attention_mask.dtype, device=device)], dim=1)

    # taps for this layer
    mlp_acc = MLPAccessor(model, layer_idx)
    res_hook = ResidualHook(model, layer_idx)
    attn_taps = AttnTaps(model, layer_idx)
    layer_out = LayerOutTap(model, layer_idx)

    sums: Dict[str, torch.Tensor] = {}

    for step in range(k):

        mask = _sanitize_attn_mask(mask)
        with torch.inference_mode():
            amp_dtype = next(model.parameters()).dtype  # bf16 or fp16, matches the loaded weights
            with torch.cuda.amp.autocast(enabled=(ids.device.type=="cuda"), dtype=amp_dtype):
                out = model(input_ids=ids, attention_mask=mask)

        Hc = res_hook.buffer          # [B,T,D] pre-MLP residual
        Ai = attn_taps.buf_in         # [B,T,D] ATTN_IN
        Ao = attn_taps.buf_out        # [B,T,D] ATTN_OUT
        Rpost = layer_out.buf         # [B,T,D] resid_post

        # compute micro-stations for the whole seq then pick the last token
        UP_seq       = mlp_acc.UP(Hc)
        GATE_PRE_seq = mlp_acc.GATE_PRE(Hc)
        GATE_ACT_seq = mlp_acc.GATE_ACT(Hc)
        POST_seq     = mlp_acc.POST(Hc)
        DOWN_seq     = mlp_acc.DOWN(POST_seq)

        last = "first_assistant_token"
        vecs = dict(
            RES=select_token_vectors(Hc, mask, last, tokenizer, ids),
            UP=select_token_vectors(UP_seq, mask, last, tokenizer, ids),
            GATE_PRE=select_token_vectors(GATE_PRE_seq, mask, last, tokenizer, ids),
            GATE_ACT=select_token_vectors(GATE_ACT_seq, mask, last, tokenizer, ids),
            POST=select_token_vectors(POST_seq, mask, last, tokenizer, ids),
            DOWN=select_token_vectors(DOWN_seq, mask, last, tokenizer, ids),
            ATTN_IN=select_token_vectors(Ai, mask, last, tokenizer, ids),
            ATTN_OUT=select_token_vectors(Ao, mask, last, tokenizer, ids),
            RESID_PRE=select_token_vectors(Ai, mask, last, tokenizer, ids),
            RESID_MID=select_token_vectors(Hc, mask, last, tokenizer, ids),
            RESID_POST=select_token_vectors(Rpost, mask, last, tokenizer, ids),
        )

        for key, V in vecs.items():
            t = V.to(torch.float32)
            sums[key] = (sums[key] + t) if key in sums else t.clone()

        # greedy next token
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)  # [B,1]
        ids  = torch.cat([ids,  next_tok.to(ids.dtype)], dim=1)
        mask = torch.cat([mask, torch.ones((B,1), dtype=mask.dtype, device=device)], dim=1)

    res_hook.close(); attn_taps.close(); layer_out.close()

    # average
    out_np = {}
    for key, S in sums.items():
        out_np[key] = (S / float(k)).cpu().numpy()
    return out_np

# Core capture across layers & site

STATIONS_MLP  = ["RES", "UP", "GATE_PRE", "GATE_ACT", "POST", "DOWN"]
STATIONS_ATTN = ["ATTN_IN", "ATTN_OUT"]
STATIONS_RESID= ["RESID_PRE", "RESID_MID", "RESID_POST"]
ALL_STATIONS  = STATIONS_MLP + STATIONS_ATTN + STATIONS_RESID

def _get_num_layers(model) -> int:
    for stem in ["model.model.layers", "model.layers", "transformer.h"]:
        try:
            base = model
            for part in stem.split("."): base = getattr(base, part)
            return len(base)
        except Exception: pass
    raise RuntimeError("Cannot determine number of layers")

@torch.no_grad()
def capture_all_stations_layer(model, tokenizer, layer_idx: int,
                               input_ids: torch.Tensor, attention_mask: torch.Tensor,
                               token_site: str,
                               assistant_ids: Optional[torch.Tensor] = None,
                               assistant_mask: Optional[torch.Tensor] = None,
                               assistant_start_token_id: Optional[int] = None,
                               gen_k: int = 10) -> Dict[str, np.ndarray]:
    if token_site == "first10_assistant_mean":
        as_id = assistant_start_token_id if assistant_start_token_id is not None else tokenizer.eos_token_id
        if as_id is None:
            raise RuntimeError("No assistant_start_token_id and tokenizer.eos_token_id is None.")
        return _mean_over_first_k_generated(
            model, tokenizer, layer_idx, input_ids, attention_mask, k=gen_k, assistant_start_id=as_id
        )

    # taps for this layer
    mlp_acc = MLPAccessor(model, layer_idx)
    res_hook = ResidualHook(model, layer_idx)
    attn_taps = AttnTaps(model, layer_idx)
    layer_out = LayerOutTap(model, layer_idx)

    amp_dtype = next(model.parameters()).dtype

    if token_site == "first_assistant_token":
        if assistant_ids is None or assistant_mask is None:
            raise RuntimeError("assistant ids/mask required for first_assistant_token site")
        assistant_mask = _sanitize_attn_mask(assistant_mask)
        with torch.cuda.amp.autocast(enabled=(assistant_ids.device.type=="cuda"), dtype=amp_dtype):
            _ = model(input_ids=assistant_ids, attention_mask=assistant_mask)

        H = res_hook.buffer
        A_in, A_out = attn_taps.buf_in, attn_taps.buf_out
        R_post = layer_out.buf

        site_for_select = "first_assistant_token"
        mask = assistant_mask
        ids_for_select = assistant_ids
    else:
        attention_mask = _sanitize_attn_mask(attention_mask)
        with torch.cuda.amp.autocast(enabled=(input_ids.device.type=="cuda"), dtype=amp_dtype):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        H = res_hook.buffer
        A_in, A_out = attn_taps.buf_in, attn_taps.buf_out
        R_post = layer_out.buf

        site_for_select = token_site
        mask = attention_mask
        ids_for_select = input_ids

    # MLP micro-stations computed from the *sequence* H corresponding to the chosen site
    UP_seq       = mlp_acc.UP(H)
    GATE_PRE_seq = mlp_acc.GATE_PRE(H)
    GATE_ACT_seq = mlp_acc.GATE_ACT(H)
    POST_seq     = mlp_acc.POST(H)
    DOWN_seq     = mlp_acc.DOWN(POST_seq)

    RES_vec        = select_token_vectors(H,       mask, site_for_select, tokenizer, ids_for_select)
    ATTN_IN_vec    = select_token_vectors(A_in,    mask, site_for_select, tokenizer, ids_for_select)
    ATTN_OUT_vec   = select_token_vectors(A_out,   mask, site_for_select, tokenizer, ids_for_select)
    RESID_PRE_vec  = select_token_vectors(A_in,    mask, site_for_select, tokenizer, ids_for_select)
    RESID_MID_vec  = select_token_vectors(H,       mask, site_for_select, tokenizer, ids_for_select)
    RESID_POST_vec = select_token_vectors(R_post,  mask, site_for_select, tokenizer, ids_for_select)

    UP_vec   = select_token_vectors(UP_seq,       mask, site_for_select, tokenizer)
    GPRE_vec = select_token_vectors(GATE_PRE_seq, mask, site_for_select, tokenizer)
    GACT_vec = select_token_vectors(GATE_ACT_seq, mask, site_for_select, tokenizer)
    POST_vec = select_token_vectors(POST_seq,     mask, site_for_select, tokenizer)
    DOWN_vec = select_token_vectors(DOWN_seq,     mask, site_for_select, tokenizer)

    res_hook.close(); attn_taps.close(); layer_out.close()

    return dict(
        RES=RES_vec.cpu().numpy(),
        UP=UP_vec.cpu().numpy(),
        GATE_PRE=GPRE_vec.cpu().numpy(),
        GATE_ACT=GACT_vec.cpu().numpy(),
        POST=POST_vec.cpu().numpy(),
        DOWN=DOWN_vec.cpu().numpy(),
        ATTN_IN=ATTN_IN_vec.cpu().numpy(),
        ATTN_OUT=ATTN_OUT_vec.cpu().numpy(),
        RESID_PRE=RESID_PRE_vec.cpu().numpy(),
        RESID_MID=RESID_MID_vec.cpu().numpy(),
        RESID_POST=RESID_POST_vec.cpu().numpy(),
    )

def _assistant_extended_batch(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, device, assistant_start_token_id: Optional[int] = None):
    """
    Append a single assistant-start token to each sequence.
    Defaults to tokenizer.eos_token if not provided.
    """
    as_id = assistant_start_token_id if assistant_start_token_id is not None else tokenizer.eos_token_id
    if as_id is None:
        raise RuntimeError("No assistant_start_token_id and tokenizer.eos_token_id is None.")
    B = input_ids.size(0)
    add = torch.full((B,1), as_id, dtype=input_ids.dtype, device=device)
    ids2  = torch.cat([input_ids, add], dim=1)
    mask2 = torch.cat([attention_mask, torch.ones(B,1, dtype=attention_mask.dtype, device=device)], dim=1)
    return ids2, mask2

# ΔΔ aggregation over prompts

def family_dispersion(mat: np.ndarray, mode: str) -> float:
    """mat shape = [N_variants, D], first row is ORIGINAL, rest are paraphrases."""
    if mode == "to_original":
        return avg_l2_to_original(mat)
    elif mode == "paraphrase_only":
        return avg_pairwise_l2_without_first(mat)
    else:
        raise ValueError("mode must be to_original | paraphrase_only")

# NEW: normalization modes
def apply_norm_mode(mat: np.ndarray, mode: str, ref_mat_for_match: Optional[np.ndarray]=None) -> np.ndarray:
    """
    mat: [N_variants, D]
    mode:
      - none: return as is
      - unit: row-wise unit L2
      - match_base: scale FT to match BASE mean row-norm (per-site/layer/station/family)
    """
    if mode == "none":
        return mat
    if mode == "unit":
        n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / n
    if mode == "match_base":
        if ref_mat_for_match is None:
            return mat
        nb = float(np.mean(np.linalg.norm(ref_mat_for_match, axis=1)))
        nf = float(np.mean(np.linalg.norm(mat,              axis=1)))
        # avoid NaNs / warnings on degenerate inputs
        if not np.isfinite(nb) or not np.isfinite(nf) or nf < 1e-12:
            return mat
        s = nb / max(nf, 1e-12)
        return mat * s
    raise ValueError("bad norm_mode")

# NEW: chat-template helpers
def format_with_chat_template(tokenizer, instruction: str, input_text: str, system_text: str = "") -> str:
    msgs = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    if input_text:
        content = f"{instruction}\n\nInput:\n{input_text}"
    else:
        content = instruction
    msgs.append({"role": "user", "content": content})
    txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return txt

def pick_assistant_start_id(tokenizer, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("No eos_token_id; set --assistant_start_token_id explicitly.")
    return tokenizer.eos_token_id

def compute_layerwise_ddiff_for_model_pair(
    base, tok_base, ft, tok_ft,
    items: List[PromptSet],
    layers: List[int],
    device: torch.device,
    token_sites: List[str],
    max_paraphrases: Optional[int],
    batch_size_texts: int,
    assistant_start_token_id: Optional[int] = None,
    norm_mode: str = "none",
    apply_chat_template_flag: int = 0,
    system_prompt: str = "",
    max_seq_len: int = 4096,
    pad_to_multiple_of: int = 8,
    families_per_batch: int = 1,
    gen_k: int = 10,
) -> Dict[str, Any]:
    """
    Returns:
      results[(token_site, mode)] -> dict with keys:
        'layers': List[int]
        'stations': ALL_STATIONS
        'D_BASE':  np.ndarray [L, S] mean over prompt families
        'D_FT':    np.ndarray [L, S]
        'DDIFF':   np.ndarray [L, S] = FT - BASE
        'CI_BASE', 'CI_FT', 'CI_DDIFF': same shapes with 95% CI
        plus per-layer per-station raw vectors in 'raw' for optional debugging.
    """
    logging.info("Starting layerwise ΔΔ over %d prompts; token sites: %s", len(items), token_sites)
    start_all = time.time()

    def _tokenize_batch(texts: List[str], tokenizer):
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            pad_to_multiple_of=pad_to_multiple_of if pad_to_multiple_of and pad_to_multiple_of > 0 else None,
        )

    # Pre-slice prompt families into batches of texts (original + paraphrases)
    families = []
    for ps in items:
        raw_texts = [ps.instruction_original] + [t for _, t in ps.paraphrases]
        if max_paraphrases is not None:
            raw_texts = raw_texts[: 1 + max_paraphrases]
        if len(raw_texts) < 2:
            continue
        if apply_chat_template_flag:
            texts = [format_with_chat_template(tok_base, t, ps.input_text, system_text=system_prompt) for t in raw_texts]
        else:
            texts = raw_texts
        families.append((ps.prompt_count, texts))

    L = len(layers)
    S = len(ALL_STATIONS)

    # per (site, mode) we collect: per-layer per-station list of per-family dispersion values for BASE/FT
    raw = {}
    for site in token_sites:
        for mode in ["to_original", "paraphrase_only"]:
            raw[(site, mode)] = dict(BASE=[[[] for _ in range(S)] for _ in range(L)],
                                     FT  =[[[] for _ in range(S)] for _ in range(L)])

    # Process families in bigger batches to saturate GPU
    total_fams = len(families)
    processed_fams = 0
    for batch_start in range(0, total_fams, max(1, families_per_batch)):
        batch_fams = families[batch_start: batch_start + max(1, families_per_batch)]
        # Stitch texts and remember per-family row ranges
        texts_all: List[str] = []
        ranges: List[Tuple[int,int]] = []
        cursor = 0
        for _, texts in batch_fams:
            texts_all.extend(texts)
            nxt = cursor + len(texts)
            ranges.append((cursor, nxt))
            cursor = nxt
        # Tokenize once; if tokenizers are identical, reuse encodings
        encB = _tokenize_batch(texts_all, tok_base)
        if tok_ft is tok_base:
            encF = encB
        else:
            encF = _tokenize_batch(texts_all, tok_ft)
        inB, mB = encB["input_ids"].to(device), encB["attention_mask"].to(device)
        inF, mF = encF["input_ids"].to(device), encF["attention_mask"].to(device)

        def _maybe_extend(tokenizer, ids, mask):
            return _assistant_extended_batch(tokenizer, ids, mask, device, assistant_start_token_id)

        # For each layer, capture stations for this batch (batched)
        for li, layer_idx in enumerate(layers):
            for site in token_sites:
                if site == "first_assistant_token":
                    ids2B, m2B = _maybe_extend(tok_base, inB, mB)
                    capB = capture_all_stations_layer(base, tok_base, layer_idx, inB, mB, site, ids2B, m2B, assistant_start_token_id, gen_k=gen_k)
                    ids2F, m2F = _maybe_extend(tok_ft, inF, mF)
                    capF = capture_all_stations_layer(ft, tok_ft, layer_idx, inF, mF, site, ids2F, m2F, assistant_start_token_id, gen_k=gen_k)
                elif site == "first10_assistant_mean":
                    capB = capture_all_stations_layer(base, tok_base, layer_idx, inB, mB, site, assistant_start_token_id=assistant_start_token_id, gen_k=gen_k)
                    capF = capture_all_stations_layer(ft, tok_ft, layer_idx, inF, mF, site, assistant_start_token_id=assistant_start_token_id, gen_k=gen_k)
                else:
                    capB = capture_all_stations_layer(base, tok_base, layer_idx, inB, mB, site)
                    capF = capture_all_stations_layer(ft, tok_ft, layer_idx, inF, mF, site)

                # Build matrices per station: [N_variants, D]
                for si, st in enumerate(ALL_STATIONS):
                    # Split back per-family using recorded ranges
                    for fam_idx, (_pc, _texts) in enumerate(batch_fams):
                        s, e = ranges[fam_idx]
                        matB = capB[st][s:e]
                        matF = capF[st][s:e]

                        # --- record mean row-norms (per family) ---
                        mean_norm_B = float(np.mean(np.linalg.norm(matB, axis=1)))
                        mean_norm_F = float(np.mean(np.linalg.norm(matF, axis=1)))

                        # Optional normalization to equalize BASE/FT scale
                        if norm_mode == "unit":
                            matBn = apply_norm_mode(matB, "unit")
                            matFn = apply_norm_mode(matF, "unit")
                        elif norm_mode == "match_base":
                            matBn = matB
                            matFn = apply_norm_mode(matF, "match_base", ref_mat_for_match=matB)
                        else:
                            matBn, matFn = matB, matF

                        # Dispersion metrics (L2 only)
                        dB_to = family_dispersion(matBn, "to_original")
                        dB_po = family_dispersion(matBn, "paraphrase_only")
                        dF_to = family_dispersion(matFn, "to_original")
                        dF_po = family_dispersion(matFn, "paraphrase_only")

                        raw[(site,"to_original")]["BASE"][li][si].append(dB_to)
                        raw[(site,"to_original")]["FT"  ][li][si].append(dF_to)
                        raw[(site,"paraphrase_only")]["BASE"][li][si].append(dB_po)
                        raw[(site,"paraphrase_only")]["FT"  ][li][si].append(dF_po)

                        # Norm stash to save later
                        raw.setdefault(("__NORMS__", site), dict(BASE=[[[] for _ in range(S)] for _ in range(L)],
                                                                 FT  =[[[] for _ in range(S)] for _ in range(L)]))
                        raw[("__NORMS__", site)]["BASE"][li][si].append(mean_norm_B)
                        raw[("__NORMS__", site)]["FT"  ][li][si].append(mean_norm_F)

        processed_fams += len(batch_fams)
        if processed_fams % max(1, 2*families_per_batch) == 0 or processed_fams == total_fams:
            logging.info("Processed %d/%d prompt families — %s",
                         processed_fams, total_fams,
                         format_eta(processed_fams, total_fams, start_all))

    # Aggregate means & CIs per-layer per-station
    out = {}
    for site in token_sites:
        for mode in ["to_original","paraphrase_only"]:
            BASE_vals = raw[(site,mode)]["BASE"]
            FT_vals   = raw[(site,mode)]["FT"]

            D_BASE = np.zeros((L,S), dtype=float)
            D_FT   = np.zeros((L,S), dtype=float)
            DDIFF  = np.zeros((L,S), dtype=float)
            CI_BASE= np.zeros((L,S), dtype=float)
            CI_FT  = np.zeros((L,S), dtype=float)
            CI_DD  = np.zeros((L,S), dtype=float)

            for li in range(L):
                for si in range(S):
                    vb = np.array(BASE_vals[li][si], dtype=float)
                    vf = np.array(FT_vals[li][si], dtype=float)
                    mb, cib = mean_ci(vb); mf, cif = mean_ci(vf)
                    dd = vf - vb
                    md, cid = mean_ci(dd)

                    D_BASE[li,si] = mb; CI_BASE[li,si] = cib
                    D_FT[li,si]   = mf; CI_FT[li,si]   = cif
                    DDIFF[li,si]  = md; CI_DD[li,si]  = cid

            out[(site,mode)] = dict(
                layers=layers, stations=ALL_STATIONS,
                D_BASE=D_BASE, D_FT=D_FT, DDIFF=DDIFF,
                CI_BASE=CI_BASE, CI_FT=CI_FT, CI_DDIFF=CI_DD,
                raw=raw[(site,mode)],
            )

    # write norm CSVs per site + store means & CI in dd_maps under ("__NORMS__", site)
    for site in token_sites:
        if ("__NORMS__", site) in raw:
            BASE_vals = raw[("__NORMS__", site)]["BASE"]  # list[L][S] -> list of per-family means
            FT_vals   = raw[("__NORMS__", site)]["FT"]

            NB  = np.zeros((L,S)); NF  = np.zeros((L,S)); ND  = np.zeros((L,S))
            CIB = np.zeros((L,S)); CIF = np.zeros((L,S)); CID = np.zeros((L,S))

            for li in range(L):
                for sj in range(S):
                    vb = np.array(BASE_vals[li][sj], dtype=float)
                    vf = np.array(FT_vals  [li][sj], dtype=float)
                    # means & CI for BASE and FT
                    mb, cib = mean_ci(vb); mf, cif = mean_ci(vf)
                    # differences FT−BASE per family
                    dd = vf - vb
                    md, cid = mean_ci(dd)

                    NB[li,sj], CIB[li,sj] = mb, cib
                    NF[li,sj], CIF[li,sj] = mf, cif
                    ND[li,sj], CID[li,sj] = md, cid

            # CSVs (as before)
            norm_dir = ensure_dir(Path(outdir) / f"{site}_norms")
            write_matrix_csv(NB, layers, ALL_STATIONS, norm_dir / "norms_BASE.csv")
            write_matrix_csv(NF, layers, ALL_STATIONS, norm_dir / "norms_FT.csv")
            write_matrix_csv(ND, layers, ALL_STATIONS, norm_dir / "norms_FT_minus_BASE.csv")

            # Also expose in dd_maps for plotting
            out[("__NORMS__", site)] = dict(
                layers=layers, stations=ALL_STATIONS,
                MEAN_BASE=NB, CI_BASE=CIB,
                MEAN_FT=NF,   CI_FT=CIF,
                MEAN_DIFF=ND, CI_DIFF=CID,
            )

    return out

def _z_from_coverage(coverage: float) -> float:
    """
    coverage is the central mass in (0,1): e.g. 0.95 (95%), 0.10 (10%), 0.05 (5%)
    returns z so that P(|Z|<=z)=coverage for Z~N(0,1)
    """
    try:
        from statistics import NormalDist
        return NormalDist().inv_cdf(0.5*(1.0+float(coverage)))
    except Exception:
        table = {0.95: 1.959964, 0.90: 1.644854, 0.10: 0.125661, 0.05: 0.062706}
        return table.get(float(coverage), 1.959964)

# Jumps & per-layer MLP ddiff stat
def compute_jumps(out_maps: Dict[Tuple[str,str], Dict[str,Any]],
                  layers: List[int]) -> Dict[str, Any]:
    key = ("last_prompt_token", "to_original")
    if key not in out_maps:
        key = ("prompt_mean", "to_original")
    d = out_maps[key]
    D = d["DDIFF"]                     # [L_all, S]
    all_layers = d["layers"]           # full layer list used in compute
    s_idx = {s:i for i,s in enumerate(ALL_STATIONS)}

    def at(layer_val, station):
        try:
            li = all_layers.index(layer_val)
        except ValueError:
            # nearest layer if exact not present
            li = int(np.argmin([abs(L-layer_val) for L in all_layers]))
        return float(D[li, s_idx[station]])

    # layer 12 slice
    dd_before_attn = at(12, "ATTN_IN")
    dd_after_attn  = at(12, "ATTN_OUT")
    j_attn = dd_after_attn - dd_before_attn
    dd_before_mlp = at(12, "RES")
    dd_after_mlp  = at(12, "RESID_POST")
    j_mlp = dd_after_mlp - dd_before_mlp

    # per-layer jumps over requested (plot) layers
    J_UP   = np.array([ at(L,"UP")       - at(L,"RES")      for L in layers ], dtype=float)
    J_GATE = np.array([ at(L,"GATE_ACT") - at(L,"GATE_PRE") for L in layers ], dtype=float)
    J_DOWN = np.array([ at(L,"DOWN")     - at(L,"POST")     for L in layers ], dtype=float)

    # find the actual layer value closest to 12 for reporting
    try:
        idx12 = layers.index(12)
    except ValueError:
        idx12 = int(np.argmin([abs(L-12) for L in layers]))
    return dict(
        idx12=idx12, layer12=layers[idx12],
        dd_before_attn=dd_before_attn, dd_after_attn=dd_after_attn, j_attn=j_attn,
        dd_before_mlp=dd_before_mlp, dd_after_mlp=dd_after_mlp, j_mlp=j_mlp,
        J_UP=J_UP, J_GATE=J_GATE, J_DOWN=J_DOWN
    )

# Plotting utilities

# Palette order (requested):
#PALETTE = ["#0413C5", "#7E74D0", "#4D5055", "#ABAABF", "#757584", "#030030", "#524384", "#50515F", "#8C89AF"]
PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # teal
    "#7f7f7f",  # gray (kept single)
    "#bcbd22",  # yellow-green
    "#393b79",  # deep indigo
    "#e6550d",  # burnt orange
    "#31a354",  # rich green
    "#756bb1",  # muted purple
    "#636363",  # mid gray
    "#3182bd",  # bright blue
]

def _bar_with_ci(ax, x, y, ci, label=None, color=None, width=0.75):
    ax.bar(x, y, yerr=ci, width=width, color=color, edgecolor="none", alpha=0.95, label=label)
    ax.axhline(0.0, color="#808080", linewidth=0.9, alpha=0.8)

def plot_layer_output_norms(dd_maps: Dict[Tuple[str,str], Dict[str,Any]],
                            outdir: Path, layers: List[int], token_site: str):
    key = ("__NORMS__", token_site)
    if key not in dd_maps:
        logging.info("No norms found for site=%s; skipping layer-output norms plot.", token_site)
        return

    nm = dd_maps[key]
    stations   = nm["stations"]           # ALL_STATIONS
    all_layers = nm["layers"]             # full computed set
    # map caller-requested layers -> row idx in the full matrices
    picked_layers = [L for L in layers if L in all_layers]
    idx = [all_layers.index(L) for L in picked_layers]
    if not idx:
        logging.warning("plot_layer_output_norms: none of requested layers found; skipping.")
        return
    x = np.array(picked_layers)

    meanB = nm["MEAN_BASE"][idx, :]
    meanF = nm["MEAN_FT"][idx, :]
    meanD = nm["MEAN_DIFF"][idx, :]
    ciB   = nm["CI_BASE"][idx, :]
    ciF   = nm["CI_FT"][idx, :]
    ciD   = nm["CI_DIFF"][idx, :]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), sharex=True)
    axL, axR = axes

    for si, st in enumerate(stations):
        c = PALETTE[si % len(PALETTE)]
        axL.plot(x, meanB[:, si], label=f"{st} BASE", linewidth=1.6, color=c, alpha=0.55)
        axL.plot(x, meanF[:, si], label=f"{st} FT",   linewidth=1.8, color=c)
        axL.fill_between(x, (meanF[:, si]-ciF[:, si]), (meanF[:, si]+ciF[:, si]), color=c, alpha=0.12, linewidth=0)

    axL.set_title(f"Layer-output norms — BASE (faint) vs FT (solid)\nsite: {token_site}")
    axL.set_xlabel("Layer"); axL.set_ylabel("Mean L2 norm")
    axL.grid(alpha=0.15); axL.legend(ncol=3, fontsize=8)

    for si, st in enumerate(stations):
        c = PALETTE[si % len(PALETTE)]
        axR.plot(x, meanD[:, si], label=st, linewidth=1.8, color=c)
        axR.fill_between(x, (meanD[:, si]-ciD[:, si]), (meanD[:, si]+ciD[:, si]), color=c, alpha=0.12, linewidth=0)

    axR.axhline(0.0, color="#666", linewidth=0.9)
    axR.set_title(f"FT − BASE norm difference (mean ± 95% CI)\nsite: {token_site}")
    axR.set_xlabel("Layer"); axR.set_ylabel("Δ norm")
    axR.grid(alpha=0.15); axR.legend(ncol=3, fontsize=8)

    # no half-layer ticks:
    for ax in (axL, axR):
        ax.set_xlim(x[0], x[-1]); ax.margins(x=0)
        ax.xaxis.set_major_locator(FixedLocator(x))

    plt.tight_layout()
    norm_dir = ensure_dir(outdir / f"{token_site}_norms")
    plt.savefig(norm_dir / f"layer_output_norms_{token_site}.png", dpi=170)
    plt.close()

def plot_station_microviews_for_site(out_maps, outdir: Path, layers: List[int],
                                     token_site: str, target_layers=(12,), mode="to_original"):
    """
    Replicates the 'layerX_mlp_micro_ddiff.png' bar plot but per TOKEN SITE.
    Saves to: <outdir>/microviews_by_site/layer{L}_mlp_micro_ddiff_{token_site}.png
    """
    key = (token_site, mode)
    if key not in out_maps:
        logging.warning("No ddiff map for site=%s, mode=%s; skipping.", token_site, mode)
        return
    d = out_maps[key]
    D = d["DDIFF"]  # [L, S]
    s_idx = [ALL_STATIONS.index(s) for s in STATIONS_MLP]

    outdir2 = ensure_dir(outdir / "microviews_by_site")
    for target in target_layers:
        all_layers = d["layers"]
        try:
            li = all_layers.index(target)
        except ValueError:
            li = int(np.argmin([abs(l-target) for l in all_layers]))
            target = all_layers[li]

        vals = [D[li, i] for i in s_idx]
        cis  = [d["CI_DDIFF"][li, i] for i in s_idx]
        labels = STATIONS_MLP
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
        plt.figure(figsize=(9.5,4.6))
        plt.bar(labels, vals, yerr=cis, color=colors, edgecolor="none", alpha=0.95)
        plt.axhline(0.0, color="#666", linewidth=0.9)
        plt.ylabel("ΔΔ (FT − BASE) — L2")
        plt.title(f"Layer {target}: six MLP micro-stations — site: {token_site} — mode: {mode}")
        plt.tight_layout()
        plt.savefig(outdir2 / f"layer{target}_mlp_micro_ddiff_{token_site}.png", dpi=170)
        plt.close()

def save_norm_adjusted_ddiff(dd_maps, outdir: Path, layers, token_site: str,
                             ref_mode: str = "layer_output",  # "layer_output" | "self"
                             ref_station: str = "RESID_POST", eps: float = 1e-9):
    """
    Create norm-adjusted ΔΔ = ΔΔ / mean_base_norm_ref.

    ref_mode="layer_output": divide every station by BASE mean norm of ref_station (default RESID_POST) per layer.
    ref_mode="self": divide each station s by its own BASE mean norm per layer.
    """
    key_norms = ("__NORMS__", token_site)
    if key_norms not in dd_maps:
        logging.info("No norms for site=%s; skipping norm-adjusted ΔΔ.", token_site)
        return

    nm = dd_maps[key_norms]
    mean_base = nm["MEAN_BASE"]  # [L, S]

    if ref_mode == "layer_output":
        s_idx_ref = ALL_STATIONS.index(ref_station)
        denom = np.maximum(mean_base[:, s_idx_ref][:, None], eps)  # [L,1]
        suffix = f"by_{ref_station.lower()}"
    elif ref_mode == "self":
        denom = np.maximum(mean_base, eps)  # [L,S]
        suffix = "by_self"
    else:
        raise ValueError("ref_mode must be 'layer_output' or 'self'")

    for mode in ["to_original", "paraphrase_only"]:
        d = dd_maps[(token_site, mode)]
        adj = d["DDIFF"] / denom
        sub = ensure_dir(outdir / f"{token_site}_{mode}")
        write_matrix_csv(adj, d["layers"], d["stations"], sub / f"DDIFF_normadj_{suffix}.csv")

        # quick plot (subselect rows to caller 'layers')
        all_layers = d["layers"]
        picked_layers = [L for L in layers if L in all_layers]
        idx = [all_layers.index(L) for L in picked_layers]
        if not idx:
            continue
        adj_sub = adj[idx, :]
        x = np.array(picked_layers)

        plt.figure(figsize=(12, 4.6))
        for si, st in enumerate(ALL_STATIONS):
            plt.plot(x, adj_sub[:, si], marker="o", linewidth=1.8, label=st)
        ax = plt.gca()
        ax.set_xlim(x[0], x[-1]); ax.margins(x=0)
        ax.xaxis.set_major_locator(FixedLocator(x))
        plt.axhline(0.0, color="#666", linewidth=0.9, alpha=0.7)
        plt.xlabel("Layer"); plt.ylabel("ΔΔ (norm-adjusted)")
        plt.title(f"ΔΔ (norm-adjusted {suffix}) — site: {token_site} — mode: {mode}")
        plt.legend(ncol=4, fontsize=9)
        plt.tight_layout()
        plt.savefig(sub / f"ddiff_normadj_{suffix}.png", dpi=170)
        plt.close()


from matplotlib.ticker import FixedLocator

def plot_main_curves(out_maps, outdir: Path, layers: List[int], token_site: str):
    site = token_site
    A = out_maps[(site,"to_original")]
    B = out_maps[(site,"paraphrase_only")]

    orig_layers = A["layers"]
    picked_layers = [L for L in layers if L in orig_layers]
    idx = [orig_layers.index(L) for L in picked_layers]
    x = np.array(picked_layers)

    def _make_panel(matrix: np.ndarray, title: str, fname: str):
        M = matrix[idx, :]  # subselect rows
        plt.figure(figsize=(12,4.6))
        for si, st in enumerate(ALL_STATIONS):
            c = PALETTE[si % len(PALETTE)]
            mks = ["o","s","^","D","v","P","X","*","+","x","<",">","h","H","1","2"]
            plt.plot(x, M[:,si], marker=mks[si % len(mks)], linewidth=1.9, label=st, color=c)

        ax = plt.gca()
        ax.set_xlim(x[0], x[-1]); ax.margins(x=0)
        ax.xaxis.set_major_locator(FixedLocator(x))  # <- no half layers
        plt.axhline(0.0, color="#666", linewidth=0.9, alpha=0.7)
        plt.xlabel("Layer"); plt.ylabel("ΔΔ (FT − BASE) — L2 dispersion")
        plt.title(title); plt.legend(ncol=4, fontsize=9)
        plt.tight_layout(); plt.savefig(outdir/fname, dpi=170); plt.close()

    _make_panel(A["DDIFF"], f"ΔΔ (to-original) — token site: {site}", f"ddiff_to_original_{site}.png")
    _make_panel(B["DDIFF"], f"ΔΔ (paraphrase-only) — token site: {site}", f"ddiff_paraphrase_only_{site}.png")
    _make_panel(A["DDIFF"] - B["DDIFF"], f"(to-original) − (paraphrase-only) — token site: {site}", f"ddiff_diff_to_minus_para_{site}.png")

def plot_resid_post_to_minus_para(out_maps, outdir: Path, layers: List[int], token_site: str):
    key_A = (token_site, "to_original")
    key_B = (token_site, "paraphrase_only")
    if key_A not in out_maps or key_B not in out_maps:
        logging.warning("Missing ddiff maps for site=%s; skipping RESID_POST diff plot.", token_site)
        return

    A = out_maps[key_A]["DDIFF"]
    B = out_maps[key_B]["DDIFF"]
    all_layers = out_maps[key_A]["layers"]
    s_idx = ALL_STATIONS.index("RESID_POST")

    picked_layers = [L for L in layers if L in all_layers]
    idx = [all_layers.index(L) for L in picked_layers]
    if not idx:
        logging.warning("resid_post_to_minus_para: none of requested layers found; skipping.")
        return
    x = np.array(picked_layers)
    y = (A[idx, s_idx] - B[idx, s_idx])

    sub = ensure_dir(outdir / f"{token_site}_resid_post_only")
    with open(sub / "resid_post_ddiff_to_minus_para.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "DDIFF_RESID_POST_(to_original_minus_paraphrase_only)"])
        for L, val in zip(x, y):
            w.writerow([L, f"{float(val):.6f}"])

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(x, y, marker="o", linewidth=2.0, label="ΔΔ (to-original − paraphrase-only) at RESID_POST")
    ax.legend(); ax.axhline(0.0, color="#666", linewidth=0.9, alpha=0.8)
    ax.set_xlabel("Layer"); ax.set_ylabel("ΔΔ (to-original − paraphrase-only) @ RESID_POST")
    ax.set_title(f"RESID_POST only — token site: {token_site}")
    ax.set_xlim(x[0], x[-1]); ax.margins(x=0)
    ax.xaxis.set_major_locator(FixedLocator(x))
    plt.tight_layout(); plt.savefig(sub / f"resid_post_ddiff_to_minus_para_{token_site}.png", dpi=170); plt.close()

def plot_within_layer_gap(dd_maps, outdir: Path, layers: List[int],
                          token_site: str, target_layer: int = 12):
    """
    Bar chart of the gap ΔΔ(to-orig) − ΔΔ(para-only) at L=target_layer
    before/after Attention and before/after MLP, all at the SAME layer,
    using stations: ATTN_IN, ATTN_OUT, RES, RESID_POST.
    """
    A = dd_maps[(token_site, "to_original")]["DDIFF"]      # [L,S]
    B = dd_maps[(token_site, "paraphrase_only")]["DDIFF"]  # [L,S]
    gap = A - B                                            # [L,S]

    all_layers = dd_maps[(token_site, "to_original")]["layers"]
    try:
        li = all_layers.index(target_layer)
    except ValueError:
        li = int(np.argmin([abs(l-target_layer) for l in all_layers]))
        target_layer = all_layers[li]

    sindex = {s:i for i,s in enumerate(ALL_STATIONS)}
    vals_before_after = {
        "Attention": (gap[li, sindex["ATTN_IN"]],  gap[li, sindex["ATTN_OUT"]]),
        "MLP":       (gap[li, sindex["RES"]],      gap[li, sindex["RESID_POST"]]),
    }

    labels = list(vals_before_after.keys())
    before = [vals_before_after[k][0] for k in labels]
    after  = [vals_before_after[k][1] for k in labels]

    import numpy as np, matplotlib.pyplot as plt
    x = np.arange(len(labels)); w = 0.38
    plt.figure(figsize=(10,4.6))
    plt.bar(x - w/2, before, width=w, label="before")
    plt.bar(x + w/2, after,  width=w, label="after")
    plt.axhline(0.0, color="#666", linewidth=0.9)
    plt.xticks(x, labels)
    plt.ylabel("ΔΔ(to-orig − para-only) @ L{}".format(target_layer))
    plt.title(f"Within layer {target_layer} gap — site: {token_site}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(outdir)/f"within_layer{target_layer}_gap_{token_site}.png", dpi=170)
    plt.close()

def plot_station_microviews(out_maps, outdir: Path, layers: List[int], default_layers=(6,12)):
    """
    Grouped bars across the six MLP stations for specific layers (default 6, 12),
    token site: primary ('last_prompt_token'), mode: to-original.
    """
    site = "last_prompt_token" if ("last_prompt_token","to_original") in out_maps else "prompt_mean"
    d = out_maps[(site,"to_original")]
    D = d["DDIFF"]  # [L,S]
    s_idx = [ALL_STATIONS.index(s) for s in STATIONS_MLP]

    outdir2 = ensure_dir(outdir / "microviews")
    for target in default_layers:
        all_layers = d["layers"]
        try:
            li = all_layers.index(target)
        except ValueError:
            li = int(np.argmin([abs(l-target) for l in all_layers]))
            target = all_layers[li]

        vals = [D[li, i] for i in s_idx]
        cis  = [d["CI_DDIFF"][li, i] for i in s_idx]
        labels = STATIONS_MLP
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
        plt.figure(figsize=(9.5,4.6))
        plt.bar(labels, vals, yerr=cis, color=colors, edgecolor="none", alpha=0.95)
        plt.axhline(0.0, color="#666", linewidth=0.9)
        plt.ylabel("ΔΔ (FT − BASE) — L2")
        plt.title(f"Layer {target}: six MLP micro-stations — token site: {site}")
        plt.tight_layout(); plt.savefig(outdir2 / f"layer{target}_mlp_micro_ddiff.png", dpi=170); plt.close()

def plot_jumps(jumps: Dict[str,Any], outdir: Path, layers: List[int], ci_map=None):
    # Layer 12 slices report
    with open(outdir / "layer12_jumps.json", "w") as f:
        json.dump(dict(
            layer12=int(jumps["layer12"]),
            dd_before_attn=jumps["dd_before_attn"],
            dd_after_attn=jumps["dd_after_attn"],
            J_attn=jumps["j_attn"],
            dd_before_mlp=jumps["dd_before_mlp"],
            dd_after_mlp=jumps["dd_after_mlp"],
            J_mlp=jumps["j_mlp"],
        ), f, indent=2)

    # Per-layer J_UP/GATE/DOWN with CI if provided
    x = layers
    plt.figure(figsize=(12,4.6))
    plt.plot(x, jumps["J_UP"],   marker="o", color=PALETTE[0], label="J_UP = ΔΔ(UP)−ΔΔ(RES)")
    plt.plot(x, jumps["J_GATE"], marker="o", color=PALETTE[1], label="J_GATE = ΔΔ(GATE_ACT)−ΔΔ(GATE_PRE)")
    plt.plot(x, jumps["J_DOWN"], marker="o", color=PALETTE[2], label="J_DOWN = ΔΔ(DOWN)−ΔΔ(POST)")
    plt.axhline(0.0, color="#666", linewidth=0.9)
    plt.xlabel("Layer"); plt.ylabel("ΔΔ jumps (FT − BASE)")
    plt.legend(); plt.title("Per-layer MLP component jumps")
    plt.tight_layout(); plt.savefig(outdir / "mlp_component_jumps.png", dpi=170); plt.close()

def plot_resid_post_last_vs_mean(dd_maps, outdir, layers):
    needed = [
        ("last_prompt_token","to_original"),
        ("last_prompt_token","paraphrase_only"),
        ("prompt_mean","to_original"),
        ("prompt_mean","paraphrase_only"),
    ]
    if any(k not in dd_maps for k in needed):
        logging.info("Skipping resid_post_last_vs_mean: required token sites not all present.")
        return

    import numpy as np, matplotlib.pyplot as plt
    s_idx = ALL_STATIONS.index("RESID_POST")
    A_last = dd_maps[("last_prompt_token","to_original")]["DDIFF"]
    B_last = dd_maps[("last_prompt_token","paraphrase_only")]["DDIFF"]
    A_mean = dd_maps[("prompt_mean","to_original")]["DDIFF"]
    B_mean = dd_maps[("prompt_mean","paraphrase_only")]["DDIFF"]
    # guard (see §3) then:
    all_layers = dd_maps[("last_prompt_token","to_original")]["layers"]
    picked_layers = [L for L in layers if L in all_layers]
    idx = [all_layers.index(L) for L in picked_layers]
    if not idx:
        logging.warning("resid_post_last_vs_mean: none of requested layers found; skipping.")
        return
    x = np.array(picked_layers)
    y1 = (A_last[idx, s_idx] - B_last[idx, s_idx])
    y2 = (A_mean[idx, s_idx] - B_mean[idx, s_idx])

    plt.figure(figsize=(10,4.2))
    plt.plot(x, y1, marker="o", linewidth=2.0, label="last prompt token")
    plt.plot(x, y2, marker="s", linewidth=2.0, label="prompt mean")
    ax = plt.gca()
    ax.set_xlim(x[0], x[-1]); ax.margins(x=0)
    ax.xaxis.set_major_locator(FixedLocator(x))
    plt.axhline(0.0, color="#666", linewidth=0.9, alpha=0.8)
    plt.xlabel("Layer"); plt.ylabel("ΔΔ(to-orig − para-only) @ RESID_POST")
    plt.title("RESID_POST: last token vs prompt mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(outdir)/"resid_post_last_vs_mean.png", dpi=170)
    plt.close()

def _gap_series_from_raw(dd_maps, token_site: str, layers: List[int], ci_coverage: float = 0.95):
    import numpy as np
    z = _z_from_coverage(ci_coverage)

    A_raw = dd_maps[(token_site, "to_original")]["raw"]
    P_raw = dd_maps[(token_site, "paraphrase_only")]["raw"]
    all_layers = dd_maps[(token_site, "to_original")]["layers"]
    SIDX = {s:i for i, s in enumerate(ALL_STATIONS)}

    # mapping: requested L -> index in raw arrays
    row_idx = [all_layers.index(L) if L in all_layers else int(np.argmin([abs(a-L) for a in all_layers]))
               for L in layers]

    def mean_ci_np(v):
        v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
        if v.size == 0: return 0.0, 0.0
        m = float(v.mean()); s = float(v.std(ddof=1)) if v.size>1 else 0.0
        return m, z * s / max(1.0, np.sqrt(v.size))

    def _gap_per_family(li, si):
        vb_to = np.array(A_raw["BASE"][li][si], dtype=float)
        vf_to = np.array(A_raw["FT"  ][li][si], dtype=float)
        vb_po = np.array(P_raw["BASE"][li][si], dtype=float)
        vf_po = np.array(P_raw["FT"  ][li][si], dtype=float)
        return (vf_to - vb_to) - (vf_po - vb_po)

    def arr(): return np.zeros(len(row_idx), dtype=float)
    G_attn_in, G_attn_out, G_res, G_respost = arr(), arr(), arr(), arr()
    C_attn_in, C_attn_out, C_res, C_respost = arr(), arr(), arr(), arr()
    J_attn, C_J_attn = arr(), arr()
    J_mlp,  C_J_mlp  = arr(), arr()

    for k, li in enumerate(row_idx):
        gi_ai = _gap_per_family(li, SIDX["ATTN_IN"])
        gi_ao = _gap_per_family(li, SIDX["ATTN_OUT"])
        gi_res= _gap_per_family(li, SIDX["RES"])
        gi_rp = _gap_per_family(li, SIDX["RESID_POST"])

        G_attn_in[k],  C_attn_in[k]  = mean_ci_np(gi_ai)
        G_attn_out[k], C_attn_out[k] = mean_ci_np(gi_ao)
        G_res[k],      C_res[k]      = mean_ci_np(gi_res)
        G_respost[k],  C_respost[k]  = mean_ci_np(gi_rp)

        J_attn[k], C_J_attn[k] = mean_ci_np(gi_ao - gi_ai)
        J_mlp[k],  C_J_mlp[k]  = mean_ci_np(gi_rp - gi_res)

    return dict(
        layers=list(layers),
        G_attn_in=(G_attn_in, C_attn_in),
        G_attn_out=(G_attn_out, C_attn_out),
        G_res=(G_res, C_res),
        G_respost=(G_respost, C_respost),
        J_attn=(J_attn, C_J_attn),
        J_mlp=(J_mlp,  C_J_mlp),
    )

def plot_inside_layer_lines(dd_maps, outdir: Path, layers: List[int], token_site: str,
                            ci_coverage: float = 0.95):
    """
    Three figures:
      A) Attention: gap before (ATTN_IN) vs after (ATTN_OUT)
      B) MLP:      gap before (RES)     vs after (RESID_POST)
      C) Jumps:    (after − before) for Attn and for MLP
    """
    import numpy as np, matplotlib.pyplot as plt
    ser = _gap_series_from_raw(dd_maps, token_site, layers, ci_coverage=ci_coverage)

    # sort x to avoid "half layers"
    idx = np.argsort(ser["layers"])
    x   = np.array(ser["layers"])[idx]

    def _plt_two(name, pair1, pair2, ylab, fname):
        (y1, c1) = pair1
        (y2, c2) = pair2
        y1, c1 = np.array(y1)[idx], np.array(c1)[idx]
        y2, c2 = np.array(y2)[idx], np.array(c2)[idx]
        plt.figure(figsize=(11,4.6))
        plt.plot(x, y1, marker="o", linewidth=2.0, label="before")
        plt.fill_between(x, y1-c1, y1+c1, alpha=0.12)
        plt.plot(x, y2, marker="s", linewidth=2.0, label="after")
        plt.fill_between(x, y2-c2, y2+c2, alpha=0.12)
        plt.axhline(0, color="#666", linewidth=0.9)
        plt.xlabel("Layer"); plt.ylabel(ylab); plt.title(name + f" — site: {token_site}")
        plt.legend(); plt.tight_layout()
        plt.savefig(Path(outdir)/fname, dpi=170); plt.close()

    def _plt_one(name, pair, ylab, fname, label):
        (y, c) = pair
        y, c = np.array(y)[idx], np.array(c)[idx]
        plt.figure(figsize=(11,4.2))
        plt.plot(x, y, marker="o", linewidth=2.0, label=label)
        plt.fill_between(x, y-c, y+c, alpha=0.12)
        plt.axhline(0, color="#666", linewidth=0.9)
        plt.xlabel("Layer"); plt.ylabel(ylab); plt.title(name + f" — site: {token_site}")
        plt.legend(); plt.tight_layout()
        plt.savefig(Path(outdir)/fname, dpi=170); plt.close()

    _plt_two("Attention gap (before vs after)",
             ser["G_attn_in"], ser["G_attn_out"],
             "ΔΔ(to-orig − para-only)", f"inside_attn_gap_{token_site}.png")

    _plt_two("MLP gap (before vs after)",
             ser["G_res"], ser["G_respost"],
             "ΔΔ(to-orig − para-only)", f"inside_mlp_gap_{token_site}.png")

    _plt_one("Attention jump (after − before)",
             ser["J_attn"], "ΔΔ gap change", f"inside_attn_jump_{token_site}.png", "J_attn")

    _plt_one("MLP jump (after − before)",
             ser["J_mlp"],  "ΔΔ gap change", f"inside_mlp_jump_{token_site}.png",  "J_mlp")

def plot_layer_bar_gap(dd_maps, outdir: Path, layers: List[int], token_site: str,
                       target_layer: int = 12, ci_coverage: float = 0.95):
    """
    Grouped bars with CI (coverage per ci_coverage) at a single layer (default L12),
    using stations: ATTN_IN/ATTN_OUT and RES/RESID_POST.
    """
    import numpy as np, matplotlib.pyplot as plt
    ser = _gap_series_from_raw(dd_maps, token_site, layers, ci_coverage=ci_coverage)
    try:
        li = ser["layers"].index(target_layer)
    except ValueError:
        li = int(np.argmin([abs(L-target_layer) for L in ser["layers"]]))
        target_layer = ser["layers"][li]

    def pick(pair):  # pair = (vals, cis)
        v, c = pair
        return float(v[li]), float(c[li])

    attn_before, ci_ab = pick(ser["G_attn_in"])
    attn_after,  ci_aa = pick(ser["G_attn_out"])
    mlp_before,  ci_mb = pick(ser["G_res"])
    mlp_after,   ci_ma = pick(ser["G_respost"])

    labels = ["Attention", "MLP"]
    before = [attn_before, mlp_before]; ci_b = [ci_ab, ci_mb]
    after  = [attn_after,  mlp_after];  ci_a = [ci_aa, ci_ma]

    x = np.arange(len(labels)); w = 0.38
    plt.figure(figsize=(9.8,4.6))
    plt.bar(x - w/2, before, yerr=ci_b, width=w, label="before")
    plt.bar(x + w/2, after,  yerr=ci_a, width=w, label="after")
    plt.axhline(0, color="#666", linewidth=0.9)
    plt.xticks(x, labels)
    plt.ylabel(f"ΔΔ(to-orig − para-only) @ L{target_layer}")
    plt.title(f"Within layer {target_layer} gap — site: {token_site}")
    plt.legend(); plt.tight_layout()
    plt.savefig(Path(outdir)/f"within_layer{target_layer}_gap_CI_{token_site}.png", dpi=170)
    plt.close()

def plot_layer12_within_block_gap(dd_maps, outdir: Path, layers: List[int], token_site: str):
    """
    For L=12: compare gap = (ΔΔ_to-original − ΔΔ_paraphrase-only) before/after attention and MLP.
    - Attention: ATTN_IN -> ATTN_OUT
    - MLP:       RES     -> DOWN
    Panel A: before vs after for both blocks
    Panel B: change (after − before); negative means 'more invariant'
    """
    keyA = (token_site, "to_original")
    keyB = (token_site, "paraphrase_only")
    if keyA not in dd_maps or keyB not in dd_maps:
        logging.warning("Missing ddiff maps for site=%s; skipping within-layer plot.", token_site)
        return

    A = dd_maps[keyA]["DDIFF"]
    B = dd_maps[keyB]["DDIFF"]
    s = {name:i for i,name in enumerate(ALL_STATIONS)}

    # find index of layer 12 (or nearest)
    all_layers = dd_maps[keyA]["layers"]
    try:
        li = all_layers.index(12)
    except ValueError:
        li = int(np.argmin([abs(l-12) for l in all_layers]))

    def gap(station):
        return float(A[li, s[station]] - B[li, s[station]])

    # before/after gaps
    gap_attn_before = gap("ATTN_IN")
    gap_attn_after  = gap("ATTN_OUT")
    gap_mlp_before  = gap("RES")
    gap_mlp_after   = gap("RESID_POST")

    # Panel A: before vs after
    labels = ["Attention", "MLP"]
    before_vals = [gap_attn_before, gap_mlp_before]
    after_vals  = [gap_attn_after,  gap_mlp_after]

    x = np.arange(len(labels)); w = 0.35
    plt.figure(figsize=(8.2, 4.4))
    plt.bar(x - w/2, before_vals, width=w, label="before")
    plt.bar(x + w/2, after_vals,  width=w, label="after")
    plt.axhline(0.0, color="#666", lw=0.9)
    plt.xticks(x, labels)
    plt.ylabel("ΔΔ(to-orig − para-only) @ L12")
    plt.title(f"Within layer 12 gap — site: {token_site}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(outdir) / f"layer12_gap_before_after_{token_site}.png", dpi=170)
    plt.close()

    # Panel B: change (after − before)
    deltas = [after_vals[0] - before_vals[0], after_vals[1] - before_vals[1]]
    plt.figure(figsize=(7.0, 4.2))
    plt.bar(labels, deltas)
    plt.axhline(0.0, color="#666", lw=0.9)
    plt.ylabel("Change in gap (after − before)")
    plt.title("Within L12 change (neg = more invariant)")
    plt.tight_layout()
    plt.savefig(Path(outdir) / f"layer12_gap_deltas_{token_site}.png", dpi=170)
    plt.close()

# Jacobian extension

@torch.no_grad()
def jacobian_layer_driver(model, tokenizer, items: List[PromptSet], layers: List[int],
                          device: torch.device, topk_pca: int, mode: str,
                          max_prompts: Optional[int], max_paraphrases: Optional[int],
                          eps: float = 1e-3, directions_random: int = 8,
                          max_seq_len: int = 4096) -> Dict[int, Dict[str, Any]]:
    """
    Reuses your per-layer Jacobian code by calling it per layer, returns a dict layer -> per_prompt rows.
    """
    out = {}
    for li, layer in enumerate(layers):
        logging.info("[jacobian] Layer %d/%d -> %d", li+1, len(layers), layer)
        res = jacobian_norms_for_model(
            model, tokenizer, items, layer, device,
            topk_pca, mode, max_prompts, max_paraphrases, eps=eps, directions_random=directions_random,
            max_seq_len=max_seq_len,
        )
        out[layer] = res["per_prompt"]
    return out

def jacobian_diff_plot(base_map: Dict[int, List[Dict[str,Any]]],
                       ft_map: Dict[int, List[Dict[str,Any]]],
                       outdir: Path):
    """
    Plot grouped bars of (FT − BASE) for (UP vs DOWN) × (MEAN vs VAR) with 95% CI (averaged over prompts),
    per requested spec. Also save CSV.
    """
    layers = sorted(base_map.keys())
    def layer_stats(rows: List[Dict[str,Any]]):
        import pandas as pd
        df = pd.DataFrame(rows)
        return dict(
            up_mean = df["jac_MEAN_up"].to_numpy(),
            up_var  = df["jac_VAR_up_mean"].to_numpy(),
            down_mean = df["jac_MEAN_down"].to_numpy(),
            down_var  = df["jac_VAR_down_mean"].to_numpy(),
        )

    means = {"UP_MEAN":[], "UP_VAR":[], "DOWN_MEAN":[], "DOWN_VAR":[]}
    cis   = {"UP_MEAN":[], "UP_VAR":[], "DOWN_MEAN":[], "DOWN_VAR":[]}

    for L in layers:
        sb = layer_stats(base_map[L])
        sf = layer_stats(ft_map[L])
        for key, (f_arr, b_arr) in {
            "UP_MEAN": (sf["up_mean"],  sb["up_mean"]),
            "UP_VAR":  (sf["up_var"],   sb["up_var"]),
            "DOWN_MEAN": (sf["down_mean"], sb["down_mean"]),
            "DOWN_VAR":  (sf["down_var"],  sb["down_var"]),
        }.items():
            dif = np.array(f_arr, dtype=float) - np.array(b_arr, dtype=float)
            m, c = mean_ci(dif)
            means[key].append(m); cis[key].append(c)

    # Plot as 2x2 grouped bars across layers
    labels = [str(l) for l in layers]
    width = 0.2
    x = np.arange(len(layers))
    series = ["UP_MEAN","UP_VAR","DOWN_MEAN","DOWN_VAR"]
    colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]

    plt.figure(figsize=(max(12, len(layers)*0.7), 5.0))
    for i, s in enumerate(series):
        plt.bar(x + (i-1.5)*width, means[s], yerr=cis[s], width=width, color=colors[i], label=s.replace("_"," / "), edgecolor="none", alpha=0.95)
    plt.axhline(0.0, color="#666", linewidth=0.9)
    plt.xticks(x, labels)
    plt.ylabel("Δ Jacobian norm (FT − BASE)")
    plt.title("Jacobian differences by layer — (UP/DOWN × MEAN/VAR)")
    plt.legend(ncol=2)
    plt.tight_layout(); plt.savefig(outdir / "jacobian_diff_layers.png", dpi=170); plt.close()


def jacobian_diff_plot_norm_adjusted(base_map, ft_map, dd_maps, token_site: str,
                                     outdir: Path,
                                     ref_mode: str = "layer_output",
                                     ref_station: str = "RESID_POST",
                                     eps: float = 1e-9):
    layers = sorted(base_map.keys())

    nm = dd_maps.get(("__NORMS__", token_site))
    if nm is None:
        logging.info("No norms for site=%s; skipping norm-adjusted Jacobian plot.", token_site)
        return

    mean_base = nm["MEAN_BASE"]          # [L_all, S]
    dd_layers = nm["layers"]             # full layer list used for dd_maps
    dd_index  = {L:i for i, L in enumerate(dd_layers)}
    s_idx_ref = ALL_STATIONS.index(ref_station)

    def layer_stats(rows):
        import pandas as pd
        df = pd.DataFrame(rows)
        return dict(
            up_mean = df["jac_MEAN_up"].to_numpy(),
            up_var  = df["jac_VAR_up_mean"].to_numpy(),
            down_mean = df["jac_MEAN_down"].to_numpy(),
            down_var  = df["jac_VAR_down_mean"].to_numpy(),
        )

    means = {"UP_MEAN":[], "UP_VAR":[], "DOWN_MEAN":[], "DOWN_VAR":[]}
    cis   = {"UP_MEAN":[], "UP_VAR":[], "DOWN_MEAN":[], "DOWN_VAR":[]}

    for L in layers:
        # find matching dd_maps row (exact or nearest)
        if L in dd_index:
            i_dd = dd_index[L]
        else:
            i_dd = int(np.argmin([abs(a - L) for a in dd_layers]))
        denom = max(mean_base[i_dd, s_idx_ref], eps)

        sb = layer_stats(base_map[L]); sf = layer_stats(ft_map[L])
        for key, (f_arr, b_arr) in {
            "UP_MEAN": (sf["up_mean"],  sb["up_mean"]),
            "UP_VAR":  (sf["up_var"],   sb["up_var"]),
            "DOWN_MEAN": (sf["down_mean"], sb["down_mean"]),
            "DOWN_VAR":  (sf["down_var"],  sb["down_var"]),
        }.items():
            dif = (np.array(f_arr, dtype=float) - np.array(b_arr, dtype=float)) / denom
            m, c = mean_ci(dif)
            means[key].append(m); cis[key].append(c)

    labels = [str(l) for l in layers]
    width = 0.2
    x = np.arange(len(layers))
    series = ["UP_MEAN","UP_VAR","DOWN_MEAN","DOWN_VAR"]

    plt.figure(figsize=(max(12, len(layers)*0.7), 5.0))
    for k, s in enumerate(series):
        plt.bar(x + (k-1.5)*width, means[s], yerr=cis[s], width=width,
                label=s.replace("_"," / "), edgecolor="none", alpha=0.95)
    plt.axhline(0.0, color="#666", linewidth=0.9)
    plt.xticks(x, labels)
    plt.ylabel("Δ Jacobian (norm-adjusted)")
    plt.title(f"Jacobian differences — norm-adjusted by {ref_station} — site: {token_site}")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / f"jacobian_diff_layers_normadj_by_{ref_station.lower()}.png", dpi=170)
    plt.close()


# CSV Writers (new) 

def write_matrix_csv(matrix: np.ndarray, layers: List[int], stations: List[str], path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + stations)
        for i, L in enumerate(layers):
            w.writerow([L] + [f"{matrix[i,j]:.6f}" for j in range(len(stations))])

# Reuse: Build models/tokenizers   

def build_models_and_tokenizers(
    base_model_name_or_path: str,
    ft_model_name_or_path: Optional[str],
    ft_lora_adapter: Optional[str],
    device: torch.device,
    dtype: str = "bf16",
):
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer_base.pad_token is None:
        tokenizer_base.pad_token = tokenizer_base.eos_token

    # Decide the model dtype
    dtype_str = (dtype or "bf16").lower()
    if dtype_str == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif dtype_str == "fp16":
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif dtype_str == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype '{dtype}'. Use one of: bf16, fp16, fp32.")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path, torch_dtype=torch_dtype
    ).to(device).eval()

    if ft_lora_adapter is not None:
        host_name = ft_model_name_or_path or base_model_name_or_path
        ft_host = AutoModelForCausalLM.from_pretrained(
            host_name, torch_dtype=torch_dtype
        ).to(device).eval()
        if _HAS_PEFT:
            ft = PeftModel.from_pretrained(ft_host, ft_lora_adapter).to(device).eval()
        else:
            raise RuntimeError("peft not available but --ft_lora_adapter provided.")
    else:
        if ft_model_name_or_path is None:
            raise ValueError("Provide --ft_model_name_or_path (merged FT) OR --ft_lora_adapter (LoRA adapter dir).")
        ft = AutoModelForCausalLM.from_pretrained(
            ft_model_name_or_path, torch_dtype=torch_dtype
        ).to(device).eval()

    tokenizer_ft = AutoTokenizer.from_pretrained(ft_model_name_or_path or base_model_name_or_path, use_fast=True)
    if tokenizer_ft.pad_token is None:
        tokenizer_ft.pad_token = tokenizer_ft.eos_token

    return (base, tokenizer_base), (ft, tokenizer_ft)

# CLI / Orchestration              

def main():
    ap = argparse.ArgumentParser(description="Layerwise ΔΔ across MLP+Attention stations with token-site control, jumps, and Jacobians")
    ap.add_argument("--selection_jsonl", type=str, required=True, help="From SCRIPT A sampler")
    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)
    ap.add_argument("--layers", type=str, default="all", help="Comma list or 'all'")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--max_paraphrases", type=int, default=None)

    # token sites (now includes first10 mean)
    ap.add_argument("--token_sites", type=str, default="last_prompt_token,prompt_mean,first_assistant_token,first10_assistant_mean")
    ap.add_argument("--assistant_start_token_id", type=int, default=None, help="If set, overrides eos token for the first-assistant-token site and generation.")

    # vector normalization (to avoid BASE/FT scale confounds)
    ap.add_argument("--norm_mode", type=str, default="none",
        choices=["none","unit","match_base"],
        help="none: raw L2; unit: per-vector unit-norm; match_base: scale FT vectors to match BASE mean norm per (layer,station,site,family)")

    # chat-template & identical tokenizer controls
    ap.add_argument("--apply_chat_template", type=int, default=0,
        help="0/1 use tokenizer.apply_chat_template (messages with roles)")
    ap.add_argument("--system_prompt", type=str, default="",
        help="optional system message when using chat template (kept empty by default)")
    ap.add_argument("--force_same_tokenizer", type=int, default=1,
        help="0/1 use the base tokenizer for FT too to guarantee identical tokenization")

    # Jacobian controls
    ap.add_argument("--compute_jacobians", type=int, default=1, help="0/1")
    ap.add_argument("--jacobian_mode", type=str, default="pca", choices=["pca","raw"])
    ap.add_argument("--topk_pca", type=int, default=8)
    ap.add_argument("--eps_jacobian", type=float, default=1e-3)
    ap.add_argument("--jacobian_layers", type=str, default="6,12", help="Comma-sep subset for Jacobians")
    ap.add_argument("--max_seq_len", type=int, default=4096,
                    help="Hard truncation length for tokenizer(..., truncation=True).")

    # perf knobs
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"])
    ap.add_argument("--use_flash_attention", type=int, default=1)
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--families_per_batch", type=int, default=1)
    ap.add_argument("--pad_to_multiple_of", type=int, default=8)
    ap.add_argument("--gen_k", type=int, default=10, help="K for firstK_assistant_mean (kept 10 by default)")

    ap.add_argument("--plot_layer_stride", type=int, default=1,
                    help="Plot every k-th layer (1=all).")
    ap.add_argument("--plot_max_layer", type=int, default=None,
                    help="If set, only plot layers <= this value.")
    ap.add_argument("--plot_force_layers", type=str, default="",
                    help="Comma list of extra layers to force-show/tick (e.g. '12,20').")
    ap.add_argument("--inside_ci_coverage", type=float, default=0.30,
        help="Central coverage for shaded bands in inside_* plots (e.g., 0.10 or 0.05).")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
        datefmt=":%H:%M:%S",
    )

    # SDPA flash attention
    if args.use_flash_attention and torch.cuda.is_available():
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
            logging.info("PyTorch SDPA: flash attention enabled.")
        except Exception as e:
            logging.warning("PyTorch SDPA: could not enable flash attention (%s).", e)
    torch.set_float32_matmul_precision("high")

    set_seed(args.seed)
    outdir_path = ensure_dir(args.outdir)

    # Load selection
    logging.info("Reading selection: %s", args.selection_jsonl)
    items = load_selection_jsonl(args.selection_jsonl, max_prompts=args.max_prompts)
    logging.info("Loaded %d prompts (families).", len(items))

    # Models
    device = torch.device(args.device)
    (base, tok_base), (ft, tok_ft) = build_models_and_tokenizers(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, device, dtype=args.dtype
    )
    if args.compile:
        try:
            base = torch.compile(base, mode="max-autotune", fullgraph=False, dynamic=True)
            ft   = torch.compile(ft,   mode="max-autotune", fullgraph=False, dynamic=True)
            logging.info("torch.compile enabled (mode=max-autotune).")
        except Exception as e:
            logging.warning("torch.compile failed, continuing without it: %s", e)

    # ---- PERF: disable KV-cache + enable TF32 ----
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for m in (base, ft):
        if hasattr(m, "config"):
            m.config.use_cache = False

    # Optionally force identical tokenization (base tokenizer for both)
    if args.force_same_tokenizer:
        tok_ft = tok_base
        logging.info("force_same_tokenizer=1 → using BASE tokenizer for both models (tokenization reused).")

    # Layers
    if args.layers.strip().lower() == "all":
        num_layers = _get_num_layers(base)
        layers = list(range(num_layers))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()!=""]

    token_sites = [t.strip() for t in args.token_sites.split(",") if t.strip()!=""]

    plot_layers = _choose_plot_layers(
        layers,
        stride=args.plot_layer_stride,
        max_layer=args.plot_max_layer,
        force_layers_csv=args.plot_force_layers
    )

    # Main computation
    logging.info("Computing layerwise ΔΔ for layers %s ; token sites: %s", layers, token_sites)
    global outdir  # used inside compute_layerwise_ddiff_for_model_pair for saving norms
    outdir = args.outdir  # simple global pass-through for norm CSV saving

    dd_maps = compute_layerwise_ddiff_for_model_pair(
        base, tok_base, ft, tok_ft,
        items, layers, device, token_sites,
        max_paraphrases=args.max_paraphrases,
        batch_size_texts=0,  # unused (we batch by family)
        assistant_start_token_id=args.assistant_start_token_id,
        norm_mode=args.norm_mode,
        apply_chat_template_flag=args.apply_chat_template,
        system_prompt=args.system_prompt,
        max_seq_len=args.max_seq_len,
        pad_to_multiple_of=args.pad_to_multiple_of,
        families_per_batch=args.families_per_batch,
        gen_k=args.gen_k,
    )

    # Persist matrices & plots per token site
    for site in token_sites:
        for mode in ["to_original","paraphrase_only"]:
            d = dd_maps[(site,mode)]
            sub = ensure_dir(outdir_path / f"{site}_{mode}")
            write_matrix_csv(d["D_BASE"], d["layers"], d["stations"], sub / "D_BASE.csv")
            write_matrix_csv(d["D_FT"],   d["layers"], d["stations"], sub / "D_FT.csv")
            write_matrix_csv(d["DDIFF"],  d["layers"], d["stations"], sub / "DDIFF.csv")

        # Main curves + derived panel
        plot_main_curves(dd_maps, outdir_path, plot_layers, site)

        # layer-output norms visualization (uses ("__NORMS__", site) entry)
        plot_layer_output_norms(dd_maps, outdir_path, plot_layers, site)

        # Norm-adjusted ΔΔ (two flavors)
        save_norm_adjusted_ddiff(dd_maps, outdir_path, plot_layers, site, ref_mode="layer_output", ref_station="RESID_POST")
        save_norm_adjusted_ddiff(dd_maps, outdir_path, plot_layers, site, ref_mode="self")

        # RESID_POST-only (to-original − paraphrase-only)
        plot_resid_post_to_minus_para(dd_maps, outdir_path, plot_layers, site)

        plot_station_microviews_for_site(dd_maps, outdir_path, plot_layers, site, target_layers=(12,), mode="to_original")

        plot_within_layer_gap(dd_maps, outdir_path, plot_layers, site, target_layer=12)

        #plot_inside_layer_lines(dd_maps, outdir_path, plot_layers, site)
        #plot_layer_bar_gap(dd_maps, outdir_path, plot_layers, site, target_layer=12)

        plot_inside_layer_lines(dd_maps, outdir_path, plot_layers, site, ci_coverage=args.inside_ci_coverage)
        plot_layer_bar_gap(dd_maps, outdir_path, plot_layers, site, target_layer=12, ci_coverage=args.inside_ci_coverage)

    # Station micro-views (layers 6 and 12 by default)
    plot_station_microviews(dd_maps, outdir_path, plot_layers, default_layers=(6,12))

    plot_resid_post_last_vs_mean(dd_maps, outdir_path, plot_layers)

    # Within-layer-12 attention vs MLP gap plots
    primary_site = "last_prompt_token" if ("last_prompt_token","to_original") in dd_maps else token_sites[0]
    plot_layer12_within_block_gap(dd_maps, outdir_path, plot_layers, primary_site)

    # Jumps
    jumps = compute_jumps(dd_maps, plot_layers)
    plot_jumps(jumps, outdir_path, plot_layers)

    # Explicit report lines in log:
    logging.info("Layer-%d attention: ΔΔ_before=%.6f ; ΔΔ_after=%.6f ; J_attn=%.6f",
                 jumps["layer12"], jumps["dd_before_attn"], jumps["dd_after_attn"], jumps["j_attn"])
    logging.info("Layer-%d MLP:       ΔΔ_before=%.6f ; ΔΔ_after=%.6f ; J_mlp=%.6f",
                 jumps["layer12"], jumps["dd_before_mlp"], jumps["dd_after_mlp"], jumps["j_mlp"])

    # Sanity: report sign balance at layer 12 across MLP stations
    try:
        site0 = "last_prompt_token" if ("last_prompt_token","to_original") in dd_maps else token_sites[0]
        D12 = dd_maps[(site0,"to_original")]["DDIFF"]
        sidx = [ALL_STATIONS.index(s) for s in STATIONS_MLP]
        li12 = layers.index(12) if 12 in layers else int(np.argmin([abs(l-12) for l in layers]))
        vals = [float(D12[li12, i]) for i in sidx]
        pos = sum(v > 0 for v in vals); neg = sum(v < 0 for v in vals)
        logging.info("Layer-%d MLP ΔΔ signs at site=%s → positive=%d, negative=%d, zeros=%d",
                     layers[li12], site0, pos, neg, len(vals)-pos-neg)
        if pos == len(vals) or neg == len(vals):
            logging.warning("All bars same sign at layer %d — check norm_mode (%s), station taps and token site.",
                            layers[li12], args.norm_mode)
    except Exception as e:
        logging.warning("Sanity sign check skipped: %s", e)

    # Jacobians
    if args.compute_jacobians:
        jac_layers = [int(x.strip()) for x in args.jacobian_layers.split(",") if x.strip()!=""]
        logging.info("Computing Jacobians on layers: %s", jac_layers)

        jacB = jacobian_layer_driver(
            base, tok_base, items, jac_layers, device,
            args.topk_pca, args.jacobian_mode, args.max_prompts, args.max_paraphrases,
            eps=args.eps_jacobian, directions_random=8, max_seq_len=args.max_seq_len
        )
        jacF = jacobian_layer_driver(
            ft, tok_ft, items, jac_layers, device,
            args.topk_pca, args.jacobian_mode, args.max_prompts, args.max_paraphrases,
            eps=args.eps_jacobian, directions_random=8, max_seq_len=args.max_seq_len
        )

        # Save quick CSVs
        for L in jac_layers:
            def _write(rows, path):
                if not rows: return
                keys = sorted(set().union(*[r.keys() for r in rows]))
                with open(path,"w",newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
            _write(jacB[L], outdir_path / f"jac_layer{L}_BASE.csv")
            _write(jacF[L], outdir_path / f"jac_layer{L}_FT.csv")

        jacobian_diff_plot(jacB, jacF, outdir_path)

        # Choose a site that has norms recorded; prefer last_prompt_token if present
        primary_site = "last_prompt_token" if ("__NORMS__", "last_prompt_token") in dd_maps else token_sites[0]
        jacobian_diff_plot_norm_adjusted(
            jacB, jacF, dd_maps, primary_site, outdir_path,
            ref_mode="layer_output", ref_station="RESID_POST"
        )

    logging.info("All done. Outputs in %s", outdir_path)

# Jacobian reuse piece (unchanged)
@torch.no_grad()
def jacobian_norms_for_model(model, tokenizer, items: List[PromptSet], layer: int,
                             device: torch.device, topk_pca: int, mode: str,
                             max_prompts: Optional[int], max_paraphrases: Optional[int],
                             eps: float = 1e-3, directions_random: int = 8, max_seq_len: int = 4096) -> Dict[str, Any]:
    accessor = MLPAccessor(model, layer)
    res_hook = ResidualHook(model, layer)

    out_rows = []
    start = time.time()
    for idx, ps in enumerate(items):
        texts = [ps.instruction_original] + [t for _, t in ps.paraphrases]
        if max_paraphrases is not None:
            texts = texts[: 1 + max_paraphrases]
        if len(texts) < 3:
            continue

        H_rows = []
        for t in texts:
            input_ids, attention_mask = encode(tokenizer, t, device, max_len=max_seq_len)
            with torch.inference_mode():
                # ensure mask is float (avoids bf16 promotion/overflow inside model)
                attention_mask = _sanitize_attn_mask(attention_mask)
                # use the model’s actual parameter dtype (bf16 on A100 if loaded that way)
                amp_dtype = next(model.parameters()).dtype
                with torch.cuda.amp.autocast(enabled=(input_ids.device.type == "cuda"), dtype=amp_dtype):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)

            H = res_hook.buffer
            # !!! JACOBIAN SITE SELECTION !!!
            #vec = mean_pool_tokens(H, attention_mask).to(torch.float32)
            #vec = select_token_vectors(H, attention_mask, "<SITE>", tokenizer, input_ids)[0].to(torch.float32)
            vec = select_token_vectors(H, attention_mask, "last_prompt_token", tokenizer, input_ids)[0].to(torch.float32)
            H_rows.append(vec)
        H_mat = torch.stack(H_rows, dim=0)
        Xc = H_mat - H_mat.mean(dim=0, keepdim=True)

        def normalize_t(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            v = v.to(torch.float32)
            n = v.norm(p=2)
            return v / (n + eps)
        def unit_random_like(h: torch.Tensor) -> torch.Tensor:
            r = torch.randn_like(h, dtype=torch.float32)
            return normalize_t(r)

        mean_dir = normalize_t(H_mat.mean(dim=0))
        dirs = [mean_dir]

        if mode == "pca":
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            k = min(topk_pca, Vh.shape[0])
            for i in range(k):
                dirs.append(normalize_t(Vh[i]))
        else:
            idxs = torch.randperm(Xc.shape[0])[:max(1, topk_pca)]
            for i in idxs:
                dirs.append(normalize_t(Xc[i]))

        for _ in range(directions_random):
            dirs.append(unit_random_like(mean_dir))

        D = torch.stack(dirs, dim=0)
        #C = H_mat.mean(dim=0, keepdim=True)
        C = H_rows[0].unsqueeze(0)

        def f_up(H_):
            return accessor.UP(H_.unsqueeze(0)).squeeze(0)
        def f_down(H_):
            post = accessor.POST(H_.unsqueeze(0)).squeeze(0)
            return accessor.DOWN(post)

        norms_up = []
        norms_down = []
        for j in range(D.shape[0]):
            d = D[j].unsqueeze(0)
            Yp = f_up(C + eps * d); Ym = f_up(C - eps * d)
            G = (Yp - Ym) / (2.0 * eps)
            norms_up.append(float(torch.linalg.vector_norm(G).item()))

            Yp2 = f_down(C + eps * d); Ym2 = f_down(C - eps * d)
            G2 = (Yp2 - Ym2) / (2.0 * eps)
            norms_down.append(float(torch.linalg.vector_norm(G2).item()))

        row = {
            "prompt_index": idx,
            "prompt_count": ps.prompt_count,
            "K": int(D.shape[0]),
            "jac_MEAN_up": norms_up[0],
            "jac_MEAN_down": norms_down[0],
        }
        var_slice = norms_up[1:1+topk_pca]
        rnd_slice = norms_up[1+topk_pca:]
        row["jac_VAR_up_mean"] = float(np.mean(var_slice)) if len(var_slice)>0 else float("nan")
        row["jac_RND_up_mean"] = float(np.mean(rnd_slice)) if len(rnd_slice)>0 else float("nan")

        var_slice_d = norms_down[1:1+topk_pca]
        rnd_slice_d = norms_down[1+topk_pca:]
        row["jac_VAR_down_mean"] = float(np.mean(var_slice_d)) if len(var_slice_d)>0 else float("nan")
        row["jac_RND_down_mean"] = float(np.mean(rnd_slice_d)) if len(rnd_slice_d)>0 else float("nan")

        out_rows.append(row)

        if (idx+1) % 20 == 0:
            logging.info("[jacobian] %d/%d prompts processed — %s", idx+1, len(items), format_eta(idx+1, len(items), start))

    res_hook.close()
    return {"per_prompt": out_rows, "layer": layer}

# Entry
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
