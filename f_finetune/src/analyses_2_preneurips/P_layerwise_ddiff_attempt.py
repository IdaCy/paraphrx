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
from contextlib import nullcontext

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
from typing import Iterable
try:
    from torch.backends.cuda import sdp_kernel
except Exception:
    sdp_kernel = None

# Global perf toggles set in main()
USE_FLASH_ATTN: bool = False
AUTOCAST_DTYPE = torch.float16  # set to bf16 on CUDA by default in main()

# Reuse: PEFT guarded import
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# ---------- Utils (unchanged except explicit pad_to_multiple_of in encode) ----------

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

def encode(tokenizer, text: str, device: torch.device, max_len: int, pad_to_multiple_of: int = 8):
    out = tokenizer(
        text, return_tensors="pt", padding=False, truncation=True,
        max_length=max_len, pad_to_multiple_of=pad_to_multiple_of
    )
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

# ---------- Accessors & Hooks (unchanged) ----------

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

# ---------- Metrics & helpers (unchanged except norm robustness) ----------

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
    if mat.shape[0] < 3:  # need at least 2 paraphrases
        return float("nan")
    X = mat[1:]
    n = X.shape[0]
    s = 0.0; k = 0
    for i in range(n):
        d = X[i+1:] - X[i]
        if d.size == 0: continue
        dist = np.linalg.norm(d, axis=1)
        s += float(dist.sum()); k += dist.size
    return s / max(k, 1)

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

# ---------- Attention taps & layer out taps (unchanged) ----------

class AttnTaps:
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
        self.buf_in = inputs[0].detach()

    def _hook_out_fn(self, module, inputs, output):
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

class LayerOutTap:
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

# ---------- Batching & token-site ops (unchanged) ----------

def build_batch(tokenizer, texts: List[str], device: torch.device, max_len: int = 4096, pad_to_multiple_of: int = 8):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len, pad_to_multiple_of=pad_to_multiple_of)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)

def select_token_vectors(seq_tensor: torch.Tensor,
                         attn_mask: torch.Tensor,
                         site: str,
                         tokenizer,
                         input_ids: torch.Tensor | None = None) -> torch.Tensor:
    B, T, D = seq_tensor.shape
    out = torch.zeros(B, D, dtype=seq_tensor.dtype, device=seq_tensor.device)
    if site == "prompt_mean":
        m = attn_mask.to(seq_tensor.dtype)
        s = (seq_tensor * m.unsqueeze(-1)).sum(1)
        c = m.sum(1).clamp(min=1.0).unsqueeze(-1)
        out = (s / c).to(torch.float32)
    elif site == "last_prompt_token":
        idxs = attn_mask.sum(1).to(torch.long) - 1
        for b in range(B):
            t = int(idxs[b].item())
            out[b] = seq_tensor[b, t]
        out = out.to(torch.float32)
    elif site == "first_assistant_token":
        out = seq_tensor[:, -1, :].to(torch.float32)
    else:
        raise ValueError(f"Unknown token site {site}")
    return out

# ---------- Generation-site averaging (unchanged structure, faster ctx) ----------

@torch.no_grad()
def _mean_over_first_k_generated(
    model, tokenizer, layer_idx: int,
    input_ids: torch.Tensor, attention_mask: torch.Tensor,
    k: int,
    assistant_start_id: int,
) -> Dict[str, np.ndarray]:
    device = input_ids.device
    B = input_ids.size(0)
    ids = torch.cat([input_ids, torch.full((B,1), assistant_start_id, dtype=input_ids.dtype, device=device)], dim=1)
    mask = torch.cat([attention_mask, torch.ones((B,1), dtype=attention_mask.dtype, device=device)], dim=1)

    mlp_acc = MLPAccessor(model, layer_idx)
    res_hook = ResidualHook(model, layer_idx)
    attn_taps = AttnTaps(model, layer_idx)
    layer_out = LayerOutTap(model, layer_idx)

    sums: Dict[str, torch.Tensor] = {}

    for _ in range(k):
        sdp_ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True) if (USE_FLASH_ATTN and device.type=="cuda") else nullcontext()
        with sdp_ctx:
            with torch.cuda.amp.autocast(enabled=(ids.device.type=="cuda"), dtype=AUTOCAST_DTYPE):
                out = model(input_ids=ids, attention_mask=mask)

        Hc = res_hook.buffer
        Ai = attn_taps.buf_in
        Ao = attn_taps.buf_out
        Rpost = layer_out.buf

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

        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        ids  = torch.cat([ids,  next_tok.to(ids.dtype)], dim=1)
        mask = torch.cat([mask, torch.ones((B,1), dtype=mask.dtype, device=device)], dim=1)

    res_hook.close(); attn_taps.close(); layer_out.close()

    out_np = {}
    for key, S in sums.items():
        out_np[key] = (S / float(k)).cpu().numpy()
    return out_np

# ---------- Core capture ----------

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
        if assistant_start_token_id is None and tokenizer.eos_token_id is None:
            raise RuntimeError("No assistant_start_token_id and tokenizer.eos_token_id is None.")
        as_id = assistant_start_token_id if assistant_start_token_id is not None else tokenizer.eos_token_id
        return _mean_over_first_k_generated(
#            model, tokenizer, layer_idx, input_ids, attention_mask, k=FIRST_ASSISTANT_MEAN_K, assistant_start_id=as_id
            model, tokenizer, layer_idx, input_ids, attention_mask, k=gen_k, assistant_start_id=as_id        )

    device = input_ids.device
    mlp_acc = MLPAccessor(model, layer_idx)
    res_hook = ResidualHook(model, layer_idx)
    attn_taps = AttnTaps(model, layer_idx)
    layer_out = LayerOutTap(model, layer_idx)

    sdp_ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True) if (USE_FLASH_ATTN and device.type=="cuda") else nullcontext()
    with sdp_ctx:
        with torch.cuda.amp.autocast(enabled=(input_ids.device.type=="cuda"), dtype=AUTOCAST_DTYPE):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    H = res_hook.buffer
    A_in  = attn_taps.buf_in
    A_out = attn_taps.buf_out
    R_post = layer_out.buf

    if token_site == "first_assistant_token":
        if assistant_ids is None or assistant_mask is None:
            raise RuntimeError("assistant ids/mask required for first_assistant_token site")
        sdp_ctx2 = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True) if (USE_FLASH_ATTN and assistant_ids.device.type=="cuda") else nullcontext()
        with sdp_ctx2:
            with torch.cuda.amp.autocast(enabled=(assistant_ids.device.type=="cuda"), dtype=AUTOCAST_DTYPE):
                _ = model(input_ids=assistant_ids, attention_mask=assistant_mask)

        H2 = res_hook.buffer
        A_in2, A_out2 = attn_taps.buf_in, attn_taps.buf_out
        R_post2 = layer_out.buf

        RES_vec = select_token_vectors(H2, assistant_mask, "first_assistant_token", tokenizer, assistant_ids)
        ATTN_IN_vec  = select_token_vectors(A_in2, assistant_mask, "first_assistant_token", tokenizer, assistant_ids)
        ATTN_OUT_vec = select_token_vectors(A_out2, assistant_mask, "first_assistant_token", tokenizer, assistant_ids)
        RESID_PRE_vec = select_token_vectors(A_in2, assistant_mask, "first_assistant_token", tokenizer, assistant_ids)
        RESID_MID_vec = select_token_vectors(H2,    assistant_mask, "first_assistant_token", tokenizer, assistant_ids)
        RESID_POST_vec= select_token_vectors(R_post2,assistant_mask, "first_assistant_token", tokenizer, assistant_ids)

        H_seq = H2; mask = assistant_mask
    else:
        RES_vec = select_token_vectors(H, attention_mask, token_site, tokenizer, input_ids)
        ATTN_IN_vec  = select_token_vectors(A_in, attention_mask, token_site, tokenizer, input_ids)
        ATTN_OUT_vec = select_token_vectors(A_out, attention_mask, token_site, tokenizer, input_ids)
        RESID_PRE_vec = select_token_vectors(A_in, attention_mask, token_site, tokenizer, input_ids)
        RESID_MID_vec = select_token_vectors(H,    attention_mask, token_site, tokenizer, input_ids)
        RESID_POST_vec= select_token_vectors(R_post,attention_mask, token_site, tokenizer, input_ids)
        H_seq = H; mask = attention_mask

    UP_seq       = mlp_acc.UP(H_seq)
    GATE_PRE_seq = mlp_acc.GATE_PRE(H_seq)
    GATE_ACT_seq = mlp_acc.GATE_ACT(H_seq)
    POST_seq     = mlp_acc.POST(H_seq)
    DOWN_seq     = mlp_acc.DOWN(POST_seq)

    site_for_select = token_site if token_site != "first_assistant_token" else "first_assistant_token"
    UP_vec       = select_token_vectors(UP_seq,       mask, site_for_select, tokenizer)
    GPRE_vec     = select_token_vectors(GATE_PRE_seq, mask, site_for_select, tokenizer)
    GACT_vec     = select_token_vectors(GATE_ACT_seq, mask, site_for_select, tokenizer)
    POST_vec     = select_token_vectors(POST_seq,     mask, site_for_select, tokenizer)
    DOWN_vec     = select_token_vectors(DOWN_seq,     mask, site_for_select, tokenizer)

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
    as_id = assistant_start_token_id if assistant_start_token_id is not None else tokenizer.eos_token_id
    if as_id is None:
        raise RuntimeError("No assistant_start_token_id and tokenizer.eos_token_id is None.")
    B = input_ids.size(0)
    add = torch.full((B,1), as_id, dtype=input_ids.dtype, device=device)
    ids2  = torch.cat([input_ids, add], dim=1)
    mask2 = torch.cat([attention_mask, torch.ones(B,1, dtype=attention_mask.dtype, device=device)], dim=1)
    return ids2, mask2

# ---------- Dispersion & normalization (unit/match_base robust) ----------

def family_dispersion(mat: np.ndarray, mode: str) -> float:
    if mode == "to_original":
        return avg_l2_to_original(mat)
    elif mode == "paraphrase_only":
        return avg_pairwise_l2_without_first(mat)
    else:
        raise ValueError("mode must be to_original | paraphrase_only")

def _mean_row_norm_finite(mat: np.ndarray) -> float:
    if mat.size == 0:
        return float("nan")
    row_ok = np.isfinite(mat).all(axis=1)
    if not row_ok.any():
        return float("nan")
    sel = mat[row_ok]
    return float(np.mean(np.linalg.norm(sel, axis=1)))

def apply_norm_mode(mat: np.ndarray, mode: str, ref_mat_for_match: Optional[np.ndarray]=None) -> np.ndarray:
    if mode == "none":
        return mat
    A = np.array(mat, dtype=np.float64, copy=True)
    if mode == "unit":
        den = np.linalg.norm(A, axis=1, keepdims=True)
        den = np.where(np.isfinite(den) & (den > 1e-12), den, 1.0)
        np.divide(A, den, out=A, where=np.isfinite(A))
        return A
    if mode == "match_base":
        if ref_mat_for_match is None:
            return A
        nb = _mean_row_norm_finite(np.asarray(ref_mat_for_match, dtype=np.float64))
        nf = _mean_row_norm_finite(A)
        if not np.isfinite(nb) or not np.isfinite(nf) or nf <= 1e-12:
            logging.warning("norm_mode=match_base skipped due to non-finite/tiny means (nb=%.3g, nf=%.3g).", nb, nf)
            return A
        s = float(np.clip(nb / nf, 1e-3, 1e3))
        mask = np.isfinite(A)
        A[mask] *= s
        return A
    raise ValueError("bad norm_mode")

# ---------- Chat-template helpers (unchanged) ----------

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

# ---------- Main ΔΔ computation (tokenization reuse + pad_to_multiple_of) ----------

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
    pad_to_multiple_of: int = 8,
    first_assistant_mean_k: int = 10,
) -> Dict[str, Any]:
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

    # make K used by generation site globally visible in this module
    global FIRST_ASSISTANT_MEAN_K
    FIRST_ASSISTANT_MEAN_K = int(first_assistant_mean_k)

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

    raw = {}
    for site in token_sites:
        for mode in ["to_original", "paraphrase_only"]:
            raw[(site, mode)] = dict(BASE=[[[] for _ in range(S)] for _ in range(L)],
                                     FT  =[[[] for _ in range(S)] for _ in range(L)])

    same_tok = (tok_ft is tok_base)

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

                for si, st in enumerate(ALL_STATIONS):
                    # Split back per-family using recorded ranges
                    for fam_idx, (_pc, _texts) in enumerate(batch_fams):
                        s, e = ranges[fam_idx]
                        matB = capB[st][s:e]
                        matF = capF[st][s:e]

                    if not (np.isfinite(matB).all() and np.isfinite(matF).all()):
                        logging.warning(
                            "[nonfinite] family=%d layer=%d site=%s station=%s  (finite_BASE=%s finite_FT=%s)",
                            f_idx+1, layer_idx, site, st,
                            str(np.isfinite(matB).all()), str(np.isfinite(matF).all())
                        )

                        mean_norm_B = float(np.mean(np.linalg.norm(matB, axis=1)))
                        mean_norm_F = float(np.mean(np.linalg.norm(matF, axis=1)))

                    if norm_mode == "unit":
                        matBn = apply_norm_mode(matB, "unit")
                        matFn = apply_norm_mode(matF, "unit")
                    elif norm_mode == "match_base":
                        matBn = matB
                        matFn = apply_norm_mode(matF, "match_base", ref_mat_for_match=matB)
                    else:
                        matBn, matFn = matB, matF

                    dB_to = family_dispersion(matBn, "to_original")
                    dB_po = family_dispersion(matBn, "paraphrase_only")
                    dF_to = family_dispersion(matFn, "to_original")
                    dF_po = family_dispersion(matFn, "paraphrase_only")

                    raw[(site,"to_original")]["BASE"][li][si].append(dB_to)
                    raw[(site,"to_original")]["FT"  ][li][si].append(dF_to)
                    raw[(site,"paraphrase_only")]["BASE"][li][si].append(dB_po)
                    raw[(site,"paraphrase_only")]["FT"  ][li][si].append(dF_po)

                    raw.setdefault(("__NORMS__", site), dict(BASE=[[[] for _ in range(S)] for _ in range(L)],
                                                             FT  =[[[] for _ in range(S)] for _ in range(L)]))
                    raw[("__NORMS__", site)]["BASE"][li][si].append(mean_norm_B)
                    raw[("__NORMS__", site)]["FT"  ][li][si].append(mean_norm_F)

        if (f_idx+1) % 10 == 0:
            logging.info("Processed %d/%d prompt families — %s",
                         f_idx+1, len(families), format_eta(f_idx+1, len(families), start_all))

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

    for site in token_sites:
        if ("__NORMS__", site) in raw:
            BASE_vals = raw[("__NORMS__", site)]["BASE"]
            FT_vals   = raw[("__NORMS__", site)]["FT"]

            NB  = np.zeros((L,S)); NF  = np.zeros((L,S)); ND  = np.zeros((L,S))
            CIB = np.zeros((L,S)); CIF = np.zeros((L,S)); CID = np.zeros((L,S))

            for li in range(L):
                for sj in range(S):
                    vb = np.array(BASE_vals[li][sj], dtype=float)
                    vf = np.array(FT_vals  [li][sj], dtype=float)
                    mb, cib = mean_ci(vb); mf, cif = mean_ci(vf)
                    dd = vf - vb
                    md, cid = mean_ci(dd)
                    NB[li,sj], CIB[li,sj] = mb, cib
                    NF[li,sj], CIF[li,sj] = mf, cif
                    ND[li,sj], CID[li,sj] = md, cid

            norm_dir = ensure_dir(Path(outdir) / f"{site}_norms")
            write_matrix_csv(NB, layers, ALL_STATIONS, norm_dir / "norms_BASE.csv")
            write_matrix_csv(NF, layers, ALL_STATIONS, norm_dir / "norms_FT.csv")
            write_matrix_csv(ND, layers, ALL_STATIONS, norm_dir / "norms_FT_minus_BASE.csv")

            out[("__NORMS__", site)] = dict(
                layers=layers, stations=ALL_STATIONS,
                MEAN_BASE=NB, CI_BASE=CIB,
                MEAN_FT=NF,   CI_FT=CIF,
                MEAN_DIFF=ND, CI_DIFF=CID,
            )
    return out

# ---------- Jumps & plotting (unchanged) ----------

def compute_jumps(out_maps: Dict[Tuple[str,str], Dict[str,Any]], layers: List[int]) -> Dict[str, Any]:
    key = ("last_prompt_token", "to_original")
    if key not in out_maps:
        key = ("prompt_mean", "to_original")
    d = out_maps[key]
    D = d["DDIFF"]
    try:
        idx12 = layers.index(12)
    except ValueError:
        idx12 = int(np.argmin([abs(l-12) for l in layers]))
    s_idx = {s:i for i,s in enumerate(ALL_STATIONS)}
    def at(li, station): return D[li, s_idx[station]]
    dd_before_attn = float(at(idx12, "ATTN_IN"))
    dd_after_attn  = float(at(idx12, "ATTN_OUT"))
    j_attn = dd_after_attn - dd_before_attn
    dd_before_mlp = float(at(idx12, "RES"))
    dd_after_mlp  = float(at(idx12, "DOWN"))
    j_mlp = dd_after_mlp - dd_before_mlp
    J_UP    = np.array([ at(i,"UP")       - at(i,"RES")      for i in range(len(layers)) ], dtype=float)
    J_GATE  = np.array([ at(i,"GATE_ACT") - at(i,"GATE_PRE") for i in range(len(layers)) ], dtype=float)
    J_DOWN  = np.array([ at(i,"DOWN")     - at(i,"POST")     for i in range(len(layers)) ], dtype=float)
    return dict(
        idx12=idx12, layer12=layers[idx12],
        dd_before_attn=dd_before_attn, dd_after_attn=dd_after_attn, j_attn=j_attn,
        dd_before_mlp=dd_before_mlp, dd_after_mlp=dd_after_mlp, j_mlp=j_mlp,
        J_UP=J_UP, J_GATE=J_GATE, J_DOWN=J_DOWN
    )

PALETTE = ["#0413C5", "#7E74D0", "#4D5055", "#ABAABF", "#757584", "#030030", "#524384", "#50515F", "#8C89AF"]

def _bar_with_ci(ax, x, y, ci, label=None, color=None, width=0.75):
    ax.bar(x, y, yerr=ci, width=width, color=color, edgecolor="none", alpha=0.95, label=label)
    ax.axhline(0.0, color="#808080", linewidth=0.9, alpha=0.8)

def plot_layer_output_norms(dd_maps: Dict[Tuple[str,str], Dict[str,Any]], outdir: Path, layers: List[int], token_site: str):
    key = ("__NORMS__", token_site)
    if key not in dd_maps:
        logging.info("No norms found for site=%s; skipping layer-output norms plot.", token_site)
        return
    nm = dd_maps[key]
    stations = nm["stations"]
    Ls = nm["layers"]
    meanB = nm["MEAN_BASE"]; meanF = nm["MEAN_FT"]; meanD = nm["MEAN_DIFF"]
    ciF   = nm["CI_FT"];     ciD   = nm["CI_DIFF"]

    cols = PALETTE
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), sharex=True)
    axL, axR = axes
    for si, st in enumerate(stations):
        c = cols[si % len(cols)]
        axL.plot(Ls, meanB[:, si], label=f"{st} BASE",  linewidth=1.6, color=c, alpha=0.55)
        axL.plot(Ls, meanF[:, si], label=f"{st} FT",    linewidth=1.8, color=c)
        axL.fill_between(Ls, (meanF[:, si] - ciF[:, si]), (meanF[:, si] + ciF[:, si]), color=c, alpha=0.12, linewidth=0)
    axL.set_title(f"Layer-output norms — BASE (faint) vs FT (solid)\nsite: {token_site}")
    axL.set_xlabel("Layer"); axL.set_ylabel("Mean L2 norm"); axL.grid(alpha=0.15); axL.legend(ncol=3, fontsize=8)

    for si, st in enumerate(stations):
        c = cols[si % len(cols)]
        axR.plot(Ls, meanD[:, si], label=st, linewidth=1.8, color=c)
        axR.fill_between(Ls, (meanD[:, si] - ciD[:, si]), (meanD[:, si] + ciD[:, si]), color=c, alpha=0.12, linewidth=0)
    axR.axhline(0.0, color="#666", linewidth=0.9)
    axR.set_title(f"FT − BASE norm difference (mean ± 95% CI)\nsite: {token_site}")
    axR.set_xlabel("Layer"); axR.set_ylabel("Δ norm"); axR.grid(alpha=0.15); axR.legend(ncol=3, fontsize=8)

    plt.tight_layout()
    norm_dir = ensure_dir(outdir / f"{token_site}_norms")
    plt.savefig(norm_dir / f"layer_output_norms_{token_site}.png", dpi=170)
    plt.close()

def plot_main_curves(out_maps: Dict[Tuple[str,str], Dict[str,Any]], outdir: Path, layers: List[int], token_site: str):
    site = token_site
    A = out_maps[(site,"to_original")]
    B = out_maps[(site,"paraphrase_only")]
    S = len(ALL_STATIONS)
    colors = PALETTE

    def _make_panel(matrix: np.ndarray, title: str, fname: str):
        plt.figure(figsize=(12,4.6))
        for si, st in enumerate(ALL_STATIONS):
            c = colors[si % len(colors)]
            plt.plot(layers, matrix[:,si], marker="o", linewidth=1.8, label=st, color=c)
        plt.axhline(0.0, color="#666", linewidth=0.9, alpha=0.7)
        plt.xlabel("Layer"); plt.ylabel("ΔΔ (FT − BASE) — L2 dispersion")
        plt.title(title)
        plt.legend(ncol=4, fontsize=9)
        plt.tight_layout(); plt.savefig(outdir/fname, dpi=170); plt.close()

    _make_panel(A["DDIFF"], f"ΔΔ (to-original) — token site: {site}", f"ddiff_to_original_{site}.png")
    _make_panel(B["DDIFF"], f"ΔΔ (paraphrase-only) — token site: {site}", f"ddiff_paraphrase_only_{site}.png")
    _make_panel(A["DDIFF"] - B["DDIFF"], f"(to-original) − (paraphrase-only) — token site: {site}", f"ddiff_diff_to_minus_para_{site}.png")

def plot_station_microviews(out_maps, outdir: Path, layers: List[int], default_layers=(6,12)):
    site = "last_prompt_token" if ("last_prompt_token","to_original") in out_maps else "prompt_mean"
    d = out_maps[(site,"to_original")]
    D = d["DDIFF"]
    s_idx = [ALL_STATIONS.index(s) for s in STATIONS_MLP]

    outdir2 = ensure_dir(outdir / "microviews")
    for target in default_layers:
        try:
            li = layers.index(target)
        except ValueError:
            li = int(np.argmin([abs(l-target) for l in layers]))
            target = layers[li]
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

    x = layers
    plt.figure(figsize=(12,4.6))
    plt.plot(x, jumps["J_UP"],   marker="o", color=PALETTE[0], label="J_UP = ΔΔ(UP)−ΔΔ(RES)")
    plt.plot(x, jumps["J_GATE"], marker="o", color=PALETTE[1], label="J_GATE = ΔΔ(GATE_ACT)−ΔΔ(GATE_PRE)")
    plt.plot(x, jumps["J_DOWN"], marker="o", color=PALETTE[2], label="J_DOWN = ΔΔ(DOWN)−ΔΔ(POST)")
    plt.axhline(0.0, color="#666", linewidth=0.9)
    plt.xlabel("Layer"); plt.ylabel("ΔΔ jumps (FT − BASE)")
    plt.legend(); plt.title("Per-layer MLP component jumps")
    plt.tight_layout(); plt.savefig(outdir / "mlp_component_jumps.png", dpi=170); plt.close()

# ---------- Jacobian extension (fixed signature + pad_to_multiple_of) ----------

@torch.no_grad()
def jacobian_layer_driver(model, tokenizer, items: List[PromptSet], layers: List[int],
                          device: torch.device, topk_pca: int, mode: str,
                          max_prompts: Optional[int], max_paraphrases: Optional[int],
                          eps: float = 1e-3, directions_random: int = 8,
                          max_seq_len: int = 4096, pad_to_multiple_of: int = 8) -> Dict[int, Dict[str, Any]]:
    out = {}
    for li, layer in enumerate(layers):
        logging.info("[jacobian] Layer %d/%d -> %d", li+1, len(layers), layer)
        res = jacobian_norms_for_model(
            model, tokenizer, items, layer, device,
            topk_pca, mode, max_prompts, max_paraphrases, eps=eps, directions_random=directions_random,
            max_seq_len=max_seq_len, pad_to_multiple_of=pad_to_multiple_of
        )
        out[layer] = res["per_prompt"]
    return out

def jacobian_diff_plot(base_map: Dict[int, List[Dict[str,Any]]],
                       ft_map: Dict[int, List[Dict[str,Any]]],
                       outdir: Path):
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

def write_matrix_csv(matrix: np.ndarray, layers: List[int], stations: List[str], path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + stations)
        for i, L in enumerate(layers):
            w.writerow([L] + [f"{matrix[i,j]:.6f}" for j in range(len(stations))])

# ---------- Models & tokenizers (dtype selectable + optional compile) ----------

def build_models_and_tokenizers(
    base_model_name_or_path: str,
    ft_model_name_or_path: Optional[str],
    ft_lora_adapter: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
    compile_flag: int = 0,
):
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer_base.pad_token is None:
        tokenizer_base.pad_token = tokenizer_base.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device).eval()

    if ft_lora_adapter is not None:
        host_name = ft_model_name_or_path or base_model_name_or_path
        ft_host = AutoModelForCausalLM.from_pretrained(host_name, torch_dtype=dtype).to(device).eval()
        if _HAS_PEFT:
            ft = PeftModel.from_pretrained(ft_host, ft_lora_adapter).to(device).eval()
        else:
            raise RuntimeError("peft not available but --ft_lora_adapter provided.")
    else:
        if ft_model_name_or_path is None:
            raise ValueError("Provide --ft_model_name_or_path (merged FT) OR --ft_lora_adapter (LoRA adapter dir).")
        ft = AutoModelForCausalLM.from_pretrained(ft_model_name_or_path, torch_dtype=dtype).to(device).eval()

    tokenizer_ft = AutoTokenizer.from_pretrained(ft_model_name_or_path or base_model_name_or_path, use_fast=True)
    if tokenizer_ft.pad_token is None:
        tokenizer_ft.pad_token = tokenizer_ft.eos_token

    if compile_flag and device.type == "cuda":
        for name, m in [("BASE", base), ("FT", ft)]:
            try:
                m = torch.compile(m, mode="max-autotune-no-cudagraphs", fullgraph=False)
                if name == "BASE": base = m
                else: ft = m
                logging.info("torch.compile succeeded for %s model.", name)
            except Exception as e:
                logging.warning("torch.compile skipped for %s (hooks/ops not supported): %s", name, e)

    return (base, tokenizer_base), (ft, tokenizer_ft)

# ---------- CLI / Orchestration ----------

def main():
    ap = argparse.ArgumentParser(description="Layerwise ΔΔ across MLP+Attention stations with token-site control, jumps, and Jacobians (A100-optimized)")
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

    ap.add_argument("--token_sites", type=str, default="last_prompt_token,prompt_mean,first_assistant_token,first10_assistant_mean")
    ap.add_argument("--assistant_start_token_id", type=int, default=None)

    ap.add_argument("--norm_mode", type=str, default="none",
        choices=["none","unit","match_base"])

    ap.add_argument("--apply_chat_template", type=int, default=0)
    ap.add_argument("--system_prompt", type=str, default="")
    ap.add_argument("--force_same_tokenizer", type=int, default=1)

    ap.add_argument("--compute_jacobians", type=int, default=1)
    ap.add_argument("--jacobian_mode", type=str, default="pca", choices=["pca","raw"])
    ap.add_argument("--topk_pca", type=int, default=8)
    ap.add_argument("--eps_jacobian", type=float, default=1e-3)
    ap.add_argument("--jacobian_layers", type=str, default="6,12")
    ap.add_argument("--max_seq_len", type=int, default=4096)

    # New perf knobs
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16","bf16","fp32"], help="Compute dtype for model & autocast (CUDA default bf16).")
    ap.add_argument("--use_flash_attention", type=int, default=1, help="Prefer Flash/mem-efficient SDPA kernels if available.")
    ap.add_argument("--pad_to_multiple_of", type=int, default=8, help="Pad sequence length to multiple-of-N (8/16) for Tensor Cores.")
    ap.add_argument("--compile", type=int, default=0, help="Try torch.compile (safe fallback if unsupported).")
    ap.add_argument("--k_first_assistant_mean", type=int, default=10, help="K for first10_assistant_mean token site.")

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
        datefmt=":%H:%M:%S",
    )
    set_seed(args.seed)
    outdir_path = ensure_dir(args.outdir)

    # Mixed-precision & SDPA preferences
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    global USE_FLASH_ATTN, AUTOCAST_DTYPE
    USE_FLASH_ATTN = bool(args.use_flash_attention)
    if args.dtype == "bf16" and torch.cuda.is_available():
        AUTOCAST_DTYPE = torch.bfloat16
        model_dtype = torch.bfloat16
    elif args.dtype == "fp16" and torch.cuda.is_available():
        AUTOCAST_DTYPE = torch.float16
        model_dtype = torch.float16
    else:
        AUTOCAST_DTYPE = torch.float32
        model_dtype = torch.float32

    # Load selection
    logging.info("Reading selection: %s", args.selection_jsonl)
    items = load_selection_jsonl(args.selection_jsonl, max_prompts=args.max_prompts)
    logging.info("Loaded %d prompts (families).", len(items))

    # Models
    device = torch.device(args.device)
    (base, tok_base), (ft, tok_ft) = build_models_and_tokenizers(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, device,
        dtype=model_dtype, compile_flag=args.compile
    )

    # Disable KV-cache to ensure hooks see full forwards
    for m in (base, ft):
        if hasattr(m, "config"):
            m.config.use_cache = False

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

    logging.info("Computing layerwise ΔΔ for layers %s ; token sites: %s", layers, token_sites)
    global outdir
    outdir = args.outdir

    dd_maps = compute_layerwise_ddiff_for_model_pair(
        base, tok_base, ft, tok_ft,
        items, layers, device, token_sites,
        max_paraphrases=args.max_paraphrases,
        batch_size_texts=0,
        assistant_start_token_id=args.assistant_start_token_id,
        norm_mode=args.norm_mode,
        apply_chat_template_flag=args.apply_chat_template,
        system_prompt=args.system_prompt,
        max_seq_len=args.max_seq_len,
        pad_to_multiple_of=args.pad_to_multiple_of,
        first_assistant_mean_k=args.k_first_assistant_mean,
    )

    for site in token_sites:
        for mode in ["to_original","paraphrase_only"]:
            d = dd_maps[(site,mode)]
            sub = ensure_dir(outdir_path / f"{site}_{mode}")
            write_matrix_csv(d["D_BASE"], d["layers"], d["stations"], sub / "D_BASE.csv")
            write_matrix_csv(d["D_FT"],   d["layers"], d["stations"], sub / "D_FT.csv")
            write_matrix_csv(d["DDIFF"],  d["layers"], d["stations"], sub / "DDIFF.csv")

        plot_main_curves(dd_maps, outdir_path, layers, site)
        plot_layer_output_norms(dd_maps, outdir_path, layers, site)

    plot_station_microviews(dd_maps, outdir_path, layers, default_layers=(6,12))
    jumps = compute_jumps(dd_maps, layers)
    plot_jumps(jumps, outdir_path, layers)

    logging.info("Layer-%d attention: ΔΔ_before=%.6f ; ΔΔ_after=%.6f ; J_attn=%.6f",
                 jumps["layer12"], jumps["dd_before_attn"], jumps["dd_after_attn"], jumps["j_attn"])
    logging.info("Layer-%d MLP:       ΔΔ_before=%.6f ; ΔΔ_after=%.6f ; J_mlp=%.6f",
                 jumps["layer12"], jumps["dd_before_mlp"], jumps["dd_after_mlp"], jumps["j_mlp"])

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

    if args.compute_jacobians:
        jac_layers = [int(x.strip()) for x in args.jacobian_layers.split(",") if x.strip()!=""]
        logging.info("Computing Jacobians on layers: %s", jac_layers)

        jacB = jacobian_layer_driver(
            base, tok_base, items, jac_layers, device,
            args.topk_pca, args.jacobian_mode, args.max_prompts, args.max_paraphrases,
            eps=args.eps_jacobian, directions_random=8, max_seq_len=args.max_seq_len, pad_to_multiple_of=args.pad_to_multiple_of
        )
        jacF = jacobian_layer_driver(
            ft, tok_ft, items, jac_layers, device,
            args.topk_pca, args.jacobian_mode, args.max_prompts, args.max_paraphrases,
            eps=args.eps_jacobian, directions_random=8, max_seq_len=args.max_seq_len, pad_to_multiple_of=args.pad_to_multiple_of
        )

        for L in jac_layers:
            def _write(rows, path):
                if not rows: return
                keys = sorted(set().union(*[r.keys() for r in rows]))
                with open(path,"w",newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
            _write(jacB[L], outdir_path / f"jac_layer{L}_BASE.csv")
            _write(jacF[L], outdir_path / f"jac_layer{L}_FT.csv")

        jacobian_diff_plot(jacB, jacF, outdir_path)

    logging.info("All done. Outputs in %s", outdir_path)

# ---------- Jacobian reuse (unchanged except encode pad & fast ctx) ----------

@torch.no_grad()
def jacobian_norms_for_model(model, tokenizer, items: List[PromptSet], layer: int,
                             device: torch.device, topk_pca: int, mode: str,
                             max_prompts: Optional[int], max_paraphrases: Optional[int],
                             eps: float = 1e-3, directions_random: int = 8, max_seq_len: int = 4096,
                             pad_to_multiple_of: int = 8) -> Dict[str, Any]:
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
            input_ids, attention_mask = encode(tokenizer, t, device, max_len=max_seq_len, pad_to_multiple_of=pad_to_multiple_of)
            sdp_ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True) if (USE_FLASH_ATTN and input_ids.device.type=="cuda") else nullcontext()
            with sdp_ctx:
                with torch.cuda.amp.autocast(enabled=(input_ids.device.type=="cuda"), dtype=AUTOCAST_DTYPE):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)

            H = res_hook.buffer
            vec = mean_pool_tokens(H, attention_mask).to(torch.float32)
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
        C = H_mat.mean(dim=0, keepdim=True)

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

# ---------- Entry ----------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
