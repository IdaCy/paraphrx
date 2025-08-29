#!/usr/bin/env python3
"""
diff-of-diffs + Robustness Mechanism Probes 
python run_mlp_diff_of_diffs_plus.py \
    --selection_jsonl selection.jsonl \
    --base_model_name_or_path google/gemma-2-2b-it \
    --ft_model_name_or_path /path/to/your/ft \
    --layer_index 6 \
    --outdir out_layer6 \
    --batch_size 4 \
    --compute_jacobians 1 \
    --compute_pair_dispersion 1 \
    --compute_dir_jac 1 \
    --compute_logits_kl 1 \
    --do_activation_patching 0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_FG  = "#228B22"   # forest green
COL_G0  = "#111111"
COL_G1  = "#555555"
COL_G2  = "#888888"
COL_G3  = "#BBBBBB"
COL_G4  = "#DDDDDD"

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# utils & helpers

def set_seed(seed: int = 42):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def now(): return time.strftime("%H:%M:%S")

def format_eta(done: int, total: int, start_time: float) -> str:
    if done <= 0: return "ETA --:--"
    elapsed = time.time() - start_time
    rate = done / max(elapsed, 1e-6); remain = (total - done) / max(rate, 1e-6)
    return f"ETA {int(remain//60):02d}:{int(remain%60):02d} (elapsed {int(elapsed//60):02d}:{int(elapsed%60):02d})"

@dataclass
class PromptSet:
    prompt_count: int
    instruction_original: str
    input_text: str
    paraphrases: List[Dict[str, Any]]  # dict with keys: key, text, (optional) paraphrase_content_score, instruct_type

def load_selection_jsonl(path: str | Path, max_prompts: Optional[int] = None,
                         allow_types: Optional[set] = None,
                         require_score5: bool = False) -> List[PromptSet]:
    """
    Expect JSONL with entries like:
      {"prompt_count":..., "instruction_original":..., "input":..., "paraphrases":[{"key":..., "text":..., "paraphrase_content_score":5, "instruct_type":"..."} ...]}
    We preserve paraphrase dicts so we can filter by score/type here if desired.
    """
    items = []; n = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            all_paras = []
            for p in obj.get("paraphrases", []):
                if not isinstance(p, dict): 
                    # backwards compatibility (["key","text"] tuples)
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        p = {"key": p[0], "text": p[1]}
                    else:
                        continue
                if allow_types is not None and p.get("key") not in allow_types:
                    continue
                if require_score5 and p.get("paraphrase_content_score", 5) != 5:
                    continue
                all_paras.append(p)
            items.append(PromptSet(
                prompt_count=int(obj["prompt_count"]),
                instruction_original=obj.get("instruction_original",""),
                input_text=obj.get("input","") or "",
                paraphrases=all_paras
            ))
            n += 1
            if max_prompts and n >= max_prompts: break
    return items

def encode_batch(tokenizer, texts: List[str], device: torch.device, max_length: Optional[int]=None):
    out = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return out["input_ids"].to(device), out["attention_mask"].to(device)

def pool_tokens(x: torch.Tensor, mask: torch.Tensor, mode: str="mean", last_only: bool=False) -> torch.Tensor:
    """
    x: [B,T,D], mask: [B,T]. Returns [B,D] pooled.
    mode: "mean" or "last"
    last_only: if True, select the last valid token (ignores 'mode').
    """
    if last_only or mode == "last":
        # pick last position where mask==1 for each row
        idx = mask.to(torch.int64).sum(dim=1) - 1  # [B]
        B = x.size(0); D = x.size(-1)
        ar = torch.arange(B, device=x.device)
        pooled = x[ar, idx.clamp(min=0), :]  # safe even if 0-length due to clamp, but mask should have at least 1
        return pooled
    else:
        m = mask.to(x.dtype)
        s = (x * m.unsqueeze(-1)).sum(dim=1)
        c = m.sum(dim=1).clamp(min=1.0)
        return (s / c.unsqueeze(-1)).to(torch.float32)

def cosine_to_original(mat: np.ndarray) -> float:
    if mat.shape[0] < 2: return float("nan")
    x0 = mat[0]; X = mat[1:]
    dists = np.linalg.norm(X - x0[None,:], axis=1)
    return float(dists.mean())

def avg_pairwise_one_minus_cos(mat: np.ndarray) -> float:
    """Average L2 distance over all unordered pairs (i<j) in mat."""
    n = mat.shape[0]
    if n < 2: return float("nan")
    iu = np.triu_indices(n, k=1)
    diffs = mat[iu[0]] - mat[iu[1]]
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.mean(dists))

def mean_ci(vals: np.ndarray) -> Tuple[float,float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: return 0.0, 0.0
    m = float(vals.mean()); s = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    ci = 1.96 * s / max(1.0, math.sqrt(vals.size))
    return m, ci

def ttest_paired(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 2: return float("nan")
    d = a - b
    m = float(np.mean(d)); s = float(np.std(d, ddof=1))
    if s == 0: return 1.0
    t = m / (s / max(math.sqrt(len(d)),1.0))
    # two-sided p via normal approx
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5*(1.0+erf(abs(t)/sqrt(2))))
    return p

# MLP accessor

class MLPAccessor:
    def __init__(self, model, layer: int, gate_mode: str = "silu"):
        self.model = model; self.layer = layer
        self.mlp = self._get_mlp_module(model, layer)
        self.kind = self._detect_kind(self.mlp)
        self.gate_mode = gate_mode
        p = next(self.mlp.parameters())
        self.device = p.device; self.dtype = p.dtype

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
        return "unknown"

    @torch.no_grad()
    def UP(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu": return self.mlp.up_proj(h)
        if hasattr(self.mlp, "wi"): return self.mlp.wi(h)
        if hasattr(self.mlp, "fc_in"): return self.mlp.fc_in(h)
        if hasattr(self.mlp, "dense_h_to_4h"): return self.mlp.dense_h_to_4h(h)
        raise RuntimeError("Cannot find upstream linear")

    @torch.no_grad()
    def GATE_PRE(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu": return self.mlp.gate_proj(h)
        if hasattr(self.mlp, "wi"): return self.mlp.wi(h)
        if hasattr(self.mlp, "fc_in"): return self.mlp.fc_in(h)
        if hasattr(self.mlp, "dense_h_to_4h"): return self.mlp.dense_h_to_4h(h)
        raise RuntimeError("Cannot find gate pre")

    @torch.no_grad()
    def GATE_ACT(self, h: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            z = self.mlp.gate_proj(h)
            mode = self.gate_mode
            if   mode == "silu":    return F.silu(z)
            elif mode == "gelu":    return F.gelu(z)
            elif mode == "sigmoid": return torch.sigmoid(z)
            elif mode == "relu":    return F.relu(z)
            elif mode == "none":    return torch.ones_like(z)
            else:                   return F.silu(z)
        else:
            if hasattr(self.mlp, "wi"): pre = self.mlp.wi(h)
            elif hasattr(self.mlp, "fc_in"): pre = self.mlp.fc_in(h)
            elif hasattr(self.mlp, "dense_h_to_4h"): pre = self.mlp.dense_h_to_4h(h)
            else: raise RuntimeError("Cannot find GeLU upstream linear")
            return torch.nn.functional.gelu(pre)

    @torch.no_grad()
    def POST(self, h: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            up = self.mlp.up_proj(h)
            gate = self.GATE_ACT(h)
            return up * gate
        else:
            if hasattr(self.mlp, "wi"):      pre = self.mlp.wi(h)
            elif hasattr(self.mlp, "fc_in"): pre = self.mlp.fc_in(h)
            elif hasattr(self.mlp, "dense_h_to_4h"): pre = self.mlp.dense_h_to_4h(h)
            else: raise RuntimeError("Cannot find GeLU upstream linear")
            return torch.nn.functional.gelu(pre)

    @torch.no_grad()
    def DOWN(self, post: torch.Tensor) -> torch.Tensor:
        post = post.to(self.device, self.dtype)
        if hasattr(self.mlp, "down_proj"): return self.mlp.down_proj(post)
        if hasattr(self.mlp, "wo"): return self.mlp.wo(post)
        if hasattr(self.mlp, "fc_out"): return self.mlp.fc_out(post)
        if hasattr(self.mlp, "dense_4h_to_h"): return self.mlp.dense_4h_to_h(post)
        raise RuntimeError("Cannot find downstream linear")

class ResidualHook:
    """Capture the residual stream input to the MLP at the chosen layer (forward_pre of the MLP)."""
    def __init__(self, model, layer):
        self.model = model; self.layer = layer
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

# capture stations (batched)

@torch.no_grad()
def capture_stations_for_model(model, tokenizer, items: List[PromptSet], layer: int, device: torch.device,
                               max_prompts: Optional[int], max_paraphrases: Optional[int],
                               token_pool: str = "mean", last_only: bool=False,
                               batch_size: int = 4, token_slices: int = 0) -> Dict[str, Any]:
    """
    Batched per-prompt capture of the six station vectors (mean-pooled over tokens or last token).
    Returns dictionaries per prompt with cos/dist to original AND paraphrase-only dispersion.
    """
    results = {"layer": layer, "per_prompt": [], "n_used": 0}
    accessor = MLPAccessor(model, layer, gate_mode=getattr(tokenizer, "_swiglu_gate_mode", "silu"))
    res_hook = ResidualHook(model, layer)

    start = time.time()
    for idx, ps in enumerate(items):
        texts_full = [ps.instruction_original] + [p["text"] for p in ps.paraphrases]
        keys_full  = ["instruction_original"] + [p.get("key","paraphrase") for p in ps.paraphrases]
        if max_paraphrases is not None:
            texts_full = texts_full[: 1 + max_paraphrases]
            keys_full  = keys_full[: 1 + max_paraphrases]
        if len(texts_full) < 2:
            logging.warning("Prompt %s has <2 texts; skipping.", ps.prompt_count); continue

        # Collect per-text pooled vectors at each station
        stages = ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]
        pooled = {s: [] for s in stages}

        # Optionally also keep matrices to compute paraphrase-only dispersion
        mats = {s: [] for s in stages}

        # Token slices view
        S = int(token_slices) if token_slices and token_slices > 0 else 0
        if S > 0:
            per_slice = {s: [[] for _ in range(S)] for s in stages}

        # Batched forward over all texts of this prompt
        BATCH = batch_size if batch_size and batch_size > 0 else 1
        total = len(texts_full)
        for start_i in range(0, total, BATCH):
            end_i = min(start_i + BATCH, total)
            batch_texts = texts_full[start_i:end_i]

            input_ids, attn = encode_batch(tokenizer, batch_texts, device)
            _ = model(input_ids=input_ids, attention_mask=attn)  # fills res_hook.buffer

            H = res_hook.buffer                           # [b, T, D]
            RES = H
            UP   = accessor.UP(H)
            GPRE = accessor.GATE_PRE(H)
            GACT = accessor.GATE_ACT(H)
            POST = accessor.POST(H)
            DOWN = accessor.DOWN(POST)

            # pooling
            for b in range(H.size(0)):
                m = attn[b:b+1]
                for name, tensor in [("RES",RES),("UP",UP),("GATE_PRE",GPRE),("GATE_ACT",GACT),("POST",POST),("DOWN",DOWN)]:
                    v = pool_tokens(tensor[b:b+1], m, mode=token_pool, last_only=last_only).squeeze(0).to(torch.float32).cpu().numpy()
                    pooled[name].append(v)
                    mats[name].append(v)

                if S > 0:
                    # slice pooling
                    valid = (attn[b] > 0).nonzero(as_tuple=False).squeeze(1).cpu().numpy()
                    n = int(valid.size)
                    cuts = np.linspace(0, n, S+1).astype(int)
                    for si in range(S):
                        sel = valid[cuts[si]:cuts[si+1]]
                        if sel.size == 0:
                            for nm in per_slice.keys(): per_slice[nm][si].append(None)
                        else:
                            idxs = torch.from_numpy(sel).to(H.device)
                            msk = torch.zeros_like(attn[b], dtype=H.dtype); msk[idxs] = 1
                            for name, tensor in [("RES",RES),("UP",UP),("GATE_PRE",GPRE),("GATE_ACT",GACT),("POST",POST),("DOWN",DOWN)]:
                                s = (tensor[b] * msk.unsqueeze(-1)).sum(dim=0)
                                c = msk.sum().clamp(min=1.0)
                                vec = (s / c).to(torch.float32).cpu().numpy()
                                per_slice[name][si].append(vec)

        # Build matrices (original first)
        mats_np = {k: np.stack(v, axis=0) for k, v in mats.items()}
        # Paraphrase-only matrices (exclude row 0)
        mats_para = {k: v[1:] for k, v in mats_np.items()}

        # Stage metrics w.r.t. original + paraphrase-only dispersion
        row = {"prompt_index": idx, "prompt_count": ps.prompt_count, "N": int(mats_np["RES"].shape[0])}
        for s in stages:
            dist = cosine_to_original(mats_np[s])  # now mean L2 to original
            row[f"dist_{s}"] = dist

            # paraphrase-only dispersion
            try:
                disp = avg_pairwise_one_minus_cos(mats_para[s])
            except Exception:
                disp = float("nan")
            row[f"disp_para_{s}"] = disp

        # Per-slice paraphrase-only dispersion if requested
        if S > 0:
            for si in range(S):
                for s in stages:
                    arr = per_slice[s][si]  # list of vectors per text (may contain None)
                    # paraphrase-only items (exclude index 0)
                    vecs = [arr[i] for i in range(1, len(arr)) if arr[i] is not None]
                    if len(vecs) >= 2:
                        try:
                            disp = avg_pairwise_one_minus_cos(np.stack(vecs, axis=0))
                        except Exception:
                            disp = float("nan")
                    else:
                        disp = float("nan")
                    row[f"disp_para_{s}_S{si+1}"] = disp

        results["per_prompt"].append(row); results["n_used"] += 1

        if (idx+1) % 10 == 0:
            logging.info("[stations %s] %d/%d prompts — %s", now(), idx+1, len(items), format_eta(idx+1, len(items), start))

    res_hook.close()
    return results

# Jacobians

def normalize_t(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v = v.to(torch.float32); n = v.norm(p=2); return v / (n + eps)

def unit_random_like(h: torch.Tensor) -> torch.Tensor:
    r = torch.randn_like(h, dtype=torch.float32); return normalize_t(r)

@torch.no_grad()
def jacobian_norms_for_model(model, tokenizer, items: List[PromptSet], layer: int,
                             device: torch.device, topk_pca: int, mode: str,
                             max_prompts: Optional[int], max_paraphrases: Optional[int],
                             eps: float = 1e-3, directions_random: int = 8,
                             last_only: bool=False, batch_size: int=4,
                             along_paraphrase_delta: bool=False) -> Dict[str, Any]:
    """
    If along_paraphrase_delta=True, we compute norms specifically along d = normalize(RES_para - RES_orig)
    Otherwise use MEAN + PCA/RAW + RANDOM
    """
    accessor = MLPAccessor(model, layer, gate_mode=getattr(tokenizer, "_swiglu_gate_mode", "silu"))
    res_hook = ResidualHook(model, layer)

    out_rows = []
    start = time.time()
    for idx, ps in enumerate(items):
        texts = [ps.instruction_original] + [p["text"] for p in ps.paraphrases]
        if max_paraphrases is not None:
            texts = texts[: 1 + max_paraphrases]
        if len(texts) < 3: continue

        # Batched forward to get pooled RES
        input_ids, attn = encode_batch(tokenizer, texts, device)
        _ = model(input_ids=input_ids, attention_mask=attn)
        H = res_hook.buffer                  # [B,T,D]
        RES_pooled = pool_tokens(H, attn, mode="mean", last_only=last_only).to(torch.float32) # [B,D]
        H_mat = RES_pooled                   # [N,D]
        Xc = H_mat - H_mat.mean(dim=0, keepdim=True)

        dirs = []
        if along_paraphrase_delta:
            # add one direction per paraphrase (exclude original idx 0)
            for j in range(1, H_mat.size(0)):
                d = normalize_t(H_mat[j] - H_mat[0])
                dirs.append(d)
        else:
            mean_dir = normalize_t(H_mat.mean(dim=0))
            dirs.append(mean_dir)
            if mode == "pca":
                U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
                k = min(topk_pca, Vh.shape[0])
                for i in range(k):
                    dirs.append(normalize_t(Vh[i]))
            else:
                idxs = torch.randperm(Xc.shape[0])[:max(1, topk_pca)]
                for i in idxs: dirs.append(normalize_t(Xc[i]))
            for _ in range(directions_random):
                dirs.append(unit_random_like(mean_dir))

        D = torch.stack(dirs, dim=0) if len(dirs)>0 else torch.zeros(1, H_mat.size(1), device=H_mat.device)
        C = H_mat.mean(dim=0, keepdim=True)  # expansion point

        def f_up(H_):   return accessor.UP(H_.unsqueeze(0)).squeeze(0)
        def f_post(H_): return accessor.POST(H_.unsqueeze(0)).squeeze(0)
        def f_down(H_): return accessor.DOWN(f_post(H_.unsqueeze(0))).squeeze(0)

        norms_up = []; norms_post = []; norms_down = []
        for j in range(D.shape[0]):
            d = D[j].unsqueeze(0)
            Yp = f_up(C + eps * d); Ym = f_up(C - eps * d);   G = (Yp - Ym) / (2.0 * eps)
            Yp2= f_post(C + eps * d); Ym2= f_post(C - eps * d);G2= (Yp2 - Ym2) / (2.0 * eps)
            Yp3= f_down(C + eps * d); Ym3= f_down(C - eps * d);G3= (Yp3 - Ym3) / (2.0 * eps)
            norms_up.append(float(torch.linalg.vector_norm(G).item()))
            norms_post.append(float(torch.linalg.vector_norm(G2).item()))
            norms_down.append(float(torch.linalg.vector_norm(G3).item()))

        row = {"prompt_index": idx, "prompt_count": ps.prompt_count, "K": int(D.shape[0])}
        row["jac_UP_mean"]   = float(np.mean(norms_up)) if norms_up else float("nan")
        row["jac_POST_mean"] = float(np.mean(norms_post)) if norms_post else float("nan")
        row["jac_DOWN_mean"] = float(np.mean(norms_down)) if norms_down else float("nan")
        out_rows.append(row)

        if (idx+1) % 20 == 0:
            logging.info("[jacobian %s] %d/%d prompts — %s", now(), idx+1, len(items), format_eta(idx+1, len(items), start))

    res_hook.close()
    return {"per_prompt": out_rows, "layer": layer, "mode": ("para_delta" if along_paraphrase_delta else mode)}

# Logits & activation patching

def softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=-1)

def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    p = p.clamp(min=eps); q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)

class DownPreHookCapturePatch:
    """
    Patch the input to down_proj (POST) at a specific layer with a provided tensor.
    - capture=True: store the incoming POST for later use.
    - patch_tensor: if provided, replace the incoming POST with this value at positions specified.
    """
    def __init__(self, model, layer_index: int, positions: str="last"):
        self.layer_index = layer_index; self.positions = positions
        self._down = self._get_down_linear(model, layer_index)
        self.captured = None
        self._hook = None
        self.patch_tensor = None
        self.capture = True

    def _get_down_linear(self, model, i):
        for stem, leaf in [("model.model.layers","mlp"), ("model.layers","mlp"), ("transformer.h","mlp")]:
            try:
                base = model
                for part in stem.split("."): base = getattr(base, part)
                mlp = getattr(base[i], leaf)
                # Try standard down module
                if hasattr(mlp, "down_proj"): return mlp.down_proj
                if hasattr(mlp, "fc_out"): return mlp.fc_out
                if hasattr(mlp, "dense_4h_to_h"): return mlp.dense_4h_to_h
            except Exception: pass
        raise RuntimeError("Could not locate down_proj-like module for patching.")

    def _fn(self, module, inputs):
        """
        Forward-pre hook on the layer's down projection.
        - Always *captures* the incoming POST (unless capture=False).
        - If a patch tensor is set, apply it according to `positions`:
            * 'last': replace only the last token, broadcasting batch if needed
            * 'all' : replace the whole sequence; if lengths mismatch, crop/pad to align time dimension
        """
        x = inputs[0]
        if self.capture and self.captured is None:
            # Keep the first captured value for the current prompt
            self.captured = x.detach()
        if self.patch_tensor is not None:
            pt = self.patch_tensor
            B, T, F = x.shape

            # Normalie pt to 3D [b, t, f]
            if pt.dim() == 1:   # [F]
                pt = pt.view(1, 1, -1)
            elif pt.dim() == 2: # [T,F]
                pt = pt.unsqueeze(0)  # -> [1,T,F]
            # else assume [B,T,F]

            if self.positions == "all":
                # Align feature dim
                if pt.shape[-1] != F:
                    if pt.shape[-1] > F:
                        pt = pt[..., :F]
                    else:
                        pad = F - pt.shape[-1]
                        pt = torch.nn.functional.pad(pt, (0, pad))
                # Align time dim by tail-cropping or padding with the last vector
                if pt.shape[1] != T:
                    if pt.shape[1] > T:
                        pt = pt[:, -T:, :]
                    else:
                        pad_t = T - pt.shape[1]
                        last_vec = pt[:, -1:, :].expand(pt.shape[0], pad_t, F)
                        pt = torch.cat([last_vec, pt], dim=1)
                # Broadcast batch if necessary
                if pt.shape[0] != B:
                    pt = pt.expand(B, -1, -1)
                return (pt.to(x.device, x.dtype),)

            # positions == 'last': take the last token from pt and stick it into the last position of x
            pt_last = pt[:, -1:, :]  # [b,1,f]
            if pt_last.shape[0] != B:
                pt_last = pt_last.expand(B, -1, -1)
            out = x.clone()
            out[:, -1:, :] = pt_last.to(x.device, x.dtype)
            return (out,)

        return (x,)

    def open(self):
        self._hook = self._down.register_forward_pre_hook(self._fn)

    def close(self):
        try:
            if self._hook is not None: self._hook.remove()
        finally:
            self._hook = None

@torch.no_grad()
def next_token_logits(model, tokenizer, text: str, device: torch.device) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
    ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(device)
        out = model(input_ids=ids, attention_mask=attn)
    else:
        out = model(input_ids=ids)
    return out.logits[0, -1, :]  # [V]

@torch.no_grad()
def logits_kl_ddiff(models_tok, items: List[PromptSet], device: torch.device,
                    max_paraphrases: Optional[int], max_prompts: Optional[int]) -> Dict[str, Any]:
    """
    For each prompt: compute next-token KL(original || paraphrase) at BASE and FT.
    Return per-prompt rows and summary.
    """
    (base, tokB), (ft, tokF) = models_tok
    rows = []; start = time.time()
    for idx, ps in enumerate(items):
        texts = [ps.instruction_original] + [p["text"] for p in ps.paraphrases]
        if max_paraphrases is not None:
            texts = texts[: 1 + max_paraphrases]
        if len(texts) < 2: continue

        logitB_orig = next_token_logits(base, tokB, texts[0], device)
        logitF_orig = next_token_logits(ft,   tokF, texts[0], device)
        pB = softmax(logitB_orig); pF = softmax(logitF_orig)

        for j, para in enumerate(texts[1:], start=1):
            logitB_para = next_token_logits(base, tokB, para, device)
            logitF_para = next_token_logits(ft,   tokF, para, device)
            qB = softmax(logitB_para); qF = softmax(logitF_para)

            KL_B = float(kl_divergence(pB, qB))
            KL_F = float(kl_divergence(pF, qF))
            rows.append({
                "prompt_index": idx, "prompt_count": ps.prompt_count, "para_index": j,
                "kl_next_base": KL_B, "kl_next_ft": KL_F, "ddiff_kl_next": KL_F - KL_B
            })

        if (idx+1) % 20 == 0:
            logging.info("[logits-KL %s] %d/%d prompts — %s", now(), idx+1, len(items), format_eta(idx+1, len(items), start))

    return {"per_prompt": rows}

@torch.no_grad()
def activation_patching_kl(models_tok, items: List[PromptSet], device: torch.device,
                           layer_index: int, positions: str="last",
                           max_paraphrases: Optional[int]=None, sample_limit_per_prompt: int=1) -> Dict[str, Any]:
    """
    Within each model separately: capture POST at layer for the original; replace paraphrase POST with captured original POST;
    measure next-token KL(original || paraphrase) before/after patch.
    """
    (base, tokB), (ft, tokF) = models_tok
    out = {"per_prompt": []}
    for model, tok, name in [(base, tokB, "BASE"), (ft, tokF, "FT")]:
        patcher = DownPreHookCapturePatch(model, layer_index, positions=positions); patcher.open()
        start = time.time()
        for idx, ps in enumerate(items):
            texts = [ps.instruction_original] + [p["text"] for p in ps.paraphrases]
            if max_paraphrases is not None: texts = texts[: 1 + max_paraphrases]
            if len(texts) < 2: continue

            # Reset capture for this prompt and capture on original
            patcher.captured = None
            patcher.capture = True
            _ = next_token_logits(model, tok, texts[0], device)  # fills patcher.captured
            captured = patcher.captured
            if captured is None:
                logging.warning("No POST captured for prompt %s on %s", ps.prompt_count, name); continue

            p_orig = softmax(next_token_logits(model, tok, texts[0], device))

            # disable capture to avoid overwriting during patching
            patcher.capture = False

            # Evaluate small set of paraphrases
            count = 0
            for j, para in enumerate(texts[1:], start=1):
                if count >= sample_limit_per_prompt: break
                q_base = softmax(next_token_logits(model, tok, para, device))
                kl_before = float(kl_divergence(p_orig, q_base))

                # Patch paraphrase with original POST
                patcher.patch_tensor = captured
                q_patch = softmax(next_token_logits(model, tok, para, device))
                patcher.patch_tensor = None

                kl_after = float(kl_divergence(p_orig, q_patch))
                out["per_prompt"].append({
                    "model": name, "prompt_index": idx, "prompt_count": ps.prompt_count, "para_index": j,
                    "kl_before": kl_before, "kl_after": kl_after, "delta_kl": kl_after - kl_before
                })
                count += 1

            if (idx+1) % 20 == 0:
                logging.info("[patch %s %s] %d/%d prompts — %s", name, now(), idx+1, len(items), format_eta(idx+1, len(items), start))

        patcher.close()
    return out

# Gate stats & DOWN SVD

def kurtosis_fisher(x: np.ndarray) -> float:
    x = x - x.mean()
    s2 = (x**2).mean()
    if s2 <= 0: return float("nan")
    k = (x**4).mean() / (s2**2) - 3.0  # Fisher
    return float(k)

@torch.no_grad()
def gate_stats(model, tokenizer, items: List[PromptSet], layer: int, device: torch.device,
               max_prompts: Optional[int], max_paraphrases: Optional[int],
               batch_size: int=4, last_only: bool=False, thresh_pre: float=0.05, thresh_act: float=0.05) -> Dict[str, Any]:
    accessor = MLPAccessor(model, layer, gate_mode=getattr(tokenizer, "_swiglu_gate_mode", "silu"))
    res_hook = ResidualHook(model, layer)
    rows = []; start = time.time()

    for idx, ps in enumerate(items):
        texts = [ps.instruction_original] + [p["text"] for p in ps.paraphrases]
        if max_paraphrases is not None: texts = texts[: 1 + max_paraphrases]
        if len(texts) < 2: continue

        B = batch_size if batch_size and batch_size > 0 else 1
        vals = {"pre": [], "act": []}
        for si in range(0, len(texts), B):
            input_ids, attn = encode_batch(tokenizer, texts[si:si+B], device)
            _ = model(input_ids=input_ids, attention_mask=attn)
            H = res_hook.buffer
            GPRE = accessor.GATE_PRE(H); GACT = accessor.GATE_ACT(H)
            if last_only:
                gpre = pool_tokens(GPRE, attn, last_only=True)  # [b,Dg]
                gact = pool_tokens(GACT, attn, last_only=True)
            else:
                # use all tokens
                gpre = GPRE.reshape(-1, GPRE.shape[-1])  # [b*T, Dg]
                gact = GACT.reshape(-1, GACT.shape[-1])
            vals["pre"].append(gpre.to(torch.float32).cpu().numpy())
            vals["act"].append(gact.to(torch.float32).cpu().numpy())

        PRE = np.concatenate(vals["pre"], axis=0); ACT = np.concatenate(vals["act"], axis=0)
        def pack(name, arr, thr):
            flat = arr.reshape(-1)
            return {
                f"{name}_mean": float(np.mean(flat)),
                f"{name}_std":  float(np.std(flat, ddof=1) if flat.size>1 else 0.0),
                f"{name}_kurtosis": kurtosis_fisher(flat),
                f"{name}_frac_near0": float(np.mean(np.abs(flat) < thr)),
                f"{name}_n": int(flat.size)
            }
        row = {"prompt_index": idx, "prompt_count": ps.prompt_count}
        row.update(pack("gate_pre", PRE, thresh_pre))
        row.update(pack("gate_act", ACT, thresh_act))
        rows.append(row)

        if (idx+1) % 20 == 0:
            logging.info("[gate-stats %s] %d/%d prompts — %s", now(), idx+1, len(items), format_eta(idx+1, len(items), start))

    res_hook.close()
    return {"per_prompt": rows}

@torch.no_grad()
def down_svd_energy(model, items: List[PromptSet], tokenizer, layer: int, device: torch.device,
                    max_paraphrases: Optional[int], last_only: bool=False, topk: int=64) -> Dict[str, Any]:
    """
    SVD of down_proj weight; energy of UP deltas (orig->para) in top-k singular subspace vs complement.
    """
    # grab down weight
    mlp = None
    for stem, leaf in [("model.model.layers","mlp"), ("model.layers","mlp"), ("transformer.h","mlp")]:
        try:
            base = model
            for part in stem.split("."): base = getattr(base, part)
            mlp = getattr(base[layer], leaf); break
        except Exception: pass
    if mlp is None: raise RuntimeError("MLP not found")
    if hasattr(mlp, "down_proj"): W = mlp.down_proj.weight.data   # [H, M]
    elif hasattr(mlp, "fc_out"):   W = mlp.fc_out.weight.data
    elif hasattr(mlp, "dense_4h_to_h"): W = mlp.dense_4h_to_h.weight.data
    else: raise RuntimeError("down weight not found")
    # SVD on CPU (safe for big)
    Wcpu = W.detach().float().cpu().numpy()
    try:
        U, S, Vt = np.linalg.svd(Wcpu, full_matrices=False)
    except np.linalg.LinAlgError:
        logging.warning("SVD did not converge; using np.linalg.eigh on W^T W")
        M = Wcpu.T @ Wcpu
        evals, V = np.linalg.eigh(M)
        idx = np.argsort(evals)[::-1]
        Vt = V[:, idx].T; S = np.sqrt(np.maximum(evals[idx], 0.0)); U = None  # Not used
    V = Vt  # [M, M] rows are right singular vectors in input space (POST)
    Vk = V[:topk]                # top-k
    # Function to project a vector (in POST/UP space) onto top-k vs complement energy
    def energy_split(vec: np.ndarray) -> Tuple[float,float]:
        # vec: [M]
        if vec.ndim == 1: v = vec[None,:]
        else: v = vec
        # project into top-k rows
        top = Vk @ v.T              # [k, 1]
        top_energy = float(np.sum(top**2))
        total = float(np.sum(v**2) + 1e-12)
        comp_energy = max(total - top_energy, 0.0)
        return top_energy, comp_energy

    accessor = MLPAccessor(model, layer, gate_mode=getattr(tokenizer, "_swiglu_gate_mode","silu"))
    res_hook = ResidualHook(model, layer)

    rows = []; start = time.time()
    for idx, ps in enumerate(items):
        texts = [ps.instruction_original] + [p["text"] for p in ps.paraphrases]
        if max_paraphrases is not None: texts = texts[: 1 + max_paraphrases]
        if len(texts) < 2: continue
        # get UP deltas (pooled)
        input_ids, attn = encode_batch(tokenizer, texts, device)
        _ = model(input_ids=input_ids, attention_mask=attn); H = res_hook.buffer
        UP = accessor.UP(H)
        UPool = pool_tokens(UP, attn, last_only=last_only).to(torch.float32).cpu().numpy()  # [N, M]
        orig = UPool[0]
        tops = []; comps = []
        for j in range(1, UPool.shape[0]):
            delta = UPool[j] - orig
            t, c = energy_split(delta)
            tops.append(t); comps.append(c)
        if tops:
            rows.append({
                "prompt_index": idx, "prompt_count": ps.prompt_count,
                "energy_topk_mean": float(np.mean(tops)), "energy_comp_mean": float(np.mean(comps)),
                "frac_topk": float(np.mean(np.array(tops) / (np.array(tops)+np.array(comps)+1e-12)))
            })
        if (idx+1) % 20 == 0:
            logging.info("[down-svd %s] %d/%d prompts — %s", now(), idx+1, len(items), format_eta(idx+1, len(items), start))

    res_hook.close()
    return {"per_prompt": rows, "topk": topk}

# plotting helpers

def write_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows: return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def summarize_numeric(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Summarise every numeric column across rows:
    mean, median, std, min, max, n, ci95.
    """
    if not rows: return {}
    agg = {}
    keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and np.isfinite(v)}
    for k in sorted(keys):
        vals = np.array([r[k] for r in rows if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])], dtype=float)
        if vals.size == 0:
            continue
        m = float(vals.mean())
        md = float(np.median(vals))
        s = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        mn = float(vals.min()) if vals.size > 0 else float("nan")
        mx = float(vals.max()) if vals.size > 0 else float("nan")
        ci = 1.96 * s / max(1.0, math.sqrt(vals.size))
        agg[k] = {"mean": m, "median": md, "std": s, "min": mn, "max": mx, "n": int(vals.size), "ci95": ci}
    return agg

def _desc(arr: Iterable[float]) -> Dict[str, float]:
    a = np.array([x for x in arr if isinstance(x,(int,float)) and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "ci95": float("nan")}
    s = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": s,
        "min": float(a.min()),
        "max": float(a.max()),
        "ci95": float(1.96 * s / max(1.0, math.sqrt(a.size))),
    }

def _write_summary_csv(label_to_desc: Dict[str, Dict[str, float]], path: Path) -> None:
    if not label_to_desc: return
    fields = ["label", "n", "mean", "median", "std", "min", "max", "ci95"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for label, d in label_to_desc.items():
            row = {"label": label}; row.update(d)
            w.writerow(row)

def _grouped_bars(xlabels: List[str], series: Dict[str, Tuple[List[float], Optional[List[float]]]], 
                  title: str, ylabel: str, outpath: Path, colors: Optional[Dict[str, str]]=None):
    """
    series: name -> (means, errs) where each is len(xlabels)
    colors: name -> color hex
    """
    if colors is None: colors = {}
    names = list(series.keys())
    m = len(names); n = len(xlabels)
    x = np.arange(n, dtype=float)
    width = 0.8 / max(m, 1)
    plt.figure(figsize=(max(8, 1.2*n), 4.3))
    for i, name in enumerate(names):
        means, errs = series[name]
        c = colors.get(name, COL_G2 if i % 2 else COL_FG)
        offset = (-0.4 + i*width + width/2.0)
        pos = x + offset
        plt.bar(pos, means, width=width, yerr=errs, capsize=3, color=c, alpha=0.95, edgecolor="none", label=name)
    plt.xticks(x, xlabels, rotation=0)
    plt.ylabel(ylabel); plt.title(title); plt.axhline(0, color=COL_G1, lw=0.8)
    plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(outpath, dpi=170); plt.close()

def _boxplot(data: Dict[str, List[float]], title: str, ylabel: str, outpath: Path):
    labels, arrays = [], []
    for k in data:
        arr = [v for v in data[k] if isinstance(v,(int,float)) and np.isfinite(v)]
        labels.append(k); arrays.append(arr)
    plt.figure(figsize=(max(6, 1.2*len(labels)), 4.0))
    bp = plt.boxplot(arrays, labels=labels, patch_artist=True)
    # grayscale + green fill
    fills = [COL_FG] + [COL_G3]*(len(labels)-1)
    for patch, c in zip(bp['boxes'], fills + [COL_G3]*(len(bp['boxes'])-len(fills))):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    plt.ylabel(ylabel); plt.title(title); plt.grid(True, axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout(); plt.savefig(outpath, dpi=170); plt.close()

def _hist(arr: List[float], title: str, xlabel: str, outpath: Path, bins: int = 40):
    a = np.array([v for v in arr if isinstance(v,(int,float)) and np.isfinite(v)], dtype=float)
    plt.figure(figsize=(7, 4))
    plt.hist(a, bins=bins, color=COL_FG, alpha=0.9, edgecolor="none")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(outpath, dpi=170); plt.close()

def save_json(obj: Any, path: Path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)

# model loading

def build_models_and_tokenizers(
    base_model_name_or_path: str,
    ft_model_name_or_path: Optional[str],
    ft_lora_adapter: Optional[str],
    device: torch.device,
):
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer_base.pad_token is None: tokenizer_base.pad_token = tokenizer_base.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32

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
    if tokenizer_ft.pad_token is None: tokenizer_ft.pad_token = tokenizer_ft.eos_token

    return (base, tokenizer_base), (ft, tokenizer_ft)

# main

def main():
    ap = argparse.ArgumentParser(description="Robustness probes across MLP micro-steps + logits (layer-centric)")
    ap.add_argument("--selection_jsonl", type=str, required=True)
    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)
    ap.add_argument("--layer_index", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)

    # limits
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--max_paraphrases", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)

    # token pooling
    ap.add_argument("--token_pool", type=str, default="mean", choices=["mean","last"])
    ap.add_argument("--last_only", type=int, default=0)
    ap.add_argument("--token_slices", type=int, default=0)

    # controls
    ap.add_argument("--allow_types_csv", type=str, default=None, help="CSV/JSON with a column/list of allowed paraphrase keys (e.g., those with >=200 and score==5).")
    ap.add_argument("--require_score5", type=int, default=0, help="Keep only paraphrases with paraphrase_content_score==5 if field is present.")

    # old metrics
    ap.add_argument("--compute_jacobians", type=int, default=1)
    ap.add_argument("--jacobian_mode", type=str, default="pca", choices=["pca","raw"])
    ap.add_argument("--topk_pca", type=int, default=8)
    ap.add_argument("--eps_jacobian", type=float, default=1e-3)

    # new probes
    ap.add_argument("--compute_pair_dispersion", type=int, default=1)
    ap.add_argument("--compute_dir_jac", type=int, default=1, help="Jacobian along paraphrase delta at UP/POST/DOWN")
    ap.add_argument("--compute_logits_kl", type=int, default=1, help="Next-token KL ΔΔ")
    ap.add_argument("--do_activation_patching", type=int, default=0, help="Causal patching at POST within each model")
    ap.add_argument("--patch_positions", type=str, default="last", choices=["last","all"])
    ap.add_argument("--patch_sample_per_prompt", type=int, default=1)

    ap.add_argument("--compute_gate_stats", type=int, default=1)
    ap.add_argument("--compute_down_svd", type=int, default=1)
    ap.add_argument("--down_topk", type=int, default=64)

    ap.add_argument("--swiglu_gate", type=str, default="gelu",
        choices=["silu", "gelu", "sigmoid", "relu", "none"],
        help="Nonlinearity used for the gate branch when applying POST offline")

    args = ap.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("Starting %s", Path(sys.argv[0]).name)
    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    # controls: load allowlist of types if provided
    allow_types = None
    if args.allow_types_csv:
        p = Path(args.allow_types_csv)
        try:
            if p.suffix.lower() == ".json":
                allow_types = set(json.loads(p.read_text()))
            else:
                # CSV with a column 'key' or first column
                rows = []
                with p.open("r", encoding="utf-8") as f:
                    header = f.readline().strip().split(",")
                    if header:
                        key_idx = 0
                        for i, h in enumerate(header):
                            if h.strip().lower() in ("key","instruct_type","paraphrase_key"): key_idx = i; break
                        for line in f:
                            cols = [c.strip() for c in line.strip().split(",")]
                            if cols and cols[key_idx]: rows.append(cols[key_idx])
                allow_types = set(rows)
            logging.info("Loaded %d allowed paraphrase keys from %s", len(allow_types), p)
        except Exception as e:
            logging.warning("Could not read allow_types file %s: %s", p, e)

    # selection
    logging.info("Reading selection from %s", args.selection_jsonl)
    items = load_selection_jsonl(args.selection_jsonl, max_prompts=args.max_prompts,
                                 allow_types=allow_types, require_score5=bool(args.require_score5))
    logging.info("Loaded %d prompts.", len(items))

    # models
    device = torch.device(args.device)
    (base, tok_base), (ft, tok_ft) = build_models_and_tokenizers(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, device
    )
    tok_base._swiglu_gate_mode = args.swiglu_gate
    tok_ft._swiglu_gate_mode   = args.swiglu_gate
    logging.info("Models ready. Layer index = %d", args.layer_index)

    # capture stations (BASE + FT)
    logging.info("Capturing MLP stations (batched) — BASE")
    base_res = capture_stations_for_model(
        base, tok_base, items, args.layer_index, device, args.max_prompts, args.max_paraphrases,
        token_pool=args.token_pool, last_only=bool(args.last_only), batch_size=args.batch_size,
        token_slices=args.token_slices
    )
    logging.info("Capturing MLP stations (batched) — FT")
    ft_res = capture_stations_for_model(
        ft, tok_ft, items, args.layer_index, device, args.max_prompts, args.max_paraphrases,
        token_pool=args.token_pool, last_only=bool(args.last_only), batch_size=args.batch_size,
        token_slices=args.token_slices
    )
    # persist
    write_rows_csv(base_res["per_prompt"], Path(outdir) / f"BASE_mlp_station_metrics_layer{args.layer_index}.csv")
    write_rows_csv(ft_res["per_prompt"],   Path(outdir) / f"FT_mlp_station_metrics_layer{args.layer_index}.csv")

    # Merge and compute ΔΔ for original-to-paraphrase dist and paraphrase-only dispersion
    logging.info("Merging BASE/FT station metrics and computing ΔΔ")
    bmap = {r["prompt_count"]: r for r in base_res["per_prompt"]}
    fmap = {r["prompt_count"]: r for r in ft_res["per_prompt"]}
    stages = ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]
    merged_rows = []
    for pc in sorted(set(bmap.keys()) & set(fmap.keys())):
        rb, rf = bmap[pc], fmap[pc]
        row = {"prompt_count": pc, "N": rb["N"]}
        for s in stages:
            row[f"dist_{s}_BASE"] = rb[f"dist_{s}"]; row[f"dist_{s}_FT"] = rf[f"dist_{s}"]
            row[f"ddiff_dist_{s}"] = rf[f"dist_{s}"] - rb[f"dist_{s}"]
            # paraphrase-only dispersion
            kb = f"disp_para_{s}"; 
            row[kb+"_BASE"] = rb.get(kb, float("nan"))
            row[kb+"_FT"]   = rf.get(kb, float("nan"))
            row[f"ddiff_disp_para_{s}"] = (rf.get(kb, float("nan")) - rb.get(kb, float("nan")))
        merged_rows.append(row)
    write_rows_csv(merged_rows, Path(outdir) / f"merged_mlp_station_metrics_layer{args.layer_index}.csv")

    # Summaries + grouped plots for station metrics
    def vec(rows, key):
        arr = [r[key] for r in rows if isinstance(r.get(key),(int,float)) and np.isfinite(r[key])]
        return np.array(arr, dtype=float)

    stages = ["RES", "UP", "GATE_PRE", "GATE_ACT", "POST", "DOWN"]

    # 1) To-original distances: BASE vs FT vs ΔΔ (per stage)
    means_base, ci_base, means_ft, ci_ft, means_dd, ci_dd = [], [], [], [], [], []
    for s in stages:
        vb = vec(base_res["per_prompt"], f"dist_{s}")
        vf = vec(ft_res["per_prompt"],   f"dist_{s}")
        vd = vf - vb[np.newaxis, ...] if vb.size and vf.size and vb.shape == vf.shape else vf - vb[:vf.size]
        # When shapes differ slightly, align to min length
        m = min(vb.size, vf.size)
        if m > 0:
            vb = vb[:m]; vf = vf[:m]; vd = vf - vb
        desc_b = _desc(vb); desc_f = _desc(vf); desc_d = _desc(vd)
        means_base.append(desc_b["mean"]); ci_base.append(desc_b["ci95"])
        means_ft.append(desc_f["mean"]);   ci_ft.append(desc_f["ci95"])
        means_dd.append(desc_d["mean"]);   ci_dd.append(desc_d["ci95"])

    # write compact summary CSV
    summ_rows = {}
    for i, s in enumerate(stages):
        summ_rows[f"dist_{s}_BASE"] = _desc(vec(base_res["per_prompt"], f"dist_{s}"))
        summ_rows[f"dist_{s}_FT"]   = _desc(vec(ft_res["per_prompt"],   f"dist_{s}"))
        # ΔΔ per stage is computed on aligned arrays above; recompute safely here:
        vb = vec(base_res["per_prompt"], f"dist_{s}"); vf = vec(ft_res["per_prompt"], f"dist_{s}")
        m = min(vb.size, vf.size)
        dd = (vf[:m] - vb[:m]) if m>0 else np.array([])
        summ_rows[f"ddiff_dist_{s}"] = _desc(dd)
    _write_summary_csv(summ_rows, Path(outdir)/f"summary_stations_to_original_layer{args.layer_index}.csv")

    _grouped_bars(
        stages,
        {
            "BASE": (means_base, ci_base),
            "FT":   (means_ft,   ci_ft),
            "ΔΔ (FT−BASE)": (means_dd,   ci_dd),
        },
        title="To-original distance (mean L2): BASE vs FT vs ΔΔ",
        ylabel="mean L2(original, paraphrase)",
        outpath=Path(outdir)/f"stations_to_original_grouped_layer{args.layer_index}.png",
        colors={"BASE": COL_G2, "FT": COL_FG, "ΔΔ (FT−BASE)": COL_G1},
    )

    # 2) Paraphrase-only dispersion: BASE vs FT vs ΔΔ (per stage)
    means_base, ci_base, means_ft, ci_ft, means_dd, ci_dd = [], [], [], [], [], []
    for s in stages:
        kb = f"disp_para_{s}"
        vb = vec(base_res["per_prompt"], kb)
        vf = vec(ft_res["per_prompt"],   kb)
        m = min(vb.size, vf.size)
        if m>0:
            dd = vf[:m] - vb[:m]
        else:
            dd = np.array([])
        db, df, ddsc = _desc(vb), _desc(vf), _desc(dd)
        means_base.append(db["mean"]); ci_base.append(db["ci95"])
        means_ft.append(df["mean"]);   ci_ft.append(df["ci95"])
        means_dd.append(ddsc["mean"]); ci_dd.append(ddsc["ci95"])

    # write compact summary CSV
    summ_rows = {}
    for s in stages:
        kb = f"disp_para_{s}"
        vb = vec(base_res["per_prompt"], kb); vf = vec(ft_res["per_prompt"], kb)
        m = min(vb.size, vf.size)
        dd = (vf[:m] - vb[:m]) if m>0 else np.array([])
        summ_rows[kb+"_BASE"] = _desc(vb)
        summ_rows[kb+"_FT"]   = _desc(vf)
        summ_rows[f"ddiff_{kb}"] = _desc(dd)
    _write_summary_csv(summ_rows, Path(outdir)/f"summary_stations_paraphrase_dispersion_layer{args.layer_index}.csv")

    _grouped_bars(
        stages,
        {
            "BASE": (means_base, ci_base),
            "FT":   (means_ft,   ci_ft),
            "ΔΔ (FT−BASE)": (means_dd,   ci_dd),
        },
        title="Paraphrase-only dispersion (avg L2): BASE vs FT vs ΔΔ",
        ylabel="avg L2 across paraphrases",
        outpath=Path(outdir)/f"stations_paraphrase_dispersion_grouped_layer{args.layer_index}.png",
        colors={"BASE": COL_G2, "FT": COL_FG, "ΔΔ (FT−BASE)": COL_G1},
    )

    # Jacobians: classic + along paraphrase delta
    if args.compute_jacobians:
        logging.info("Computing Jacobians (classic PCA/raw+random) — BASE")
        jac_base = jacobian_norms_for_model(
            base, tok_base, items, args.layer_index, device, args.topk_pca, args.jacobian_mode,
            args.max_prompts, args.max_paraphrases, eps=args.eps_jacobian,
            last_only=bool(args.last_only), batch_size=args.batch_size, along_paraphrase_delta=False
        )
        logging.info("Computing Jacobians (classic) — FT")
        jac_ft = jacobian_norms_for_model(
            ft, tok_ft, items, args.layer_index, device, args.topk_pca, args.jacobian_mode,
            args.max_prompts, args.max_paraphrases, eps=args.eps_jacobian,
            last_only=bool(args.last_only), batch_size=args.batch_size, along_paraphrase_delta=False
        )
        write_rows_csv(jac_base["per_prompt"], Path(outdir)/f"jacobian_classic_BASE_layer{args.layer_index}.csv")
        write_rows_csv(jac_ft["per_prompt"],   Path(outdir)/f"jacobian_classic_FT_layer{args.layer_index}.csv")

    # Summaries + plot: Jacobian norms (classic)
    parts = ["jac_UP_mean", "jac_POST_mean", "jac_DOWN_mean"]
    def _vec_j(rows, k): return np.array([r[k] for r in rows if isinstance(r.get(k),(int,float)) and np.isfinite(r[k])], float)

    means_base = []; ci_base = []; means_ft = []; ci_ft = []; means_dd = []; ci_dd = []
    # align by prompt_count for ΔΔ
    mapB = {r["prompt_count"]: r for r in jac_base["per_prompt"]}
    mapF = {r["prompt_count"]: r for r in jac_ft["per_prompt"]}
    overlap = sorted(set(mapB) & set(mapF))
    for p in parts:
        vb = np.array([mapB[pc][p] for pc in overlap], float)
        vf = np.array([mapF[pc][p] for pc in overlap], float)
        db, df, dd = _desc(vb), _desc(vf), _desc(vf - vb)
        means_base.append(db["mean"]); ci_base.append(db["ci95"])
        means_ft.append(df["mean"]);   ci_ft.append(df["ci95"])
        means_dd.append(dd["mean"]);   ci_dd.append(dd["ci95"])

    # summary CSV
    summ = {}
    for p in parts:
        vb = _vec_j(jac_base["per_prompt"], p)
        vf = _vec_j(jac_ft["per_prompt"],   p)
        m = min(vb.size, vf.size)
        dd = vf[:m] - vb[:m] if m>0 else np.array([])
        summ[p+"_BASE"] = _desc(vb)
        summ[p+"_FT"]   = _desc(vf)
        summ["ddiff_"+p] = _desc(dd)
    _write_summary_csv(summ, Path(outdir)/f"summary_jacobian_classic_layer{args.layer_index}.csv")

    _grouped_bars(
        ["UP","POST","DOWN"],
        {"BASE": (means_base, ci_base), "FT": (means_ft, ci_ft), "ΔΔ (FT−BASE)": (means_dd, ci_dd)},
        title="Jacobian norms (classic): BASE vs FT vs ΔΔ",
        ylabel="||∂f/∂h|| (finite-diff approx)",
        outpath=Path(outdir)/f"jacobian_classic_grouped_layer{args.layer_index}.png",
        colors={"BASE": COL_G2, "FT": COL_FG, "ΔΔ (FT−BASE)": COL_G1},
    )

    if args.compute_dir_jac:
        logging.info("Computing Jacobians along paraphrase delta — BASE")
        jacB = jacobian_norms_for_model(
            base, tok_base, items, args.layer_index, device, args.topk_pca, args.jacobian_mode,
            args.max_prompts, args.max_paraphrases, eps=args.eps_jacobian,
            last_only=bool(args.last_only), batch_size=args.batch_size, along_paraphrase_delta=True
        )
        logging.info("Computing Jacobians along paraphrase delta — FT")
        jacF = jacobian_norms_for_model(
            ft, tok_ft, items, args.layer_index, device, args.topk_pca, args.jacobian_mode,
            args.max_prompts, args.max_paraphrases, eps=args.eps_jacobian,
            last_only=bool(args.last_only), batch_size=args.batch_size, along_paraphrase_delta=True
        )
        write_rows_csv(jacB["per_prompt"], Path(outdir)/f"jacobian_paradelta_BASE_layer{args.layer_index}.csv")
        write_rows_csv(jacF["per_prompt"], Path(outdir)/f"jacobian_paradelta_FT_layer{args.layer_index}.csv")

        # Summaries + plot: Jacobian norms (along paraphrase delta)
        parts = ["jac_UP_mean", "jac_POST_mean", "jac_DOWN_mean"]
        mapB = {r["prompt_count"]: r for r in jacB["per_prompt"]}
        mapF = {r["prompt_count"]: r for r in jacF["per_prompt"]}
        overlap = sorted(set(mapB) & set(mapF))
        means_base = []; ci_base = []; means_ft = []; ci_ft = []; means_dd = []; ci_dd = []
        for p in parts:
            vb = np.array([mapB[pc][p] for pc in overlap], float)
            vf = np.array([mapF[pc][p] for pc in overlap], float)
            db, df, dd = _desc(vb), _desc(vf), _desc(vf - vb)
            means_base.append(db["mean"]); ci_base.append(db["ci95"])
            means_ft.append(df["mean"]);   ci_ft.append(df["ci95"])
            means_dd.append(dd["mean"]);   ci_dd.append(dd["ci95"])

        summ = {}
        for p in parts:
            vb = np.array([mapB[pc][p] for pc in overlap], float)
            vf = np.array([mapF[pc][p] for pc in overlap], float)
            summ[p+"_BASE"] = _desc(vb)
            summ[p+"_FT"]   = _desc(vf)
            summ["ddiff_"+p] = _desc(vf - vb)
        _write_summary_csv(summ, Path(outdir)/f"summary_jacobian_paradelta_layer{args.layer_index}.csv")

        _grouped_bars(
            ["UP","POST","DOWN"],
            {"BASE": (means_base, ci_base), "FT": (means_ft, ci_ft), "ΔΔ (FT−BASE)": (means_dd, ci_dd)},
            title="Jacobian norms (along paraphrase delta): BASE vs FT vs ΔΔ",
            ylabel="||∂f/∂h|| (finite-diff approx)",
            outpath=Path(outdir)/f"jacobian_paradelta_grouped_layer{args.layer_index}.png",
            colors={"BASE": COL_G2, "FT": COL_FG, "ΔΔ (FT−BASE)": COL_G1},
        )

    # Logit-level ΔΔ (next-token)
    if args.compute_logits_kl:
        logging.info("Computing next-token KL ΔΔ (BASE vs FT)")
        kl_rows = logits_kl_ddiff(((base,tok_base),(ft,tok_ft)), items, device, args.max_paraphrases, args.max_prompts)
        write_rows_csv(kl_rows["per_prompt"], Path(outdir)/"logits_next_token_kl_ddiff.csv")

        dd = np.array([r["ddiff_kl_next"] for r in kl_rows["per_prompt"] if isinstance(r.get("ddiff_kl_next"), (int,float)) and np.isfinite(r["ddiff_kl_next"])], float)
        kb = np.array([r["kl_next_base"] for r in kl_rows["per_prompt"] if isinstance(r.get("kl_next_base"), (int,float)) and np.isfinite(r["kl_next_base"])], float)
        kf = np.array([r["kl_next_ft"]   for r in kl_rows["per_prompt"] if isinstance(r.get("kl_next_ft"), (int,float)) and np.isfinite(r["kl_next_ft"])], float)

        # summary CSV
        _write_summary_csv({
            "kl_next_base": _desc(kb),
            "kl_next_ft":   _desc(kf),
            "ddiff_kl_next": _desc(dd),
        }, Path(outdir)/"summary_logits_next_token_kl.csv")

        logging.info("KL ΔΔ mean=%.4f (n=%d)", float(np.nanmean(dd)) if dd.size else float("nan"), int(dd.size))

        # graphics
        _hist(dd.tolist(), title="Next-token KL ΔΔ (FT−BASE)", xlabel="KL ΔΔ", outpath=Path(outdir)/"kl_ddiff_hist.png")
        _boxplot({"ΔΔ (FT−BASE)": dd.tolist(), "BASE": kb.tolist(), "FT": kf.tolist()},
                title="Next-token KL: BASE vs FT vs ΔΔ",
                ylabel="KL",
                outpath=Path(outdir)/"kl_base_ft_ddiff_box.png")

        # Means with CI
        ddb, ddc = _desc(kb), _desc(kf)
        ddsc     = _desc(dd)
        _grouped_bars(
            ["BASE", "FT", "ΔΔ (FT−BASE)"],
            {"mean": ([ddb["mean"], ddc["mean"], ddsc["mean"]], [ddb["ci95"], ddc["ci95"], ddsc["ci95"]])},
            title="Next-token KL (means with 95% CI)",
            ylabel="KL",
            outpath=Path(outdir)/"kl_means_ci.png",
            colors={"mean": COL_FG},
        )

    # Activation patching (optional)
    if args.do_activation_patching:
        logging.info("Activation patching at POST (%s positions)", args.patch_positions)
        patch_rows = activation_patching_kl(((base,tok_base),(ft,tok_ft)), items, device, args.layer_index,
                                            positions=args.patch_positions, max_paraphrases=args.max_paraphrases,
                                            sample_limit_per_prompt=args.patch_sample_per_prompt)
        write_rows_csv(patch_rows["per_prompt"], Path(outdir)/f"activation_patching_layer{args.layer_index}.csv")

        # Summaries + plots for activation patching
        rows = patch_rows["per_prompt"]
        base_b = [r["kl_before"] for r in rows if r["model"]=="BASE"]
        base_a = [r["kl_after"]  for r in rows if r["model"]=="BASE"]
        ft_b   = [r["kl_before"] for r in rows if r["model"]=="FT"]
        ft_a   = [r["kl_after"]  for r in rows if r["model"]=="FT"]
        base_d = [r["delta_kl"]  for r in rows if r["model"]=="BASE"]
        ft_d   = [r["delta_kl"]  for r in rows if r["model"]=="FT"]

        _write_summary_csv({
            "BASE_kl_before": _desc(base_b), "BASE_kl_after": _desc(base_a), "BASE_delta_kl": _desc(base_d),
            "FT_kl_before":   _desc(ft_b),   "FT_kl_after":   _desc(ft_a),   "FT_delta_kl":   _desc(ft_d),
        }, Path(outdir)/f"summary_activation_patching_layer{args.layer_index}.csv")

        # Boxplot of ΔKL by model
        _boxplot({"ΔKL BASE": base_d, "ΔKL FT": ft_d}, 
                title=f"Activation patching @POST layer {args.layer_index}: ΔKL (after − before)",
                ylabel="ΔKL", outpath=Path(outdir)/f"activation_patching_delta_box_layer{args.layer_index}.png")

        # Means with CI for before vs after (per model)
        desc_BASE_b, desc_BASE_a = _desc(base_b), _desc(base_a)
        desc_FT_b,   desc_FT_a   = _desc(ft_b),   _desc(ft_a)
        _grouped_bars(
            ["BASE", "FT"],
            {
                "KL before": ([desc_BASE_b["mean"], desc_FT_b["mean"]], [desc_BASE_b["ci95"], desc_FT_b["ci95"]]),
                "KL after":  ([desc_BASE_a["mean"], desc_FT_a["mean"]], [desc_BASE_a["ci95"], desc_FT_a["ci95"]]),
            },
            title=f"Activation patching @POST layer {args.layer_index}: before vs after",
            ylabel="KL",
            outpath=Path(outdir)/f"activation_patching_before_after_layer{args.layer_index}.png",
            colors={"KL before": COL_G2, "KL after": COL_FG},
        )

    # Gate stats
    if args.compute_gate_stats:
        logging.info("Gate stats — BASE")
        gsB = gate_stats(base, tok_base, items, args.layer_index, device, args.max_prompts, args.max_paraphrases,
                         batch_size=args.batch_size, last_only=bool(args.last_only))
        logging.info("Gate stats — FT")
        gsF = gate_stats(ft, tok_ft, items, args.layer_index, device, args.max_prompts, args.max_paraphrases,
                         batch_size=args.batch_size, last_only=bool(args.last_only))
        write_rows_csv(gsB["per_prompt"], Path(outdir)/f"gate_stats_BASE_layer{args.layer_index}.csv")
        write_rows_csv(gsF["per_prompt"], Path(outdir)/f"gate_stats_FT_layer{args.layer_index}.csv")

        # Summaries + plots for gate statistics
        def _pull(rows, key): return [r[key] for r in rows["per_prompt"] if isinstance(r.get(key),(int,float)) and np.isfinite(r[key])]

        metrics = ["mean", "std", "kurtosis", "frac_near0"]
        # Gate PRE
        desc_B_pre = {m: _desc(_pull(gsB, f"gate_pre_{m}")) for m in metrics}
        desc_F_pre = {m: _desc(_pull(gsF, f"gate_pre_{m}")) for m in metrics}
        # Gate ACT
        desc_B_act = {m: _desc(_pull(gsB, f"gate_act_{m}")) for m in metrics}
        desc_F_act = {m: _desc(_pull(gsF, f"gate_act_{m}")) for m in metrics}

        # Write compact summaries
        _write_summary_csv({f"BASE_gate_pre_{m}": desc_B_pre[m] for m in metrics} |
                        {f"FT_gate_pre_{m}":   desc_F_pre[m] for m in metrics},
                        Path(outdir)/f"summary_gate_pre_layer{args.layer_index}.csv")
        _write_summary_csv({f"BASE_gate_act_{m}": desc_B_act[m] for m in metrics} |
                        {f"FT_gate_act_{m}":   desc_F_act[m] for m in metrics},
                        Path(outdir)/f"summary_gate_act_layer{args.layer_index}.csv")

        # Grouped bars PRE
        _grouped_bars(
            ["mean","std","kurtosis","frac<ε"],
            {
                "BASE": ([desc_B_pre["mean"]["mean"], desc_B_pre["std"]["mean"], desc_B_pre["kurtosis"]["mean"], desc_B_pre["frac_near0"]["mean"]],
                        [desc_B_pre["mean"]["ci95"],  desc_B_pre["std"]["ci95"],  desc_B_pre["kurtosis"]["ci95"],  desc_B_pre["frac_near0"]["ci95"]]),
                "FT":   ([desc_F_pre["mean"]["mean"], desc_F_pre["std"]["mean"], desc_F_pre["kurtosis"]["mean"], desc_F_pre["frac_near0"]["mean"]],
                        [desc_F_pre["mean"]["ci95"],  desc_F_pre["std"]["ci95"],  desc_F_pre["kurtosis"]["ci95"],  desc_F_pre["frac_near0"]["ci95"]]),
            },
            title=f"Gate PRE summary @layer {args.layer_index}",
            ylabel="value",
            outpath=Path(outdir)/f"gate_pre_summary_grouped_layer{args.layer_index}.png",
            colors={"BASE": COL_G2, "FT": COL_FG},
        )

        # Grouped bars ACT
        _grouped_bars(
            ["mean","std","kurtosis","frac<ε"],
            {
                "BASE": ([desc_B_act["mean"]["mean"], desc_B_act["std"]["mean"], desc_B_act["kurtosis"]["mean"], desc_B_act["frac_near0"]["mean"]],
                        [desc_B_act["mean"]["ci95"],  desc_B_act["std"]["ci95"],  desc_B_act["kurtosis"]["ci95"],  desc_B_act["frac_near0"]["ci95"]]),
                "FT":   ([desc_F_act["mean"]["mean"], desc_F_act["std"]["mean"], desc_F_act["kurtosis"]["mean"], desc_F_act["frac_near0"]["mean"]],
                        [desc_F_act["mean"]["ci95"],  desc_F_act["std"]["ci95"],  desc_F_act["kurtosis"]["ci95"],  desc_F_act["frac_near0"]["ci95"]]),
            },
            title=f"Gate ACT summary @layer {args.layer_index}",
            ylabel="value",
            outpath=Path(outdir)/f"gate_act_summary_grouped_layer{args.layer_index}.png",
            colors={"BASE": COL_G2, "FT": COL_FG},
        )

    # DOWN SVD energy split
    if args.compute_down_svd:
        logging.info("DOWN SVD energy split — BASE")
        svdB = down_svd_energy(base, items, tok_base, args.layer_index, device, args.max_paraphrases,
                               last_only=bool(args.last_only), topk=args.down_topk)
        logging.info("DOWN SVD energy split — FT")
        svdF = down_svd_energy(ft, items, tok_ft, args.layer_index, device, args.max_paraphrases,
                               last_only=bool(args.last_only), topk=args.down_topk)
        write_rows_csv(svdB["per_prompt"], Path(outdir)/f"down_svd_energy_BASE_layer{args.layer_index}.csv")
        write_rows_csv(svdF["per_prompt"], Path(outdir)/f"down_svd_energy_FT_layer{args.layer_index}.csv")

        # Summaries + plots for DOWN SVD energy split
        def _pullv(rows, key): return [r[key] for r in rows["per_prompt"] if isinstance(r.get(key),(int,float)) and np.isfinite(r[key])]
        eB_top, eB_comp, fB = _pullv(svdB, "energy_topk_mean"), _pullv(svdB, "energy_comp_mean"), _pullv(svdB, "frac_topk")
        eF_top, eF_comp, fF = _pullv(svdF, "energy_topk_mean"), _pullv(svdF, "energy_comp_mean"), _pullv(svdF, "frac_topk")

        _write_summary_csv({
            "BASE_energy_topk_mean": _desc(eB_top), "BASE_energy_comp_mean": _desc(eB_comp), "BASE_frac_topk": _desc(fB),
            "FT_energy_topk_mean":   _desc(eF_top), "FT_energy_comp_mean":   _desc(eF_comp), "FT_frac_topk":   _desc(fF),
        }, Path(outdir)/f"summary_down_svd_energy_layer{args.layer_index}.csv")

        # Energies (topk vs comp), grouped by model
        _grouped_bars(
            ["energy_topk", "energy_comp"],
            {
                "BASE": ([ _desc(eB_top)["mean"],  _desc(eB_comp)["mean"]], [ _desc(eB_top)["ci95"], _desc(eB_comp)["ci95"]]),
                "FT":   ([ _desc(eF_top)["mean"],  _desc(eF_comp)["mean"]], [ _desc(eF_top)["ci95"], _desc(eF_comp)["ci95"]]),
            },
            title=f"DOWN SVD energy split @layer {args.layer_index}",
            ylabel="energy (sum of squares)",
            outpath=Path(outdir)/f"down_svd_energy_grouped_layer{args.layer_index}.png",
            colors={"BASE": COL_G2, "FT": COL_FG},
        )

        # Fraction in top-k
        _grouped_bars(
            ["frac_topk"],
            {
                "BASE": ([ _desc(fB)["mean"] ], [ _desc(fB)["ci95"] ]),
                "FT":   ([ _desc(fF)["mean"] ], [ _desc(fF)["ci95"] ]),
            },
            title=f"DOWN SVD: fraction of delta energy in top-{args.down_topk}",
            ylabel="fraction",
            outpath=Path(outdir)/f"down_svd_fraction_grouped_layer{args.layer_index}.png",
            colors={"BASE": COL_G2, "FT": COL_FG},
        )

    # Summaries
    logging.info("Writing compact numeric summaries + index")
    summaries = {
        "stations_BASE": summarize_numeric(base_res["per_prompt"]),
        "stations_FT":   summarize_numeric(ft_res["per_prompt"]),
        "merged_ddiff":  summarize_numeric(merged_rows),
        "files_summary_csv": [
            f"summary_stations_to_original_layer{args.layer_index}.csv",
            f"summary_stations_paraphrase_dispersion_layer{args.layer_index}.csv",
            "summary_logits_next_token_kl.csv",
            f"summary_activation_patching_layer{args.layer_index}.csv" if args.do_activation_patching else None,
            f"summary_gate_pre_layer{args.layer_index}.csv" if args.compute_gate_stats else None,
            f"summary_gate_act_layer{args.layer_index}.csv" if args.compute_gate_stats else None,
            f"summary_down_svd_energy_layer{args.layer_index}.csv" if args.compute_down_svd else None,
            f"summary_jacobian_classic_layer{args.layer_index}.csv" if args.compute_jacobians else None,
            f"summary_jacobian_paradelta_layer{args.layer_index}.csv" if args.compute_dir_jac else None,
        ],
        "files_figures": [
            f"stations_to_original_grouped_layer{args.layer_index}.png",
            f"stations_paraphrase_dispersion_grouped_layer{args.layer_index}.png",
            "kl_ddiff_hist.png",
            "kl_base_ft_ddiff_box.png",
            "kl_means_ci.png",
            f"activation_patching_delta_box_layer{args.layer_index}.png" if args.do_activation_patching else None,
            f"activation_patching_before_after_layer{args.layer_index}.png" if args.do_activation_patching else None,
            f"gate_pre_summary_grouped_layer{args.layer_index}.png" if args.compute_gate_stats else None,
            f"gate_act_summary_grouped_layer{args.layer_index}.png" if args.compute_gate_stats else None,
            f"down_svd_energy_grouped_layer{args.layer_index}.png" if args.compute_down_svd else None,
            f"down_svd_fraction_grouped_layer{args.layer_index}.png" if args.compute_down_svd else None,
            f"jacobian_classic_grouped_layer{args.layer_index}.png" if args.compute_jacobians else None,
            f"jacobian_paradelta_grouped_layer{args.layer_index}.png" if args.compute_dir_jac else None,
        ]
    }
    # prune None
    summaries["files_summary_csv"] = [p for p in summaries["files_summary_csv"] if p]
    summaries["files_figures"]     = [p for p in summaries["files_figures"] if p]
    save_json(summaries, Path(outdir)/"summaries_overall.json")


    logging.info("All done. Outputs in %s", outdir)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
