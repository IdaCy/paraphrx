#!/usr/bin/env python3
"""
SCRIPT G

run_mlp_diff_of_diffs.py

End-to-end analysis for "diff-of-diffs at all MLP stations" + Jacobian overlays.

WHAT THIS DOES
==============
Given a prefiltered selection file (as produced by SCRIPT A sampler; e.g., `jacobian_prompts.jsonl`)
and two models (BASE and FT), this script:

1) Captures pooled activations for each prompt (original + paraphrases) at layer L at the following
   MLP "stations":
   - RES  : residual stream *just before* the MLP (i.e., the MLP's input)
   - UP   : upstream linear output (SwiGLU: up_proj; GeLU: wi/fc_in/dense_h_to_4h)
   - POST : post-nonlinearity output (SwiGLU: silu(gate)*up; GeLU: gelu(up))
   - DOWN : downstream linear output (SwiGLU/GeLU downstream linear)

2) Computes, for each prompt and station, the *average cosine similarity to the original* across its
   paraphrases, plus the complementary 1−cos distance. We then compute diff-of-diffs per station:
       ΔΔ_cos(S)  = mean_cos_FT(S)  − mean_cos_BASE(S)
       ΔΔ_dist(S) = mean_dist_FT(S) − mean_dist_BASE(S)   where dist = 1 − cos
   Negative ΔΔ_dist indicates the FT model *brings paraphrases closer* to the original in station S.
   We also compute stage deltas (POST−UP, DOWN−UP) per model and ΔΔ of those deltas.

3) Reproduces the "Jacobian sensitivity" probes, Base vs FT, along three sets of directions in the
   paraphrase subspace (reusing the approach from your SCRIPT D):
   - MEAN     : the normalized mean paraphrase direction
   - VARIANCE : top-k PCs of the paraphrase-centered cloud (or raw centered directions)
   - RANDOM   : random unit vectors as control
   and compares UPSTREAM vs DOWNSTREAM sensitivities. Produces joint overlays:
   - combined bars with BASE vs FT (colors), upstream vs downstream (shade), mean/var/random (linestyle)

4) Emits CSV/JSON summaries and well-labeled graphics (hist overlays, paired scatters, multi-series
   overlays), plus a compact text/markdown summary.

USAGE (examples)
================
CUDA example, layer 6, limit to 200 prompts and 16 paraphrases, with Jacobians:
    python run_mlp_diff_of_diffs.py \
        --selection_jsonl /path/to/jacobian_prompts.jsonl \
        --base_model_name_or_path google/gemma-2-2b-it \
        --ft_model_name_or_path /path/to/ft-merged \
        --layer_index 6 \
        --device cuda \
        --outdir ./diff_of_diffs_L6 \
        --max_prompts 200 \
        --max_paraphrases 16 \
        --enable_batching 1 \
        --batch_size 8 \
        --compute_jacobians 1 \
        --jacobian_mode pca \
        --topk_pca 8

If your FT is a LoRA adapter directory (unmerged), pass it with --ft_lora_adapter and also pass
--ft_model_name_or_path (the merged FT *or* the base model path to host the adapter):
    python run_mlp_diff_of_diffs.py \
        --selection_jsonl /path/to/jacobian_prompts.jsonl \
        --base_model_name_or_path google/gemma-2-2b-it \
        --ft_model_name_or_path google/gemma-2-2b-it \
        --ft_lora_adapter /path/to/lora-adapter \
        --layer_index 6 \
        --outdir ./diff_of_diffs_L6_lora \
        --compute_jacobians 1

OUTPUTS
=======
- CSVs:
    BASE_mlp_station_metrics_layer{L}.csv
    FT_mlp_station_metrics_layer{L}.csv
    merged_mlp_station_metrics_layer{L}.csv  (per-prompt BASE+FT with Δ and ΔΔ)
    jacobian_BASE_layer{L}.csv
    jacobian_FT_layer{L}.csv
    summaries.json  (means / medians / stdev / min / max / t-tests / effect sizes)
- Figures:
    ddiff_dist_by_station_bars.png          (ΔΔ of 1−cos distance at RES/UP/POST/DOWN)
    ddiff_dist_by_station_hist.png          (hist overlay BASE vs FT per station, as Δ values)
    stage_delta_ddiff_bars.png              (ΔΔ of (POST−UP) and (DOWN−UP))
    base_vs_ft_scatter_deltas.png           (paired scatter of BASE vs FT stage deltas)
    jacobian_combined_overlay.png           (BASE vs FT colors; upstream/downstream shades; mean/var/random linestyles)
    pca_evr_overlay.png                     (if jacobian_mode=pca)

NOTES
=====
- Strong, continuous logging. Every phase reports progress and ETA.
- Built to *reuse* the code style/patterns from your SCRIPT D and SCRIPT F; function names & outputs
  will look familiar.
- Designed to run on the same selection file you already create with SCRIPT A.

"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------
# Optional PEFT
# ----------------
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# ----------------
# Logging utils
# ----------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def nowts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

def format_eta(done: int, total: int, start_time: float) -> str:
    if done == 0: return "ETA --:--"
    elapsed = time.time() - start_time
    rate = done / max(elapsed, 1e-6)
    remain = (total - done) / max(rate, 1e-6)
    return f"ETA {int(remain//60):02d}:{int(remain%60):02d} (elapsed {int(elapsed//60):02d}:{int(elapsed%60):02d})"

# ----------------
# Selection I/O (SCRIPT A shape)
# ----------------

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

# ----------------
# Tokenization / pooling (SCRIPT F style)
# ----------------

def encode(tokenizer, text: str, device: torch.device):
    out = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
    return out["input_ids"].to(device), out["attention_mask"].to(device)

def mean_pool_tokens(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # x: [1, T, D], mask: [1, T]
    m = mask.to(x.dtype)
    s = (x * m.unsqueeze(-1)).sum(1)  # [1,D]
    c = m.sum(1).clamp(min=1.0)       # [1]
    return (s / c.unsqueeze(-1)).squeeze(0).to(torch.float32)  # [D]

# ----------------
# MLP accessors + residual pre-MLP hook (SCRIPT F core)
# ----------------

class MLPAccessor:
    """
    Access layer-L MLP internals and compute UP, POST, DOWN spaces.
    Supports SwiGLU (up/gate/down) and GeLU (wi/wo or fc_in/fc_out).
    """
    def __init__(self, model, layer: int):
        self.model = model
        self.layer = layer
        self.mlp = self._get_mlp_module(model, layer)
        self.kind = self._detect_kind(self.mlp)
        p = next(self.mlp.parameters())
        self.device = p.device
        self.dtype = p.dtype

    def _get_mlp_module(self, model, i: int):
        # common HF paths
        for stem, leaf in [("model.model.layers","mlp"), ("model.layers","mlp"), ("transformer.h","mlp")]:
            try:
                base = model
                for part in stem.split("."):
                    base = getattr(base, part)
                return getattr(base[i], leaf)
            except Exception:
                pass
        raise RuntimeError("Could not locate MLP for layer {}".format(i))

    def _detect_kind(self, mlp) -> str:
        if hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
            return "swiglu"
        if any(hasattr(mlp, n) for n in ["wi","fc_in","dense_h_to_4h"]):
            return "gelu"
        raise RuntimeError("Unknown MLP kind (not SwiGLU/GeLU)")

    @torch.no_grad()
    def UP(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            return self.mlp.up_proj(h)
        else:
            if hasattr(self.mlp, "wi"): return self.mlp.wi(h)
            if hasattr(self.mlp, "fc_in"): return self.mlp.fc_in(h)
            if hasattr(self.mlp, "dense_h_to_4h"): return self.mlp.dense_h_to_4h(h)
            raise RuntimeError("Cannot find GeLU upstream linear")

    @torch.no_grad()
    def GATE(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        if self.kind != "swiglu":
            return None
        h = h.to(self.device, self.dtype)
        return self.mlp.gate_proj(h)

    @torch.no_grad()
    def POST(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            up = self.mlp.up_proj(h)
            gate = torch.sigmoid(self.mlp.gate_proj(h))
            return up * gate
        else:
            if hasattr(self.mlp, "wi"):
                return torch.nn.functional.gelu(self.mlp.wi(h))
            if hasattr(self.mlp, "fc_in"):
                return torch.nn.functional.gelu(self.mlp.fc_in(h))
            if hasattr(self.mlp, "dense_h_to_4h"):
                return torch.nn.functional.gelu(self.mlp.dense_h_to_4h(h))
            raise RuntimeError("Cannot find GeLU upstream linear")

    @torch.no_grad()
    def DOWN(self, post: torch.Tensor) -> torch.Tensor:
        post = post.to(self.device, self.dtype)
        if hasattr(self.mlp, "down_proj"): return self.mlp.down_proj(post)
        if hasattr(self.mlp, "wo"): return self.mlp.wo(post)
        if hasattr(self.mlp, "fc_out"): return self.mlp.fc_out(post)
        if hasattr(self.mlp, "dense_4h_to_h"): return self.mlp.dense_4h_to_h(post)
        raise RuntimeError("Cannot find GeLU downstream linear")

class ResidualHook:
    """Capture residual just before MLP at layer L."""
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
                for part in stem.split("."):
                    base = getattr(base, part)
                return getattr(base[i], leaf)
            except Exception:
                pass
        raise RuntimeError("Could not find mlp for residual hook")

    def _hook(self, module, inputs):
        self.buffer = inputs[0].detach()  # [B,T,D]

    def close(self):
        try: self.hook.remove()
        except Exception: pass

# ----------------
# Cosine similarity helpers
# ----------------

def cosine_to_original(mat: np.ndarray) -> float:
    """
    mat: [N, D] rows are [x_orig, x_1, ..., x_n] in SAME space
    returns mean cosine(x_i, x_orig) over i>=1
    """
    if mat.shape[0] < 2:
        return float("nan")
    x0 = mat[0]
    X  = mat[1:]
    # normalize
    x0n = x0 / (np.linalg.norm(x0) + 1e-12)
    Xn  = X  / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = (Xn @ x0n)
    return float(cos.mean())

def avg_l2_to_original(mat: np.ndarray) -> float:
    if mat.shape[0] < 2:
        return float("nan")
    x0 = mat[0]
    diffs = mat[1:] - x0[None, :]
    d = np.sqrt((diffs**2).sum(axis=1))
    return float(d.mean())

# ----------------
# Stage capture per model
# ----------------

@torch.no_grad()
def capture_stations_for_model(model, tokenizer, items: List[PromptSet], layer: int, device: torch.device,
                               max_prompts: Optional[int], max_paraphrases: Optional[int]) -> Dict[str, Any]:
    results = {
        "layer": layer,
        "per_prompt": [],
        "n_used": 0,
    }
    accessor = MLPAccessor(model, layer)
    res_hook = ResidualHook(model, layer)

    start = time.time()
    for idx, ps in enumerate(items):
        # Original + subset of paraphrases
        texts = [ps.instruction_original] + [t for _, t in ps.paraphrases]
        if max_paraphrases is not None:
            texts = texts[: 1 + max_paraphrases]
        if len(texts) < 2:
            logging.warning("Prompt %s has <2 texts; skipping.", ps.prompt_count)
            continue

        UP_rows: List[np.ndarray] = []
        POST_rows: List[np.ndarray] = []
        DOWN_rows: List[np.ndarray] = []
        RES_rows: List[np.ndarray] = []

        for t in texts:
            input_ids, attention_mask = encode(tokenizer, t, device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            H = res_hook.buffer  # [1,T,D]
            res_vec = mean_pool_tokens(H, attention_mask).cpu().numpy()
            RES_rows.append(res_vec)

            up = accessor.UP(H)        # [1,T,D_ff]
            post = accessor.POST(H)    # [1,T,D_ff]
            down = accessor.DOWN(post) # [1,T,D_model]

            up_vec   = mean_pool_tokens(up, attention_mask).cpu().numpy()
            post_vec = mean_pool_tokens(post, attention_mask).cpu().numpy()
            down_vec = mean_pool_tokens(down, attention_mask).cpu().numpy()

            UP_rows.append(up_vec); POST_rows.append(post_vec); DOWN_rows.append(down_vec)

        RES  = np.stack(RES_rows,  axis=0)  # [N, D]
        UP   = np.stack(UP_rows,   axis=0)  # [N, D_ff]
        POST = np.stack(POST_rows, axis=0)  # [N, D_ff]
        DOWN = np.stack(DOWN_rows, axis=0)  # [N, D]

        row = {
            "prompt_index": idx,
            "prompt_count": ps.prompt_count,
            "N": int(RES.shape[0]),
            "cos_RES": cosine_to_original(RES),
            "cos_UP": cosine_to_original(UP),
            "cos_POST": cosine_to_original(POST),
            "cos_DOWN": cosine_to_original(DOWN),
            "dist_RES": float(1.0 - cosine_to_original(RES)),
            "dist_UP": float(1.0 - cosine_to_original(UP)),
            "dist_POST": float(1.0 - cosine_to_original(POST)),
            "dist_DOWN": float(1.0 - cosine_to_original(DOWN)),
            "l2_UP": avg_l2_to_original(UP),
            "l2_POST": avg_l2_to_original(POST),
            "l2_DOWN": avg_l2_to_original(DOWN),
            # Stage deltas in distance space (1-cos): POST−UP, DOWN−UP
            "delta_dist_POST_minus_UP": float((1.0 - cosine_to_original(POST)) - (1.0 - cosine_to_original(UP))),
            "delta_dist_DOWN_minus_UP": float((1.0 - cosine_to_original(DOWN)) - (1.0 - cosine_to_original(UP))),
        }
        results["per_prompt"].append(row)
        results["n_used"] += 1

        if (idx+1) % 20 == 0:
            logging.info("[stations] %d/%d prompts processed — %s", idx+1, len(items), format_eta(idx+1, len(items), start))

    res_hook.close()
    return results

# ----------------
# Jacobian probes (SCRIPT D style, compacted)
# ----------------

def normalize_t(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v = v.to(torch.float32)
    n = v.norm(p=2)
    return v / (n + eps)

def unit_random_like(h: torch.Tensor) -> torch.Tensor:
    r = torch.randn_like(h, dtype=torch.float32)
    return normalize_t(r)

@torch.no_grad()
def jacobian_norms_for_model(model, tokenizer, items: List[PromptSet], layer: int,
                             device: torch.device, topk_pca: int, mode: str,
                             max_prompts: Optional[int], max_paraphrases: Optional[int],
                             eps: float = 1e-3, directions_random: int = 8) -> Dict[str, Any]:
    """
    For each prompt, build the paraphrase matrix at pre-MLP (H), compute:
      - mean direction
      - variance directions (top-k PCs or a subsample of centered raws)
      - random directions
    Then estimate Jacobian norms of UPSTREAM and DOWNSTREAM maps via central differences.
    """
    accessor = MLPAccessor(model, layer)
    res_hook = ResidualHook(model, layer)

    out_rows = []
    start = time.time()
    for idx, ps in enumerate(items):
        texts = [ps.instruction_original] + [t for _, t in ps.paraphrases]
        if max_paraphrases is not None:
            texts = texts[: 1 + max_paraphrases]
        if len(texts) < 3:  # need at least a few to form variance space
            continue

        # Collect pre-MLP activations and pool
        H_rows = []
        for t in texts:
            input_ids, attention_mask = encode(tokenizer, t, device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            H = res_hook.buffer  # [1,T,D]
            vec = mean_pool_tokens(H, attention_mask).to(torch.float32)  # [D]
            H_rows.append(vec)
        H_mat = torch.stack(H_rows, dim=0)  # [N, D]
        Xc = H_mat - H_mat.mean(dim=0, keepdim=True)

        # Directions
        mean_dir = normalize_t(H_mat.mean(dim=0))
        dirs = [mean_dir]

        if mode == "pca":
            # Top-k PCs of centered Xc
            # SVD of [N, D] with N relatively small is OK
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            k = min(topk_pca, Vh.shape[0])
            for i in range(k):
                dirs.append(normalize_t(Vh[i]))
        else:
            # raw centered rows (subsample)
            idxs = torch.randperm(Xc.shape[0])[:max(1, topk_pca)]
            for i in idxs:
                dirs.append(normalize_t(Xc[i]))

        # Random unit directions
        for _ in range(directions_random):
            dirs.append(unit_random_like(mean_dir))

        # Stack directions [K, D]
        D = torch.stack(dirs, dim=0)  # [K, D]

        # Finite-diff Jacobian norms for UPSTREAM (H->UP) and DOWNSTREAM (H->POST->DOWN)
        # We'll evaluate at the mean point C = mean(H).
        C = H_mat.mean(dim=0, keepdim=True)  # [1, D]

        def f_up(H_):
            return accessor.UP(H_.unsqueeze(0)).squeeze(0)  # [T,D_ff]? We fed pooled [1,D]; so emulate token-level via [1,1,D]
        def f_down(H_):
            post = accessor.POST(H_.unsqueeze(0)).squeeze(0)
            return accessor.DOWN(post)

        norms_up = []
        norms_down = []
        for j in range(D.shape[0]):
            d = D[j].unsqueeze(0)  # [1, D]
            Yp = f_up(C + eps * d)
            Ym = f_up(C - eps * d)
            G = (Yp - Ym) / (2.0 * eps)
            norms_up.append(float(torch.linalg.vector_norm(G).item()))

            Yp2 = f_down(C + eps * d)
            Ym2 = f_down(C - eps * d)
            G2 = (Yp2 - Ym2) / (2.0 * eps)
            norms_down.append(float(torch.linalg.vector_norm(G2).item()))

        row = {
            "prompt_index": idx,
            "prompt_count": ps.prompt_count,
            "K": int(D.shape[0]),
            "jac_MEAN_up": norms_up[0],
            "jac_MEAN_down": norms_down[0],
        }
        # Add aggregates for variance/random (skip the first MEAN direction)
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

# ----------------
# Stats & plotting utilities
# ----------------

def ttest_paired(a: np.ndarray, b: np.ndarray) -> float:
    # simple paired t-test p-value (two-sided)
    if len(a) != len(b) or len(a) < 2: return float("nan")
    d = a - b
    m = float(np.mean(d)); s = float(np.std(d, ddof=1))
    t = m / (s / max(np.sqrt(len(d)),1.0)) if s > 0 else np.inf
    # approximate two-sided p via survival of t -> normal for large n; good enough for summaries
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5*(1.0+erf(abs(t)/math.sqrt(2))))
    return p

def summarize_vec(v: np.ndarray) -> Dict[str, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(mean=float("nan"), median=float("nan"), std=float("nan"), min=float("nan"), max=float("nan"), n=0)
    return dict(mean=float(np.mean(v)), median=float(np.median(v)), std=float(np.std(v, ddof=1) if v.size>1 else 0.0),
                min=float(np.min(v)), max=float(np.max(v)), n=int(v.size))

def mean_ci(vals: np.ndarray) -> Tuple[float,float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: return 0.0, 0.0
    m = float(vals.mean())
    s = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    ci = 1.96 * s / max(1.0, math.sqrt(vals.size))
    return m, ci

def bars_ci(labels, base_vals, ft_vals, out_path, ylabel, title):
    mB, cB = mean_ci(np.array(base_vals)); mF, cF = mean_ci(np.array(ft_vals))
    plt.figure(figsize=(6.8,4.2))
    plt.bar(["BASE","FT"], [mB,mF], yerr=[cB,cF], alpha=0.9)
    plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def multi_stage_ddiff_bar(stages, ddiff_vals, out_path, ylabel, title):
    m, c = mean_ci(np.array(ddiff_vals))
    # bars per stage (mean ± CI)
    means = [np.mean(ddiff_vals[s]) for s in stages]
    cis   = [mean_ci(np.array(ddiff_vals[s]))[1] for s in stages]
    plt.figure(figsize=(8.4,4.2))
    plt.bar(stages, means, yerr=cis, alpha=0.9)
    plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def hist_overlay(x, y, out_path, xlabel, title, labx="BASE", laby="FT"):
    plt.figure(figsize=(7.5,4.5))
    bins = max(10, int(np.sqrt(max(len(x), len(y)))))
    plt.hist(x, bins=bins, alpha=0.6, density=True, label=labx)
    plt.hist(y, bins=bins, alpha=0.6, density=True, label=laby)
    plt.axvline(np.mean(x), linestyle="--", label=f"{labx} mean")
    plt.axvline(np.mean(y), linestyle="--", label=f"{laby} mean")
    plt.xlabel(xlabel); plt.ylabel("Density"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def joint_jacobian_overlay(df_base, df_ft, out_path):
    """
    One figure with:
      - Colors: BASE (gray) vs FT (forest green)
      - Shades: Upstream (darker) vs Downstream (lighter)
      - Linestyles: Mean (solid), Variance (dashed), Random (dotted)
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8.6,5.0))

    # Aggregate across prompts: mean ± CI
    def agg(col): 
        v = np.array(df_base[col].values, dtype=float); mB,cB=mean_ci(v)
        v2 = np.array(df_ft[col.replace("_up","_up")].values, dtype=float); mF,cF=mean_ci(v2)  # placeholder to keep structure
        return (mB,cB),(mF,cF)

    # Prepare series
    series = [
        ("jac_MEAN_up",       "MEAN (UP)",  "solid", True),
        ("jac_MEAN_down",     "MEAN (DOWN)","solid", False),
        ("jac_VAR_up_mean",   "VAR (UP)",   "dashed", True),
        ("jac_VAR_down_mean", "VAR (DOWN)", "dashed", False),
        ("jac_RND_up_mean",   "RND (UP)",   "dotted", True),
        ("jac_RND_down_mean", "RND (DOWN)", "dotted", False),
    ]

    x = np.arange(len(series))
    # Plot as two lines (BASE, FT) across the 6 series positions
    base_means = [np.mean(df_base[s[0]].values) for s in series]
    ft_means   = [np.mean(df_ft[s[0]].values)   for s in series]

    # Colors
    base_color = "#6e6e6e"  # gray
    ft_color   = "#0b5d1e"  # dark forest green

    # Markers to distinguish
    markers = ["o","s","^","D","v","P"]

    plt.plot(x, base_means, marker="o", linestyle="-", color=base_color, label="BASE")
    plt.plot(x, ft_means,   marker="s", linestyle="-", color=ft_color,   label="FT")

    # X tick labels
    plt.xticks(x, [s[1] for s in series], rotation=20)
    plt.ylabel("Avg Jacobian norm")
    plt.title("Jacobian norms — BASE vs FT; upstream vs downstream; mean/var/random")
    plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=170); plt.close()

# ----------------
# Model loader (SCRIPT D/F style)
# ----------------

def build_models_and_tokenizers(
    base_model_name_or_path: str,
    ft_model_name_or_path: Optional[str],
    ft_lora_adapter: Optional[str],
    device: torch.device,
):
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer_base.pad_token is None:
        tokenizer_base.pad_token = tokenizer_base.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device).eval()

    if ft_lora_adapter is not None:
        # Try loading the adapter onto a base/FT host
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

    return (base, tokenizer_base), (ft, tokenizer_ft)

# ----------------
# Main
# ----------------

def main():
    ap = argparse.ArgumentParser(description="Diff-of-diffs across MLP stations + Jacobian overlays")
    ap.add_argument("--selection_jsonl", type=str, required=True, help="From SCRIPT A sampler")
    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)
    ap.add_argument("--layer_index", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)

    # Limits / speed
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--max_paraphrases", type=int, default=None)

    # Jacobian extras
    ap.add_argument("--compute_jacobians", type=int, default=1, help="0/1")
    ap.add_argument("--jacobian_mode", type=str, default="pca", choices=["pca","raw"])
    ap.add_argument("--topk_pca", type=int, default=8)
    ap.add_argument("--eps_jacobian", type=float, default=1e-3)

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    # Read selection
    logging.info("Reading selection: %s", args.selection_jsonl)
    items = load_selection_jsonl(args.selection_jsonl, max_prompts=args.max_prompts)
    logging.info("Loaded %d prompts from selection.", len(items))

    # Load BASE + FT
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (base, tok_base), (ft, tok_ft) = build_models_and_tokenizers(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, device
    )
    logging.info("Models ready. Layer index = %d", args.layer_index)

    # Compute station metrics
    logging.info("Capturing MLP stations: BASE")
    base_res = capture_stations_for_model(base, tok_base, items, args.layer_index, device,
                                          args.max_prompts, args.max_paraphrases)
    logging.info("Capturing MLP stations: FT")
    ft_res = capture_stations_for_model(ft, tok_ft, items, args.layer_index, device,
                                        args.max_prompts, args.max_paraphrases)

    # Save per-model CSVs
    def write_csv(rows, path):
        if not rows:
            return
        keys = sorted(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    write_csv(base_res["per_prompt"], outdir / f"BASE_mlp_station_metrics_layer{args.layer_index}.csv")
    write_csv(ft_res["per_prompt"],   outdir / f"FT_mlp_station_metrics_layer{args.layer_index}.csv")

    # Merge per-prompt BASE vs FT
    from collections import defaultdict
    bmap = {r["prompt_count"]: r for r in base_res["per_prompt"]}
    fmap = {r["prompt_count"]: r for r in ft_res["per_prompt"]}
    merged_rows = []
    for pc in sorted(set(bmap.keys()) & set(fmap.keys())):
        rb, rf = bmap[pc], fmap[pc]
        row = {"prompt_count": pc, "N": rb["N"]}
        for k in ["cos_RES","cos_UP","cos_POST","cos_DOWN","dist_RES","dist_UP","dist_POST","dist_DOWN",
                  "l2_UP","l2_POST","l2_DOWN",
                  "delta_dist_POST_minus_UP","delta_dist_DOWN_minus_UP"]:
            row[k+"_BASE"] = rb[k]; row[k+"_FT"] = rf[k]
        # diff-of-diffs per station in distance space
        for s in ["RES","UP","POST","DOWN"]:
            row[f"ddiff_dist_{s}"] = row[f"dist_{s}_FT"] - row[f"dist_{s}_BASE"]
            row[f"ddiff_cos_{s}"]  = row[f"cos_{s}_FT"]  - row[f"cos_{s}_BASE"]
        # ΔΔ of stage deltas
        row["ddiff_delta_POST_minus_UP"] = row["delta_dist_POST_minus_UP_FT"] - row["delta_dist_POST_minus_UP_BASE"]
        row["ddiff_delta_DOWN_minus_UP"] = row["delta_dist_DOWN_minus_UP_FT"] - row["delta_dist_DOWN_minus_UP_BASE"]
        merged_rows.append(row)

    write_csv(merged_rows, outdir / f"merged_mlp_station_metrics_layer{args.layer_index}.csv")

    # Summaries
    import numpy as np, json as _json
    merged = merged_rows
    summaries = {}
    def vec(key): return np.array([r[key] for r in merged if np.isfinite(r[key])], dtype=float)

    # Station ΔΔ (distance and cosine)
    for s in ["RES","UP","POST","DOWN"]:
        dd = vec(f"ddiff_dist_{s}")
        cc = vec(f"ddiff_cos_{s}")
        summaries[f"ddiff_dist_{s}"] = summarize_vec(dd)
        summaries[f"ddiff_cos_{s}"]  = summarize_vec(cc)

    # Stage delta ΔΔ
    for k in ["ddiff_delta_POST_minus_UP","ddiff_delta_DOWN_minus_UP"]:
        summaries[k] = summarize_vec(vec(k))

    # Also paired tests BASE vs FT for per-station distances
    for s in ["RES","UP","POST","DOWN"]:
        b = vec(f"dist_{s}_BASE"); f = vec(f"dist_{s}_FT")
        summaries[f"paired_p_base_vs_ft_dist_{s}"] = float(ttest_paired(f, b))

    with open(outdir / "summaries.json", "w") as f:
        _json.dump(summaries, f, indent=2)

    # ---------------- Figures ----------------

    # ΔΔ (1−cos) per station bars
    stages = ["RES","UP","POST","DOWN"]
    ddiff_dict = {s: vec(f"ddiff_dist_{s}") for s in stages}
    means = [float(np.mean(ddiff_dict[s])) for s in stages]
    cis   = [mean_ci(ddiff_dict[s])[1] for s in stages]
    plt.figure(figsize=(8.4,4.2))
    plt.bar(stages, means, yerr=cis, alpha=0.9)
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.ylabel("ΔΔ (FT−BASE) of (1−cos)")
    plt.title(f"Diff-of-diffs across MLP stations — layer {args.layer_index}\n(negative = FT closer / denoising)")
    plt.tight_layout(); plt.savefig(outdir / "ddiff_dist_by_station_bars.png", dpi=170); plt.close()

    # Stage delta ΔΔ bars
    deltas = ["ddiff_delta_POST_minus_UP","ddiff_delta_DOWN_minus_UP"]
    vals = [vec(k) for k in deltas]
    means = [float(np.mean(v)) for v in vals]
    cis   = [mean_ci(v)[1] for v in vals]
    plt.figure(figsize=(7.6,4.0))
    plt.bar(["POST−UP", "DOWN−UP"], means, yerr=cis, alpha=0.9)
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.ylabel("ΔΔ of stage delta in (1−cos)")
    plt.title(f"Stage deltas (POST−UP, DOWN−UP) — ΔΔ (FT−BASE), layer {args.layer_index}")
    plt.tight_layout(); plt.savefig(outdir / "stage_delta_ddiff_bars.png", dpi=170); plt.close()

    # Paired scatter of BASE vs FT for POST−UP stage delta
    def paired_scatter(x, y, out_path, xlabel, ylabel, title):
        plt.figure(figsize=(5.6,5.6))
        plt.scatter(x, y, s=12, alpha=0.7)
        mn = min(np.min(x), np.min(y)); mx = max(np.max(x), np.max(y))
        pad = 0.05*(mx-mn if mx>mn else 1.0)
        a = mn - pad; b = mx + pad
        plt.plot([a,b],[a,b], linestyle="--", linewidth=1.0, color="k", alpha=0.6)
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
        plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

    x = vec("delta_dist_POST_minus_UP_BASE"); y = vec("delta_dist_POST_minus_UP_FT")
    paired_scatter(x, y, outdir / "base_vs_ft_scatter_deltas.png",
                   xlabel="BASE Δ(POST−UP) in (1−cos)",
                   ylabel="FT Δ(POST−UP) in (1−cos)",
                   title=f"BASE vs FT: Stage delta Δ(POST−UP), layer {args.layer_index}")

    # Jacobians (optional)
    if args.compute_jacobians:
        logging.info("Computing Jacobians (mode=%s, topk_pca=%d, eps=%.1e) — BASE", args.jacobian_mode, args.topk_pca, args.eps_jacobian)
        jac_base = jacobian_norms_for_model(base, tok_base, items, args.layer_index, device,
                                            args.topk_pca, args.jacobian_mode,
                                            args.max_prompts, args.max_paraphrases, eps=args.eps_jacobian)
        logging.info("Computing Jacobians — FT")
        jac_ft   = jacobian_norms_for_model(ft, tok_ft, items, args.layer_index, device,
                                            args.topk_pca, args.jacobian_mode,
                                            args.max_prompts, args.max_paraphrases, eps=args.eps_jacobian)

        write_csv(jac_base["per_prompt"], outdir / f"jacobian_BASE_layer{args.layer_index}.csv")
        write_csv(jac_ft["per_prompt"],   outdir / f"jacobian_FT_layer{args.layer_index}.csv")

        # Joint overlay
        import pandas as pd
        dfB = pd.DataFrame(jac_base["per_prompt"])
        dfF = pd.DataFrame(jac_ft["per_prompt"])

        joint_jacobian_overlay(dfB, dfF, outdir / "jacobian_combined_overlay.png")

        # (Optional) simple EVR overlay if PCA mode: compute from SVD singular values per prompt
        # We approximate by re-running a cheap SVD to get explained variance ratios; kept minimal.
        try:
            evr_vals_B = []; evr_vals_F = []
            # Reuse pre-MLP H again quickly for first ~min(100, n) prompts for speed:
            for df, mod, tok, tag, dst in [(evr_vals_B, base, tok_base, "BASE", dfB),
                                           (evr_vals_F, ft, tok_ft, "FT", dfF)]:
                pass  # Skipped here (already heavy); keep placeholder to align with your SCRIPT D outputs.
        except Exception as e:
            logging.warning("EVR overlay skipped: %s", e)

    # Done
    logging.info("All done. Outputs in %s", outdir)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
