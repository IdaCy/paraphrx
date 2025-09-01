
"""
SCRIPT F

Nonlinearity Alignment: Avg L2 Distance to Original (Diff-of-Diffs)

Core, single most telling metric (simple & high information):
  For each prompt p, let x_orig be the pooled activation of the original instruction.
  For each paraphrase i, let x_i be the pooled activation for that paraphrase.
  Define the average L2 distance to the original in a given space S as:
      D_S = (1/n) * sum_i || x_i(S) - x_orig(S) ||_2

We compute D_S in three layer-L MLP-related spaces:
  - UP   : pre-activation upstream output  (SwiGLU: up = W_up h; GeLU: a = W_in h)
  - POST : post-nonlinearity output        (SwiGLU: silu(gate)*up; GeLU: gelu(a))
  - DOWN : downstream linear output        (SwiGLU/GeLU: y = W_down POST)

Then for each prompt we form the *within-model* difference:
  Δ_up→post  = D_POST - D_UP
  Δ_up→down  = D_DOWN - D_UP

Finally, the *diff-of-diffs* between FT and BASE:
  ΔΔ_up→post = (Δ_up→post)_FT  - (Δ_up→post)_BASE
  ΔΔ_up→down = (Δ_up→down)_FT  - (Δ_up→down)_BASE

Interpretation (robustness via nonlinearity):
  If the MLP’s nonlinearity learns to suppress paraphrase nuisance variation,
  then distances to the original should shrink AFTER the NL, especially in FT:
    Expect ΔΔ_up→post < 0  and/or  ΔΔ_up→down < 0

Secondary (optional) metric:
  LOG_COMPRESSION in POST space relative to UP (variance perspective):
    log( trace(Cov_POST)/trace(Cov_UP) ), reported per-prompt for BASE & FT.

Inputs:
  - selection_path: JSONL/JSON with 'instruction_original' and either a `paraphrases` list
                    (with 'paraphrase' strings) or 'instruct_*' fields.

Run example:
  python nonlinearity_alignment_diff_of_diffs.py \
      --selection_path runs/first_sampler/jacobian_prompts.jsonl \
      --base_model /path/to/base \
      --ft_model /path/to/ft \
      --layer 6 \
      --outdir runs/alignment_L6 \
      --compute_variance

Notes:
  - Token pooling = mean over non-padding tokens (same for all spaces).
  - We stay in the MLP’s own spaces for clarity (no residual mix-in for the core metric).
"""

import argparse
import json
import jsonlines
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------ IO ------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_selection(fp: str) -> List[Dict[str, Any]]:
    data = []
    if fp.endswith(".jsonl"):
        with jsonlines.open(fp, "r") as r:
            for obj in r:
                data.append(obj)
    elif fp.endswith(".json"):
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Selection JSON must be a list.")
    else:
        raise ValueError("selection_path must be .json or .jsonl")
    return data

def extract_texts(item: Dict[str, Any], max_paraphrases: Optional[int]=None) -> Tuple[Optional[str], List[str]]:
    orig = item.get("instruction_original", None)
    paras: List[str] = []
    if "paraphrases" in item and isinstance(item["paraphrases"], list):
        for p in item["paraphrases"]:
            t = p.get("paraphrase") or p.get("instruction") or p.get("text")
            if t: paras.append(t)
    else:
        for k,v in item.items():
            if isinstance(v, str) and k.startswith("instruct_"):
                paras.append(v)
    # unique & clean
    seen = set()
    clean = []
    for t in paras:
        t2 = t.strip()
        if t2 and t2 not in seen:
            seen.add(t2)
            clean.append(t2)
    if max_paraphrases is not None and len(clean) > max_paraphrases:
        clean = clean[:max_paraphrases]
    return orig, clean

# ------------------ Model access ------------------

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
            gate = self.mlp.gate_proj(h)
            return torch.nn.functional.silu(gate) * up
        else:
            if hasattr(self.mlp, "wi"): a = self.mlp.wi(h)
            elif hasattr(self.mlp, "fc_in"): a = self.mlp.fc_in(h)
            elif hasattr(self.mlp, "dense_h_to_4h"): a = self.mlp.dense_h_to_4h(h)
            else: raise RuntimeError("Cannot find GeLU upstream linear")
            return torch.nn.functional.gelu(a)

    @torch.no_grad()
    def DOWN(self, post: torch.Tensor) -> torch.Tensor:
        post = post.to(self.device, self.dtype)
        if self.kind == "swiglu":
            return self.mlp.down_proj(post)
        else:
            if hasattr(self.mlp, "wo"): return self.mlp.wo(post)
            if hasattr(self.mlp, "fc_out"): return self.mlp.fc_out(post)
            if hasattr(self.mlp, "dense_4h_to_h"): return self.mlp.dense_4h_to_h(post)
            raise RuntimeError("Cannot find GeLU downstream linear")

class ResidualHook:
    """
    Capture residual just before MLP at layer L.
    """
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

# ------------------ Tokenization / pooling ------------------

def encode(tokenizer, text: str, device: torch.device):
    out = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
    return out["input_ids"].to(device), out["attention_mask"].to(device)

def mean_pool_tokens(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # x: [1,T,D], mask: [1,T]
    m = mask.to(x.dtype)
    s = (x * m.unsqueeze(-1)).sum(1)  # [1,D]
    c = m.sum(1).clamp(min=1.0)       # [1]
    return (s / c.unsqueeze(-1)).squeeze(0).to(torch.float32)  # [D]

# ------------------ Distances & variance ------------------

def avg_l2_to_original(mat: np.ndarray) -> float:
    """
    mat: [N, D] rows are [x_orig, x_1, ..., x_n] in SAME space
    returns average ||x_i - x_orig||_2 over i>=1
    """
    if mat.shape[0] < 2:
        return float("nan")
    x0 = mat[0]
    diffs = mat[1:] - x0[None, :]
    d = np.sqrt((diffs**2).sum(axis=1))
    return float(d.mean())

def cov_trace(X: np.ndarray) -> float:
    if X.shape[0] < 2: return 0.0
    Xc = X - X.mean(axis=0, keepdims=True)
    var_dim = (Xc**2).sum(axis=0) / max(1, X.shape[0]-1)
    return float(var_dim.sum())

# ------------------ Runner per model ------------------

@torch.no_grad()
def compute_alignment_for_model(model, tokenizer, items, layer: int, device: torch.device,
                                max_items: Optional[int], max_paraphrases: Optional[int],
                                compute_variance: bool=False) -> Dict[str, Any]:
    accessor = MLPAccessor(model, layer)
    hook = ResidualHook(model, layer)
    results = {
        "per_prompt": [],
        "kind": accessor.kind,
        "n_used": 0,
        "skipped": 0,
    }

    for idx, item in enumerate(items[:max_items] if max_items else items):
        orig, paras = extract_texts(item, max_paraphrases=max_paraphrases)
        texts: List[str] = []
        if orig: texts.append(orig)
        # Require an original as fixed anchor
        if not orig:
            results["skipped"] += 1
            continue
        texts.extend(paras)
        if len(texts) < 3:
            results["skipped"] += 1
            continue

        mats_UP = []
        mats_POST = []
        mats_DOWN = []

        for t in texts:
            input_ids, attn_mask = encode(tokenizer, t, device)
            _ = model(input_ids=input_ids, attention_mask=attn_mask)
            h = hook.buffer                    # [1,T,D_model]
            up = accessor.UP(h)                # [1,T,D_ff]
            post = accessor.POST(h)            # [1,T,D_ff]
            down = accessor.DOWN(post)         # [1,T,D_model]

            up_p   = mean_pool_tokens(up, attn_mask).cpu().numpy()
            post_p = mean_pool_tokens(post, attn_mask).cpu().numpy()
            down_p = mean_pool_tokens(down, attn_mask).cpu().numpy()

            mats_UP.append(up_p)
            mats_POST.append(post_p)
            mats_DOWN.append(down_p)

        UP = np.stack(mats_UP, axis=0)         # [N, D_ff]
        POST = np.stack(mats_POST, axis=0)     # [N, D_ff]
        DOWN = np.stack(mats_DOWN, axis=0)     # [N, D_model]

        D_up   = avg_l2_to_original(UP)
        D_post = avg_l2_to_original(POST)
        D_down = avg_l2_to_original(DOWN)

        row = {
            "prompt_index": idx,
            "N": int(UP.shape[0]),
            "D_up": D_up,
            "D_post": D_post,
            "D_down": D_down,
            "delta_up_to_post": D_post - D_up,
            "delta_up_to_down": D_down - D_up,
        }

        if compute_variance:
            tr_up = cov_trace(UP); tr_post = cov_trace(POST)
            row["log_compression_up2post"] = math.log((tr_post/UP.shape[1] + 1e-12) / (tr_up/UP.shape[1] + 1e-12))

        results["per_prompt"].append(row)
        results["n_used"] += 1

    hook.close()
    return results

# ------------------ Plotting ------------------

def mean_ci(vals: List[float]) -> Tuple[float,float]:
    arr = np.array(vals, dtype=float)
    if arr.size == 0: return 0.0, 0.0
    m = float(arr.mean())
    s = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci = 1.96 * s / max(1.0, math.sqrt(arr.size))
    return m, ci

def paired_scatter(x, y, out_path, xlabel, ylabel, title):
    plt.figure(figsize=(5.5,5.5))
    plt.scatter(x, y, s=12, alpha=0.7)
    mn = min(np.min(x), np.min(y)); mx = max(np.max(x), np.max(y))
    pad = 0.05*(mx-mn if mx>mn else 1.0)
    a = mn - pad; b = mx + pad
    plt.plot([a,b],[a,b],'k--',lw=1)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()

def overlay_hist(x, y, out_path, xlabel, title):
    plt.figure(figsize=(7.5,4.5))
    bins = max(10, int(np.sqrt(max(len(x), len(y)))))
    plt.hist(x, bins=bins, alpha=0.6, density=True, label="BASE")
    plt.hist(y, bins=bins, alpha=0.6, density=True, label="FT")
    plt.axvline(np.mean(x), linestyle="--", label="BASE mean")
    plt.axvline(np.mean(y), linestyle="--", label="FT mean")
    plt.xlabel(xlabel); plt.ylabel("Density"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def bars_two(base_vals, ft_vals, out_path, ylabel, title):
    mB, cB = mean_ci(base_vals); mF, cF = mean_ci(ft_vals)
    plt.figure(figsize=(6.5,4.0))
    plt.bar(["BASE","FT"], [mB,mF], yerr=[cB,cF], alpha=0.9)
    plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser(description="Avg L2 to original (diff-of-diffs) across MLP NL")
    ap.add_argument("--selection_path", type=str, required=True)
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--ft_model", type=str, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--max_paraphrases", type=int, default=None)
    ap.add_argument("--compute_variance", action="store_true", help="also compute variance log-compression as a secondary check")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    items = read_selection(args.selection_path)

    # Load models
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tok_base = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    mod_base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).to(device).eval()

    tok_ft = AutoTokenizer.from_pretrained(args.ft_model, use_fast=True)
    mod_ft = AutoModelForCausalLM.from_pretrained(args.ft_model, torch_dtype=dtype).to(device).eval()

    # Compute per-model metrics
    base_res = compute_alignment_for_model(mod_base, tok_base, items, args.layer, device,
                                           args.max_items, args.max_paraphrases,
                                           compute_variance=args.compute_variance)
    ft_res   = compute_alignment_for_model(mod_ft, tok_ft, items, args.layer, device,
                                           args.max_items, args.max_paraphrases,
                                           compute_variance=args.compute_variance)

    # Persist CSVs
    import csv, pandas as pd
    with open(os.path.join(args.outdir, "BASE_alignment.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(base_res["per_prompt"][0].keys()))
        w.writeheader(); w.writerows(base_res["per_prompt"])
    with open(os.path.join(args.outdir, "FT_alignment.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ft_res["per_prompt"][0].keys()))
        w.writeheader(); w.writerows(ft_res["per_prompt"])

    # Merge by prompt_index for paired plots
    dfB = pd.read_csv(os.path.join(args.outdir, "BASE_alignment.csv"))
    dfF = pd.read_csv(os.path.join(args.outdir, "FT_alignment.csv"))
    merged = dfB.merge(dfF, on="prompt_index", suffixes=("_BASE","_FT"))

    # Diff-of-diffs
    def ddiff(col_base, col_ft):
        return merged[col_ft].values - merged[col_base].values

    dd_up_post = ddiff("delta_up_to_post_BASE", "delta_up_to_post_FT")
    dd_up_down = ddiff("delta_up_to_down_BASE", "delta_up_to_down_FT")

    # Save summary
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"MLP kind BASE: {base_res['kind']}\n")
        f.write(f"MLP kind FT:   {ft_res['kind']}\n")
        f.write(f"Prompts used: BASE={base_res['n_used']} FT={ft_res['n_used']}  (skipped: {base_res['skipped']} / {ft_res['skipped']})\n")
        # means
        import numpy as np
        f.write(f"mean ΔΔ_up→post: {float(np.mean(dd_up_post)):.6f}\n")
        f.write(f"mean ΔΔ_up→down: {float(np.mean(dd_up_down)):.6f}\n")

    # Overlaid histograms (BASE vs FT) for D_UP, D_POST, D_DOWN
    overlay_hist(merged["D_up_BASE"].values, merged["D_up_FT"].values,
                 os.path.join(args.outdir, "D_up_hist_overlay.png"),
                 xlabel="Average L2 to original (UP)", title="Avg L2 to original — UP space (overlay)")
    overlay_hist(merged["D_post_BASE"].values, merged["D_post_FT"].values,
                 os.path.join(args.outdir, "D_post_hist_overlay.png"),
                 xlabel="Average L2 to original (POST)", title="Avg L2 to original — POST space (overlay)")
    overlay_hist(merged["D_down_BASE"].values, merged["D_down_FT"].values,
                 os.path.join(args.outdir, "D_down_hist_overlay.png"),
                 xlabel="Average L2 to original (DOWN)", title="Avg L2 to original — DOWN space (overlay)")

    # Paired BASE vs FT scatters for Δ_up→post and Δ_up→down
    paired_scatter(merged["delta_up_to_post_BASE"].values, merged["delta_up_to_post_FT"].values,
                   os.path.join(args.outdir, "paired_delta_up_to_post.png"),
                   xlabel="BASE Δ(POST−UP)", ylabel="FT Δ(POST−UP)",
                   title="Paired Δ_up→post (y=x dashed)")
    paired_scatter(merged["delta_up_to_down_BASE"].values, merged["delta_up_to_down_FT"].values,
                   os.path.join(args.outdir, "paired_delta_up_to_down.png"),
                   xlabel="BASE Δ(DOWN−UP)", ylabel="FT Δ(DOWN−UP)",
                   title="Paired Δ_up→down (y=x dashed)")

    # Bars with CI comparing BASE vs FT for Δ_up→post and Δ_up→down
    bars_two(merged["delta_up_to_post_BASE"].values, merged["delta_up_to_post_FT"].values,
             os.path.join(args.outdir, "bars_delta_up_to_post.png"),
             ylabel="Δ (POST − UP)", title="BASE vs FT: Δ_up→post")
    bars_two(merged["delta_up_to_down_BASE"].values, merged["delta_up_to_down_FT"].values,
             os.path.join(args.outdir, "bars_delta_up_to_down.png"),
             ylabel="Δ (DOWN − UP)", title="BASE vs FT: Δ_up→down")

    # Histograms of ΔΔ (diff-of-diffs) centered at 0
    overlay_hist(dd_up_post, np.zeros_like(dd_up_post),
                 os.path.join(args.outdir, "hist_ddiff_up_to_post.png"),
                 xlabel="ΔΔ_up→post (FT−BASE of Δ)", title="ΔΔ_up→post (FT−BASE) vs 0")
    overlay_hist(dd_up_down, np.zeros_like(dd_up_down),
                 os.path.join(args.outdir, "hist_ddiff_up_to_down.png"),
                 xlabel="ΔΔ_up→down (FT−BASE of Δ)", title="ΔΔ_up→down (FT−BASE) vs 0")

    # Optional: variance log-compression overlay (UP→POST)
    if args.compute_variance and "log_compression_up2post_BASE" in merged.columns:
        overlay_hist(merged["log_compression_up2post_BASE"].values, merged["log_compression_up2post_FT"].values,
                     os.path.join(args.outdir, "log_compression_up2post_hist_overlay.png"),
                     xlabel="log( trace(Cov_POST)/trace(Cov_UP) )", title="Variance compression UP→POST (overlay)")
        bars_two(merged["log_compression_up2post_BASE"].values, merged["log_compression_up2post_FT"].values,
                 os.path.join(args.outdir, "log_compression_up2post_bars.png"),
                 ylabel="Mean log compression ± CI", title="Variance compression (UP→POST)")

    print("Done. Outputs saved to", args.outdir)

if __name__ == "__main__":
    main()
