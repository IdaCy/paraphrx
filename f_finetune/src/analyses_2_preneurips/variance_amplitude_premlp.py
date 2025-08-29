#!/usr/bin/env python3
"""
SCRIPT E

variance_amplitude_premlp.py

Goal:
- Quantify **amplitude** of paraphrase variance at the **pre-MLP** input of layer L.
- Compare BASE vs FT on per-prompt paraphrase sets.
- Outputs CSVs and figures; logging mirrors your other scripts.

Metrics per prompt:
- n_paraphrases
- trace(Σ) where Σ = cov(pre-MLP activations) across paraphrases
- top-k eigenvalues of Σ (via SVD of centered V)
- ||mean(h)||_2  (context)

Figures:
- trace_mean_bars_layer{L}.png  (BASE vs FT, mean±95% CI)
- trace_scatter_layer{L}.png     (BASE vs FT, y=x)
- trace_diff_hist_layer{L}.png   (FT−BASE)
- trace_ratio_bar_layer{L}.png   (FT/BASE mean±CI)
- eig_overlay_layer{L}.png       (mean eigenvalue by index, BASE vs FT)
- eig_cum_overlay_layer{L}.png   (cumulative eigenvalue mass vs k, BASE vs FT)
- eig_overlay_log_layer{L}.png   (same as overlay but log-y)

Notes:
- Uses the *FIRST* dataset selection JSONL (same as SCRIPT D).
"""

from __future__ import annotations
import argparse, json, logging, math, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# ---------------- utils ----------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def nowts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

@dataclass
class PromptSet:
    prompt_count: int
    instruction_original: str
    input_text: str
    paraphrases: List[Tuple[str,str]]

def load_selection_jsonl(path: str | Path, max_prompts: Optional[int] = None) -> List[PromptSet]:
    items = []
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
    if max_prompts is not None:
        items = items[:max_prompts]
    return items

def build_prompt_text(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return f"{instruction}\n\nInput: {input_text}"
    return instruction

@dataclass
class Encoded:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_mask: torch.Tensor

def encode_prompt(tokenizer, text: str, device: str, prompt_span: str = "no_bos") -> Encoded:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0]
    attn_mask = enc["attention_mask"][0]
    T = input_ids.shape[0]
    prompt_mask = torch.ones(T, dtype=torch.bool)
    if prompt_span == "no_bos" and T > 0:
        prompt_mask[0] = False
    elif prompt_span == "pre_eos":
        eos_id = tokenizer.eos_token_id
        eos_pos = (input_ids == eos_id).nonzero(as_tuple=True)[0]
        if eos_pos.numel() > 0:
            first_eos = int(eos_pos[0].item())
            prompt_mask[first_eos:] = False
    return Encoded(
        input_ids=input_ids.to(device),
        attention_mask=attn_mask.to(device),
        prompt_mask=prompt_mask.to(device),
    )

class BlockAccessor:
    def __init__(self, model: nn.Module, layer_index: int):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.layers = model.transformer.h
        else:
            raise TypeError("Unsupported model architecture.")
        self.block = self.layers[layer_index]
        self.mlp  = getattr(self.block, "mlp", None) or getattr(self.block, "feed_forward", None)
        if self.mlp is None:
            raise TypeError("Could not access MLP submodule.")

def build_model_and_tokenizer(
    base_model_name_or_path: str,
    ft_model_name_or_path: Optional[str],
    ft_lora_adapter: Optional[str],
    device: str,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device).eval()

    if ft_model_name_or_path is None and ft_lora_adapter is None:
        raise ValueError("Provide --ft_model_name_or_path or --ft_lora_adapter.")
    if ft_lora_adapter is not None:
        apath = Path(ft_lora_adapter)
        if apath.exists() and (apath / "adapter_config.json").exists():
            if not _HAS_PEFT:
                raise RuntimeError("peft not installed but --ft_lora_adapter was provided.")
            ft = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
            ft = PeftModel.from_pretrained(ft, str(apath))
            ft = ft.merge_and_unload().eval()
            return base, ft, tokenizer
        if apath.exists() and (apath / "config.json").exists():
            ft = AutoModelForCausalLM.from_pretrained(str(apath), torch_dtype=dtype).to(device).eval()
            return base, ft, tokenizer
    if ft_model_name_or_path is None:
        raise ValueError("When --ft_lora_adapter is not a local adapter dir, you must provide --ft_model_name_or_path.")
    ft = AutoModelForCausalLM.from_pretrained(ft_model_name_or_path, torch_dtype=dtype).to(device).eval()
    return base, ft, tokenizer

def capture_pre_mlp_activations(model: nn.Module, layer_index: int, enc: Encoded) -> torch.Tensor:
    ba = BlockAccessor(model, layer_index)
    captured = {"x": None}
    def pre_hook(module, inputs):
        x = inputs[0]
        captured["x"] = x.detach().to("cpu")
    h = ba.mlp.register_forward_pre_hook(pre_hook, with_kwargs=False)
    with torch.inference_mode():
        _ = model(input_ids=enc.input_ids.unsqueeze(0), attention_mask=enc.attention_mask.unsqueeze(0))
    h.remove()
    x = captured["x"]
    if x is None:
        raise RuntimeError("Failed to capture pre-MLP activations.")
    return x[0]  # [T, d]

def pool_tokens(H: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    H = H.to(torch.float32)
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        idx = torch.arange(H.shape[0])
    if pooling == "mean":
        return H[idx].mean(0)
    elif pooling == "first":
        return H[int(idx[0].item())]
    else:
        return H[int(idx[-1].item())]

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Variance amplitude of paraphrase pre-MLP activations (BASE vs FT)")
    ap.add_argument("--selection_jsonl", type=str, required=True, help="jacobian_prompts.jsonl from sampler (FIRST dataset)")
    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)
    ap.add_argument("--layer_index", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--prompt_span", type=str, default="no_bos", choices=["no_bos","pre_eos","all"])
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean","first","last"])
    ap.add_argument("--topk_eigs", type=int, default=16)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("varamp")

    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    log.info("[%s] Loading selection: %s", nowts(), args.selection_jsonl)
    sets = load_selection_jsonl(args.selection_jsonl, max_prompts=args.max_prompts)
    log.info("Loaded %d prompt sets.", len(sets))

    log.info("[%s] Loading models...", nowts())
    base, ft, tokenizer = build_model_and_tokenizer(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, args.device
    )
    log.info("Models ready. Layer=%d. Pooling=%s", args.layer_index, args.pooling)

    # Per-model per-prompt rows
    rows_base: List[dict] = []
    rows_ft:   List[dict] = []

    t0 = time.time()
    for idx_ps, ps in enumerate(sets, 1):
        if idx_ps == 1:
            log.info("Beginning capture loop over prompts (n=%d).", len(sets))
        if idx_ps % 5 == 0 or idx_ps == len(sets):
            done = idx_ps; total = len(sets)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            remain = (total - done) / max(rate, 1e-6)
            log.info("[%s] progress %d/%d (%.1f%%)  ETA %02d:%02d",
                     nowts(), done, total, 100.0*done/total, int(remain//60), int(remain%60))

        def collect(model, tag: str) -> dict:
            hs = []
            for key, txt in ps.paraphrases:
                enc = encode_prompt(tokenizer, build_prompt_text(txt, ps.input_text), args.device, args.prompt_span)
                Htok = capture_pre_mlp_activations(model, args.layer_index, enc)  # [T, d]
                h = pool_tokens(Htok, enc.prompt_mask.cpu(), args.pooling)        # [d]
                hs.append(h.to(torch.float32).cpu())
            if len(hs) < 2:
                return {"prompt_count": ps.prompt_count, "n": len(hs), "trace": float("nan")}
            Hmat = torch.stack(hs, dim=0)                  # [n, d]
            mu = Hmat.mean(0, keepdim=True)                # [1, d]
            V  = (Hmat - mu)                                # [n, d]
            n = Hmat.shape[0]
            # SVD of V (thin): eigenvalues of covariance = (S^2)/(n-1)
            # Work in float32 on CPU for stability
            U, S, VT = torch.linalg.svd(V.to(torch.float32), full_matrices=False)
            eigs = (S**2) / max(n-1, 1)
            trace = float(eigs.sum().item())
            row = {"prompt_count": ps.prompt_count, "n": int(n), "trace": trace, "mean_norm_sq": float(mu.pow(2).sum().item())}
            k = min(int(args.topk_eigs), eigs.shape[0])
            for i in range(k):
                row[f"eig_{i+1}"] = float(eigs[i].item())
            return row

        rows_base.append(collect(base, "BASE"))
        rows_ft.append(collect(ft, "FT"))

    df_base = pd.DataFrame(rows_base)
    df_ft   = pd.DataFrame(rows_ft)

    df_base.to_csv(outdir / f"BASE_varamp_layer{args.layer_index}.csv", index=False)
    df_ft.to_csv(outdir / f"FT_varamp_layer{args.layer_index}.csv", index=False)

    # Merge for paired analyses
    merged = df_base.merge(df_ft, on=["prompt_count"], suffixes=("_BASE","_FT"))
    merged.to_csv(outdir / f"varamp_merged_layer{args.layer_index}.csv", index=False)

    # ---- Figures ----
    def mean_ci(vals: np.ndarray) -> Tuple[float,float]:
        vals = vals[np.isfinite(vals)]
        if vals.size == 0: return 0.0, 0.0
        m = float(vals.mean())
        s = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        ci = 1.96 * s / math.sqrt(vals.size) if vals.size > 1 else 0.0
        return m, ci

    # 1) trace mean±CI
    tB, ciB = mean_ci(merged["trace_BASE"].values)
    tF, ciF = mean_ci(merged["trace_FT"].values)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["BASE","FT"], [tB, tF], yerr=[ciB, ciF], color=["#34495e","#2e7d32"], edgecolor="#2c3e50")
    ax.set_title(f"Pre-MLP paraphrase variance (trace Σ) — layer {args.layer_index}")
    ax.set_ylabel("trace(Σ)")
    fig.tight_layout(); fig.savefig(outdir / f"trace_mean_bars_layer{args.layer_index}.png", dpi=170); plt.close(fig)

    # 2) scatter BASE vs FT
    x = merged["trace_BASE"].values; y = merged["trace_FT"].values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x, y, s=12, alpha=0.6, color="#2e7d32")
    mn = float(min(x.min(), y.min())); mx = float(max(x.max(), y.max()))
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="#2c3e50")
    ax.set_xlabel("trace(Σ) — BASE"); ax.set_ylabel("trace(Σ) — FT")
    ax.set_title(f"BASE vs FT (trace Σ) — layer {args.layer_index}")
    fig.tight_layout(); fig.savefig(outdir / f"trace_scatter_layer{args.layer_index}.png", dpi=170); plt.close(fig)

    # 3) histogram of differences FT−BASE
    diff = (merged["trace_FT"] - merged["trace_BASE"]).values
    diff = diff[np.isfinite(diff)]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(diff, bins=30, alpha=0.8, color="#2e7d32", edgecolor="#2c3e50")
    ax.axvline(0.0, linestyle="--", color="#34495e")
    ax.set_title(f"Distribution of FT − BASE trace(Σ) — layer {args.layer_index}")
    ax.set_xlabel("FT − BASE"); ax.set_ylabel("count")
    fig.tight_layout(); fig.savefig(outdir / f"trace_diff_hist_layer{args.layer_index}.png", dpi=170); plt.close(fig)

    # 4) ratio bar mean±CI
    ratio = (merged["trace_FT"] / merged["trace_BASE"]).values
    ratio = ratio[np.isfinite(ratio) & (merged["trace_BASE"].values > 0)]
    rM, rCI = mean_ci(ratio)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(["FT/BASE"], [rM], yerr=[rCI], color="#2e7d32", edgecolor="#2c3e50")
    ax.axhline(1.0, linestyle="--", color="#34495e")
    ax.set_title(f"FT / BASE trace(Σ) — layer {args.layer_index}")
    fig.tight_layout(); fig.savefig(outdir / f"trace_ratio_bar_layer{args.layer_index}.png", dpi=170); plt.close(fig)

    # 5) eigenvalue overlays (mean across prompts)
    maxk = min(args.topk_eigs, len([c for c in merged.columns if c.startswith("eig_") and c.endswith("_BASE")]))
    if maxk > 0:
        xs = np.arange(1, maxk+1)
        base_means = []; ft_means = []
        for i in range(1, maxk+1):
            b = merged[f"eig_{i}_BASE"].values
            f = merged[f"eig_{i}_FT"].values
            base_means.append(float(np.nanmean(b)))
            ft_means.append(float(np.nanmean(f)))
        # linear overlay
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(xs, base_means, marker="o", label="BASE", color="#34495e")
        ax.plot(xs, ft_means, marker="o", label="FT", color="#2e7d32")
        ax.set_xticks(xs); ax.set_xlabel("eigen index")
        ax.set_ylabel("eigenvalue (variance)")
        ax.set_title(f"Top-{maxk} eigenvalues (mean across prompts) — layer {args.layer_index}")
        ax.legend()
        fig.tight_layout(); fig.savefig(outdir / f"eig_overlay_layer{args.layer_index}.png", dpi=170); plt.close(fig)

        # log overlay
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(xs, base_means, marker="o", label="BASE", color="#34495e")
        ax.plot(xs, ft_means, marker="o", label="FT", color="#2e7d32")
        ax.set_yscale("log")
        ax.set_xticks(xs); ax.set_xlabel("eigen index")
        ax.set_ylabel("eigenvalue (log scale)")
        ax.set_title(f"Top-{maxk} eigenvalues (log) — layer {args.layer_index}")
        ax.legend()
        fig.tight_layout(); fig.savefig(outdir / f"eig_overlay_log_layer{args.layer_index}.png", dpi=170); plt.close(fig)

        # cumulative mass
        base_cum = np.cumsum(base_means); ft_cum = np.cumsum(ft_means)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(xs, base_cum, marker="o", label="BASE", color="#34495e")
        ax.plot(xs, ft_cum, marker="o", label="FT", color="#2e7d32")
        ax.set_xticks(xs); ax.set_xlabel("eigen index")
        ax.set_ylabel("cumulative variance")
        ax.set_title(f"Cumulative eigenvalue mass — layer {args.layer_index}")
        ax.legend()
        fig.tight_layout(); fig.savefig(outdir / f"eig_cum_overlay_layer{args.layer_index}.png", dpi=170); plt.close(fig)

    # README
    lines = []
    lines.append("# Variance amplitude at pre-MLP — outputs")
    lines.append(f"- `BASE_varamp_layer{args.layer_index}.csv`")
    lines.append(f"- `FT_varamp_layer{args.layer_index}.csv`")
    lines.append(f"- `varamp_merged_layer{args.layer_index}.csv`")
    lines.append("Figures:")
    lines.append(f"- `trace_mean_bars_layer{args.layer_index}.png`")
    lines.append(f"- `trace_scatter_layer{args.layer_index}.png`")
    lines.append(f"- `trace_diff_hist_layer{args.layer_index}.png`")
    lines.append(f"- `trace_ratio_bar_layer{args.layer_index}.png`")
    lines.append(f"- `eig_overlay_layer{args.layer_index}.png`")
    lines.append(f"- `eig_overlay_log_layer{args.layer_index}.png`")
    lines.append(f"- `eig_cum_overlay_layer{args.layer_index}.png`")
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    log.info("[%s] Done. Outputs in %s", nowts(), outdir)

if __name__ == "__main__":
    main()
