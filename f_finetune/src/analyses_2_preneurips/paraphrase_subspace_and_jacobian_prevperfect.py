#!/usr/bin/env python3
"""
SCRIPT D

paraphrase_subspace_and_jacobian.py

Purpose:
- For the **FIRST** dataset selection (from prep_first_dataset_sampler.py: `jacobian_prompts.jsonl`),
  probe what the **layer-L MLP** learned by estimating its directional sensitivity (Jacobian norms)
  along three families of directions:
    (1) Mean direction (normalize(mean(h))).   [semantic core]
    (2) Paraphrase-variance directions:
        - `--jacobian_mode pca`: top-K PCs of centered paraphrase activations.
        - `--jacobian_mode raw`: normalized centered vectors themselves (up to --max_directions).
    (3) Random directions (control).
  Compare **BASE** vs **FT** (and optionally multiple FT runs by re-running with different `--tag`).

- Also compute and compare the **norm of the MLP output** ||MLP(h)||₂ distributions (BASE vs FT).

Kept from your tuned setup:
- Comprehensive logging, seed=42 default.
- Robust model loading & LoRA merging; device & dtype handling.
- Layer index flag; pooling over prompt tokens (mean by default).
- Strong separation of concerns: no unrelated extras; outputs are CSVs and small, focused plots.

Outputs (to --outdir):
- Per model tag:
  * `{TAG}_paraphrase_subspace_jacobian_layer{L}.csv`   (per-prompt results)
  * `{TAG}_mlp_out_norm_stats_layer{L}.csv`             (per-prompt ||MLP(h)||₂ stats)
  * `{TAG}_pca_stats_layer{L}.csv` (if mode=pca): per-prompt explained variance ratios
- Aggregated plots:
  * `jacobian_var_bars_layer{L}.png`          (VAR across models, mean±CI)
  * `jacobian_rand_bars_layer{L}.png`         (RANDOM across models, mean±CI)
  * `jacobian_mean_dir_bars_layer{L}.png`     (MEAN across models, mean±CI)
  * `mlp_out_norm_mean_bars_layer{L}.png`     (BASE vs FT mean ||MLP(h)||₂)
  * `pca_scree_mean_layer{L}.png`             (if mode=pca; mean EVR bars)
- NEW comparative figures (single-figure overlays & deeper comparisons):
  * `jacobian_combined_layer{L}.png`                    (MEAN/VAR/RAND in one plot, colored, BASE/FT)
  * `jacobian_ratio_ft_over_base_layer{L}.png`          (FT/BASE ratios with 95% CI, per family)
  * `jacobian_diff_hist_layer{L}.png`                   (overlaid histograms of FT−BASE for MEAN & VAR)
  * `jacobian_scatter_VAR_layer{L}.png`                 (scatter BASE vs FT for VAR with y=x line)
  * `jacobian_var_box_layer{L}.png`                     (boxplots of VAR per tag, same figure)
  * `mlp_out_norm_combined_bars_layer{L}.png`           (BASE & FT norms in same plot)
  * `pca_scree_overlay_layer{L}.png`                    (if mode=pca; EVR overlay BASE vs FT)
  * `pca_cum_evr_overlay_layer{L}.png`                  (if mode=pca; cumulative EVR overlay)

Key fixes kept:
- Perform PCA/SVD on float32 CPU tensors to avoid
  `RuntimeError: "linalg_svd_cpu" not implemented for 'Half'`.
- Normalize and Jacobian arithmetic in float32 for stability.
- Ensure inputs to the MLP are cast to the module's parameter dtype
  inside `get_mlp_function` (so half-precision models work cleanly).
"""
from __future__ import annotations
import argparse, json, logging, math, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v = v.to(torch.float32)
    n = v.norm(p=2)
    return v / (n + eps)

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

class BlockAccessor:
    def __init__(self, model: nn.Module, layer_index: int):
        self.model = model
        self.layer_index = layer_index
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.layers = model.transformer.h
        else:
            raise TypeError("Unsupported model architecture.")
        self.block = self.layers[layer_index]
        self.attn = getattr(self.block, "self_attn", None) or getattr(self.block, "attention", None)
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
        raise ValueError("Provide --ft_model_name_or_path (merged FT) OR --ft_lora_adapter (LoRA adapter dir).")

    if ft_lora_adapter is not None:
        apath = Path(ft_lora_adapter)
        if apath.exists() and (apath / "adapter_config.json").exists():
            if not _HAS_PEFT:
                raise RuntimeError("peft not installed but --ft_lora_adapter was provided.")
            ft = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
            from peft import PeftModel  # lazy import to avoid hard dep if not used
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

@dataclass
class Encoded:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_mask: torch.Tensor
    tokens: List[str]

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
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    return Encoded(
        input_ids=input_ids.to(device),
        attention_mask=attn_mask.to(device),
        prompt_mask=prompt_mask.to(device),
        tokens=tokens
    )

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
    T = x.shape[1]
    return x[0, :T, :].contiguous()

def get_mlp_function(model: nn.Module, layer_index: int):
    ba = BlockAccessor(model, layer_index)
    mlp = ba.mlp
    mlp_device = next(mlp.parameters()).device
    mlp_dtype  = next(mlp.parameters()).dtype
    def f(h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        x = h.to(device=mlp_device, dtype=mlp_dtype).unsqueeze(0)
        with torch.inference_mode():
            y = mlp(x)
        return y[0,0,:].to("cpu", dtype=torch.float32)
    return f

def estimate_jacobian_norm(f, center: torch.Tensor, direction: torch.Tensor, eps: float = 1e-3) -> float:
    c = center.to(torch.float32)
    d = normalize(direction)
    y1 = f(c + eps * d)
    y2 = f(c - eps * d)
    g = (y1 - y2) / (2.0 * eps)
    return float(g.norm(p=2).item())

def main():
    ap = argparse.ArgumentParser(description="Paraphrase subspace + Jacobian probes for MLP")
    ap.add_argument("--selection_jsonl", type=str, required=True, help="jacobian_prompts.jsonl from sampler")
    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)
    ap.add_argument("--tag", type=str, default="FT")
    ap.add_argument("--layer_index", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--prompt_span", type=str, default="no_bos", choices=["no_bos","pre_eos","all"])
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean","first","last"])
    ap.add_argument("--jacobian_mode", type=str, default="pca", choices=["pca","raw"])
    ap.add_argument("--topk_pca", type=int, default=8)
    ap.add_argument("--max_directions", type=int, default=50)
    ap.add_argument("--eps_jacobian", type=float, default=1e-3)
    ap.add_argument("--n_random_dirs", type=int, default=8)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("jacobian")

    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    sets = load_selection_jsonl(args.selection_jsonl, max_prompts=args.max_prompts)
    log.info("Loaded %d prompt sets for Jacobian.", len(sets))
    base, ft, tokenizer = build_model_and_tokenizer(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, args.device
    )

    per_model_rows: Dict[str, List[dict]] = {"BASE": [], f"FT_{args.tag}": []}
    pca_rows: Dict[str, List[dict]] = {"BASE": [], f"FT_{args.tag}": []}
    mlp_norm_rows: Dict[str, List[dict]] = {"BASE": [], f"FT_{args.tag}": []}

    def pool_tokens(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        H = H.to(torch.float32)
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            idx = torch.arange(H.shape[0], device=H.device)
        if args.pooling == "mean":
            return H[idx].mean(0)
        elif args.pooling == "first":
            return H[int(idx[0].item())]
        else:
            return H[int(idx[-1].item())]

    def run_for_model(tag: str, model: nn.Module):
        mlp_f = get_mlp_function(model, args.layer_index)
        for ps in sets:
            hs: List[torch.Tensor] = []
            for key, txt in ps.paraphrases:
                enc = encode_prompt(tokenizer, build_prompt_text(txt, ps.input_text), args.device, args.prompt_span)
                H = capture_pre_mlp_activations(model, args.layer_index, enc)
                h = pool_tokens(H, enc.prompt_mask.cpu())
                hs.append(h)
            if len(hs) < 3:
                log.warning("prompt_count=%s has <3 paraphrases captured; skipping.", ps.prompt_count)
                continue
            Hmat = torch.stack(hs, dim=0)
            mean_h = Hmat.mean(0)
            V = Hmat - mean_h.unsqueeze(0)

            norms = []
            for i in range(Hmat.shape[0]):
                y = mlp_f(Hmat[i])
                norms.append(float(y.norm(p=2).item()))
            mlp_norm_rows[tag].append({
                "prompt_count": ps.prompt_count,
                "n": int(Hmat.shape[0]),
                "mlp_out_norm_mean": float(np.mean(norms)),
                "mlp_out_norm_std": float(np.std(norms, ddof=1) if len(norms)>1 else 0.0),
                "mlp_out_norm_min": float(np.min(norms)),
                "mlp_out_norm_max": float(np.max(norms)),
            })

            dirs_mean = [normalize(mean_h)]
            dirs_random = []
            D = Hmat.shape[1]
            for _ in range(args.n_random_dirs):
                r = torch.randn(D, dtype=torch.float32)
                dirs_random.append(normalize(r))

            if args.jacobian_mode == "pca":
                Vc = (V - V.mean(0, keepdim=True)).to(torch.float32)
                U, S, VT = torch.linalg.svd(Vc, full_matrices=False)
                comps = VT[:args.topk_pca, :]
                dirs_var = [normalize(comps[k]) for k in range(comps.shape[0])]
                var = (S**2)
                total = var.sum().item() + 1e-8
                ratios = (var / total).cpu().numpy()
                row = {"prompt_count": ps.prompt_count, "evr_total_topk": float(ratios[:args.topk_pca].sum())}
                for i in range(min(len(ratios), args.topk_pca)):
                    row[f"evr_{i+1}"] = float(ratios[i])
                pca_rows[tag].append(row)
            else:
                idxs = np.random.permutation(V.shape[0])[:args.max_directions]
                dirs_var = [normalize(V[i]) for i in idxs]

            eps = float(args.eps_jacobian)
            jac_mean_vals = [estimate_jacobian_norm(mlp_f, mean_h, d, eps) for d in dirs_mean]
            jac_var_vals  = [estimate_jacobian_norm(mlp_f, mean_h, d, eps) for d in dirs_var]
            jac_rand_vals = [estimate_jacobian_norm(mlp_f, mean_h, d, eps) for d in dirs_random]

            per_model_rows[tag].append({
                "prompt_count": ps.prompt_count,
                "n": int(Hmat.shape[0]),
                "jac_MEAN": float(np.mean(jac_mean_vals)),
                "jac_VAR_mean": float(np.mean(jac_var_vals)) if len(jac_var_vals)>0 else np.nan,
                "jac_VAR_std": float(np.std(jac_var_vals, ddof=1)) if len(jac_var_vals)>1 else 0.0,
                "jac_RAND_mean": float(np.mean(jac_rand_vals)),
                "jac_RAND_std": float(np.std(jac_rand_vals, ddof=1)) if len(jac_rand_vals)>1 else 0.0,
                "mode": args.jacobian_mode,
            })

    logging.getLogger("jacobian").info("Running Jacobian probes for BASE...")
    run_for_model("BASE", base)
    logging.getLogger("jacobian").info("Running Jacobian probes for FT (%s)...", args.tag)
    run_for_model(f"FT_{args.tag}", ft)

    # --- Persist CSVs (unchanged) ---
    for tag, rows in per_model_rows.items():
        pd.DataFrame(rows).to_csv(outdir / f"{tag}_paraphrase_subspace_jacobian_layer{args.layer_index}.csv", index=False)
    for tag, rows in mlp_norm_rows.items():
        pd.DataFrame(rows).to_csv(outdir / f"{tag}_mlp_out_norm_stats_layer{args.layer_index}.csv", index=False)
    if args.jacobian_mode == "pca":
        for tag, rows in pca_rows.items():
            pd.DataFrame(rows).to_csv(outdir / f"{tag}_pca_stats_layer{args.layer_index}.csv", index=False)

    # --- Existing mean±CI bar helpers (kept) ---
    def agg_bar(data: Dict[str, List[dict]], metric: str, title: str, fname: str):
        tags = list(data.keys())
        means = []
        cis = []
        for tag in tags:
            df = pd.DataFrame(data[tag])
            vals = df[metric].dropna().values
            if len(vals) == 0:
                means.append(0.0); cis.append(0.0); continue
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
            ci = 1.96 * s / math.sqrt(len(vals)) if len(vals)>1 else 0.0
            means.append(m); cis.append(ci)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(tags, means, yerr=cis)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=170)
        plt.close(fig)

    # Original single-metric figures
    agg_bar(per_model_rows, "jac_VAR_mean",  f"Jacobian (VAR — mode={args.jacobian_mode}) — layer {args.layer_index}", f"jacobian_var_bars_layer{args.layer_index}.png")
    agg_bar(per_model_rows, "jac_RAND_mean", f"Jacobian (RANDOM) — layer {args.layer_index}", f"jacobian_rand_bars_layer{args.layer_index}.png")
    agg_bar(per_model_rows, "jac_MEAN",      f"Jacobian (MEAN dir) — layer {args.layer_index}", f"jacobian_mean_dir_bars_layer{args.layer_index}.png")

    tags = list(mlp_norm_rows.keys())
    means = []; cis = []
    for tag in tags:
        df = pd.DataFrame(mlp_norm_rows[tag])
        vals = df["mlp_out_norm_mean"].dropna().values
        m = float(np.mean(vals)) if len(vals)>0 else 0.0
        s = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
        ci = 1.96 * s / math.sqrt(len(vals)) if len(vals)>1 else 0.0
        means.append(m); cis.append(ci)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(tags, means, yerr=cis)
    ax.set_title(f"||MLP(h)||₂ (mean over paraphrases) — layer {args.layer_index}")
    fig.tight_layout()
    fig.savefig(outdir / f"mlp_out_norm_mean_bars_layer{args.layer_index}.png", dpi=170)
    plt.close(fig)

    if args.jacobian_mode == "pca":
        evr_cols = [f"evr_{i+1}" for i in range(args.topk_pca)]
        evr_means = []; evr_cis = []
        base_pca = pd.DataFrame(pca_rows["BASE"])
        if not base_pca.empty:
            for c in evr_cols:
                vals = base_pca[c].dropna().values
                m = float(np.mean(vals)) if len(vals)>0 else 0.0
                s = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
                ci = 1.96 * s / math.sqrt(len(vals)) if len(vals)>1 else 0.0
                evr_means.append(m); evr_cis.append(ci)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar([f"PC{i+1}" for i in range(len(evr_means))], evr_means, yerr=evr_cis)
            ax.set_title(f"PCA explained variance ratio — mean across prompts (layer {args.layer_index})")
            fig.tight_layout()
            fig.savefig(outdir / f"pca_scree_mean_layer{args.layer_index}.png", dpi=170)
            plt.close(fig)

    # ======================
    # NEW comparative plots
    # ======================

    # Helper: aligned per-prompt dataframe for paired diffs/ratios
    base_df = pd.DataFrame(per_model_rows["BASE"])
    ft_df   = pd.DataFrame(per_model_rows[f"FT_{args.tag}"])
    merged  = None
    if not base_df.empty and not ft_df.empty:
        merged = base_df.merge(ft_df, on="prompt_count", suffixes=("_BASE","_FT"))

    # Paired statistical tests (BASE vs FT) for MEAN/VAR/RANDOM
    try:
        import numpy as _np
        try:
            from scipy.stats import ttest_rel, wilcoxon
            _HAS_SCIPY = True
        except Exception:
            _HAS_SCIPY = False

        stats_rows = []
        def paired_tests(bcol, fcol, label):
            xb = merged[bcol].values
            xf = merged[fcol].values
            mask = _np.isfinite(xb) & _np.isfinite(xf)
            xb = xb[mask]; xf = xf[mask]
            diffs = xf - xb
            res = {"metric": label,
                   "n": int(diffs.size),
                   "mean_base": float(xb.mean()) if xb.size else _np.nan,
                   "mean_ft": float(xf.mean()) if xf.size else _np.nan,
                   "mean_diff": float(diffs.mean()) if diffs.size else _np.nan,
                   "std_diff": float(diffs.std(ddof=1)) if diffs.size>1 else 0.0}
            if _HAS_SCIPY and diffs.size >= 2:
                res["ttest_t"], res["ttest_p"] = ttest_rel(xf, xb)
                try:
                    w = wilcoxon(diffs)
                    res["wilcoxon_stat"], res["wilcoxon_p"] = w.statistic, w.pvalue
                except Exception:
                    res["wilcoxon_stat"], res["wilcoxon_p"] = _np.nan, _np.nan
            else:
                res["ttest_t"] = res["ttest_p"] = _np.nan
                res["wilcoxon_stat"] = res["wilcoxon_p"] = _np.nan
            return res

        stats_rows.append(paired_tests("jac_MEAN_BASE", "jac_MEAN_FT", "MEAN"))
        stats_rows.append(paired_tests("jac_VAR_mean_BASE", "jac_VAR_mean_FT", "VAR"))
        stats_rows.append(paired_tests("jac_RAND_mean_BASE", "jac_RAND_mean_FT", "RANDOM"))

        stats_df = pd.DataFrame(stats_rows)
        stats_path = outdir / f"paired_stats_layer{args.layer_index}.csv"
        stats_df.to_csv(stats_path, index=False)
        logging.getLogger("jacobian").info("Paired stats written: %s\n%s", stats_path, stats_df.to_string(index=False))
    except Exception as e:
        logging.getLogger("jacobian").warning("Paired tests failed: %s", e)

    # 1) Combined jacobian plot in one figure: MEAN vs VAR vs RANDOM for BASE and FT
    try:
        if merged is not None and not merged.empty:
            # compute group means ± CI
            def mean_ci(vals):
                vals = np.asarray(vals, dtype=float)
                if len(vals) == 0: return 0.0, 0.0
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                ci = 1.96 * s / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
                return m, ci

            metrics = [
                ("jac_MEAN_BASE","jac_MEAN_FT","MEAN"),
                ("jac_VAR_mean_BASE","jac_VAR_mean_FT","VAR"),
                ("jac_RAND_mean_BASE","jac_RAND_mean_FT","RANDOM"),
            ]
            x = np.arange(len(metrics))
            width = 0.35

            fig, ax = plt.subplots(figsize=(9,5))
            base_means = []; base_cis = []
            ft_means   = []; ft_cis   = []
            labels = []
            for mb, mf, lab in metrics:
                mB, cB = mean_ci(merged[mb].dropna().values)
                mF, cF = mean_ci(merged[mf].dropna().values)
                base_means.append(mB); base_cis.append(cB)
                ft_means.append(mF);   ft_cis.append(cF)
                labels.append(lab)
            ax.bar(x - width/2, base_means, width, yerr=base_cis, label="BASE")
            ax.bar(x + width/2, ft_means,   width, yerr=ft_cis,   label=f"FT_{args.tag}")
            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.set_title(f"Jacobian sensitivities by direction — layer {args.layer_index}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / f"jacobian_combined_layer{args.layer_index}.png", dpi=170)
            plt.close(fig)
    except Exception as e:
        logging.warning("Failed to plot jacobian_combined: %s", e)

    # 2) Ratios FT/BASE for MEAN/VAR/RANDOM (paired per prompt), mean±CI
    try:
        if merged is not None and not merged.empty:
            ratios = {}
            for base_col, ft_col, lab in [
                ("jac_MEAN_BASE","jac_MEAN_FT","MEAN"),
                ("jac_VAR_mean_BASE","jac_VAR_mean_FT","VAR"),
                ("jac_RAND_mean_BASE","jac_RAND_mean_FT","RANDOM"),
            ]:
                # Avoid div-by-zero; drop rows with base<=0
                valid = merged[merged[base_col] > 0]
                if valid.empty:
                    ratios[lab] = np.array([])
                else:
                    ratios[lab] = (valid[ft_col].values / valid[base_col].values)

            labs = ["MEAN","VAR","RANDOM"]
            x = np.arange(len(labs)); width=0.6
            means = []; cis=[]
            for lab in labs:
                arr = ratios[lab]
                if arr.size == 0:
                    means.append(0.0); cis.append(0.0)
                else:
                    m = float(np.mean(arr))
                    s = float(np.std(arr, ddof=1)) if arr.size>1 else 0.0
                    ci = 1.96 * s / math.sqrt(arr.size) if arr.size>1 else 0.0
                    means.append(m); cis.append(ci)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(x, means, yerr=cis)
            ax.set_xticks(x); ax.set_xticklabels(labs)
            ax.axhline(1.0, linestyle="--")
            ax.set_title(f"FT / BASE ratio of Jacobian sensitivities — layer {args.layer_index}")
            fig.tight_layout()
            fig.savefig(outdir / f"jacobian_ratio_ft_over_base_layer{args.layer_index}.png", dpi=170)
            plt.close(fig)
    except Exception as e:
        logging.warning("Failed to plot jacobian_ratio_ft_over_base: %s", e)

    # 3) Overlaid histograms of FT−BASE per prompt for MEAN & VAR (and optionally RANDOM)
    try:
        if merged is not None and not merged.empty:
            diffs = {
                "MEAN": merged["jac_MEAN_FT"] - merged["jac_MEAN_BASE"],
                "VAR":  merged["jac_VAR_mean_FT"] - merged["jac_VAR_mean_BASE"],
                # "RANDOM": merged["jac_RAND_mean_FT"] - merged["jac_RAND_mean_BASE"],  # uncomment if desired
            }
            fig, ax = plt.subplots(figsize=(9,5))
            for lab, series in diffs.items():
                vals = series.dropna().values
                if len(vals) == 0: continue
                ax.hist(vals, bins=30, alpha=0.5, label=lab)
            ax.axvline(0.0, linestyle="--")
            ax.set_title(f"Distribution of FT − BASE Jacobian sensitivities — layer {args.layer_index}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / f"jacobian_diff_hist_layer{args.layer_index}.png", dpi=170)
            plt.close(fig)
    except Exception as e:
        logging.warning("Failed to plot jacobian_diff_hist: %s", e)

    # 4) Scatter plot: BASE vs FT (VAR)
    try:
        if merged is not None and not merged.empty:
            x = merged["jac_VAR_mean_BASE"].values
            y = merged["jac_VAR_mean_FT"].values
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]; y = y[mask]
            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(x, y, s=12, alpha=0.6)
            mn = min(x.min(), y.min()); mx = max(x.max(), y.max())
            ax.plot([mn, mx], [mn, mx], linestyle="--")
            ax.set_xlabel("BASE VAR Jacobian")
            ax.set_ylabel(f"FT_{args.tag} VAR Jacobian")
            ax.set_title(f"BASE vs FT (VAR) — layer {args.layer_index}")
            fig.tight_layout()
            fig.savefig(outdir / f"jacobian_scatter_VAR_layer{args.layer_index}.png", dpi=170)
            plt.close(fig)
    except Exception as e:
        logging.warning("Failed to plot jacobian_scatter_VAR: %s", e)

    # 5) Boxplots: VAR sensitivity distributions (BASE vs FT) in one figure
    try:
        if merged is not None and not merged.empty:
            data = [
                merged["jac_VAR_mean_BASE"].dropna().values,
                merged["jac_VAR_mean_FT"].dropna().values
            ]
            fig, ax = plt.subplots(figsize=(7,5))
            ax.boxplot(data, labels=["BASE","FT"])
            ax.set_title(f"VAR Jacobian distribution (boxplot) — layer {args.layer_index}")
            fig.tight_layout()
            fig.savefig(outdir / f"jacobian_var_box_layer{args.layer_index}.png", dpi=170)
            plt.close(fig)
    except Exception as e:
        logging.warning("Failed to plot jacobian_var_box: %s", e)

    # 6) Combined MLP ||h|| bars in one figure (BASE & FT)
    try:
        tags = list(mlp_norm_rows.keys())
        means = []; cis = []; labels=[]
        for tag in tags:
            df = pd.DataFrame(mlp_norm_rows[tag])
            vals = df["mlp_out_norm_mean"].dropna().values
            m = float(np.mean(vals)) if len(vals)>0 else 0.0
            s = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
            ci = 1.96 * s / math.sqrt(len(vals)) if len(vals)>1 else 0.0
            means.append(m); cis.append(ci); labels.append(tag)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(labels, means, yerr=cis)
        ax.set_title(f"||MLP(h)||₂ (mean over paraphrases) — BASE vs FT — layer {args.layer_index}")
        fig.tight_layout()
        fig.savefig(outdir / f"mlp_out_norm_combined_bars_layer{args.layer_index}.png", dpi=170)
        plt.close(fig)
    except Exception as e:
        logging.warning("Failed to plot mlp_out_norm_combined_bars: %s", e)

    # 7) PCA overlays (if applicable): EVR and cumulative EVR for BASE vs FT
    if args.jacobian_mode == "pca":
        try:
            base_pca = pd.DataFrame(pca_rows["BASE"])
            ft_pca   = pd.DataFrame(pca_rows[f"FT_{args.tag}"])
            if not base_pca.empty and not ft_pca.empty:
                maxk = args.topk_pca
                evr_cols = [f"evr_{i+1}" for i in range(maxk)]
                base_means = []; ft_means = []
                for c in evr_cols:
                    base_means.append(float(base_pca[c].dropna().mean() if c in base_pca else 0.0))
                    ft_means.append(float(ft_pca[c].dropna().mean() if c in ft_pca else 0.0))
                # Overlay EVR
                fig, ax = plt.subplots(figsize=(8,4))
                xs = np.arange(1, len(base_means)+1)
                ax.plot(xs, base_means, marker="o", label="BASE")
                ax.plot(xs, ft_means, marker="o", label=f"FT_{args.tag}")
                ax.set_xticks(xs); ax.set_xlabel("PC index")
                ax.set_title(f"EVR overlay (mean across prompts) — layer {args.layer_index}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(outdir / f"pca_scree_overlay_layer{args.layer_index}.png", dpi=170)
                plt.close(fig)
                # Cumulative EVR
                base_cum = np.cumsum(base_means)
                ft_cum   = np.cumsum(ft_means)
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(xs, base_cum, marker="o", label="BASE")
                ax.plot(xs, ft_cum, marker="o", label=f"FT_{args.tag}")
                ax.set_xticks(xs); ax.set_ylim(0, 1.05)
                ax.set_title(f"Cumulative EVR overlay — layer {args.layer_index}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(outdir / f"pca_cum_evr_overlay_layer{args.layer_index}.png", dpi=170)
                plt.close(fig)
        except Exception as e:
            logging.warning("Failed to plot PCA overlays: %s", e)

    # README (kept + augmented list)
    lines = []
    lines.append("# Paraphrase subspace + Jacobian — outputs")
    lines.append("")
    lines.append("Per-model CSVs (per prompt):")
    lines.append(f"- `BASE_paraphrase_subspace_jacobian_layer{args.layer_index}.csv`")
    lines.append(f"- `FT_{args.tag}_paraphrase_subspace_jacobian_layer{args.layer_index}.csv`")
    lines.append(f"- `BASE_mlp_out_norm_stats_layer{args.layer_index}.csv`")
    lines.append(f"- `FT_{args.tag}_mlp_out_norm_stats_layer{args.layer_index}.csv`")
    if args.jacobian_mode == "pca":
        lines.append(f"- `BASE_pca_stats_layer{args.layer_index}.csv`")
        lines.append(f"- `FT_{args.tag}_pca_stats_layer{args.layer_index}.csv`")
    lines.append("")
    lines.append("Figures (original):")
    lines.append(f"- `jacobian_var_bars_layer{args.layer_index}.png`")
    lines.append(f"- `jacobian_rand_bars_layer{args.layer_index}.png`")
    lines.append(f"- `jacobian_mean_dir_bars_layer{args.layer_index}.png`")
    lines.append(f"- `mlp_out_norm_mean_bars_layer{args.layer_index}.png`")
    if args.jacobian_mode == "pca":
        lines.append(f"- `pca_scree_mean_layer{args.layer_index}.png`")
    lines.append("")
    lines.append("Figures (new, comparative):")
    lines.append(f"- `jacobian_combined_layer{args.layer_index}.png`  (MEAN/VAR/RAND, BASE & FT, colored)")
    lines.append(f"- `jacobian_ratio_ft_over_base_layer{args.layer_index}.png`  (FT/BASE ratio with 95% CI)")
    lines.append(f"- `jacobian_diff_hist_layer{args.layer_index}.png`  (FT−BASE distribution overlays)")
    lines.append(f"- `jacobian_scatter_VAR_layer{args.layer_index}.png`  (per-prompt VAR BASE vs FT, y=x)")
    lines.append(f"- `jacobian_var_box_layer{args.layer_index}.png`  (VAR boxplots BASE vs FT)")
    lines.append(f"- `mlp_out_norm_combined_bars_layer{args.layer_index}.png`  (BASE & FT together)")
    if args.jacobian_mode == "pca":
        lines.append(f"- `pca_scree_overlay_layer{args.layer_index}.png`  (EVR overlay BASE vs FT)")
        lines.append(f"- `pca_cum_evr_overlay_layer{args.layer_index}.png`  (cumulative EVR overlay)")
    lines.append("")
    lines.append("Interpretation sketch:")
    lines.append("- If `FT` shows **lower** Jacobian on VAR than BASE but similar on MEAN, it supports 'denoising paraphrase variance'.")
    lines.append("- If `FT/BASE < 1` for VAR and ~1 for MEAN, same story. Scatter below y=x for VAR reinforces it.")
    lines.append("- Shifts in `||MLP(h)||` indicate selective amplification/suppression (semantic core vs variance).")
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    logging.getLogger("jacobian").info("Done. CSVs and figures written to %s", outdir)

if __name__ == "__main__":
    main()
