#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  python3 b_tests/graphics/plot_ac_caus_results.py \
  --outdir b_tests/graphics/ac \
    f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2_preneurips/ac_cause_I_DIAG_mlp_only
    f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2_preneurips/ac_cause_D_attn_only_14_18_22_last
    f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2_preneurips/ac_cause_H4_noblock_noL22
    f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2_preneurips/ac_L_opt_mid_attn_mlp_noBlock_noL22_last
    f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2_preneurips/ac_cause_G_OPT_l6_plus_lateATTN_last_K4

Outputs (all-runs combined):
- <outdir>/ladder_scrubbed_all.png
- <outdir>/ladder_patched_all.png
- <outdir>/ladder_ablated_all.png
- <outdir>/suff_nec_all.png
- <outdir>/portability_all.png
- <outdir>/gate_heatmap_all.png
- <outdir>/effect_portraits_all.png
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

# colors
FOREST = "#228B22"
GRAYS = ["#4a4a4a", "#6a6a6a", "#7f7f7f", "#9a9a9a", "#b5b5b5", "#cfcfcf"]
GRID = "#e6e6e6"

# IO helpers

def ensure_outdir(outdir: str) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    return out

def load_histories(paths: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    loaded = []
    for p in paths:
        pth = Path(p)
        hp = pth / "history.json" if pth.is_dir() else pth
        if not hp.exists():
            print(f"[warn] missing history.json: {pth}")
            continue
        try:
            with hp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[warn] failed to load {hp}: {e}")
            continue
        name = hp.parent.name
        loaded.append((name, data))
    return loaded

# Metrics helpers

def get_baselines(hist: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    base = hist.get("baseline", {})
    return base.get("BASE_sKL", None), base.get("FT_sKL", None)

def get_eval_numbers(hist: Dict[str, Any]) -> Dict[str, Optional[float]]:
    evals = hist.get("eval", {})
    out = {
        "FT_patched_sKL": evals.get("FT_patched_sKL"),
        "FT_scrubbed_sKL": evals.get("FT_scrubbed_sKL"),
        "FT_ablated_sKL":  evals.get("FT_ablated_sKL"),
    }
    # Try to locate controls if present under various keys
    ctrl_keys = [
        "all_scrub_sKL", "All-scrub", "all_scrub_keep_nothing_sKL",
        "random_mask_sKL", "Random mask (size-matched)"
    ]
    # search eval, then top-level "controls", then root
    for k in ctrl_keys:
        if k in evals:
            out[k] = evals.get(k)
    controls = hist.get("controls", {})
    for k in ctrl_keys:
        if k not in out and k in controls:
            out[k] = controls.get(k)
    for k in ctrl_keys:
        if k not in out and k in hist:
            out[k] = hist.get(k)
    return out

def suff_nec_from_history(hist: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    base_skl, ft_skl = get_baselines(hist)
    ev = get_eval_numbers(hist)
    if base_skl is None or ft_skl is None:
        return None, None
    denom = (base_skl - ft_skl)
    if denom == 0:
        return None, None

    scrubbed = ev.get("FT_scrubbed_sKL")
    patched  = ev.get("FT_patched_sKL")
    ablated  = ev.get("FT_ablated_sKL")

    # prefer scrubbed for sufficiency
    suff_src = scrubbed if (isinstance(scrubbed, (int, float)) and not math.isnan(scrubbed)) else \
               (patched if (isinstance(patched, (int, float)) and not math.isnan(patched)) else None)
    suff = None if suff_src is None else (base_skl - float(suff_src)) / denom
    nec  = None if (ablated is None or math.isnan(ablated)) else (float(ablated) - ft_skl) / denom
    return suff, nec

def parse_tag(tag: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        if not tag.startswith("L"):
            return None, None
        rest = tag[1:]
        L_str, after = rest.split(":", 1)
        mod = after.split(":", 1)[0]
        return int(L_str), mod
    except Exception:
        return None, None

# Combined plots

def pick_gray(i: int) -> str:
    return GRAYS[i % len(GRAYS)]

def best_run_by_median_metric(runs: List[Tuple[str, Dict[str, Any]]], metric_key: str) -> Optional[int]:
    best_idx, best_val = None, None
    for i, (_, hist) in enumerate(runs):
        ladder = hist.get("budget_ladder", [])
        vals = [rec.get(metric_key) for rec in ladder
                if rec.get(metric_key) is not None and not (isinstance(rec.get(metric_key), float) and math.isnan(rec.get(metric_key)))]
        if not vals:
            continue
        m = float(np.median(vals))
        if best_val is None or m < best_val:
            best_val = m
            best_idx = i
    return best_idx

def plot_ladder_all(runs, outdir):
    series = [
        ("ladder_scrubbed_all.png", "FT_scrubbed_sKL", "Scrubbed (only path kept)"),
        ("ladder_patched_all.png",  "FT_patched_sKL",  "Patched"),
        ("ladder_ablated_all.png",  "FT_ablated_sKL",  "Ablated"),
    ]
    for fname, key, label in series:
        if not any(hist.get("budget_ladder") for _, hist in runs):
            print("[warn] no budget_ladder in any run; skipping ladder plots.")
            return
        best_idx = best_run_by_median_metric(runs, key)
        plt.figure(figsize=(7.6, 4.8))
        for i, (runname, hist) in enumerate(runs):
            ladder = hist.get("budget_ladder", [])
            if not ladder:
                continue
            Ks = [rec.get("K") for rec in ladder]
            vals = [rec.get(key, np.nan) for rec in ladder]
            col = FOREST if i == best_idx else pick_gray(i)
            lw = 2.4 if i == best_idx else 1.8
            plt.plot(Ks, vals, marker="o", label=runname, color=col, linewidth=lw, alpha=0.95 if i == best_idx else 0.85)
        plt.gca().invert_xaxis()
        plt.xlabel("K (mask size)")
        plt.ylabel("sKL (lower is better)")
        plt.title(f"{label} vs K — all runs")
        plt.grid(True, color=GRID, linewidth=0.7)
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=170)
        plt.close()

def plot_suff_nec_all(runs, outdir):
    rows = []
    for runname, hist in runs:
        s, n = suff_nec_from_history(hist)
        if s is None and n is None:
            continue
        rows.append((runname, s, n))
    if not rows:
        print("[warn] no suff/nec data; skipping suff_nec_all.")
        return
    names = [r[0] for r in rows]
    suffs = [100.0 * r[1] if r[1] is not None else np.nan for r in rows]
    necs  = [100.0 * r[2] if r[2] is not None else np.nan for r in rows]
    x = np.arange(len(names))
    w = 0.42
    best_idx = int(np.nanargmax(suffs)) if any(not np.isnan(v) for v in suffs) else None
    plt.figure(figsize=(max(7.6, 0.9*len(names)), 4.8))
    # suff
    for i, v in enumerate(suffs):
        col = FOREST if best_idx is not None and i == best_idx else pick_gray(i)
        plt.bar(x[i] - w/2, v, width=w, color=col, label="Sufficiency" if i == 0 else None)
    # nec
    for i, v in enumerate(necs):
        plt.bar(x[i] + w/2, v, width=w, color=pick_gray(i+1), label="Necessity" if i == 0 else None)
    plt.axhline(60, linestyle="--", color="#bfbfbf", linewidth=1.0, label="Pre-reg 60%")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("% of BASE->FT gap explained")
    plt.title("Sufficiency & Necessity — all runs")
    plt.grid(axis="y", color=GRID, linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "suff_nec_all.png", dpi=170)
    plt.close()

def plot_portability_all(runs, outdir):
    rows = []
    for runname, hist in runs:
        port = hist.get("portability", {})
        if not port:
            continue
        u = port.get("BASE_unpatched_sKL_to_FTorig", port.get("BASE_unpatched_sKL"))
        p = port.get("BASE_patched_sKL_to_FTorig", port.get("BASE_patched_sKL"))
        if u is None or p is None:
            continue
        rows.append((runname, float(u), float(p)))
    if not rows:
        print("[warn] no portability data; skipping portability_all.")
        return
    names = [r[0] for r in rows]
    unp   = [r[1] for r in rows]
    pat   = [r[2] for r in rows]
    x = np.arange(len(names)); w = 0.42
    best_idx = int(np.argmin(pat))
    plt.figure(figsize=(max(7.6, 0.9*len(names)), 4.8))
    for i, v in enumerate(unp):
        plt.bar(x[i] - w/2, v, width=w, color=pick_gray(i+2), label="BASE unpatched" if i == 0 else None)
    for i, v in enumerate(pat):
        col = FOREST if i == best_idx else pick_gray(i)
        plt.bar(x[i] + w/2, v, width=w, color=col, label="BASE + path" if i == 0 else None)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("sKL vs FT-original (lower is better)")
    plt.title("Portability — all runs")
    plt.grid(axis="y", color=GRID, linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "portability_all.png", dpi=170)
    plt.close()

def plot_gate_heatmap_all(runs, outdir):
    mats = []
    titles = []
    all_layers = set()
    modules = ["attn_out", "mlp_post", "block_out"]
    for runname, hist in runs:
        lm = hist.get("learned_mask", {})
        tags = lm.get("tags", [])
        gates = lm.get("gates", {})
        if not tags or not gates:
            continue
        layers_here = sorted({parse_tag(t)[0] for t in tags if parse_tag(t)[0] is not None})
        for L in layers_here:
            all_layers.add(L)
        mats.append((runname, tags, gates))
        titles.append(runname)
    if not mats:
        print("[warn] no learned_mask data; skipping gate_heatmap_all.")
        return
    layers_sorted = sorted(all_layers)
    def build_mat(tags, gates):
        agg = {m: {L: [] for L in layers_sorted} for m in modules}
        for t in tags:
            L, mod = parse_tag(t)
            if L is None or mod not in agg:
                continue
            val = gates.get(t)
            try:
                v = float(val)
            except Exception:
                continue
            agg[mod][L].append(v)
        M = np.zeros((len(modules), len(layers_sorted)), dtype=float)
        for i, m in enumerate(modules):
            for j, L in enumerate(layers_sorted):
                arr = agg[m][L]
                M[i, j] = float(np.mean(arr)) if arr else 0.0
        return M
    mats_built = [(name, build_mat(tags, gates)) for name, tags, gates in mats]
    vmax = max((m.max() for _, m in mats_built), default=1.0)
    n = len(mats_built); ncol = min(3, n); nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol + 1.0, 2.9*nrow + 0.8), squeeze=False)
    for idx, (name, M) in enumerate(mats_built):
        r, c = divmod(idx, ncol)
        ax = axes[r][c]
        im = ax.imshow(M, aspect="auto", cmap="Greens", vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
        ax.set_xticks(np.arange(len(layers_sorted)))
        ax.set_xticklabels([str(L) for L in layers_sorted])
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels(["attn_out", "mlp_post", "block_out"])
        ax.set_title(name, fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if v >= 0.5:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black")
    for k in range(n, nrow*ncol):
        r, c = divmod(k, ncol)
        axes[r][c].axis("off")
    fig.suptitle("Gate values by layer × module — all runs", y=0.995, fontsize=12)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="Greens",
                             norm=plt.Normalize(vmin=0.0, vmax=vmax if vmax>0 else 1.0)),
                        ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Gate value (0–1)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir / "gate_heatmap_all.png", dpi=170)
    plt.close(fig)

# Effect Portraits

def effect_numbers(hist: Dict[str, Any]) -> Optional[Dict[str, float]]:
    base, ft = get_baselines(hist)
    if base is None or ft is None:
        return None
    ev = get_eval_numbers(hist)
    res = {
        "BASE": float(base),
        "FT": float(ft),
        "Patched": ev.get("FT_patched_sKL"),
        "Scrubbed": ev.get("FT_scrubbed_sKL"),
        "Ablated": ev.get("FT_ablated_sKL"),
        "AllScrub": ev.get("all_scrub_sKL", ev.get("All-scrub")),
        "RandomMask": ev.get("random_mask_sKL", ev.get("Random mask (size-matched)")),
    }
    return res

def choose_highlight_idx(runs: List[Tuple[str, Dict[str, Any]]], highlight: Optional[str]) -> Optional[int]:
    if highlight:
        for i, (name, _) in enumerate(runs):
            if name == highlight:
                return i
    # else: pick the one with highest sufficiency
    best_idx, best_s = None, None
    for i, (_, hist) in enumerate(runs):
        s, _ = suff_nec_from_history(hist)
        if s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        if best_s is None or s > best_s:
            best_s = s; best_idx = i
    return best_idx

def plot_effect_portraits_all(runs: List[Tuple[str, Dict[str, Any]]], outdir: Path, highlight: Optional[str]):
    # gather usable runs
    rows = []
    for runname, hist in runs:
        nums = effect_numbers(hist)
        if nums is None:
            continue
        if nums["Patched"] is None and nums["Scrubbed"] is None and nums["Ablated"] is None:
            continue
        rows.append((runname, nums, hist))
    if not rows:
        print("[warn] no evaluable runs for effect portraits.")
        return

    n = len(rows)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6*ncol + 0.6, 3.6*nrow + 0.8), squeeze=False)
    hi = choose_highlight_idx(rows, highlight)

    for idx, (runname, nums, hist) in enumerate(rows):
        r, c = divmod(idx, ncol)
        ax = axes[r][c]
        base = nums["BASE"]; ft = nums["FT"]
        patched = nums["Patched"]; scrubbed = nums["Scrubbed"]; ablated = nums["Ablated"]
        allscrub = nums["AllScrub"]; rand = nums["RandomMask"]

        # y-range padding
        y_vals = [v for v in [base, ft, patched, scrubbed, ablated, allscrub, rand] if isinstance(v, (int, float))]
        ymin = min(y_vals) - 0.5
        ymax = max(y_vals) + 0.5

        # x positions for markers
        x_scrub = 0.8; x_abl = 1.2
        # reference lines
        ax.axhline(base, color="#bfbfbf", linestyle="--", linewidth=1.0, label="BASE")
        ax.axhline(ft,   color="#d0d0d0", linestyle=":",  linewidth=1.0, label="FT")

        # arrows & points
        is_hi = (idx == hi)
        col_suff = FOREST if is_hi else "#7f7f7f"
        col_nec  = FOREST if is_hi else "#9a9a9a"

        # sufficiency: BASE -> Scrubbed (or Patched if Scrubbed missing)
        suff_src = scrubbed if isinstance(scrubbed, (int, float)) else patched
        if isinstance(suff_src, (int, float)):
            ax.plot([x_scrub], [suff_src], marker="o", color=col_suff)
            ax.annotate("",
                xy=(x_scrub, suff_src), xytext=(x_scrub, base),
                arrowprops=dict(arrowstyle="->", color=col_suff, lw=2.2 if is_hi else 1.8))
        # necessity: FT -> Ablated
        if isinstance(ablated, (int, float)):
            ax.plot([x_abl], [ablated], marker="o", color=col_nec)
            ax.annotate("",
                xy=(x_abl, ablated), xytext=(x_abl, ft),
                arrowprops=dict(arrowstyle="->", color=col_nec, lw=2.2 if is_hi else 1.8))

        # optional controls
        if isinstance(allscrub, (int, float)):
            ax.plot([0.55], [allscrub], marker="s", color="#cfcfcf", ms=6, label="All-scrub")
        if isinstance(rand, (int, float)):
            ax.plot([1.45], [rand], marker="^", color="#b5b5b5", ms=6, label="Random")

        # labels: % explained
        s, n = suff_nec_from_history(hist)
        if s is not None:
            ax.text(x_scrub, (base + (suff_src if isinstance(suff_src, (int, float)) else base))/2,
                    f"Suff {100*s:.0f}%", ha="center", va="center",
                    color=col_suff, fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.5))
        if n is not None:
            ax.text(x_abl, (ft + (ablated if isinstance(ablated, (int, float)) else ft))/2,
                    f"Nec {100*n:.0f}%", ha="center", va="center",
                    color=col_nec, fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.5))

        ax.set_xlim(0.4, 1.6)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([0.8, 1.2])
        ax.set_xticklabels(["Scrubbed/Patched", "Ablated"])
        ax.set_ylabel("sKL (lower is better)")
        ax.set_title(runname, fontsize=10)
        ax.grid(True, axis="y", color=GRID, linewidth=0.7)

    # hide spare axes
    for k in range(n, nrow*ncol):
        r, c = divmod(k, ncol)
        axes[r][c].axis("off")

    fig.suptitle("Effect Portraits — Before->After causal impact per run", y=0.995, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir / "effect_portraits_all.png", dpi=170)
    plt.close(fig)

# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True, help="Where to save plots")
    ap.add_argument("--highlight", type=str, default=None, help="Run name to highlight (defaults to highest sufficiency)")
    ap.add_argument("paths", nargs="+", help="history.json files or run directories")
    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)
    runs = load_histories(args.paths)
    if not runs:
        print("[error] no valid histories given.")
        return

    # Combined charts
    plot_ladder_all(runs, outdir)
    plot_suff_nec_all(runs, outdir)
    plot_portability_all(runs, outdir)
    plot_gate_heatmap_all(runs, outdir)
    plot_effect_portraits_all(runs, outdir, highlight=args.highlight)

if __name__ == "__main__":
    main()

