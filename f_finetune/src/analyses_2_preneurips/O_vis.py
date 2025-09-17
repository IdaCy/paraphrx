#!/usr/bin/env python3

"""
python3 f_finetune/src/analyses_2_preneurips/O_vis.py \
  --in_dir f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2/O2_ex8_2/DAMPENING \
  --out f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2/O2_ex8_2/DAMPENING/DAMPENING_FIG_D1.redraw.png
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# I/O helpers

def find_csv(path_dir, preferred_name):
    cand = os.path.join(path_dir, preferred_name)
    if os.path.isfile(cand):
        return cand
    alt = {
        "dampening_curves.csv": ["curves.csv"],
        "alpha_sweep.csv": ["alphas.csv", "alpha.csv"],
        "survival.csv": ["surv.csv"],
    }
    for a in alt.get(preferred_name, []):
        cand2 = os.path.join(path_dir, a)
        if os.path.isfile(cand2):
            return cand2
    return None


def load_csvs(in_dir, curves_csv=None, alpha_csv=None, survival_csv=None):
    if curves_csv or alpha_csv or survival_csv:
        curves = pd.read_csv(curves_csv) if curves_csv else pd.DataFrame()
        alphas = pd.read_csv(alpha_csv)  if alpha_csv  else pd.DataFrame()
        surv   = pd.read_csv(survival_csv) if survival_csv else pd.DataFrame()
        return curves, alphas, surv

    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Directory not found: {in_dir}")

    curves_path = find_csv(in_dir, "dampening_curves.csv")
    alpha_path  = find_csv(in_dir, "alpha_sweep.csv")
    surv_path   = find_csv(in_dir, "survival.csv")

    if curves_path is None and alpha_path is None and surv_path is None:
        raise FileNotFoundError(f"No expected CSVs found in: {in_dir}")

    curves = pd.read_csv(curves_path) if curves_path else pd.DataFrame()
    alphas = pd.read_csv(alpha_path)  if alpha_path  else pd.DataFrame()
    surv   = pd.read_csv(surv_path)   if surv_path   else pd.DataFrame()
    return curves, alphas, surv


# Plotting

def replot(curves_df, alphas_df, surv_df, out_path, dpi):
    # Colors (swapped for FT wrong / BASE patched)
    COLORS = {
        "FT patched": "#04129B",   # strong blue
        "FT baseline": "#9496B4",  # light gray
        "FT wrong": "#2D2E39",     # super dark gray (swapped)
        "FT noise": "#5A5B6F",     # medium gray
        "BASE patched": "#704EEB", # violet (swapped)
    }

    def plot_left(ax_left):
        if len(curves_df):
            sub = curves_df[curves_df.get("site", "layer_input") == "layer_input"]

            # BASE patched first (behind)
            sub_base = sub[sub.get("model", "FT") == "BASE"]
            if len(sub_base):
                patched_base = sub_base.groupby("layer_k")["kl_patched_correct"].mean()
                if len(patched_base):
                    ax_left.plot(
                        patched_base.index, patched_base.values,
                        marker="s", label="BASE patched",
                        color=COLORS["BASE patched"],
                    )

            # FT lines next; FT patched last (on top)
            sub_ft = sub[sub.get("model", "FT") == "FT"]
            if len(sub_ft):
                # FT wrong
                if "kl_patched_wrong" in sub_ft.columns:
                    wrong_ft = sub_ft.groupby("layer_k")["kl_patched_wrong"].mean()
                    if len(wrong_ft):
                        ax_left.plot(
                            wrong_ft.index, wrong_ft.values,
                            linestyle=":", marker=".",
                            label="FT wrong",
                            color=COLORS["FT wrong"],
                        )
                # FT noise
                if "kl_patched_noise" in sub_ft.columns:
                    noise_ft = sub_ft.groupby("layer_k")["kl_patched_noise"].mean()
                    if len(noise_ft):
                        ax_left.plot(
                            noise_ft.index, noise_ft.values,
                            linestyle="--", marker="x",
                            label="FT noise",
                            color=COLORS["FT noise"],
                        )
                # FT baseline
                if "kl_baseline" in sub_ft.columns and len(sub_ft["kl_baseline"]):
                    base_level = sub_ft.groupby("layer_k")["kl_baseline"].mean().mean()
                    ax_left.axhline(
                        base_level, linestyle="--",
                        label="FT baseline",
                        color=COLORS["FT baseline"], linewidth=1.5,
                    )
                # FT patched (topmost)
                patched_ft = sub_ft.groupby("layer_k")["kl_patched_correct"].mean()
                if len(patched_ft):
                    ax_left.plot(
                        patched_ft.index, patched_ft.values,
                        marker="o", label="FT patched",
                        color=COLORS["FT patched"],
                    )

        ax_left.set_title("Specificity & controls")
        ax_left.set_xlabel("Layer k")
        ax_left.set_ylabel("Mean symmetric KL")
        ax_left.grid(alpha=0.2)
        ax_left.legend(fontsize=8, loc="upper left")

    def plot_mid(ax_mid):
        if len(alphas_df):
            alphas_ft = alphas_df[alphas_df.get("model", "FT") == "FT"] if "model" in alphas_df.columns else alphas_df
            if len(alphas_ft):
                ks = sorted(set(alphas_ft["layer_k"].tolist()))
                for idx, k in enumerate(ks):
                    subk = alphas_ft[alphas_ft["layer_k"] == k]
                    if not len(subk):
                        continue
                    g = subk.groupby("alpha")["kl_alpha"].mean().reset_index()
                    # draw non-FT (BASE color) first if idx > 0
                    if idx > 0:
                        ax_mid.plot(g["alpha"], g["kl_alpha"], marker="o",
                                    label=f"layer {k}", color=COLORS["BASE patched"])
                # FT last (first layer in list)
                if ks:
                    subk = alphas_ft[alphas_ft["layer_k"] == ks[0]]
                    g = subk.groupby("alpha")["kl_alpha"].mean().reset_index()
                    ax_mid.plot(g["alpha"], g["kl_alpha"], marker="o",
                                label=f"layer {ks[0]}", color=COLORS["FT patched"])
                ax_mid.legend(fontsize=8)
        ax_mid.set_title("α-sweep (FT)")
        ax_mid.set_xlabel("α")
        ax_mid.set_ylabel("Mean KL")
        ax_mid.set_xlim(0, 1)
        ax_mid.grid(alpha=0.2)

    def plot_right(ax_right):
        if len(surv_df):
            if "model" in surv_df.columns and "BASE" in surv_df["model"].unique():
                models = ["BASE", "FT"]  # BASE first (behind), FT last (top)
            else:
                models = ["FT"]

            for m in models:
                subm = surv_df[surv_df["model"] == m] if "model" in surv_df.columns else surv_df
                if not len(subm):
                    continue
                g = subm.groupby("layer_k")["survival"].mean().reset_index()
                color = COLORS["FT patched"] if m == "FT" else COLORS["BASE patched"]
                ax_right.plot(g["layer_k"], g["survival"], marker="o", label=m, color=color)
            ax_right.legend(fontsize=8)

        ax_right.set_title("Activation survival")
        ax_right.set_xlabel("Layer k")
        ax_right.set_ylabel("Relative L2 difference (vs. layer-6 residual input)")
        ax_right.set_ylim(bottom=0)
        ax_right.grid(alpha=0.2)

    # 3-panel figure (original)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    plot_left(axs[0])
    plot_mid(axs[1])
    plot_right(axs[2])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote figure: {out_path}")

    # 2-panel figure (no middle)
    out_noalpha = _derive_noalpha_path(out_path)
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 4))
    plot_left(axs2[0])
    plot_right(axs2[1])
    fig2.tight_layout()
    fig2.savefig(out_noalpha, dpi=dpi)
    plt.close(fig2)
    print(f"Wrote figure (no α-sweep): {out_noalpha}")


def _derive_noalpha_path(out_path: str) -> str:
    p = Path(out_path)
    if p.suffix:
        return str(p.with_name(p.stem + ".noalpha" + p.suffix))
    # no extension case
    return out_path + ".noalpha"


# CLI

def main():
    p = argparse.ArgumentParser(description="Re-plot dampening figure from saved CSVs (no model needed).")
    p.add_argument("--in_dir", type=str, default=".", help="Directory containing the CSVs (…/DAMPENING).")
    p.add_argument("--curves_csv", type=str, default=None, help="Optional explicit path to dampening_curves.csv")
    p.add_argument("--alpha_csv", type=str, default=None, help="Optional explicit path to alpha_sweep.csv")
    p.add_argument("--survival_csv", type=str, default=None, help="Optional explicit path to survival.csv")
    p.add_argument("--out", type=str, default=None, help="Output PNG path (defaults to <in_dir>/DAMPENING_FIG_D1.redraw.png)")
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    out_path = args.out or os.path.join(args.in_dir, "DAMPENING_FIG_D1.redraw.png")

    try:
        curves, alphas, surv = load_csvs(
            args.in_dir,
            curves_csv=args.curves_csv,
            alpha_csv=args.alpha_csv,
            survival_csv=args.survival_csv,
        )
    except Exception as e:
        print(f"ERROR loading CSVs: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        replot(curves, alphas, surv, out_path, dpi=args.dpi)
    except Exception as e:
        print(f"ERROR while plotting: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
