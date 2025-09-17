#!/usr/bin/env python3
"""
python3 f_finetune/src/analyses_2_preneurips/O_vis.py \
  --csv_dir f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2_preneurips/O2_ex8_2/DAMPENING
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

# Styling helpers

def build_palette(primary=None):
    """
    Provides three distinct forest-green hues (bright/mid/dark) and a long gray ramp.
    Also supplies 'primary' for legacy callers (defaults to mid green or user-provided).
    """
    palette = {
        # three clearly separated forest greens
        "green_bright": "#4CAF50",  # bright foresty green
        "green_mid":    "#228B22",  # classic forest green
        "green_dark":   "#0B5D0B",  # dark evergreen

        # long gray ramp (dark -> light)
        "gray0": "#111111",
        "gray1": "#2B2B2B",
        "gray2": "#444444",
        "gray3": "#666666",
        "gray4": "#888888",
        "gray5": "#AAAAAA",
        "gray6": "#CCCCCC",
        "gray7": "#E0E0E0",
    }

    # convenience lists
    palette["greens"] = [palette["green_bright"], palette["green_mid"], palette["green_dark"]]
    palette["grays"]  = [palette[f"gray{i}"] for i in range(0, 8)]

    # legacy/CLI compatibility: set 'primary'
    if primary and primary.strip():
        # allow either a named option or a hex
        named = {
            "bright": palette["green_bright"],
            "mid":    palette["green_mid"],
            "dark":   palette["green_dark"],
        }
        palette["primary"] = named.get(primary.lower(), primary)  # hex or unknown string -> use as-is
    else:
        palette["primary"] = palette["green_mid"]

    return palette

    # convenience lists
    palette["greens"] = [palette["green_bright"], palette["green_mid"], palette["green_dark"]]
    palette["grays"]  = [palette[f"gray{i}"] for i in range(0, 8)]
    return palette

def configure_matplotlib():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "figure.dpi": 120,
    })

# Plotters

def panel_specificity(curves_df, ax, palette):
    """
    Recreates panel (i): per-layer mean KL for various conditions.
    We place the legend outside the axes to avoid covering curves.
    """
    if curves_df is None or len(curves_df) == 0:
        ax.text(0.5, 0.5, "No data in dampening_curves.csv", ha="center", va="center", transform=ax.transAxes)
        return

    # determine which patch sites are present to loop deterministically
    patch_sites = list(sorted(curves_df["site"].dropna().unique().tolist()))
    # For each site, draw FT patched (primary), FT baseline (gray), FT wrong (dotted gray), FT noise (x gray),
    # and BASE patched (darker gray square) if present.
    handles = []
    labels = []

    for site_idx, site in enumerate(patch_sites):
        sub = curves_df[curves_df["site"] == site]
        sub_ft = sub[sub["model"] == "FT"]

        # FT baseline (mean over layer, gray dashed)
        if "kl_baseline" in sub_ft.columns and len(sub_ft):
            g = sub_ft.groupby("layer_k")["kl_baseline"].mean()
            h, = ax.plot(
                g.index, g.values,
                linestyle="--", color=palette["gray2"],
                label=f"FT baseline ({site})"
            )
            handles.append(h); labels.append(h.get_label())

        # FT patched (correct, one green per site)
        if "kl_patched_correct" in sub_ft.columns and len(sub_ft):
            g = sub_ft.groupby("layer_k")["kl_patched_correct"].mean()
            marker = ["o", "D", "^"][site_idx % 3]
            site_colors = {
                "layer_input": palette["green_bright"],
                "attn_in":     palette["green_mid"],
                "mlp_in":      palette["green_dark"],
            }
            h, = ax.plot(
                g.index, g.values,
                marker=marker, linewidth=2.5,
                color=site_colors.get(site, palette["green_mid"]),
                label=f"FT patched ({site})"
            )
            handles.append(h); labels.append(h.get_label())


        # FT wrong
        if "kl_patched_wrong" in sub_ft.columns and sub_ft["kl_patched_wrong"].notna().any():
            g = sub_ft.dropna(subset=["kl_patched_wrong"]).groupby("layer_k")["kl_patched_wrong"].mean()
            h, = ax.plot(g.index, g.values, linestyle=":", marker=".", color=palette["gray1"],
                         label=f"FT wrong ({site})")
            handles.append(h); labels.append(h.get_label())

        # FT noise
        if "kl_patched_noise" in sub_ft.columns and sub_ft["kl_patched_noise"].notna().any():
            g = sub_ft.dropna(subset=["kl_patched_noise"]).groupby("layer_k")["kl_patched_noise"].mean()
            h, = ax.plot(g.index, g.values, linestyle="-.", marker="x", color=palette["gray3"],
                         label=f"FT noise ({site})")
            handles.append(h); labels.append(h.get_label())

        # BASE patched
        sub_base = sub[sub["model"] == "BASE"]
        if len(sub_base):
            g = sub_base.groupby("layer_k")["kl_patched_correct"].mean()
            h, = ax.plot(g.index, g.values, marker="s", linewidth=2.0, color=palette["gray0"],
                         label=f"BASE patched ({site})")
            handles.append(h); labels.append(h.get_label())

    ax.set_title("(i) Specificity & controls")
    ax.set_xlabel("Layer k")
    ax.set_ylabel("Mean symmetric KL")
    # Legend outside to the right
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=1, frameon=True)

def panel_alpha(alphas_df, ax, palette, layers_to_show=None, site_filter=None):
    """
    Recreates panel (ii): alpha sweep curves for selected layers (FT), optionally filtering by site
    """
    if alphas_df is None or len(alphas_df) == 0:
        ax.text(0.5, 0.5, "No data in alpha_sweep.csv", ha="center", va="center", transform=ax.transAxes)
        return

    # filter to FT and optional site
    sub = alphas_df[alphas_df["model"] == "FT"].copy()
    if site_filter is not None:
        sub = sub[sub["site"] == site_filter]

    if len(sub) == 0:
        msg = "No FT rows"
        if site_filter is not None:
            msg += f" for site={site_filter!r}"
        ax.text(0.5, 0.5, msg + " in alpha_sweep.csv", ha="center", va="center", transform=ax.transAxes)
        return

    layers = sorted(sub["layer_k"].dropna().unique().tolist())
    if layers_to_show:
        layers = [k for k in layers if k in set(layers_to_show)]
        if not layers:
            layers = sorted(sub["layer_k"].dropna().unique().tolist())

    # color ramp: same primary but vary intensity using grays for variety
    greens_cycle = palette["greens"]  # [bright, mid, dark]
    grays_cycle  = [palette[g] for g in ["gray1","gray2","gray3","gray4","gray5"]]
    color_cycle  = greens_cycle + grays_cycle
    markers = ["o","s","^","D","v","P"]
    lines = []

    for idx, k in enumerate(layers):
        g = sub[sub["layer_k"] == k].groupby("alpha")["kl_alpha"].mean().reset_index().sort_values("alpha")
        c = color_cycle[idx % len(color_cycle)]
        m = markers[idx % len(markers)]
        h, = ax.plot(g["alpha"], g["kl_alpha"], marker=m, linewidth=2.5, color=c, label=f"layer {k}")
        lines.append(h)

    ax.set_title("(ii) α-sweep (FT)")
    ax.set_xlabel("α")
    ax.set_ylabel("Mean KL")
    ax.set_xlim(0, 1)
    ax.legend(lines, [h.get_label() for h in lines], loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, ncol=1, frameon=True)

def panel_survival(surv_df, ax, palette):
    """
    Recreates panel (iii): survival curve mean ||Δh_k|| / ||Δh_anchor|| for FT (and BASE if present).
    """
    if surv_df is None or len(surv_df) == 0:
        ax.text(0.5, 0.5, "No data in survival.csv", ha="center", va="center", transform=ax.transAxes)
        return

    models = []
    if "FT" in surv_df["model"].unique():
        models.append(("FT", palette["primary"], "o"))
    for other in [m for m in surv_df["model"].unique() if m != "FT"]:
        models.append((other, palette["gray0"], "s"))

    handles = []
    labels = []
    for (mname, color, marker) in models:
        sub = surv_df[surv_df["model"] == mname]
        g = sub.groupby("layer_k")["survival"].mean().reset_index().sort_values("layer_k")
        h, = ax.plot(g["layer_k"], g["survival"], marker=marker, linewidth=2.5, color=color, label=mname)
        handles.append(h); labels.append(mname)

    ax.set_title("(iii) Activation survival")
    ax.set_xlabel("Layer k")
    ax.set_ylabel("mean ||Δh_k|| / ||Δh_anchor||")
    ax.set_ylim(bottom=0)
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=1, frameon=True)

# Main

def main():
    parser = argparse.ArgumentParser(description="Redraw downstream dampening panels from CSVs.")
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory containing dampening_curves.csv, alpha_sweep.csv, survival.csv")
    parser.add_argument("--out", type=str, default=None, help="Output path for the combined figure (PNG). Defaults to <csv_dir>/DAMPENING_PANELS_redrawn.png")
    parser.add_argument("--figwidth", type=float, default=20.0, help="Figure width in inches")
    parser.add_argument("--figheight", type=float, default=6.5, help="Figure height in inches")
    parser.add_argument("--primary", type=str, default="#228B22", help="Primary color (forest green default)")
    parser.add_argument("--alpha_layers", type=str, default=None, help="Comma-separated subset of layers to show in panel (ii). If omitted, show all present.")
    parser.add_argument("--alpha_site", type=str, default="layer_input", help="Which patch site to show in panel (ii). Options: layer_input, mlp_in, attn_in")
    args = parser.parse_args()

    configure_matplotlib()
    palette = build_palette(primary=args.primary)

    curves_path = os.path.join(args.csv_dir, "dampening_curves.csv")
    alphas_path = os.path.join(args.csv_dir, "alpha_sweep.csv")
    surv_path = os.path.join(args.csv_dir, "survival.csv")

    # Load (tolerant if missing)
    def read_csv_safe(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return None

    curves_df = read_csv_safe(curves_path)
    alphas_df = read_csv_safe(alphas_path)
    surv_df = read_csv_safe(surv_path)

    layers_to_show = None
    if args.alpha_layers:
        try:
            layers_to_show = [int(x) for x in args.alpha_layers.split(",") if x.strip()]
        except Exception:
            layers_to_show = None

        # Make 1x3 composite figure
    fig, axs = plt.subplots(1, 3, figsize=(args.figwidth, args.figheight))
    panel_specificity(curves_df, axs[0], palette)
    panel_alpha(alphas_df, axs[1], palette, layers_to_show=layers_to_show, site_filter=args.alpha_site)  # <-- pass it here
    panel_survival(surv_df, axs[2], palette)

    fig.tight_layout()
    out_path = args.out or os.path.join(args.csv_dir, "DAMPENING_PANELS_redrawn.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    main()
