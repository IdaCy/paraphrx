#!/usr/bin/env python3
"""
FILTER_KEYS="instruct_coord_to_subord,instruct_double_negative,\
instruct_polite_request,instruct_dramatic,instruct_leet_speak,\
instruct_formal_memo,instruct_future_tense,instruct_joke,\
instruct_one_typo_punctuation,instruct_output_markdown,\
instruct_sardonic"

FILTER_TAGS="format,length_brevity,style\
dialect,emotion,genre,obfuscation,playfulness,\
slangy,syntax_tweeks,tone"
FILTER_KEYS="\
instruct_dramatic,instruct_leet_speak,\
instruct_future_tense,instruct_joke"
python e_eval/src/eval_paras3.py \
  --prompts a_data/alpaca/paraphrases_500.json \
  --scores Alpaca=c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
           GSM8K=c_assess_inf/output/gsm8k_answer_scores/gemma-2-2b-it.json \
           MMLU=c_assess_inf/output/mmlu_answer_scores/gemma-2-2b-it.json \
  --tags-json a_data/paraphrases_tagged.json \
  --content-preservation a_data/alpaca/equi_scores_paraphrases_500.json \
  --output-dir b_tests/xxgraphics/eval_paras/out17/ \
  --filter-keys "$FILTER_KEYS" \
  --filter-tags "$FILTER_TAGS" \
  --max-samples 450
"""
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set, Any

import matplotlib.pyplot as plt
import numpy as np

def _norm_tag(s: str) -> str:
    return (s or "").strip().lower()

#FOREST_GREEN = "#228B22"
FOREST_GREEN = "#4B6EAF"
DARK_GRAY = "#2B2B2B"
LIGHT_GRAY = "#E5E5E5"

def parse_scores_arg(scores_list: List[str]) -> List[Tuple[str, str]]:
    """Parse scores entries like 'Alpaca=path/to.json' preserving order."""
    pairs = []
    for item in scores_list:
        if "=" not in item:
            raise ValueError(f"--scores entries must be NAME=PATH; got: {item}")
        name, path = item.split("=", 1)
        pairs.append((name.strip(), path.strip()))
    if len(pairs) != 3:
        # Not strictly required, but the spec expects three datasets
        print(f"[warn] Expected 3 datasets in --scores; got {len(pairs)}. Proceeding anyway.", flush=True)
    return pairs

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def prettify_key(key: str) -> str:
    """Remove instruct_/instruction_ prefix. Special-case instruction_original -> Original
       E.g., instruct_polite_request -> Polite Request; instruct_rude -> Rude; instruction_* -> * (title)"""
    if key == "instruction_original":
        return "Original"
    if key.startswith("instruct_"):
        core = key[len("instruct_"):]
    elif key.startswith("instruction_"):
        core = key[len("instruction_"):]
    else:
        core = key
    # special-cases mentioned by user
    if key == "instruct_polite":
        return "Polite"
    if key == "instruct_rude":
        return "Rude"
    # general title-case; keep common acronyms nicer
    parts = core.split("_")
    titled = []
    for p in parts:
        if p.lower() == "aave":
            titled.append("AAVE")
        else:
            titled.append(p.capitalize())
    return " ".join(titled)

def collect_styles_from_prompts(prompts: List[dict]) -> Set[str]:
    styles = set()
    for obj in prompts:
        for k in obj.keys():
            if k.startswith("instruct_") or k == "instruction_original":
                styles.add(k)
    return styles

def build_cp_maps(cp_list: List[dict]) -> Tuple[Dict[int, Dict[str, int]], Dict[str, int]]:
    """Return: (cp_by_id[pid][style]=score, pass_counts[style]=count of score in {4,5})."""
    cp_by_id: Dict[int, Dict[str, int]] = {}
    pass_counts: Dict[str, int] = {}
    for item in cp_list:
        pid = item.get("prompt_count")
        scores = item.get("scores", {})
        if pid is None:
            continue
        cp_by_id[pid] = {}
        for style, sc in scores.items():
            try:
                sc_int = int(sc)
            except Exception:
                # tolerate e.g. float
                sc_int = int(round(float(sc)))
            cp_by_id[pid][style] = sc_int
            if sc_int in (4, 5):
                pass_counts[style] = pass_counts.get(style, 0) + 1
    return cp_by_id, pass_counts

def build_scores_map(scores_list: List[dict]) -> Dict[int, Dict[str, List[float]]]:
    """Map prompt_count -> {style -> [10 scores]}"""
    out: Dict[int, Dict[str, List[float]]] = {}
    for item in scores_list:
        pid = item.get("prompt_count")
        if pid is None:
            continue
        out[pid] = {}
        for k, v in item.items():
            if k in ("prompt_count",):
                continue
            if isinstance(v, list) and len(v) == 10:
                out[pid][k] = [float(x) for x in v]
    return out

def tf_from(scores_10: List[float]) -> float:
    return float(scores_10[0]) if scores_10 else float("nan")

def intersect_prompt_ids(*maps: Iterable[Dict[int, Any]]) -> List[int]:
    sets = [set(m.keys()) for m in maps if m]
    if not sets:
        return []
    inter = set.intersection(*sets)
    return sorted(list(inter))

def choose_prompt_ids(ids: List[int], max_samples: int = None) -> List[int]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(ids):
        return ids
    return ids[:max_samples]

def select_styles(all_styles: Set[str], pass_counts: Dict[str, int], min_ok: int, filter_keys: Set[str]) -> List[str]:
    """Return ordered list with 'instruction_original' first, then other selected styles sorted by pretty name."""
    qualified = {s for s in all_styles if s.startswith("instruct_") and pass_counts.get(s, 0) >= min_ok}
    if filter_keys:
        # intersect while preserving only instruct_* (Original added later)
        qualified = {s for s in qualified if s in filter_keys}
    # Always include original
    final_styles = ["instruction_original"] + sorted(qualified, key=lambda s: prettify_key(s))
    return final_styles

def build_dataset_tf_vectors(
    dataset_scores: Dict[int, Dict[str, List[float]]],
    selected_styles: List[str],
    cp_by_id: Dict[int, Dict[str, int]],
    prompt_ids: List[int],
) -> Dict[str, List[float]]:
    """For each style, collect TF scores for those prompt_ids that have cp score in {4,5} for that style (except Original)."""
    result: Dict[str, List[float]] = {style: [] for style in selected_styles}
    for pid in prompt_ids:
        per_id = dataset_scores.get(pid, {})
        for style in selected_styles:
            # 'Original' has no cp score filter
            if style != "instruction_original":
                cp_style_score = cp_by_id.get(pid, {}).get(style, None)
                if cp_style_score not in (4, 5):
                    continue
            scores10 = per_id.get(style)
            if scores10 is None:
                continue
            tf = tf_from(scores10)
            if not (math.isnan(tf)):
                result[style].append(tf)
    return result

def grouped_boxplot_by_styles(
    tf_data_per_dataset: List[Tuple[str, Dict[str, List[float]]]],
    styles_order: List[str],
    out_path: Path,
    title: str,
    legend_loc: str = "upper right",
):
    """Draw grouped boxplots: x=styles, 3 boxes per style (datasets)."""
    plt.figure(figsize=(max(10, 1.2 * len(styles_order)), 6))
    ax = plt.gca()

    num_datasets = len(tf_data_per_dataset)
    width = 0.18
    gap = 0.10
    positions_base = np.arange(len(styles_order))

    # Colors/hatches per dataset: first = green, second = dark gray, third = light gray + hatch
    facecolors = [FOREST_GREEN, DARK_GRAY, LIGHT_GRAY]
    hatches = [None, None, "///"]

    # For legend means
    means_for_legend = []

    # Plot per dataset
    for di, (dname, tf_map) in enumerate(tf_data_per_dataset):
        pos = positions_base + (di - (num_datasets-1)/2) * (width + 0.02)

        data = [tf_map.get(style, []) for style in styles_order]

        bp = ax.boxplot(
            data,
            positions=pos,
            widths=width,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            whis=1.5,
            manage_ticks=False
        )

        # Style
        fc = facecolors[di % len(facecolors)]
        for patch in bp['boxes']:
            patch.set_facecolor(fc)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.8)
            if di < len(hatches) and hatches[di]:
                patch.set_hatch(hatches[di])
        for med in bp['medians']:
            med.set_color("#222222")
            med.set_linewidth(1.2)
        for mean in bp['means']:
            mean.set_color("#111111")
            mean.set_linewidth(1.2)
        for w in bp['whiskers']:
            w.set_color("#333333")
            w.set_linewidth(0.8)
        for cap in bp['caps']:
            cap.set_color("#333333")
            cap.set_linewidth(0.8)
        for fl in bp['fliers']:
            fl.set_markerfacecolor("#666666")
            fl.set_markeredgecolor("#666666")
            fl.set_alpha(0.5)

        # Dataset-wide mean TF across all shown style data points
        flat = [v for style in styles_order for v in tf_map.get(style, [])]
        mu = np.nan if not flat else float(np.mean(flat))
        means_for_legend.append((dname, mu, fc, hatches[di] if di < len(hatches) else None))

    # Axes / labels
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel("TF score", fontsize=12)
    # X tick labels: prettified keys
    pretty_labels = [prettify_key(s) for s in styles_order]
    ax.set_xticks(positions_base)
    ax.set_xticklabels(pretty_labels, rotation=30, ha="right")

    ax.set_ylim(0, 10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # Legend: include means
    from matplotlib.patches import Patch
    legend_patches = []
    for name, mu, fc, hatch in means_for_legend:
        label = f"{name} (μ={mu:.2f})" if not math.isnan(mu) else f"{name}"
        patch = Patch(facecolor=fc, edgecolor="#333333", hatch=hatch if hatch else None, label=label)
        legend_patches.append(patch)
    ax.legend(handles=legend_patches, loc=legend_loc, frameon=False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def grouped_boxplot_by_tags(
    tf_data_per_dataset_by_style: List[Tuple[str, Dict[str, List[float]]]],
    styles_order: List[str],
    tags_map: Dict[str, List[str]],
    out_path: Path,
    title: str,
    filter_tags: set | None = None,
):
    """Aggregate per tag across all selected styles. 'Original' becomes its own pseudotag 'Original'
       Tag filtering is case/whitespace-insensitive; Original is always shown."""
    # Build canonical tag names and a normalization map
    canonical_for_norm = {}  # norm_tag -> canonical as first seen
    def canon(tag: str) -> str:
        n = _norm_tag(tag)
        if n not in canonical_for_norm:
            canonical_for_norm[n] = tag
        return canonical_for_norm[n]

    # Normalise requested filter tags (if any)
    norm_filter = set(_norm_tag(t) for t in (filter_tags or set()) if t)

    datasets_tag_map: List[Tuple[str, Dict[str, List[float]]]] = []
    all_tags = set()
    for dname, style_map in tf_data_per_dataset_by_style:
        tmap: Dict[str, List[float]] = {}
        # Ensure 'Original' aggregation bucket exists
        tmap.setdefault("Original", [])
        for style in styles_order:
            tfs = style_map.get(style, [])
            if style == "instruction_original":
                tmap["Original"].extend(tfs)
                all_tags.add("Original")
                continue
            # Map each style to its tags, normalize, then filter if requested
            for tag in tags_map.get(style, []) or []:
                ctag = canon(tag)
                if norm_filter and _norm_tag(ctag) not in norm_filter:
                    continue
                tmap.setdefault(ctag, []).extend(tfs)
                all_tags.add(ctag)
        datasets_tag_map.append((dname, tmap))

    # If user asked to filter tags, keep only those (plus Original)
    if norm_filter:
        all_tags = {t for t in all_tags if (_norm_tag(t) in norm_filter) or (t == "Original")}

    # Sort with Original first then alpha
    tags_sorted = sorted(all_tags, key=lambda s: ("~" if s=="Original" else "") + s.lower())

    # Diagnostics to stdout
    try:
        print(f"[info] Tags in plot: {tags_sorted}", flush=True)
    except Exception:
        pass

    # Reuse grouped boxplot drawing; legend bottom-right for tags
    grouped_boxplot_by_styles(datasets_tag_map, tags_sorted, out_path, title, legend_loc="lower right")

def radar_prepare_axes(labels: List[str]):
    """Return angles (closed) and set up polar axes."""
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 10)
    ax.grid(color="#AAAAAA", linestyle="--", linewidth=0.6, alpha=0.6)
    return ax, angles

def radar_plot(ax, angles, values: List[float], line_color: str, fill_alpha: float = 0.25, label: str = None):
    vals = list(values) + values[:1]
    ax.plot(angles, vals, linewidth=1.6, color=line_color, label=label)
    ax.fill(angles, vals, color=line_color, alpha=fill_alpha)

def compute_mean_10(scores_map: Dict[int, Dict[str, List[float]]], style: str, cp_by_id: Dict[int, Dict[str, int]], prompt_ids: List[int]) -> List[float]:
    rows = []
    for pid in prompt_ids:
        sc = scores_map.get(pid, {}).get(style)
        if sc is None:
            continue
        if style != "instruction_original":
            cp = cp_by_id.get(pid, {}).get(style)
            if cp not in (4, 5):
                continue
        rows.append(sc)
    if not rows:
        return [float("nan")]*10
    arr = np.array(rows, dtype=float)
    return list(np.nanmean(arr, axis=0))

def distinct_gray(i: int, total: int) -> str:
    """Generate a set of distinct gray hues across [DARK_GRAY..LIGHT_GRAY]"""
    if total <= 1:
        return DARK_GRAY
    t = i / (total - 1)
    # interpolate between dark and light
    def interp_hex(h1, h2, t):
        c1 = tuple(int(h1[i:i+2], 16) for i in (1,3,5))
        c2 = tuple(int(h2[i:i+2], 16) for i in (1,3,5))
        c = tuple(int(round(c1[j] + (c2[j]-c1[j])*t)) for j in range(3))
        return "#" + "".join(f"{v:02X}" for v in c)
    return interp_hex(DARK_GRAY, LIGHT_GRAY, t)

def main():
    parser = argparse.ArgumentParser(description="Paraphrase robustness graphics")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSON (Alpaca-style paraphrases)")
    parser.add_argument("--scores", nargs="+", required=True, help="Dataset scores as NAME=PATH, e.g., Alpaca=... GSM8K=... MMLU=...")
    parser.add_argument("--tags-json", required=True, help="Path to tags JSON mapping instruct_* to tags")
    parser.add_argument("--filter-tags", default="", help="Comma-separated tags; if provided, only these tags are shown in the tag-based plot (Original always included)")
    parser.add_argument("--content-preservation", required=True, help="Path to content-preservation JSON")
    parser.add_argument("--output-dir", required=True, help="Directory to save graphics")
    parser.add_argument("--filter-keys", default="", help="Comma-separated instruct_* keys to include (intersected with threshold filter). 'instruction_original' is always included.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max number of prompt_count IDs to use")
    parser.add_argument("--min-occurrences", type=int, default=200, help="Min # of cp 4/5 occurrences required for a style (default 200)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load JSONs
    prompts = load_json(args.prompts)
    tags_map_raw = load_json(args.tags_json)
    cp_list = load_json(args.content_preservation)

    # Build maps
    all_styles = collect_styles_from_prompts(prompts)
    cp_by_id, pass_counts = build_cp_maps(cp_list)

    # Filter keys (normalize input list to exact keys)
    filter_keys = set([s.strip() for s in args.filter_keys.split(",") if s.strip()])
    # Warn about unknown filter keys
    unknown = [k for k in filter_keys if k not in all_styles]
    if unknown:
        print(f"[warn] Some --filter-keys not in prompts styles and will be ignored: {unknown}", flush=True)
        filter_keys = {k for k in filter_keys if k in all_styles}

    selected_styles = select_styles(all_styles, pass_counts, args.min_occurrences, filter_keys)
    if len(selected_styles) <= 1:
        print("[error] After filtering, only 'Original' remained. Consider adjusting --filter-keys or --min-occurrences.", flush=True)

    # Load scores per dataset, preserving order
    dataset_pairs = parse_scores_arg(args.scores)
    datasets_scores: List[Tuple[str, Dict[int, Dict[str, List[float]]]]] = []
    for name, path in dataset_pairs:
        scores_list = load_json(path)
        ds_map = build_scores_map(scores_list)
        datasets_scores.append((name, ds_map))

    # Common prompt_count IDs present in ALL datasets and the CP file
    common_ids = intersect_prompt_ids(*(dict(m) for _, m in datasets_scores), cp_by_id)
    if not common_ids:
        print("[error] No overlapping prompt_count IDs across datasets and content-preservation JSON.", flush=True)
        return
    # Limit by max samples
    prompt_ids = choose_prompt_ids(common_ids, args.max_samples)

    # Prepare TF vectors per dataset and style
    tf_maps_per_dataset: List[Tuple[str, Dict[str, List[float]]]] = []
    for dname, dmap in datasets_scores:
        tf_map = build_dataset_tf_vectors(dmap, selected_styles, cp_by_id, prompt_ids)
        tf_maps_per_dataset.append((dname, tf_map))

    # Graphic 1: by styles
    out1 = out_dir / "tf_scores_by_dataset_styles.png"
    grouped_boxplot_by_styles(tf_maps_per_dataset, selected_styles, out1, "TF Scores By Dataset")

    # Tags map: tags keyed by instruct_*; ensure default of []
    tags_map = {k: v for k, v in tags_map_raw.items() if isinstance(v, list)}
    out2 = out_dir / "tf_scores_by_dataset_tags.png"
    # Build tag filter set and warn on unknowns
    # Parse and normalize filter tags
    filter_tags_raw = [s for s in (args.filter_tags or '').split(',') if s.strip()]
    filter_tags = set(_norm_tag(s) for s in filter_tags_raw)
    # Build known tag set (normalized)
    known_tags = set()
    for v in tags_map.values():
        if isinstance(v, list):
            known_tags.update(_norm_tag(x) for x in v)
    if filter_tags and not (filter_tags <= known_tags):
        unknown_tags = sorted(list(filter_tags - known_tags))
        if unknown_tags:
            print(f"[warn] Some --filter-tags not recognized in tags JSON (normalized): {unknown_tags}", flush=True)
    grouped_boxplot_by_tags(tf_maps_per_dataset, selected_styles, tags_map, out2, "TF Scores By Dataset (by Tags)", filter_tags=filter_tags)

    # Graphic 3: Radar overlay (Alpaca assumed first dataset in args)
    if not datasets_scores:
        print("[error] No datasets provided for radar charts.", flush=True)
        return
    alpaca_name, alpaca_scores = datasets_scores[0]  # first dataset uses forest green hue
    metric_labels = [
        "Task Fulfilment/Relevance",   # TF (index 0)
        "Usefulness/Actionability",
        "Factual Accuracy/Verifiability",
        "Efficiency, Depth, & Completeness",
        "Reasoning Quality & Transparency",
        "Tone & Likeability",
        "Adaption to Context",
        "Safety & Bias Avoidance",
        "Structuring, Formating, & UX",
        "Creativity"
    ]
    plt.figure(figsize=(8, 8))
    ax, angles = radar_prepare_axes(metric_labels)

    # Original first (green), then others (distinct grays)
    for idx, style in enumerate(selected_styles):
        mean10 = compute_mean_10(alpaca_scores, style, cp_by_id, prompt_ids)
        if all(math.isnan(x) for x in mean10):
            continue
        if style == "instruction_original":
            color = FOREST_GREEN
        else:
            # spread gray hues
            color = distinct_gray(idx-1, max(1, len(selected_styles)-1))
        radar_plot(ax, angles, mean10, line_color=color, fill_alpha=0.25, label=prettify_key(style))

    # Optional: emphasize TF spoke (angle at first axis) in forest green
    rmax = 10
    ax.plot([angles[0], angles[0]], [0, rmax], color=FOREST_GREEN, linewidth=2.0, alpha=0.6)

    ax.set_title(f"Alpaca Radar: All Metrics (N={len(prompt_ids)} prompts)", va='bottom', fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False)
    plt.tight_layout()
    out3 = out_dir / "radar_alpaca_all_styles.png"
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close()

    # Additional: per-style radars (Alpaca only)
    # Original
    for style in selected_styles:
        mean10 = compute_mean_10(alpaca_scores, style, cp_by_id, prompt_ids)
        if all(math.isnan(x) for x in mean10):
            continue
        plt.figure(figsize=(6.8, 6.8))
        ax, angles = radar_prepare_axes(metric_labels)
        # draw all axis lines in gray; TF spoke highlighted in green
        rmax = 10
        ax.plot([angles[0], angles[0]], [0, rmax], color=FOREST_GREEN, linewidth=2.0, alpha=0.6)
        # plot the polygon
        color = FOREST_GREEN if style == "instruction_original" else DARK_GRAY
        radar_plot(ax, angles, mean10, line_color=color, fill_alpha=0.20, label=prettify_key(style))
        ax.set_title(f"Alpaca Radar: {prettify_key(style)} (N={len(prompt_ids)} prompts)", va='bottom', fontsize=13, pad=18)
        ax.legend(loc="upper right", frameon=False)
        plt.tight_layout()
        safe_name = prettify_key(style).lower().replace(" ", "_")
        outp = out_dir / f"radar_alpaca_{safe_name}.png"
        plt.savefig(outp, dpi=200, bbox_inches="tight")
        plt.close()

    # Console summary
    print("[done] Saved:")
    print(f" - {out1}")
    print(f" - {out2}")
    print(f" - {out3}")
    print(f" - {out_dir} / radar_alpaca_*.png")

if __name__ == "__main__":
    main()
