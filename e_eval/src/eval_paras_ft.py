#!/usr/bin/env python3
"""
FILTER_KEYS="instruct_coord_to_subord,\
instruct_dramatic,instruct_leet_speak,\
instruct_formal_memo,instruct_future_tense,instruct_joke,\
instruct_sardonic"

FULL:
FILTER_KEYS="instruct_coord_to_subord,instruct_double_negative,\
instruct_polite_request,instruct_dramatic,instruct_leet_speak,\
instruct_formal_memo,instruct_future_tense,instruct_joke,\
instruct_one_typo_punctuation,instruct_output_markdown,\
instruct_sardonic"

python e_eval/src/eval_paras.py \
  --prompts a_data/alpaca/50k_phrxed.json \
  --scores MAIN-FT=f_finetune/outputs_clean/9x_a1stop_ft/scores512.json \
  --tags-json a_data/paraphrases_tagged.json \
  --content-preservation a_data/alpaca/equi_scores_paraphrases_500.json \
  --output-dir f_finetune/outputs_clean/9x_a1stop_ft/calcoutput7ONLYFT/ \
 --filter-keys "$FILTER_KEYS"

python e_eval/src/eval_paras.py \
  --prompts a_data/alpaca/50k_phrxed.json \
  --scores BASE=c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
  --tags-json a_data/paraphrases_tagged.json \
  --content-preservation a_data/alpaca/equi_scores_paraphrases_500.json \
  --output-dir f_finetune/outputs_clean/9x_a1stop_ft/calBASE7/ \
 --filter-keys "$FILTER_KEYS"

python e_eval/src/eval_paras.py \
  --prompts a_data/alpaca/50k_phrxed.json \
  --scores Lay6-FT=f_finetune/outputs_clean/9xlay6_a1stop_ft/scores512.json \
  --tags-json a_data/paraphrases_tagged.json \
  --content-preservation a_data/alpaca/equi_scores_paraphrases_500.json \
  --output-dir f_finetune/outputs_clean/9xlay6_a1stop_ft/calcoutput7ONLYFT/ \
 --filter-keys "$FILTER_KEYS"

FILTER_KEYS="instruct_double_negative,\
instruct_formal_memo,instruct_future_tense,instruct_joke,\
instruct_one_typo_punctuation"
python e_eval/src/eval_paras_ft.py \
  --prompts a_data/alpaca/50k_phrxed.json \
  --scores BASE=c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
        Lay6-FT=f_finetune/outputs_clean/9x_a1stop_ft/scores512.json \
        MAIN-FT=f_finetune/outputs_clean/9xlay6_a1stop_ft/scores512.json \
  --tags-json a_data/paraphrases_tagged.json \
  --content-preservation a_data/alpaca/equi_scores_paraphrases_500.json \
  --output-dir f_finetune/outputs_clean/9xlay6_a1stop_ft/calcoutput9ALL/ \
 --filter-keys "$FILTER_KEYS"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def setup_logging(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized. Writing to %s", log_path)


class KVMap(argparse.Action):
    """Parse KEY=VALUE tokens into an ordered dict (to preserve dataset order)."""
    def __call__(self, parser, namespace, values, option_string=None):
        od = OrderedDict()
        for v in values:
            if "=" not in v:
                parser.error(f"Expected KEY=VALUE under {option_string}, got '{v}'")
            k, val = v.split("=", 1)
            if not k:
                parser.error(f"Empty key in '{v}'")
            if k in od:
                parser.error(f"Duplicate dataset name '{k}' in --scores")
            od[k] = val
        setattr(namespace, self.dest, od)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LLM prompt paraphrase robustness.")
    p.add_argument("--prompts", required=True, help="Path to prompts JSON (for reference)")
    p.add_argument("--answers", nargs="*", default=[], help="Optional answers JSON(s) – not required")
    p.add_argument("--scores", nargs="+", action=KVMap, required=True,
                   help="One or more Dataset=path mappings to score JSON files")
    p.add_argument("--tags-json", required=True, help="JSON mapping instruct_* -> [tags]")
    p.add_argument("--content-preservation", required=True,
                   help="JSON of CP scores per prompt_count with a 'scores' dict per record")
    p.add_argument("--output-dir", required=True, help="Directory to write outputs")
    p.add_argument("--filter-keys", default="", help="Comma-separated instruct_* keys to limit GRAPHICS to")
    p.add_argument("--cp-threshold", type=float, default=4.0, help="CP threshold for inclusion (default 4)")
    p.add_argument("--cp-min-count", type=int, default=200, help="Minimum count above threshold to include a type (default 200)")
    p.add_argument("--forest-green", default="#1B5E20", help="Hex color for forest green emphasis")
    return p.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


NAME_TRIM = 20  # max chars after removing instruct_/instruction_ prefix


def friendly_type_name(key: str) -> str:
    """Map JSON keys to display names: 'instruction_original'->'Original',
    'instruct_apologetic'->'Apologetic' (trim to <=20 chars)."""
    if key == "instruction_original":
        return "Original"
    # Strip common prefixes
    base = re.sub(r"^(instruct_|instruction_)", "", key)
    # Title-case with underscores treated as spaces
    base = base.replace("_", " ")
    base = base.strip()
    # Trim
    if len(base) > NAME_TRIM:
        base = base[:NAME_TRIM - 1] + "…"
    # Capitalize words
    return " ".join(w.capitalize() for w in base.split())


@dataclass
class ScoreRow:
    prompt_count: int
    dataset: str
    type_key: str
    tf: float


def build_tf_dataframe(dataset_to_scores_path: "OrderedDict[str, str]") -> pd.DataFrame:
    rows: List[ScoreRow] = []
    missing_tf_counter = 0
    for dataset, path in dataset_to_scores_path.items():
        obj = load_json(path)
        if not isinstance(obj, list):
            logging.warning("Scores JSON %s for dataset %s is not a list", path, dataset)
        for rec in obj:
            if "prompt_count" not in rec:
                logging.warning("Record without prompt_count in %s", path)
                continue
            pc = rec["prompt_count"]
            for key, val in rec.items():
                if key == "prompt_count":
                    continue
                # Only consider keys that look like instruction_original or instruct_*
                if key == "instruction_original" or key.startswith("instruct_"):
                    if not isinstance(val, (list, tuple)) or not val:
                        missing_tf_counter += 1
                        continue
                    tf = val[0]
                    try:
                        tf_f = float(tf)
                    except Exception:
                        missing_tf_counter += 1
                        continue
                    rows.append(ScoreRow(prompt_count=int(pc), dataset=dataset, type_key=key, tf=tf_f))
    if missing_tf_counter:
        logging.warning("Encountered %d items without a usable TF score (index 0)", missing_tf_counter)
    df = pd.DataFrame([r.__dict__ for r in rows])
    # Flag original
    df["is_original"] = df["type_key"] == "instruction_original"
    df["display_type"] = df["type_key"].map(friendly_type_name)
    return df


def build_cp_dataframe(cp_path: str) -> pd.DataFrame:
    obj = load_json(cp_path)
    recs = []
    for rec in obj:
        pc = rec.get("prompt_count")
        scores = rec.get("scores", {})
        if pc is None or not isinstance(scores, dict):
            logging.warning("CP record missing prompt_count or scores dict")
            continue
        for tkey, cp in scores.items():
            if not tkey.startswith("instruct_"):
                # Content preservation is defined for paraphrases, skip others
                continue
            try:
                recs.append({"prompt_count": int(pc), "type_key": tkey, "cp": float(cp)})
            except Exception:
                logging.warning("Bad CP value for %s pc=%s: %r", tkey, pc, cp)
                continue
    df = pd.DataFrame(recs)
    return df


def load_tags(tags_path: str) -> Dict[str, List[str]]:
    obj = load_json(tags_path)
    if not isinstance(obj, dict):
        logging.warning("Tags JSON is not a dict: %s", tags_path)
        return {}
    # Normalize tag names minimally (keep as given but strip/space)
    clean = {}
    for k, v in obj.items():
        if not isinstance(v, list):
            logging.warning("Tags for %s are not a list; skipping", k)
            continue
        tags = [str(t).strip() for t in v if str(t).strip()]
        clean[k] = tags
    return clean


def compute_allowed_types(cp_df: pd.DataFrame, cp_threshold: float, min_count: int) -> List[str]:
    if cp_df.empty:
        logging.warning("No CP data available; will treat all instruct_* types as disallowed except Original.")
        return []
    grp = cp_df.assign(hit=(cp_df["cp"] >= cp_threshold)).groupby("type_key")["hit"].sum()
    allowed = grp[grp >= min_count].sort_values(ascending=False)
    logging.info("Types meeting CP>=%.2f at least %d times: %d", cp_threshold, min_count, len(allowed))
    for tkey, cnt in allowed.items():
        logging.info("  %-40s %5d", tkey, int(cnt))
    return list(allowed.index)


def parse_filter_keys(filter_csv: str) -> Optional[set]:
    if not filter_csv:
        return None
    keys = [k.strip() for k in filter_csv.split(",") if k.strip()]
    return set(keys)


def series_descriptives(s: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"count": 0, "mean": np.nan, "median": np.nan, "std": np.nan,
                "min": np.nan, "25%": np.nan, "50%": np.nan, "75%": np.nan, "max": np.nan}
    desc = {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
        "min": float(s.min()),
        "25%": float(s.quantile(0.25)),
        "50%": float(s.quantile(0.50)),
        "75%": float(s.quantile(0.75)),
        "max": float(s.max()),
    }
    return desc


def df_descriptives(df: pd.DataFrame, value_col: str, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if group_cols is None or not group_cols:
        return pd.DataFrame([series_descriptives(df[value_col])])
    out_rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        desc = series_descriptives(sub[value_col])
        row = {**{gc: k for gc, k in zip(group_cols, keys)}, **desc}
        out_rows.append(row)
    res = pd.DataFrame(out_rows)
    return res


def df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Minimal, dependency-free Markdown table rendering."""
    if df.empty:
        return "(no data)"
    if index:
        data = df.reset_index()
    else:
        data = df.copy()
    cols = list(data.columns)
    # Build header
    header = "| " + " | ".join(str(c) for c in cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = []
    for _, r in data.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return header + sep + "\n".join(rows)


GREYS = ["#111111", "#444444", "#777777", "#999999", "#BBBBBB", "#DDDDDD"]

def dataset_color_map(datasets: List[str], forest_green: str) -> Dict[str, str]:
    cmap = {}
    for i, ds in enumerate(datasets):
        if i == 0:
            cmap[ds] = forest_green
        else:
            cmap[ds] = GREYS[min(i - 1, len(GREYS) - 1)]
    return cmap


def ensure_range(v: float, lo: float, hi: float) -> float:
    try:
        vf = float(v)
    except Exception:
        return lo
    return min(max(vf, lo), hi)


def main():
    args = parse_args()
    setup_logging(args.output_dir)

    logging.info("Loading inputs…")
    # Prompts are currently unused; we still load to validate presence
    try:
        _prompts = load_json(args.prompts)
        logging.info("Loaded prompts JSON with %d top-level records", len(_prompts) if isinstance(_prompts, list) else -1)
    except Exception as e:
        logging.warning("Failed to load prompts JSON: %s", e)

    if args.answers:
        logging.info("Answers JSON(s) provided but not required for this analysis: %s", ", ".join(args.answers))

    tags_map = load_tags(args.tags_json)
    cp_df = build_cp_dataframe(args.content_preservation)
    tf_df = build_tf_dataframe(args.scores)

    if tf_df.empty:
        logging.error("No TF data. Aborting.")
        return

    datasets = list(args.scores.keys())
    color_map = dataset_color_map(datasets, args.forest_green)

    # Decide the paraphrase types to analyze
    filter_set = parse_filter_keys(args.filter_keys)
    if filter_set:
        # Bypass CP gating entirely: use only user-specified instruct_* keys
        present_para_types = {t for t in tf_df["type_key"].unique() if t.startswith("instruct_")}
        missing = sorted(filter_set - present_para_types)
        if missing:
            logging.warning(
                "The following --filter-keys are not present in your score JSONs and will be ignored: %s",
                ", ".join(missing)
            )
        paraphrase_types = filter_set & present_para_types
        logging.info("Using ONLY user-specified paraphrase types: %s",
                    ", ".join(sorted(paraphrase_types)) or "(none)")
    else:
        # Fall back to CP gating
        allowed_types = compute_allowed_types(cp_df, args.cp_threshold, args.cp_min_count)
        paraphrase_types = set(allowed_types)
        logging.info("Using CP-allowed paraphrase types: %s",
                    ", ".join(sorted(paraphrase_types)) or "(none)")

    # Prepare merged frames
    # Add tags to TF rows for paraphrases
    tf_df["tags"] = tf_df["type_key"].map(lambda k: tags_map.get(k, []) if k.startswith("instruct_") else [])

    # subset for paraphrase-only rows meeting the chosen set (user filter or CP-gated)
    tf_para_allowed = tf_df[tf_df["type_key"].isin(paraphrase_types)]

    # For tables that require Original vs Paraphrase comparisons include Original rows too
    tf_orig = tf_df[tf_df["type_key"] == "instruction_original"]

    # TABLES
    # TF for Original vs Paraphrased (across all datasets)
    tbl1 = pd.DataFrame([
        {"Group": "Original", **series_descriptives(tf_orig["tf"])},
        {"Group": "Paraphrased", **series_descriptives(tf_para_allowed["tf"])},
    ])

    # TF by Dataset (Original + allowed paraphrases)
    tf_all_for_dataset = pd.concat([tf_orig, tf_para_allowed], ignore_index=True)
    tbl2 = df_descriptives(tf_all_for_dataset, value_col="tf", group_cols=["dataset"]).sort_values("mean", ascending=False)

    # TF by Individual Tag (across datasets). Only paraphrases (original has no tags)
    tf_para_allowed_with_tags = tf_para_allowed.explode("tags")
    tf_para_allowed_with_tags = tf_para_allowed_with_tags[tf_para_allowed_with_tags["tags"].notna() & (tf_para_allowed_with_tags["tags"] != "")]
    if tf_para_allowed_with_tags.empty:
        logging.warning("No tag data found for allowed types – Tag tables/plots will be empty.")
        tbl3 = pd.DataFrame()
    else:
        tbl3 = df_descriptives(tf_para_allowed_with_tags, value_col="tf", group_cols=["tags"]).rename(columns={"tags": "Tag"}).sort_values("mean", ascending=False)

    # Mean ΔTF vs Original — By Type × Dataset
    # Match paraphrase TF with original TF by prompt_count within each dataset
    orig_by_ds_pc = tf_orig[["dataset", "prompt_count", "tf"]].rename(columns={"tf": "tf_orig"})
    para_vs_orig = tf_para_allowed.merge(orig_by_ds_pc, on=["dataset", "prompt_count"], how="inner")
    para_vs_orig["delta_tf"] = para_vs_orig["tf"] - para_vs_orig["tf_orig"]
    g = para_vs_orig.groupby(["dataset", "type_key"])
    rows = []
    for (ds, tkey), sub in g:
        rows.append({
            "Dataset": ds,
            "Type": friendly_type_name(tkey),
            "n": int(sub.shape[0]),
            "mean_TF": round(float(sub["tf"].mean()), 4),
            "mean_ΔTF": round(float(sub["delta_tf"].mean()), 4),
        })
    tbl4 = pd.DataFrame(rows).sort_values(["Dataset", "mean_ΔTF"], ascending=[True, False])

    # CP stats per paraphrase type (allowed types only)
    cp_allowed = cp_df[cp_df["type_key"].isin(paraphrase_types)]
    tbl5 = df_descriptives(cp_allowed.rename(columns={"cp": "value"}), value_col="value", group_cols=["type_key"]).rename(columns={"type_key": "TypeKey"})
    if not tbl5.empty:
        tbl5.insert(1, "Type", tbl5["TypeKey"].map(friendly_type_name))
        tbl5 = tbl5.drop(columns=["TypeKey"]).sort_values("mean", ascending=False)

    # CP stats per tag (allowed types only)
    if not cp_allowed.empty:
        cp_tag_rows = []
        for _, row in cp_allowed.iterrows():
            tkey = row["type_key"]
            tags = tags_map.get(tkey, [])
            for tag in tags:
                cp_tag_rows.append({"tag": tag, "cp": row["cp"]})
        cp_tag_df = pd.DataFrame(cp_tag_rows)
        if cp_tag_df.empty:
            tbl6 = pd.DataFrame()
        else:
            tbl6 = df_descriptives(cp_tag_df.rename(columns={"cp": "value"}), value_col="value", group_cols=["tag"]).rename(columns={"tag": "Tag"}).sort_values("mean", ascending=False)
    else:
        tbl6 = pd.DataFrame()

    # Correlation CP vs TF (overall and per dataset)
    cp_tf = cp_df.merge(tf_para_allowed, on=["prompt_count", "type_key"], how="inner")
    cor_rows = []
    def corr_pair(x: pd.DataFrame) -> Tuple[float, float, int]:
        if x.empty:
            return (np.nan, np.nan, 0)
        pear = x[["cp", "tf"]].corr(method="pearson").iloc[0, 1]
        spear = x[["cp", "tf"]].corr(method="spearman").iloc[0, 1]
        return (float(pear), float(spear), int(x.shape[0]))

    pear, spear, n = corr_pair(cp_tf)
    cor_rows.append({"Dataset": "ALL", "Pearson_r": round(pear, 4) if not math.isnan(pear) else np.nan,
                     "Spearman_ρ": round(spear, 4) if not math.isnan(spear) else np.nan, "n": n})
    for ds, sub in cp_tf.groupby("dataset"):
        pear, spear, n = corr_pair(sub)
        cor_rows.append({"Dataset": ds, "Pearson_r": round(pear, 4) if not math.isnan(pear) else np.nan,
                         "Spearman_ρ": round(spear, 4) if not math.isnan(spear) else np.nan, "n": n})
    tbl7 = pd.DataFrame(cor_rows).sort_values("Dataset")

    # Determine types for graphics (Original + intersection of allowed types and filter_set if provided)
    # Determine types for graphics (Original + selected paraphrase types)
    types_for_graphs = ["instruction_original"] + sorted(list(paraphrase_types))


    # Helper for sorted display order by overall mean TF (excluding Original first)
    type_means = tf_df[tf_df["type_key"].isin(types_for_graphs)].groupby("type_key")["tf"].mean().sort_values(ascending=False)
    ordered_types = ["instruction_original"] + [t for t in type_means.index if t != "instruction_original"]

    # Colors for types in boxplots: Original forest green, others greys cycling
    type_color_map = {"instruction_original": args.forest_green}
    grey_cycle = [GREYS[i % len(GREYS)] for i in range(max(1, len(ordered_types)))]
    gi = 0
    for t in ordered_types:
        if t == "instruction_original":
            continue
        type_color_map[t] = grey_cycle[gi]
        gi += 1

    # Per dataset boxplot by type
    for ds in datasets:
        sub = tf_df[(tf_df["dataset"] == ds) & (tf_df["type_key"].isin(ordered_types))]
        data = []
        labels = []
        colors = []
        for t in ordered_types:
            vals = sub[sub["type_key"] == t]["tf"].dropna().tolist()
            if not vals:
                continue
            data.append(vals)
            labels.append(friendly_type_name(t))
            colors.append(type_color_map.get(t, GREYS[-1]))
        if not data:
            logging.warning("No data for dataset %s in boxplot by type", ds)
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 6))
        bp = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=False)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
        ax.set_title(f"TF by Type — {ds}")
        ax.set_ylabel("TF score")
        ax.set_ylim(0, 10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"box_types_{ds}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        logging.info("Saved %s", out)

    # Per dataset boxplot by tag
    tags_sorted = sorted({tag for tags in tags_map.values() for tag in tags})
    if tags_sorted:
        for ds in datasets:
            sub = tf_para_allowed_with_tags[tf_para_allowed_with_tags["dataset"] == ds]
            data = []
            labels = []
            colors = []
            for i, tag in enumerate(tags_sorted):
                vals = sub[sub["tags"] == tag]["tf"].dropna().tolist()
                if not vals:
                    continue
                data.append(vals)
                labels.append(tag)
                colors.append(GREYS[i % len(GREYS)])
            if not data:
                logging.warning("No tag data for dataset %s in boxplot by tag", ds)
                continue
            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 6))
            bp = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=False)
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
            ax.set_title(f"TF by Tag — {ds}")
            ax.set_ylabel("TF score")
            ax.set_ylim(0, 10)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            out = os.path.join(args.output_dir, f"box_tags_{ds}.png")
            fig.savefig(out, dpi=200)
            plt.close(fig)
            logging.info("Saved %s", out)
    else:
        logging.warning("No tags available for plotting boxplots by tag.")

    # Grouped bars by type (datasets as colors)
    # Build matrix: rows = types (ordered), cols = datasets
    mean_by_type_ds = tf_df[tf_df["type_key"].isin(ordered_types)].groupby(["type_key", "dataset"])['tf'].mean().unstack(fill_value=np.nan)
    if not mean_by_type_ds.empty:
        labels = [friendly_type_name(t) for t in mean_by_type_ds.index]
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(datasets))
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 6))
        for i, ds in enumerate(datasets):
            ys = mean_by_type_ds.get(ds)
            if ys is None:
                ys = [np.nan] * len(labels)
            ax.bar(x + (i - len(datasets)/2) * width + width/2, ys, width, label=ds, color=color_map[ds])
        ax.set_title("Mean TF by Type × Dataset")
        ax.set_ylabel("Mean TF")
        ax.set_ylim(0, 10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"bars_type_by_dataset.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        logging.info("Saved %s", out)

    # Grouped bars by tag (datasets as colors)
    if not tf_para_allowed_with_tags.empty:
        mean_by_tag_ds = tf_para_allowed_with_tags.groupby(["tags", "dataset"])['tf'].mean().unstack(fill_value=np.nan)
        labels = list(mean_by_tag_ds.index)
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(datasets))
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), 6))
        for i, ds in enumerate(datasets):
            ys = mean_by_tag_ds.get(ds)
            if ys is None:
                ys = [np.nan] * len(labels)
            ax.bar(x + (i - len(datasets)/2) * width + width/2, ys, width, label=ds, color=color_map[ds])
        ax.set_title("Mean TF by Tag × Dataset")
        ax.set_ylabel("Mean TF")
        ax.set_ylim(0, 10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"bars_tag_by_dataset.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        logging.info("Saved %s", out)

    # CP vs TF scatter
    if not cp_tf.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, ds in enumerate(datasets):
            sub = cp_tf[cp_tf["dataset"] == ds]
            ax.scatter(sub["cp"], sub["tf"], s=8, alpha=0.3, label=ds, color=color_map[ds])
        # Simple OLS fit across all points
        try:
            coef = np.polyfit(cp_tf["cp"], cp_tf["tf"], 1)
            xs = np.linspace(cp_tf["cp"].min(), cp_tf["cp"].max(), 100)
            ys = coef[0] * xs + coef[1]
            ax.plot(xs, ys, linestyle="--", linewidth=1, color="#2E7D32")  # dashed green fit
        except Exception as e:
            logging.warning("Could not fit regression line: %s", e)
        ax.set_title("Content Preservation vs TF")
        ax.set_xlabel("Content Preservation score")
        ax.set_ylabel("TF score")
        ax.set_xlim(left=cp_tf["cp"].min() - 0.5, right=cp_tf["cp"].max() + 0.5)
        ax.set_ylim(0, 10)
        ax.legend()
        plt.tight_layout()
        out = os.path.join(args.output_dir, f"scatter_cp_vs_tf.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        logging.info("Saved %s", out)

    def fmt_float_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = out[c].map(lambda x: ("" if pd.isna(x) else f"{x:.3f}") if isinstance(x, (float, np.floating)) else x)
        return out

    # Friendly versions of tables
    tbl2f = tbl2.copy()
    tbl2f["count"] = tbl2f["count"].astype(int)
    tbl2f = fmt_float_cols(tbl2f, ["mean", "median", "std", "min", "25%", "50%", "75%", "max"]) 

    tbl3f = tbl3.copy()
    if not tbl3f.empty:
        tbl3f["count"] = tbl3f["count"].astype(int)
        tbl3f = fmt_float_cols(tbl3f, ["mean", "median", "std", "min", "25%", "50%", "75%", "max"]) 

    tbl4f = tbl4.copy()

    tbl5f = tbl5.copy()
    if not tbl5f.empty:
        tbl5f["count"] = tbl5f["count"].astype(int)
        tbl5f = fmt_float_cols(tbl5f, ["mean", "median", "std", "min", "25%", "50%", "75%", "max"]) 

    tbl6f = tbl6.copy()
    if not tbl6f.empty:
        tbl6f["count"] = tbl6f["count"].astype(int)
        tbl6f = fmt_float_cols(tbl6f, ["mean", "median", "std", "min", "25%", "50%", "75%", "max"]) 

    tbl7f = tbl7.copy()

    # Compose markdown and txt
    md = []
    md.append("# Paraphrase Robustness Evaluation Summary\n")
    md.append(f"**Datasets:** {', '.join(datasets)}  ")
    if filter_set:
        md.append(f"**Types included (user filter):** "
                f"{', '.join(friendly_type_name(t) for t in sorted(paraphrase_types))}  ")
    else:
        md.append(f"**Types included (CP>={args.cp_threshold}, count>={args.cp_min_count}):** "
                f"{', '.join(friendly_type_name(t) for t in sorted(paraphrase_types))}  ")

    md.append("\n## TF — Original vs Paraphrased\n")
    md.append(df_to_markdown(tbl1, index=False))

    md.append("\n## TF by Dataset (Original + allowed paraphrases)\n")
    md.append(df_to_markdown(tbl2f, index=False))

    # Per-dataset TF tables
    md.append("\n### Per-Dataset TF Descriptives\n")
    for ds in datasets:
        sub = tf_all_for_dataset[tf_all_for_dataset["dataset"] == ds]
        ds_tbl = df_descriptives(sub, value_col="tf")
        ds_tbl["count"] = ds_tbl["count"].astype(int)
        ds_tbl = fmt_float_cols(ds_tbl, ["mean","median","std","min","25%","50%","75%","max"])
        md.append(f"#### {ds}\n" + df_to_markdown(ds_tbl, index=False))

    # Per-dataset TF by Type (Original + selected paraphrases)
    md.append("\n### Per-Dataset TF by Type\n")
    for ds in datasets:
        sub = tf_all_for_dataset[tf_all_for_dataset["dataset"] == ds]
        # keep Original + selected types
        keep_types = {"instruction_original"} | set(paraphrase_types)
        sub = sub[sub["type_key"].isin(keep_types)]

        if sub.empty:
            md.append(f"#### {ds}\n(no data)")
            continue

        sub = sub.assign(Type=sub["type_key"].map(friendly_type_name))
        ds_type_tbl = df_descriptives(sub, value_col="tf", group_cols=["Type"]).sort_values("mean", ascending=False)
        ds_type_tbl["count"] = ds_type_tbl["count"].astype(int)
        ds_type_tbl = fmt_float_cols(ds_type_tbl, ["mean","median","std","min","25%","50%","75%","max"])
        md.append(f"#### {ds}\n" + df_to_markdown(ds_type_tbl, index=False))

    md.append("\n## TF by Individual Tag (paraphrases only)\n")
    md.append(df_to_markdown(tbl3f, index=False))

    md.append("\n## Mean ΔTF vs Original — By Type × Dataset\n")
    md.append(df_to_markdown(tbl4f, index=False))

    md.append("\n## Content Preservation — by Type (allowed)\n")
    md.append(df_to_markdown(tbl5f, index=False))

    md.append("\n## Content Preservation — by Tag (allowed)\n")
    md.append(df_to_markdown(tbl6f, index=False))

    md.append("\n## Correlation: CP vs TF\n")
    md.append(df_to_markdown(tbl7f, index=False))

    md_txt = "\n".join(md)

    # TXT version: strip markdown tables to plain text
    txt = []
    txt.append("Paraphrase Robustness Evaluation Summary\n")
    txt.append(f"Datasets: {', '.join(datasets)}")
    if filter_set:
        txt.append(f"Types included (user filter): {', '.join(sorted(paraphrase_types))}")
    else:
        txt.append(f"Types included (CP>={args.cp_threshold}, count>={args.cp_min_count}): "
                f"{', '.join(sorted(paraphrase_types))}")

    txt.append("\nTF — Original vs Paraphrased")
    txt.append(tbl1.to_string(index=False))
    txt.append("\nTF by Dataset (Original + allowed paraphrases)")
    txt.append(tbl2f.to_string(index=False))
    # Per-dataset TF tables (expanded)
    txt.append("\nPer-Dataset TF Descriptives")
    for ds in datasets:
        sub = tf_all_for_dataset[tf_all_for_dataset["dataset"] == ds]
        ds_tbl = df_descriptives(sub, value_col="tf")
        txt.append(f"\n{ds}\n" + ds_tbl.to_string(index=False))

    # Per-dataset TF by Type
    txt.append("\nPer-Dataset TF by Type")
    for ds in datasets:
        sub = tf_all_for_dataset[tf_all_for_dataset["dataset"] == ds]
        keep_types = {"instruction_original"} | set(paraphrase_types)
        sub = sub[sub["type_key"].isin(keep_types)]

        if sub.empty:
            txt.append(f"\n{ds}\n(no data)")
            continue

        sub = sub.assign(Type=sub["type_key"].map(friendly_type_name))
        ds_type_tbl = df_descriptives(sub, value_col="tf", group_cols=["Type"]).sort_values("mean", ascending=False)
        txt.append(f"\n{ds}\n" + ds_type_tbl.to_string(index=False))

    txt.append("\nTF by Individual Tag (paraphrases only)")
    txt.append(tbl3f.to_string(index=False) if not tbl3f.empty else "(no data)")
    txt.append("\nMean ΔTF vs Original — By Type × Dataset")
    txt.append(tbl4f.to_string(index=False))
    txt.append("\nContent Preservation — by Type (allowed)")
    txt.append(tbl5f.to_string(index=False) if not tbl5f.empty else "(no data)")
    txt.append("\nContent Preservation — by Tag (allowed)")
    txt.append(tbl6f.to_string(index=False) if not tbl6f.empty else "(no data)")
    txt.append("\nCorrelation: CP vs TF")
    txt.append(tbl7f.to_string(index=False))
    txt_txt = "\n\n".join(txt)

    # Write files
    md_path = os.path.join(args.output_dir, "summary.md")
    txt_path = os.path.join(args.output_dir, "summary.txt")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_txt)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_txt)
    logging.info("Wrote %s and %s", md_path, txt_path)

    logging.info("Done.")


if __name__ == "__main__":
    main()
