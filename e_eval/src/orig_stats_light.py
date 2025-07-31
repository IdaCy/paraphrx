#!/usr/bin/env python3
"""
compute statistical summaries, significance tests, and graphics for prompt-robustness paraphrase datasets

Inputs:
  --data_jsons: One or more main-data JSON files for the *same* model but different source datasets
  --paraphrase_tags: JSON mapping paraphrase type (key) -> list of tags
  --content_stats: JSON with content/equivalence score stats for each paraphrase type
  --output_dir: Directory where figures and csv/tsv outputs will be written (default: "./results")

python e_eval/src/orig_stats.py \
    --data_jsons f_finetune/data/alpaca_gemma-2-2b-it.json \
        f_finetune/data/gsm8k_gemma-2-2b-it.json \
        f_finetune/data/mmlu_gemma-2-2b-it.json \
    --paraphrase_tags paraphrase_tags.json \
    --content_stats a_data/paraphrases_500_phrx_scores_stats.json

The script prints tables to stdout and writes:
  - <output_dir>/summary_stats.tsv
  - <output_dir>/t_tests.tsv
  - Plots (PNG files)
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

SELECTED_TYPES = [
    "instruction_original",
    "instruct_output_markdown",
    "instruct_one_typo_punctuation",
    "instruct_coord_to_subord",
    "instruct_future_tense",
    "instruct_polite_request",
    "instruct_dramatic",
    "instruct_sardonic",
    "instruct_joke",
    "instruct_formal_demo",
    "instruct_double_negative",
    "instruct_leet_speak",
]

BUCKETS = [1, 2, 3, 4, 5]
CONTENT_SCORES = [0, 1, 2, 3, 4, 5]


def load_dataset(path: str, tags_map: dict) -> pd.DataFrame:
    """Load a single main-data JSON produced by the pipeline"""
    dataset_name = Path(path).stem.split("_", 1)[0]  # alpaca, gsm8k, ...
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for obj in data:
        pc_id = obj["prompt_count"]
        inp_present = bool(obj.get("input") or obj.get("scenarios"))
        # Attempt to capture original score if present
        orig_score = None
        for k in ("original_task_score", "task_score_original", "task_score"):
            v = obj.get(k)
            if isinstance(v, (int, float)):
                orig_score = v
                break
        if orig_score is not None:
            rows.append(
                {
                    "dataset": dataset_name,
                    "prompt_count": pc_id,
                    "paraphrase_type": "instruction_original",
                    "bucket": 0,
                    "content_score": np.nan,
                    "task_score": orig_score,
                    "perplexity": np.nan,
                    "input_present": inp_present,
                }
            )

        for p in obj["paraphrases"]:
            p_type = p["instruct_type"]
            rows.append(
                {
                    "dataset": dataset_name,
                    "prompt_count": pc_id,
                    "paraphrase_type": p_type,
                    "bucket": p.get("bucket"),
                    "content_score": p.get("paraphrase_content_score"),
                    "task_score": p.get("task_score") or (p["answer_scores"][0] if p.get("answer_scores") else np.nan),
                    "perplexity": p.get("perplexity"),
                    "input_present": inp_present,
                }
            )

    df = pd.DataFrame(rows)
    # attach tags
    df["tags"] = df["paraphrase_type"].map(tags_map).fillna("").apply(lambda x: x if isinstance(x, list) else [])
    return df


def describe_series(s: pd.Series) -> pd.Series:
    """Return count, mean, std, min, 25%, 50%, 75%, max"""
    return s.describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]


def save_plot(fig, outdir: Path, fname: str):
    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_jsons", nargs="+", required=True, help="Main data JSONs (same model).")
    parser.add_argument("--paraphrase_tags", required=True, help="Paraphrase-type -> tags mapping JSON.")
    parser.add_argument("--content_stats", required=True, help="JSON with content score stats by type.")
    parser.add_argument("--output_dir", default="results", help="Directory for outputs.")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tags_map = json.load(open(args.paraphrase_tags, "r", encoding="utf-8"))
    content_stats_map = json.load(open(args.content_stats, "r", encoding="utf-8"))

    # Load and merge
    frames = [load_dataset(p, tags_map) for p in args.data_jsons]
    df = pd.concat(frames, ignore_index=True)

    # Input-field presence
    prompt_input_stats = (
        df[["dataset", "prompt_count", "input_present"]]
        .drop_duplicates()
        .groupby("dataset")["input_present"]
        .agg(total="count", with_input="sum")
        .reset_index()
    )
    prompt_input_stats["percentage"] = 100 * prompt_input_stats["with_input"] / prompt_input_stats["total"]
    print("\n=== Input-field presence ===")
    print(prompt_input_stats.to_markdown(index=False))

    # TF stats for selected types
    tf_stats_selected = (
        df[df["paraphrase_type"].isin(SELECTED_TYPES)]
        .groupby("paraphrase_type")["task_score"]
        .apply(describe_series)
        .unstack()
        .reset_index()
        .round(3)
    )
    print("\n=== TF stats for selected types ===")
    print(tf_stats_selected.to_markdown(index=False))

    # Perplexity stats for selected types
    perp_stats_selected = (
        df[df["paraphrase_type"].isin(SELECTED_TYPES)]
        .dropna(subset=["perplexity"])
        .groupby("paraphrase_type")["perplexity"]
        .apply(describe_series)
        .unstack()
        .reset_index()
        .round(3)
    )
    print("\n=== Perplexity stats for selected types ===")
    print(perp_stats_selected.to_markdown(index=False))

    # Correlation perplexity vs TF
    corr = df[["perplexity", "task_score"]].dropna().corr().iloc[0, 1]
    print(f"\n=== Pearson correlation (perplexity, TF) : {corr:.3f} ===")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(data=df, x="perplexity", y="task_score", scatter_kws={"alpha": 0.2}, ax=ax)
    ax.set_title(f"Perplexity vs Task Fulfilment (r={corr:.3f})")
    save_plot(fig, outdir, "perplexity_vs_tf.png")

    # Bucket distribution stats
    bucket_stats = (
        df.dropna(subset=["bucket"])
        .groupby("bucket")["paraphrase_type"]
        .count()
        .reindex(BUCKETS)
        .rename("count")
        .reset_index()
    )
    bucket_stats["label"] = bucket_stats["bucket"].astype(int).astype(str)
    print("\n=== Paraphrase count per bucket ===")
    print(bucket_stats.to_markdown(index=False))

    # Histogram
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=bucket_stats, x="label", y="count", ax=ax)
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Paraphrase count")
    ax.set_title("Paraphrase counts by TF bucket")
    save_plot(fig, outdir, "bucket_counts.png")

    # Content score distribution stats
    content_stats_df = (
        df.dropna(subset=["content_score"])
        .groupby("content_score")["paraphrase_type"]
        .count()
        .reindex(CONTENT_SCORES)
        .rename("count")
        .reset_index()
    )
    print("\n=== Paraphrase count per content score ===")
    print(content_stats_df.to_markdown(index=False))

    # Boxplot TF by content score
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df.dropna(subset=["content_score"]), x="content_score", y="task_score", ax=ax)
    ax.set_xlabel("Content equivalence score")
    ax.set_ylabel("Task Fulfilment (TF)")
    save_plot(fig, outdir, "tf_by_content_score.png")

    # Counts of score 4 & 5 by paraphrase type
    high_content = df[df["content_score"].isin([4, 5])]
    high_counts = (
        high_content.groupby(["paraphrase_type", "content_score"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={4: "score4", 5: "score5"})
    )
    high_counts["score4+5"] = high_counts["score4"] + high_counts["score5"]
    print("\n=== Content score 4/5 counts by type ===")
    print(high_counts.sort_values("score4+5", ascending=False).head(20).to_markdown(index=False))

    # TF stats for types with >=100 high scores
    eligible_types = high_counts[high_counts["score4+5"] >= 100]["paraphrase_type"].tolist()
    eligible_df = df[df["paraphrase_type"].isin(eligible_types)]

    overall_stats = (
        eligible_df.groupby("paraphrase_type")["task_score"]
        .apply(describe_series)
        .unstack()
        .reset_index()
        .round(3)
    )
    print("\n=== TF stats for high-content paraphrase types (all data) ===")
    print(overall_stats.to_markdown(index=False))

    # Stats per dataset
    per_dataset_stats = (
        eligible_df.groupby(["dataset", "paraphrase_type"])["task_score"]
        .apply(describe_series)
        .unstack()
        .reset_index()
        .round(3)
    )
    print("\n=== TF stats for high-content types by dataset ===")
    print(per_dataset_stats.to_markdown(index=False))

    # Boxplot of TF by paraphrase type (eligible only)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=eligible_df, x="paraphrase_type", y="task_score", ax=ax)
    ax.set_xlabel("Paraphrase type")
    ax.set_ylabel("Task Fulfilment (TF)")
    save_plot(fig, outdir, "tf_by_paraphrase_type.png")
