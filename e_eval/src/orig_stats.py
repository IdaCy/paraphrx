# -*- coding: utf-8 -*-
"""
analyse paraphrase robustness data +
produce descriptive statistics, significance tests and graphics

- or more *main* dataset JSON files (e.g. generated answers for the same
  model on Alpaca, GSM8K, MMLU …)
- A JSON that maps each 'instruct_*' paraphrase type to a list of tags
- A JSON with content-equivalence score statistics for every paraphrase type
- Optionally, one or more JSON files that hold the scored answers for the
  **original** instructions ('instruction_original'); if these are not
  supplied, the script will try to find TF scores inside the main files; if
  none are found, original-instruction statistics are skipped with a
  warning

produces:
1. tabular summaries (CSV & pretty-printed to stdout) of counts and the
     standard five-number summary for:
     * presence of the 'input' field in original prompts;
     * Task Fulfilment (TF) scores for the 12 selected paraphrase types;
     * perplexity for the same 12 types;
     * number of paraphrases per bucket (1-5) per prompt;
     * number of paraphrases per content-equivalence score (0-5) per prompt;
     * TF score distributions for paraphrase types with ≥100 high-quality
       (score 4 + 5) instances - overall and per dataset;
     * TF score distributions aggregated by *tag* for those types.
2. Welch two-sample *t*-tests for each comparison, always using
     'instruction_original' as the reference where meaningful.  All
     p-values are Bonferroni-corrected
3. A Pearson correlation between perplexity and TF scores (scatter plot
     + numerical coefficients)
4. Publication-ready graphics (PNG) for every section:
     * boxplots of TF and of perplexity for the 12 paraphrase types;
     * scatter plot TF vs. perplexity with regression line;
     * bar charts of bucket counts and content-equivalence score counts;
     * boxplots of TF by tag

python e_eval/src/orig_stats.py \
  --data-files e_eval/data/alpaca_gemma-2-2b-it.json \
               e_eval/data/gsm8k_gemma-2-2b-it.json \
               e_eval/data/mmlu_gemma-2-2b-it.json \
  --tag-file a_data/paraphrases_tagged.json \
  --content-score-file a_data/paraphrases_500_phrx_scores_stats.json \
  --original-score-files e_eval/data/ppl_mock_alpaca_gemma-2-2b-it.json \
                         e_eval/data/ppl_mock_gsm8k_gemma-2-2b-it.json \
                         e_eval/data/ppl_mock_mmlu_gemma-2-2b-it.json \
  --output-dir e_eval/analysis_figures
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy import stats

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO, datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
# CONSTANTS & HELPERS
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

TF_INDEX = 0  # position in answer_scores list


def describe_series(series: pd.Series) -> pd.Series:
    """Return count, mean, std, min, 25%, 50%, 75%,
    max just like pandas.describe"""
    desc = series.describe()
    return desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]


# DATA LOADING
def _infer_dataset_name(path: str | Path) -> str:
    """Infer dataset name (alpaca/gsm8k/mmlu) from file path."""
    fname = Path(path).name.lower()
    for name in ("alpaca", "gsm8k", "mmlu"):
        if name in fname:
            return name
    return "unknown"


def load_answer_scores(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load separate answer-score JSONs for original instructions.

    Expected format per file:
    ```
    [
      {
        "prompt_count": 1,
        "answer_scores": [10, 9, ...],
        "perplexity": 123.4
      },
      ...
    ]
    ```
    """
    records: list[dict] = []
    for path in paths:
        dataset = _infer_dataset_name(path)
        logger.info(
            "Loading original-instruction scores from %s (dataset=%s)",
            path, dataset)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for item in data:
            records.append(
                {
                    "dataset": dataset,
                    "prompt_count": item["prompt_count"],
                    "original_tf": item[
                        "answer_scores"][
                            TF_INDEX] if "answer_scores" in item else math.nan,
                    "original_perplexity": item.get("perplexity", math.nan),
                }
            )
    df_scores = pd.DataFrame.from_records(records)
    return df_scores


def load_main_data(path: str | Path) -> pd.DataFrame:
    """Flatten one main JSON file into a long DataFrame."""
    dataset = _infer_dataset_name(path)
    logger.info("Loading main data from %s (dataset=%s)", path, dataset)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    rows: list[dict] = []
    for obj in data:
        prompt_id = obj["prompt_count"]
        input_present = bool(obj.get("input"))
        scenarios_present = bool(obj.get("scenarios"))

        # original instruction row (may not have scores!)
        rows.append(
            {
                "dataset": dataset,
                "prompt_count": prompt_id,
                "paraphrase_type": "instruction_original",
                "bucket": 0,  # N/A; use 0 for original
                "content_score": 5,  # by definition fully equivalent
                "tf_score": obj.get("answer_scores", [math.nan])[
                    TF_INDEX] if "answer_scores" in obj else math.nan,
                "perplexity": obj.get("perplexity", math.nan),
                "input_present": input_present,
                "scenarios_present": scenarios_present,
            }
        )

        # paraphrases
        for p in obj.get("paraphrases", []):
            rows.append(
                {
                    "dataset": dataset,
                    "prompt_count": prompt_id,
                    "paraphrase_type": p["instruct_type"],
                    "bucket": p.get("bucket", math.nan),
                    "content_score": p.get(
                        "paraphrase_content_score", math.nan),
                    "tf_score": p.get("task_score", p.get(
                        "answer_scores", [math.nan])[TF_INDEX]),
                    "perplexity": p.get("perplexity", math.nan),
                    "input_present": input_present,
                    "scenarios_present": scenarios_present,
                }
            )
    df = pd.DataFrame.from_records(rows)
    return df

# ANALYSIS FUNCTIONS
def bucket_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return stats of paraphrase counts per bucket over prompts"""
    # Build table of (# paraphrases in bucket) per prompt
    bucket_counts = (
        df[df["bucket"].between(1, 5)]
        .groupby(["dataset", "prompt_count", "bucket"])
        .size()
        .unstack(fill_value=0)
    )
    stats_rows = {}
    for b in range(1, 6):
        desc = describe_series(bucket_counts.get(b, pd.Series(dtype=int)))
        stats_rows[f"bucket_{b}"] = desc
    return pd.DataFrame(stats_rows).T


def content_score_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """stats of counts per content-equivalence score over prompts"""
    content_counts = (
        df[df["content_score"].between(0, 5)]
        .groupby(["dataset", "prompt_count", "content_score"])
        .size()
        .unstack(fill_value=0)
    )
    stats_rows = {}
    for c in range(0, 6):
        desc = describe_series(content_counts.get(c, pd.Series(dtype=int)))
        stats_rows[f"content_score_{c}"] = desc
    return pd.DataFrame(stats_rows).T


def tf_perplex_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return describe() of TF or perplexity per paraphrase type (selected)"""
    subset = df[df["paraphrase_type"].isin(SELECTED_TYPES)]
    grouped = subset.groupby("paraphrase_type")[col]
    stats = grouped.apply(describe_series).unstack()
    return stats


def correlation_tf_perplex(df: pd.DataFrame) -> Tuple[float, float]:
    """Pearson r and p-value for TF vs perplexity (drop NaNs)."""
    cleaned = df.dropna(subset=["tf_score", "perplexity"])
    if cleaned.empty:
        return math.nan, math.nan
    return stats.pearsonr(cleaned["tf_score"], cleaned["perplexity"])


def ttests_against_reference(
        df: pd.DataFrame, col: str, reference: str = "instruction_original"
        ) -> pd.DataFrame:
    """Welch t-tests of each paraphrase_type against `reference`

    Returns a DataFrame with p-values (Bonferroni-corrected) and effect size
    (Cohen d)
    """
    from statsmodels.stats.multitest import multipletests  # lazy import

    pvals = {}
    ds = df.dropna(subset=[col])
    ref_values = ds[ds["paraphrase_type"] == reference][col]
    for pt, grp in ds.groupby("paraphrase_type"):
        if pt == reference:
            continue
        if len(grp[col]) < 2 or len(ref_values) < 2:
            pvals[pt] = (math.nan, math.nan)
            continue
        tstat, p = stats.ttest_ind(grp[col], ref_values, equal_var=False)
        # Cohen d
        d = (
            grp[col].mean() - ref_values.mean()
        ) / math.sqrt((grp[col].var() + ref_values.var()) / 2)
        pvals[pt] = (p, d)
    # Bonferroni correction
    if pvals:
        names, raw_p = zip(*[(k, v[0]) for k, v in pvals.items()])
        adj = multipletests(raw_p, method="bonferroni")[1]
        pvals = {n: (adj[i], pvals[n][1]) for i, n in enumerate(names)}
    return pd.DataFrame.from_dict(pvals, orient="index", columns=[
        "p_value", "cohen_d"])


# GRAPHICS
def save_boxplot(df: pd.DataFrame, col: str, title: str, path: Path):
    sns_args = dict(vert=True, patch_artist=True)
    plt.figure(figsize=(10, 6))
    order = SELECTED_TYPES
    data = [df[df["paraphrase_type"] == t][col].dropna() for t in order]
    plt.boxplot(data, labels=order, **sns_args)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(col.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_scatter(df: pd.DataFrame, x: str, y: str, title: str, path: Path):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x], df[y], alpha=0.3, s=10)
    # regression line
    m, b = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
    xs = np.array([df[x].min(), df[x].max()])
    plt.plot(xs, m * xs + b, linestyle="--")
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_bar(df_counts: pd.Series, title: str, path: Path):
    plt.figure(figsize=(7, 5))
    ax = df_counts.plot(kind="bar")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# TAG-BASED ANALYSIS
def tag_stats(df: pd.DataFrame, tag_map: dict[str, list[str]]) -> pd.DataFrame:
    """Aggregate TF scores by individual tags for qualif. paraphrase types"""
    tag_to_scores: dict[str, list[float]] = defaultdict(list)
    for pt, tags in tag_map.items():
        if pt not in df["paraphrase_type"].values:
            continue
        scores = df[df["paraphrase_type"] == pt]["tf_score"].dropna().tolist()
        for tag in tags:
            tag_to_scores[tag].extend(scores)
    # Build DataFrame
    records = []
    for tag, vals in tag_to_scores.items():
        series = pd.Series(vals)
        desc = describe_series(series)
        desc_dict = desc.to_dict()
        desc_dict.update({"tag": tag, "n": len(series)})
        records.append(desc_dict)
    result = pd.DataFrame.from_records(records).set_index("tag")
    return result


# MAIN
def main():
    parser = argparse.ArgumentParser(description="Prompt-robustness statistics & plots")
    parser.add_argument("--data-files", nargs="+", required=True, help="Main dataset JSONs (same model)")
    parser.add_argument("--tag-file", required=True, help="JSON mapping paraphrase type ➔ tags")
    parser.add_argument("--content-score-file", required=True, help="JSON with content/equivalence score stats")
    parser.add_argument("--original-score-files", nargs="*", default=[], help="JSONs with original-instruction scores")
    parser.add_argument("--output-dir", required=True, help="Directory to write CSVs and PNGs")
    parser.add_argument("--quiet", action="store_true", help="Reduce log verbosity")
    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # LOAD DATA
    dfs = [load_main_data(p) for p in args.data_files]
    df = pd.concat(dfs, ignore_index=True)

    # enrich with original scores if provided
    if args.original_score_files:
        df_scores = load_answer_scores(args.original_score_files)
        df = df.merge(
            df_scores,
            how="left",
            left_on=["dataset", "prompt_count"],
            right_on=["dataset", "prompt_count"],
        )
        # fill missing values for original rows
        mask = df["paraphrase_type"] == "instruction_original"
        df.loc[mask & df["tf_score"].isna(), "tf_score"] = df.loc[mask, "original_tf"]
        df.loc[mask & df["perplexity"].isna(), "perplexity"] = df.loc[mask, "original_perplexity"]
        df.drop(columns=["original_tf", "original_perplexity"], inplace=True)

    # INPUT FIELD PRESENCE
    input_stats = (
        df[df["paraphrase_type"] == "instruction_original"][["dataset", "input_present"]]
        .drop_duplicates()
        .groupby("dataset")["input_present"]
        .agg(["count", "sum"])
    )
    input_stats["percentage"] = 100 * input_stats["sum"] / input_stats["count"]
    input_stats.to_csv(outdir / "input_field_presence.csv")
    logger.info("Saved input-field presence stats → %s", outdir / "input_field_presence.csv")

    # TF & PERPLEXITY SUMMARY FOR SELECTED TYPES
    tf_stats = tf_perplex_stats(df, "tf_score")
    tf_stats.to_csv(outdir / "tf_stats_selected_types.csv")

    ppl_stats = tf_perplex_stats(df, "perplexity")
    ppl_stats.to_csv(outdir / "perplexity_stats_selected_types.csv")

    # t-tests
    tf_ttests = ttests_against_reference(df, "tf_score")
    tf_ttests.to_csv(outdir / "tf_ttests_vs_original.csv")

    ppl_ttests = ttests_against_reference(df, "perplexity")
    ppl_ttests.to_csv(outdir / "perplexity_ttests_vs_original.csv")

    # plots
    save_boxplot(df, "tf_score", "Task Fulfilment by Paraphrase Type", outdir / "tf_boxplot.png")
    save_boxplot(df, "perplexity", "Perplexity by Paraphrase Type", outdir / "perplexity_boxplot.png")

    # CORRELATION TF ↔ perplexity
    r, p = correlation_tf_perplex(df)
    with open(outdir / "tf_perplexity_correlation.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Pearson r = {r:.4f}\np-value = {p:.4e}\n")
    save_scatter(
        df.dropna(subset=["tf_score", "perplexity"]),
        "perplexity",
        "tf_score",
        "TF vs Perplexity (r = {:.2f})".format(r),
        outdir / "tf_vs_perplexity.png",
    )

    # BUCKET STATISTICS
    bucket_stats_df = bucket_statistics(df)
    bucket_stats_df.to_csv(outdir / "bucket_stats.csv")

    bucket_counts = df[df["bucket"].between(1, 5)]["bucket"].value_counts().sort_index()
    save_bar(bucket_counts, "Paraphrase Count per Bucket", outdir / "bucket_counts.png")

    # CONTENT-EQUIVALENCE SCORE STATISTICS
    content_stats_df = content_score_statistics(df)
    content_stats_df.to_csv(outdir / "content_score_stats.csv")

    content_counts = df[df["content_score"].between(0, 5)]["content_score"].value_counts().sort_index()
    save_bar(content_counts, "Paraphrase Count per Content-Equivalence Score", outdir / "content_score_counts.png")

    # COUNTS OF SCORE 4 & 5 BY PARAPHRASE TYPE
    high_content = df[df["content_score"].isin([4, 5])]
    type_high_counts = high_content.groupby("paraphrase_type").size()
    type_high_counts.to_csv(outdir / "type_high_content_counts.csv")

    qualifying_types = type_high_counts[type_high_counts >= 100].index.tolist()

    # TF SUMMARY FOR QUALIFYING TYPES (OVERALL)
    qual_df = df[df["paraphrase_type"].isin(qualifying_types)]
    qual_tf_stats = qual_df.groupby("paraphrase_type")["tf_score"].apply(describe_series).unstack()
    qual_tf_stats.to_csv(outdir / "qualifying_types_tf_stats_overall.csv")

    # TF SUMMARY FOR QUALIFYING TYPES (PER DATASET)
    per_dataset_stats = (
        qual_df.groupby(["dataset", "paraphrase_type"])["tf_score"].apply(describe_series).unstack()
    )
    per_dataset_stats.to_csv(outdir / "qualifying_types_tf_stats_per_dataset.csv")

    # TAG-BASED ANALYSIS
    with open(args.tag_file, "r", encoding="utf-8") as fh:
        tag_map = json.load(fh)

    tag_df = tag_stats(qual_df, tag_map)
    tag_df.to_csv(outdir / "tag_tf_stats.csv")

    # boxplot TF by tag (top 20 tags by n)
    top_tags = tag_df.sort_values("n", ascending=False).head(20).index

    plt.figure(figsize=(12, 6))
    data = []
    for tag in top_tags:
        # mask rows whose paraphrase_type carries this tag
        mask = qual_df["paraphrase_type"].map(lambda pt: tag in tag_map.get(pt, []))
        data.append(qual_df.loc[mask, "tf_score"].dropna())

    plt.boxplot(data, labels=list(top_tags), patch_artist=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("TF Score")
    plt.title("Task Fulfilment by Tag (Top 20)")
    plt.tight_layout()
    plt.savefig(outdir / "tf_by_tag_boxplot.png", dpi=300)
    plt.close()

    logger.info("All results written to '%s'", outdir)


if __name__ == "__main__":
    main()
