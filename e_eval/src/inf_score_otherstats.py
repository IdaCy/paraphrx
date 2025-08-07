"""
python e_eval/src/inf_score_otherstats.py \
    --in-baseline \
    c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
    --in-jsons \
    f_finetune/output_inf_ft_50k_scores/all_a1_notarg_inf.json \
    f_finetune/output_inf_ft_50k_scores/all_b2_2_sptarg.json \
    f_finetune/output_inf_ft_50k_scores/li9x_a1_notarg_inf.json \
    --out-dir f_finetune/out_inf_score_otherstats_spec_3 \
    --instruct-keys \
    instruction_original \
    instruct_coord_to_subord instruct_dramatic \
    instruct_future_tense instruct_joke \
    instruct_one_typo_punctuation \
    instruct_polite_request instruct_double_negative \
    instruct_leet_speak \
    instruct_formal_memo
"""

#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Helper Functions


def get_file_name(path_str: str) -> str:
    """Extracts the file name without the extension and path"""
    return Path(path_str).stem


def calculate_score(scores: list, mode: str, error_log: list, context: str) -> float:
    """Calculates the score based on the chosen mode, with error handling"""
    if not isinstance(scores, list):
        error_log.append(f"'{context}': Value is not a list, but {type(scores)}.")
        return np.nan
    if len(scores) != 10:
        error_log.append(f"'{context}': Expected 10 scores, but found {len(scores)}.")
        return np.nan

    try:
        # Ensure all elements are numbers
        numeric_scores = [float(s) for s in scores]
    except (ValueError, TypeError) as e:
        error_log.append(f"'{context}': Contains non-numeric values. Error: {e}")
        return np.nan

    if mode == 'first':
        return numeric_scores[0]
    elif mode == 'all':
        return np.mean(numeric_scores)
    else:
        # This case should be prevented by argparse choices
        raise ValueError("Invalid score mode")

# Data Loading and Processing


def load_and_process_data(
    json_files: list[str],
    baseline_file: str,
    score_mode: str,
    instruct_keys_to_process: list[str] | None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Loads data from JSON files, processes it into a pandas DataFrame,
    and logs any encountered errors.
    """
    all_data = []
    error_log = []

    all_input_files = [baseline_file] + json_files

    # Discover instruct keys from the first file if not provided
    if not instruct_keys_to_process:
        print("No --instruct-keys provided. Discovering them from the baseline file...")
        try:
            with open(baseline_file, 'r') as f:
                sample_data = json.load(f)
            if not sample_data:
                sys.exit(f"ERROR: Baseline file '{baseline_file}' is empty. Cannot discover keys.")
            
            # instruction_original is treated like any other instruct* key
            instruct_keys_to_process = [
                key for key in sample_data[0].keys() if key.startswith("instruct")
            ]
            print(f"Found {len(instruct_keys_to_process)} keys to process: {instruct_keys_to_process}")
        except (IOError, json.JSONDecodeError, IndexError) as e:
            sys.exit(f"ERROR: Could not read or parse baseline file '{baseline_file}' to discover keys. Details: {e}")

    # Ensure a set for faster lookups
    instruct_keys_set = set(instruct_keys_to_process)

    for file_path in all_input_files:
        file_name = get_file_name(file_path)
        print(f"Processing '{file_path}' as '{file_name}'...")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            error_log.append(f"CRITICAL: Could not read or parse file '{file_path}'. Skipping. Error: {e}")
            continue

        for record in data:
            prompt_count = record.get("prompt_count")
            if prompt_count is None:
                error_log.append(f"'{file_name}': Found a record without 'prompt_count'. Skipping record.")
                continue

            for key, value in record.items():
                if key in instruct_keys_set:
                    context = f"File '{file_name}', prompt_count {prompt_count}, key '{key}'"
                    score = calculate_score(value, score_mode, error_log, context)
                    if not np.isnan(score):
                        all_data.append({
                            "ft_regime": file_name,
                            "prompt_count": prompt_count,
                            "instruct_key": key,
                            "score": score
                        })
                        
    if not all_data:
        sys.exit("ERROR: No valid data could be processed. Please check your input files and parameters. Check error log for details.")

    return pd.DataFrame(all_data), error_log


# Analysis and Output Functions
def generate_statistics_report(df: pd.DataFrame, out_dir: str, score_mode: str):
    """Generates and saves the main statistics table to a Markdown file"""
    print("Generating statistics report...")
    stats_df = df.groupby(['ft_regime', 'instruct_key'])['score'].agg(
        ['mean', 'median', 'min', 'max', 'std']
    ).reset_index()
    stats_df['std'] = stats_df['std'].fillna(0) # Std dev is NaN for single-entry groups

    # Pivot for better comparison
    mean_pivot = stats_df.pivot(index='instruct_key', columns='ft_regime', values='mean')
    median_pivot = stats_df.pivot(index='instruct_key', columns='ft_regime', values='median')

    report_path = os.path.join(out_dir, "statistics_report.md")
    with open(report_path, 'w') as f:
        f.write(f"# Analysis Report\n\n")
        f.write(f"This report was generated using the **'{score_mode}'** score mode.\n\n")

        f.write("## Mean Scores\n\n")
        f.write(mean_pivot.to_markdown(floatfmt=".2f"))
        f.write("\n\n")

        f.write("## Median Scores\n\n")
        f.write(median_pivot.to_markdown(floatfmt=".2f"))
        f.write("\n\n")

        f.write("## Full Statistics\n\n")
        f.write(stats_df.to_markdown(index=False, floatfmt=".2f"))
        f.write("\n")

    print(f"  -> Saved report to {report_path}")


def analyze_and_plot_percentage_change(df: pd.DataFrame, baseline_name: str, out_dir: str):
    """Calculates and plots the percentage change from the baseline"""
    print("Analyzing and plotting percentage change from baseline...")

    mean_scores = df.groupby(['ft_regime', 'instruct_key'])['score'].mean().reset_index()

    baseline_scores = mean_scores[mean_scores['ft_regime'] == baseline_name].set_index('instruct_key')['score']
    other_scores = mean_scores[mean_scores['ft_regime'] != baseline_name]

    if baseline_scores.empty:
        print("  -> WARNING: No baseline scores found. Skipping percentage change analysis.")
        return

    # Calculate percentage change
    results = []
    for _, row in other_scores.iterrows():
        key = row['instruct_key']
        if key in baseline_scores:
            base_score = baseline_scores[key]
            if base_score > 0:
                change = ((row['score'] - base_score) / base_score) * 100
            elif row['score'] > 0:
                change = np.inf # Or some large number to indicate change from zero
            else:
                change = 0.0 # Zero to zero is no change
        else:
            change = np.nan # Baseline doesn't have this key

        results.append({
            'ft_regime': row['ft_regime'],
            'instruct_key': key,
            '%_change': change
        })

    if not results:
        print("  -> No other files to compare with baseline. Skipping percentage change analysis.")
        return

    change_df = pd.DataFrame(results).dropna()

    # Save table to report
    change_pivot = change_df.pivot(index='instruct_key', columns='ft_regime', values='%_change')
    report_path = os.path.join(out_dir, "statistics_report.md")
    with open(report_path, 'a') as f:
        f.write("\n## Percentage Change from Baseline\n\n")
        f.write(f"Percentage change in mean scores relative to **'{baseline_name}'**.\n\n")
        f.write(change_pivot.to_markdown(floatfmt=".2f"))
        f.write("\n")
    print(f"  -> Appended percentage change table to report.")

    # Create plot
    plt.figure(figsize=(15, 8))
    sns.barplot(data=change_df, x='instruct_key', y='%_change', hue='ft_regime')
    plt.title(f"Percentage Change in Mean Score vs. Baseline ('{baseline_name}')")
    plt.ylabel("% Change")
    plt.xlabel("Instruction Key")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "percentage_change_from_baseline.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  -> Saved plot to {plot_path}")


def plot_score_distributions(df: pd.DataFrame, out_dir: str):
    """Plots histograms, violin plots, and box plots of the score distributions"""
    print("Plotting score distributions...")

    # Histogram
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Use a color palette
    palette = sns.color_palette("viridis", n_colors=df['ft_regime'].nunique())

    for i, (name, group) in enumerate(df.groupby('ft_regime')):
        # Calculate stats for the legend and lines first
        mean_val = group['score'].mean()
        median_val = group['score'].median()
        count_val = len(group['score'])

        # Create a single, informative label for the legend
        legend_label = f'{name} (n={count_val}, mean={mean_val:.2f}, median={median_val:.2f})'

        # Plot histogram with the CORRECTED binrange parameter
        sns.histplot(group['score'], bins=11, binrange=(0,11), stat='count', 
                     alpha=0.5, label=legend_label, color=palette[i], ax=ax, 
                     element="step", fill=True)

        # Add lines for mean and median (without adding them to the legend)
        ax.axvline(median_val, color=palette[i], linestyle='-', linewidth=2)
        ax.axvline(mean_val, color=palette[i], linestyle=':', linewidth=2, alpha=0.8)

    # Add a simple legend with a title
    ax.legend(title="File Name")
    ax.set_title("Score Distribution Histogram Comparison")
    ax.set_xlabel("Score (Dotted line: Mean, Solid line: Median)")
    ax.set_ylabel("Count")
    ax.set_xticks(range(0, 11))

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "distribution_histogram.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  -> Saved histogram to {plot_path}")

    # Violin Plot
    plt.figure(figsize=(12, 7))
    sns.violinplot(data=df, x='ft_regime', y='score', inner='quartile')
    plt.title("Score Distribution Violin Plot")
    plt.xlabel("File Name")
    plt.ylabel("Score")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "distribution_violin_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  -> Saved violin plot to {plot_path}")

    # Box Plot
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x='ft_regime', y='score')
    plt.title("Score Distribution Box Plot")
    plt.xlabel("File Name")
    plt.ylabel("Score")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "distribution_box_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  -> Saved box plot to {plot_path}")


def plot_heatmap(df: pd.DataFrame, out_dir: str):
    """Plots a heatmap of mean scores: instruct keys vs. file names"""
    print("Generating bonus plot: Mean Score Heatmap...")
    pivot_table = df.pivot_table(
        values='score', 
        index='instruct_key', 
        columns='ft_regime', 
        aggfunc='mean'
    )
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
    plt.title("Heatmap of Mean Scores")
    plt.xlabel("File Name")
    plt.ylabel("Instruction Key")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "bonus_heatmap.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  -> Saved heatmap to {plot_path}")


def plot_paired_scatter(df: pd.DataFrame, baseline_name: str, out_dir: str):
    """Plots scatter plots comparing each file to the baseline on a per-item basis"""
    print("Generating bonus plot: Paired Score Scatter Plots...")

    baseline_df = df[df['ft_regime'] == baseline_name].set_index(['prompt_count', 'instruct_key'])['score']
    other_files = df[df['ft_regime'] != baseline_name]['ft_regime'].unique()

    for other_file in other_files:
        other_df = df[df['ft_regime'] == other_file].set_index(['prompt_count', 'instruct_key'])['score']

        # Merge on the shared index (prompt_count, instruct_key)
        comparison_df = pd.concat([baseline_df, other_df], axis=1, keys=[baseline_name, other_file]).dropna()

        if comparison_df.empty:
            print(f"  -> No common (prompt_count, instruct_key) pairs between '{baseline_name}' and '{other_file}'. Skipping scatter plot.")
            continue
            
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=comparison_df, x=baseline_name, y=other_file, alpha=0.6)
        # Add y=x line for reference
        max_val = df['score'].max()
        if pd.isna(max_val): max_val = 10 # Fallback
        plt.plot([0, max_val], [0, max_val], 'r--', label='y=x (No Change)')
        plt.title(f"Paired Scores: '{other_file}' vs. Baseline '{baseline_name}'")
        plt.xlabel(f"Score in '{baseline_name}'")
        plt.ylabel(f"Score in '{other_file}'")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.axis('equal') # Ensure a square plot for easy comparison
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"bonus_scatter_{other_file}_vs_{baseline_name}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  -> Saved scatter plot for '{other_file}' to {plot_path}")


# Main Execution
def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize scores from JSON evaluation files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--in-baseline',
        required=True,
        help="Path to the baseline JSON file. This file is used for comparison."
    )
    parser.add_argument(
        '--in-jsons',
        nargs='*',
        default=[],
        help="Paths to other JSON files to compare against the baseline."
    )
    parser.add_argument(
        '--out-dir',
        required=True,
        help="Directory to save the output reports and plots."
    )
    parser.add_argument(
        '--score-mode',
        choices=['first', 'all'],
        default='first',
        help="How to calculate the score from the 10-number array:\n"
             "'first': Use only the first number (default).\n"
             "'all': Use the average of all 10 numbers."
    )
    parser.add_argument(
        '--instruct-keys',
        nargs='*',
        help="Specific instruct* keys to process. If not provided, all keys starting with 'instruct' will be processed."
    )

    args = parser.parse_args()

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Data Processing
    df, error_log = load_and_process_data(
        args.in_jsons,
        args.in_baseline,
        args.score_mode,
        args.instruct_keys
    )

    baseline_name = get_file_name(args.in_baseline)

    # Analysis & Visualization
    generate_statistics_report(df, args.out_dir, args.score_mode)
    if args.in_jsons:
        analyze_and_plot_percentage_change(df, baseline_name, args.out_dir)

    plot_score_distributions(df, args.out_dir)
    plot_heatmap(df, args.out_dir)

    if args.in_jsons:
        plot_paired_scatter(df, baseline_name, args.out_dir)

    # Final Error Summary
    if error_log:
        print("\n" + "="*50)
        print(f"Completed with {len(error_log)} non-critical errors/warnings.")
        print("="*50)
        report_path = os.path.join(args.out_dir, "statistics_report.md")
        with open(report_path, 'a') as f:
            f.write("\n## Data Quality Issues\n\n")
            f.write(f"A total of **{len(error_log)}** non-critical issues were found during data processing.\n")
            f.write("A sample of these issues is listed below:\n\n")
            for i, error in enumerate(error_log[:20]):  # Log first 20 errors
                f.write(f"- `{error}`\n")
            if len(error_log) > 20:
                f.write(f"- ... and {len(error_log) - 20} more.\n")
    else:
        print("\n" + "="*50)
        print("Completed successfully with no data processing errors.")
        print("="*50)

    print(f"All outputs have been saved to the '{args.out_dir}' directory.")


if __name__ == "__main__":
    main()
