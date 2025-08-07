#!/usr/bin/env python3
"""
By default, only the first element of each instruct* array is used
If --all-scores is specified, uses the average of all 10 values
If --instruct-keys is omitted, processes all keys starting with "instruct"

Outputs:
  - Markdown file "stats.md" in the output directory with tables for count, mean, median, min, max, stdev
  - Three PNG graphics: distribution.png, boxplot.png, cdf.png
  - Logs unexpected events to "script.log" in the output directory

python e_eval/src/inf_score_stats.py \
    --in-jsons \
    c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
    f_finetune/output_inf_ft_50k_scores/8x_attn8_sptarg.json \
    f_finetune/output_inf_ft_50k_scores/all_a1_notarg_inf.json \
    f_finetune/output_inf_ft_50k_scores/all_b2_2_sptarg.json \
    f_finetune/output_inf_ft_50k_scores/h9x_loramlp12_sptarg.json \
    f_finetune/output_inf_ft_50k_scores/l9x_b2_sptarg_50k_ft.json \
    f_finetune/output_inf_ft_50k_scores/li9x_a1_notarg_inf.json \
    --out-dir f_finetune/out_inf_score_stats \
    --instruct-keys \
    instruction_original \
    instruct_coord_to_subord instruct_dramatic \
    instruct_future_tense instruct_joke \
    instruct_one_typo_punctuation instruct_sardonic \
    instruct_polite_request instruct_double_negative \
    instruct_leet_speak instruct_output_markdown \
    instruct_formal_memo
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Global Issue Logger
ISSUES_LOG = []


def log_issue(message):
    """Accumulates unexpected events or errors for a final summary"""
    ISSUES_LOG.append(message)


def get_file_basename(filepath):
    """Extracts a unique name from the file path, e.g., 'path/to/xyz.json' -> 'xyz'"""
    return os.path.basename(filepath).rsplit('.', 1)[0]


def parse_arguments():
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize instruction-following scores from JSON files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--in-jsons',
        nargs='+',
        required=True,
        help="One or more paths to the input JSON files."
    )
    parser.add_argument(
        '--out-dir',
        required=True,
        help="Directory to save the output markdown report and graphics."
    )
    parser.add_argument(
        '--score-mode',
        choices=['first', 'average'],
        default='first',
        help="How to calculate the score from the 10-number array:\n"
             "'first': Use only the first value (default).\n"
             "'average': Use the average of all 10 values."
    )
    parser.add_argument(
        '--instruct-keys',
        nargs='*',
        default=None,
        help="Specific instruct* keys to process. If not provided, all keys\n"
             "starting with 'instruct_' will be processed."
    )
    return parser.parse_args()


def process_files(in_files, score_mode, specific_instruct_keys):
    """
    Loads data from JSON files, processes scores, and returns a structured DataFrame
    """
    all_scores_data = []
    
    for file_path in in_files:
        basename = get_file_basename(file_path)
        print(f"Processing '{basename}'...")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            log_issue(f"File not found: {file_path}")
            continue
        except json.JSONDecodeError:
            log_issue(f"Invalid JSON in file: {file_path}")
            continue

        if not isinstance(data, list):
            log_issue(f"JSON in {file_path} is not a list of objects.")
            continue

        for i, record in enumerate(data):
            keys_to_process_for_this_record = []
            if specific_instruct_keys is not None:
                # If user specified keys, that's our list.
                keys_to_process_for_this_record = specific_instruct_keys
            else:
                # Otherwise, discover keys from this specific record.
                keys_to_process_for_this_record = [k for k in record.keys() if k.startswith('instruct_')]

            for key in keys_to_process_for_this_record:
                # If a key is not in the record, just silently skip it and check the next key
                if key not in record:
                    continue
                
                scores_array = record[key]
                if not isinstance(scores_array, list) or not scores_array:
                    log_issue(f"Key '{key}' in record {i} of {basename}.json is not a non-empty list.")
                    continue

                try:
                    numeric_scores = [float(s) for s in scores_array]
                    
                    score = 0.0
                    if score_mode == 'first':
                        score = numeric_scores[0]
                    elif score_mode == 'average':
                        score = np.mean(numeric_scores)
                    
                    all_scores_data.append({
                        'file': basename,
                        'instruct_key': key,
                        'score': score
                    })
                except (ValueError, TypeError) as e:
                    log_issue(f"Non-numeric value in scores for key '{key}' in record {i} of {basename}.json. Error: {e}")
                except IndexError:
                    log_issue(f"Score array for key '{key}' in record {i} of {basename}.json is empty.")

    if not all_scores_data:
        return None

    return pd.DataFrame(all_scores_data)


def generate_markdown_report(df, out_dir):
    """Generates and saves the statistics tables to a markdown file"""
    if df is None or df.empty:
        return

    stats = df.groupby(['instruct_key', 'file'])['score'].agg(['count', 'mean', 'median', 'min', 'max', 'std']).reset_index()
    stats['std'] = stats['std'].fillna(0) 

    report_path = os.path.join(out_dir, "statistics_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Instruction Score Analysis Report\n\n")
        f.write("This report summarizes the performance scores across different models and instruction types.\n\n")

        stat_metrics = ['mean', 'median', 'std', 'count', 'min', 'max']
        for metric in stat_metrics:
            f.write(f"## Table of {metric.title()} Scores\n\n")
            pivot_table = stats.pivot(index='instruct_key', columns='file', values=metric)
            f.write(pivot_table.to_markdown(floatfmt=".2f"))
            f.write("\n\n")

    print(f"Statistics report saved to: {report_path}")


def create_distribution_plot(df, out_dir):
    """Creates a KDE plot comparing the overall score distributions for each file"""
    if df is None or df.empty:
        return
        
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    files = df['file'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))

    for i, file_name in enumerate(files):
        file_df = df[df['file'] == file_name]
        scores = file_df['score']
        
        if len(scores) < 2:
            log_issue(f"Cannot draw distribution for '{file_name}' as it has fewer than 2 data points.")
            continue

        mean_val, median_val, count_val = scores.mean(), scores.median(), len(scores)
        label = f"{file_name} (n={count_val}, median={median_val:.2f}, mean={mean_val:.2f})"
        
        sns.kdeplot(scores, ax=ax, label=label, color=colors[i], fill=True, alpha=0.1)
        ax.axvline(median_val, color=colors[i], linestyle='-', linewidth=2)
        ax.axvline(mean_val, color=colors[i], linestyle=':', linewidth=2, alpha=0.8)

    ax.set_title('Score Distribution Comparison by File', fontsize=16)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(title="File (Count, Median, Mean)")
    ax.set_xlim(0, 10.5)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "distribution_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Distribution plot saved to: {plot_path}")


def create_box_plot(df, out_dir):
    """Creates a box plot to compare score distributions across files"""
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=df, x='file', y='score', palette='viridis')
    plt.title('Score Spread Comparison (Box Plot)', fontsize=16)
    plt.xlabel('File Name', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "score_boxplot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Box plot saved to: {plot_path}")


def create_barchart_comparison(df, out_dir):
    """Creates a grouped bar chart to compare mean scores per instruction key"""
    if df is None or df.empty: return
    num_keys = df['instruct_key'].nunique()
    if num_keys > 20: print(f"Warning: Barchart comparison might be crowded with {num_keys} instruction keys.")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(max(12, num_keys * 0.5), 8))
    sns.barplot(data=df, x='instruct_key', y='score', hue='file', palette='muted', estimator=np.mean, errorbar=None)
    plt.title('Mean Score by Instruction Type', fontsize=16)
    plt.xlabel('Instruction Key', fontsize=12)
    plt.ylabel('Mean Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='File')
    plt.ylim(0, 10.5)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "mean_score_by_instruction.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Bar chart saved to: {plot_path}")


def main():
    args = parse_arguments()
    try:
        os.makedirs(args.out_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory '{args.out_dir}'. {e}", file=sys.stderr)
        sys.exit(1)

    print(f"--- Starting Analysis ---")
    print(f"Score Mode: '{args.score_mode}'")
    if args.instruct_keys:
        print(f"Processing ONLY specified keys: {', '.join(args.instruct_keys)}")
    else:
        print("Processing all 'instruct_*' keys found in each record.")
    
    master_df = process_files(args.in_jsons, args.score_mode, args.instruct_keys)

    if master_df is None or master_df.empty:
        print("\nNo valid data could be processed from the input files.", file=sys.stderr)
    else:
        print("\n--- Generating Outputs ---")
        generate_markdown_report(master_df, args.out_dir)
        create_distribution_plot(master_df, args.out_dir)
        create_box_plot(master_df, args.out_dir)
        create_barchart_comparison(master_df, args.out_dir)
    
    print("\n--- Analysis Complete ---")
    if ISSUES_LOG:
        print(f"\nEncountered {len(ISSUES_LOG)} issue(s) during processing:")
        issue_counts = defaultdict(int)
        for issue in ISSUES_LOG: issue_counts[issue] += 1
        for issue, count in issue_counts.items():
            print(f"  - {issue} (occurred {count} time(s))")
    else:
        print("Processing completed without any issues.")

if __name__ == '__main__':
    main()
