#!/usr/bin/env python3
"""
python3 e_eval/src/eval_scores_direct.py \
    f_finetune/output_inf_ft_50k_scores/8x_attn8_sptarg.json

python3 e_eval/src/eval_scores_direct.py \
    c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
    --keys \
    instruct_original \
    instruct_coord_to_subord instruct_dramatic \
    instruct_future_tense instruct_joke \
    instruct_one_typo_punctuation instruct_sardonic \
    instruct_polite_request instruct_double_negative \
    instruct_leet_speak instruct_output_markdown \
    instruct_formal_memo
"""
import json
import argparse
import logging
import statistics
import sys


def compute_stats(values):
    """Compute statistics for vals"""
    stats = {}
    try:
        stats['count'] = len(values)
        if stats['count'] == 0:
            stats.update({'mean': None, 'median': None, 'min': None, 'max': None, 'stdev': None})
        else:
            stats['mean'] = statistics.mean(values)
            stats['median'] = statistics.median(values)
            stats['min'] = min(values)
            stats['max'] = max(values)
            stats['stdev'] = statistics.stdev(values) if stats['count'] > 1 else 0.0
    except statistics.StatisticsError as e:
        logging.warning(f"Statistics error for values {values}: {e}")
        stats.update({'mean': None, 'median': None, 'min': None, 'max': None, 'stdev': None})
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute stats for JSON score data, optionally filtered by specific keys, including first-score stats."
    )
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument(
        '--keys', '-k',
        nargs='+',
        help='List of keys to include (default: all except prompt_count)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load JSON data
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON file: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        logging.error("Unexpected JSON format: top-level element is not a list")
        sys.exit(1)

    # Determine which keys to process
    filter_keys = set(args.keys) if args.keys else None
    if filter_keys:
        logging.info(f"Filtering to keys: {', '.join(filter_keys)}")

    per_key = {}
    per_key_first = {}
    overall_values = []
    overall_first_values = []

    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            logging.warning(f"Skipping non-dict entry at index {idx}: {entry}")
            continue

        for key, val in entry.items():
            if key == 'prompt_count':
                continue
            if filter_keys and key not in filter_keys:
                continue
            if not isinstance(val, list):
                logging.warning(f"Expected list for key '{key}' at index {idx}, got {type(val).__name__}")
                continue

            # Collect all scores
            clean_vals = []
            for i, x in enumerate(val):
                try:
                    clean_vals.append(float(x))
                except (TypeError, ValueError):
                    logging.warning(f"Non-numeric item for key '{key}' at entry {idx}, index {i}: {x}")
            if clean_vals:
                per_key.setdefault(key, []).extend(clean_vals)
                overall_values.extend(clean_vals)

                # Collect first score only
                first = clean_vals[0]
                per_key_first.setdefault(key, []).append(first)
                overall_first_values.append(first)
            else:
                logging.warning(f"No valid scores for key '{key}' at entry {idx}")

    if not per_key:
        logging.warning("No data collected for the specified keys.")

    # Print per-key stats
    header = f"{'Key':<30} {'Count':>7} {'Mean':>10} {'Median':>10} {'Min':>7} {'Max':>7} {'Stdev':>10}"

    print("Per-key statistics:")
    print(header)
    print('-' * len(header))
    for key in sorted(per_key):
        stats = compute_stats(per_key[key])
        print(f"{key:<30} {stats['count']:7d} {stats['mean'] or 0:10.2f} {stats['median'] or 0:10.2f} {stats['min'] or 0:7.2f} {stats['max'] or 0:7.2f} {stats['stdev'] or 0:10.2f}")

    # Print overall stats
    print("\nOverall statistics across all selected keys:")
    overall_stats = compute_stats(overall_values)
    print(f"Count: {overall_stats['count']}")
    print(f"Mean: {overall_stats['mean']:.2f}")
    print(f"Median: {overall_stats['median']:.2f}")
    print(f"Min: {overall_stats['min']:.2f}")
    print(f"Max: {overall_stats['max']:.2f}")
    print(f"Stdev: {overall_stats['stdev']:.2f}")

    # Print per-key first-score stats
    print("\nPer-key first-score (Task Fulfilment / Relevance) statistics:")
    print(header)
    print('-' * len(header))
    for key in sorted(per_key_first):
        stats = compute_stats(per_key_first[key])
        print(f"{key:<30} {stats['count']:7d} {stats['mean'] or 0:10.2f} {stats['median'] or 0:10.2f} {stats['min'] or 0:7.2f} {stats['max'] or 0:7.2f} {stats['stdev'] or 0:10.2f}")

    # Print overall first-score stats
    print("\nOverall first-score statistics:")
    overall_first_stats = compute_stats(overall_first_values)
    print(f"Count: {overall_first_stats['count']}")
    print(f"Mean: {overall_first_stats['mean']:.2f}")
    print(f"Median: {overall_first_stats['median']:.2f}")
    print(f"Min: {overall_first_stats['min']:.2f}")
    print(f"Max: {overall_first_stats['max']:.2f}")
    print(f"Stdev: {overall_first_stats['stdev']:.2f}")

if __name__ == '__main__':
    main()
