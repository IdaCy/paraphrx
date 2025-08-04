#!/usr/bin/env python3
"""
python3 e_eval/src/eval_scores_direct.py \
    f_finetune/output_inf_ft_50k_scores/8x_attn8_sptarg.json
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
            # stdev requires at least two data points
            stats['stdev'] = statistics.stdev(values) if stats['count'] > 1 else 0.0
    except statistics.StatisticsError as e:
        logging.warning(f"Statistics error for values {values}: {e}")
        stats.update({'mean': None, 'median': None, 'min': None, 'max': None, 'stdev': None})
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute stats for JSON score data.")
    parser.add_argument('input_file', help='Path to the input JSON file')
    args = parser.parse_args()

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

    # Aggregate values per key and overall
    per_key = {}
    overall_values = []

    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            logging.warning(f"Skipping non-dict entry at index {idx}: {entry}")
            continue

        for key, val in entry.items():
            if key == 'prompt_count':
                continue
            if not isinstance(val, list):
                logging.warning(f"Expected list for key '{key}' at index {idx}, got {type(val).__name__}")
                continue
            # Validate each value
            clean_vals = []
            for i, x in enumerate(val):
                try:
                    num = float(x)
                    clean_vals.append(num)
                except (TypeError, ValueError):
                    logging.warning(f"Non-numeric item for key '{key}' at entry {idx}, index {i}: {x}")
            if key not in per_key:
                per_key[key] = []
            per_key[key].extend(clean_vals)
            overall_values.extend(clean_vals)

    # Compute and print per-key stats
    print("Per-key statistics:")
    header = f"{'Key':<30} {'Count':>7} {'Mean':>10} {'Median':>10} {'Min':>7} {'Max':>7} {'Stdev':>10}"
    print(header)
    print('-' * len(header))
    for key, values in sorted(per_key.items()):
        stats = compute_stats(values)
        print(f"{key:<30} {stats['count']:7d} {stats['mean'] or 0:10.2f} {stats['median'] or 0:10.2f} {stats['min'] or 0:7.2f} {stats['max'] or 0:7.2f} {stats['stdev'] or 0:10.2f}")

    # Compute and print overall stats
    print("\nOverall statistics across all keys:")
    overall_stats = compute_stats(overall_values)
    print(f"Count: {overall_stats['count']}")
    print(f"Mean: {overall_stats['mean']:.2f}")
    print(f"Median: {overall_stats['median']:.2f}")
    print(f"Min: {overall_stats['min']:.2f}")
    print(f"Max: {overall_stats['max']:.2f}")
    print(f"Stdev: {overall_stats['stdev']:.2f}")


if __name__ == '__main__':
    main()
