"""

python3 e_eval/src/tf_stats.py \
<<<<<<< HEAD
    --orig_json e_eval/data/alpaca_gemma-2-2b-it.json \
=======
    --orig_json f_finetune/data/all_alpaca_gemma-2-2b-it_all_ppl.json \
>>>>>>> 04f99b35685c6c73ef5f8a29ebc98ca5580f0fc1
    --finetune_jsons \
    f_finetune/outputs/1st_run/alpaca/all_layers/ft_inf_scores/buckets_1-1.json \
    f_finetune/outputs/1st_run/alpaca/all_layers/ft_inf_scores/buckets_1-2.json \
    f_finetune/outputs/1st_run/alpaca/all_layers/ft_inf_scores/buckets_1-3.json \
    f_finetune/outputs/1st_run/alpaca/all_layers/ft_inf_scores/buckets_1-4.json \
<<<<<<< HEAD
    f_finetune/outputs/1st_run/alpaca/all_layers/ft_inf_scores/buckets_1-5.json \
    --equiv_json a_data/alpaca/equi_scores/median_equi_integer.json \
    --log_file logs/stats_run.log \
=======
    equiv_json/outputs/1st_run/alpaca/all_layers/ft_inf_scores/buckets_1-5.json \
    --equivalence a_data/alpaca/equi_scores/median_equi_integer.json \
    --log-log_file logs/stats_run.log \
>>>>>>> 04f99b35685c6c73ef5f8a29ebc98ca5580f0fc1
    --output_dir e_eval/output/alpaca/tf_stats
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized to {log_file}")


def load_original_paraphrases(path: str) -> pd.DataFrame:
    """
    Load the JSON where each prompt_count entry lists its paraphrases and TF scores.
    Returns a DataFrame with columns: prompt_count, instruct_type, tf_score
    """
    logging.info(f"Loading original paraphrases from {path}")
    with open(path) as f:
        data = json.load(f)

    records = []
    for entry in data:
        pc = entry['prompt_count']
        for p in entry.get('paraphrases', []):
            tf = p['answer_scores'][0]
            records.append({'prompt_count': pc,
                            'instruct_type': p['instruct_type'],
                            'tf_score': tf})
    df = pd.DataFrame(records)
    df['stage'] = 'original'
    logging.info(f"Original paraphrases loaded: {len(df)} records")
    return df


def load_finetune_scores(paths: list) -> pd.DataFrame:
    """
    Load post-finetuning JSONs. Each is a list of dicts, each dict has keys instruct_type -> [scores], plus prompt_count.
    Returns a DataFrame with columns: prompt_count, instruct_type, tf_score, stage
    """
    all_records = []
    for path in paths:
        label = os.path.splitext(os.path.basename(path))[0]
        logging.info(f"Loading fine-tuned scores from {path} as stage {label}")
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            pc = entry['prompt_count']
            for instr, scores in entry.items():
                if instr == 'prompt_count':
                    continue
                tf = scores[0]
                all_records.append({'prompt_count': pc,
                                    'instruct_type': instr,
                                    'tf_score': tf,
                                    'stage': label})
    df = pd.DataFrame(all_records)
    logging.info(f"Fine-tuned scores loaded: {len(df)} records")
    return df


def load_equivalence(path: str) -> pd.DataFrame:
    """
    Load JSON mapping instruct_type to equivalence score.
    Returns a DataFrame with columns: instruct_type, eq_score
    """
    logging.info(f"Loading equivalence scores from {path}")
    with open(path) as f:
        eq_map = json.load(f)
    df = pd.DataFrame([{'instruct_type': k, 'eq_score': v} for k, v in eq_map.items()])
    logging.info(f"Equivalence mapping loaded: {len(df)} types")
    return df


def compute_stats(df: pd.DataFrame) -> dict:
    """
    Compute mean, median, min, max of tf_score in df. Returns dict.
    """
    return {
        'mean': df['tf_score'].mean(),
        'median': df['tf_score'].median(),
        'min': df['tf_score'].min(),
        'max': df['tf_score'].max(),
        'count': len(df)
    }


def main():
    parser = argparse.ArgumentParser(description="Compute prompt robustness statistics and plots.")
    parser.add_argument('--orig_json', required=True, help="Path to original paraphrases JSON")
    parser.add_argument('--finetune_jsons', nargs='+', required=True, help="Paths to fine-tuned scores JSONs")
    parser.add_argument('--equiv_json', required=True, help="Path to paraphrase equivalence JSON")
    parser.add_argument('--log_file', required=True, help="Path to log file")
    parser.add_argument('--output_dir', default='output', help="Directory to save tables and plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.log_file)

    # Load data
    orig_df = load_original_paraphrases(args.orig_json)
    ft_df = load_finetune_scores(args.finetune_jsons)
    eq_df = load_equivalence(args.equiv_json)

    # Combine original and finetuned
    all_df = pd.concat([orig_df, ft_df], ignore_index=True)

    # Merge equivalence scores
    all_df = all_df.merge(eq_df, on='instruct_type', how='left')
    logging.info("Merged equivalence scores; NaNs if instr type not found.")

    # Define high/low equivalence
    high_eq = all_df[all_df['eq_score'] >= 4]
    low_eq = all_df[all_df['eq_score'] <= 3]

    # Prepare tables
    stages = all_df['stage'].unique()

    table1 = []  # overall paraphrase TF stats by stage
    table2 = []  # high_eq stats by stage
    table3 = []  # low_eq stats by stage
    table4 = []  # comparison high_eq original vs after-ft

    # Table 1-3
    for s in stages:
        sub = all_df[all_df['stage'] == s]
        stats_all = compute_stats(sub)
        stats_high = compute_stats(high_eq[high_eq['stage'] == s])
        stats_low = compute_stats(low_eq[low_eq['stage'] == s])
        table1.append({'stage': s, **stats_all})
        table2.append({'stage': s, **stats_high})
        table3.append({'stage': s, **stats_low})

    df1 = pd.DataFrame(table1).set_index('stage')
    df2 = pd.DataFrame(table2).set_index('stage')
    df3 = pd.DataFrame(table3).set_index('stage')

    # Table 4: compare high_eq before vs each ft stage
    orig_high = high_eq[high_eq['stage'] == 'original']['tf_score']
    for s in stages:
        if s == 'original':
            continue
        after_high = high_eq[high_eq['stage'] == s]['tf_score']
        stats_orig = compute_stats(orig_high.to_frame(name='tf_score'))
        stats_after = compute_stats(after_high.to_frame(name='tf_score'))
        # t-test
        t_stat, p_val = ttest_ind(orig_high, after_high, equal_var=False, nan_policy='omit')
        table4.append({
            'stage': s,
            'orig_mean': stats_orig['mean'],
            'after_mean': stats_after['mean'],
            'mean_diff': stats_after['mean'] - stats_orig['mean'],
            'orig_median': stats_orig['median'],
            'after_median': stats_after['median'],
            'median_diff': stats_after['median'] - stats_orig['median'],
            't_stat': t_stat,
            'p_value': p_val
        })
    df4 = pd.DataFrame(table4).set_index('stage')

    # Save tables
    df1.to_csv(os.path.join(args.output_dir, 'table1_overall_tf_stats.csv'))
    df2.to_csv(os.path.join(args.output_dir, 'table2_high_eq_tf_stats.csv'))
    df3.to_csv(os.path.join(args.output_dir, 'table3_low_eq_tf_stats.csv'))
    df4.to_csv(os.path.join(args.output_dir, 'table4_high_eq_comparison.csv'))
    logging.info("Saved statistical tables to CSV")

    # Plotting
<<<<<<< HEAD
    # 1. Mean TF by stage
=======
    # Mean TF by stage
>>>>>>> 04f99b35685c6c73ef5f8a29ebc98ca5580f0fc1
    plt.figure()
    df1['mean'].plot(kind='bar')
    plt.title('Mean TF Score by Stage (All paraphrases)')
    plt.ylabel('Mean TF')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'mean_tf_by_stage.png'))
    plt.close()
    logging.info("Saved plot mean_tf_by_stage.png")

<<<<<<< HEAD
    # 2. Mean TF by stage for high_eq
=======
    # Mean TF by stage for high_eq
>>>>>>> 04f99b35685c6c73ef5f8a29ebc98ca5580f0fc1
    plt.figure()
    df2['mean'].plot(kind='bar')
    plt.title('Mean TF Score by Stage (High-Equivalence)')
    plt.ylabel('Mean TF')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'mean_tf_high_eq_by_stage.png'))
    plt.close()
    logging.info("Saved plot mean_tf_high_eq_by_stage.png")

<<<<<<< HEAD
    # 3. Distribution boxplot of TF scores for high_eq across stages
=======
    # Distribution boxplot of TF scores for high_eq across stages
>>>>>>> 04f99b35685c6c73ef5f8a29ebc98ca5580f0fc1
    plt.figure()
    high_pivot = high_eq.pivot(columns='stage', values='tf_score')
    high_pivot.boxplot()
    plt.title('TF Score Distribution (High-Equivalence)')
    plt.ylabel('TF Score')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'boxplot_high_eq_tf.png'))
    plt.close()
    logging.info("Saved plot boxplot_high_eq_tf.png")

    logging.info("All done!")


if __name__ == '__main__':
    main()
