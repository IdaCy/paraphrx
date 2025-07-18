"""

pip install ijson

python e_eval/src/analyses_mem_test.py --run_all --data_dir e_eval/data --output_dir e_eval/output2
python e_eval/src/analyses_mem_test.py --do_descriptive --do_perplexity --data_dir e_eval/data --output_dir e_eval/output
python e_eval/src/analyses_mem_test.py --run_all --data_dir e_eval/data --ft_dir f_finetune/outputs --output_dir e_eval/output


"""


import argparse
import logging
import gc
import json
from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import ijson  # For memory-efficient JSON parsing

try:
    from scipy import stats
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    ADVANCED_LIBS_AVAILABLE = True
except ImportError:
    ADVANCED_LIBS_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# GLOBAL CONFIGURATIONS
METRIC_NAMES = [
    "Task Fulfilment / Relevance", "Usefulness & Actionability", "Factual Accuracy & Verifiabiliy",
    "Efficiency / Depth & Completeness", "Reasoning Quality / Transparency", "Tone & Likeability",
    "Adaptation to Context", "Safety & Bias Avoidance", "Structure & Formatting & UX Extras", "Creativity"
]
METRIC_COLS_SHORT = [f"m{i}" for i in range(1, 11)]
METRIC_MAP = {old: new for old, new in zip(METRIC_NAMES, METRIC_COLS_SHORT)}
METRIC_MAP_REVERSE = {v: k for k, v in METRIC_MAP.items()}


# LOGGING SETUP
def setup_logging(log_dir: Path):
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analysis_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")


# DATA LOADING

def stream_and_parse_main_data(data_path: Path):
    """Generator to stream a large JSON and yield processed records."""
    logging.info(f"Streaming main data from: {data_path}")
    dataset_name = data_path.stem.split('_')[0]
    model_name = '_'.join(data_path.stem.split('_')[1:])

    with open(data_path, 'r', encoding='utf-8') as f:
        parser = ijson.items(f, 'item')
        for prompt_obj in parser:
            for para_obj in prompt_obj.get("paraphrases", []):
                record = {
                    "prompt_count": prompt_obj["prompt_count"],
                    "dataset": dataset_name,
                    "model": f"{model_name}_baseline",
                    "instruct_type": para_obj["instruct_type"],
                    "p_content_score": para_obj.get("paraphrase_content_score"),
                    "bucket": para_obj.get("bucket"),
                    "tags": tuple(para_obj.get("tags", [])),
                    "perplexity": para_obj.get("perplexity"),
                    "p_len": len(para_obj.get("paraphrase", "").split())
                }
                scores = para_obj.get("answer_scores", [np.nan] * 10)
                for i, score in enumerate(scores):
                    record[METRIC_COLS_SHORT[i]] = score
                yield record

def load_ft_data(ft_path: Path):
    """Loads and transforms a fine-tuning scores JSON."""
    records = []
    parts = ft_path.parts
    dataset_group, layers_group, bucket_file = parts[-4], parts[-3], parts[-1]
    bucket_num = int(bucket_file.split('.')[0].split('-')[-1])
    model_config_name = f"ft_{dataset_group}_{layers_group}_b{bucket_num}"
    
    with open(ft_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        prompt_id = item.pop("prompt_count")
        for instruct_type, scores in item.items():
            record = {"prompt_count": prompt_id, "instruct_type": instruct_type, "model": model_config_name}
            for i, score in enumerate(scores):
                record[METRIC_COLS_SHORT[i]] = score
            records.append(record)
    return pd.DataFrame(records)


def load_original_scores(score_dir: Path) -> pd.DataFrame:
    """Loads scores for 'instruction_original' from a directory of JSONs."""
    records = []
    score_files = list(score_dir.glob("*.json"))
    if not score_files:
        logging.warning(f"No original score files found in {score_dir}")
        return pd.DataFrame()
    
    logging.info(f"Loading original instruction scores from {score_dir}...")
    for f in score_files:
        dataset_name = f.stem
        try:
            with open(f, 'r', encoding='utf-8') as file_handle:
                data = json.load(file_handle)
                for item in data:
                    if "instruction_original" in item:
                        scores = item["instruction_original"]
                        records.append({'dataset': dataset_name, 'tf_score': scores[0]})
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Could not read or parse file {f}: {e}")
            
    return pd.DataFrame(records)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Optimizing DataFrame data types...")
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'tags':  # Exclude tags from category conversion
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif col.startswith('m') or col in ['bucket', 'p_content_score']:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
    
    mem_usage = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info(f"Optimized DataFrame memory usage: {mem_usage:.2f} MB")
    return df

def load_and_prepare_data(args: argparse.Namespace) -> pd.DataFrame:
    """Loads all data sources using streaming and merges them."""
    main_files = list(Path(args.data_dir).glob("*.json"))
    if not main_files: raise FileNotFoundError(f"No main data files found in {args.data_dir}")
    
    all_records = (record for f in main_files for record in stream_and_parse_main_data(f))
    baseline_df = pd.DataFrame(all_records)
    
    ft_files = list(Path(args.ft_dir).glob("**/*buckets*.json")) if args.ft_dir else []
    if not ft_files:
        logging.warning("No FT data found. Continuing with baseline data only.")
        return optimize_dtypes(baseline_df)
    
    logging.info(f"Loading {len(ft_files)} fine-tuning score files...")
    ft_df_list = [load_ft_data(f) for f in ft_files]
    ft_df = pd.concat(ft_df_list, ignore_index=True)
    
    metadata_cols = ["prompt_count", "instruct_type", "dataset", "p_content_score", "bucket", "tags", "perplexity", "p_len"]
    merged_df = pd.merge(
        ft_df,
        baseline_df[metadata_cols].drop_duplicates(subset=["prompt_count", "instruct_type"]),
        on=["prompt_count", "instruct_type"],
        how="left"
    )
    
    full_df = pd.concat([baseline_df, merged_df], ignore_index=True)
    full_df = optimize_dtypes(full_df)
    
    logging.info(f"Data loading complete. Final DataFrame shape: {full_df.shape}")
    logging.info(f"Models found: {full_df['model'].unique().tolist()}")
    gc.collect()
    return full_df


# ANALYSIS & VISUALIZATION

class Analyzer:
    def __init__(self, df: pd.DataFrame, output_dir: Path, args: argparse.Namespace):
        self.df = df
        self.output_dir = output_dir
        self.args = args
        self.results_log = []
        self.graphics_log = {}
        (self.output_dir / "plots").mkdir(exist_ok=True)

    def run_all(self):
        """Runs all selected analysis modules."""
        logging.info("Starting analysis modules...")
        if self.args.do_descriptive: self.run_descriptive_stats()
        if self.args.do_ft_compare: self.run_ft_comparison()
        if self.args.do_perplexity: self.run_perplexity_analysis()
        if self.args.do_advanced and ADVANCED_LIBS_AVAILABLE: self.run_advanced_stats()
        self.save_summary_report()

    def _add_result(self, title, content):
        self.results_log.append(f"\n{'='*80}\n## {title}\n{'='*80}\n\n{content}\n")

    def _save_plot(self, fig, name, description):
        filepath = self.output_dir / "plots" / f"{name}.png"
        fig.tight_layout()
        fig.savefig(filepath, dpi=120)
        plt.close(fig)
        self.graphics_log[f"plots/{name}.png"] = description
        logging.info(f"Saved plot: {filepath}")

    def run_descriptive_stats(self):
        logging.info("Running Descriptive Statistics...")
        
        content = ""
        if self.args.original_scores_dir:
            original_scores_df = load_original_scores(Path(self.args.original_scores_dir))
            if not original_scores_df.empty:
                content += "### TF Score for Original (Un-paraphrased) Instructions:\n"
                overall_desc = original_scores_df['tf_score'].describe().to_frame().T
                overall_desc.index = ['overall']
                per_dataset_desc = original_scores_df.groupby('dataset')['tf_score'].describe()
                full_desc = pd.concat([overall_desc, per_dataset_desc])
                content += full_desc.to_string() + "\n\n"
        
        df_high_eq = self.df[self.df['p_content_score'].isin([4, 5])].copy()
        if df_high_eq.empty: return

        title = "Descriptive Stats for High-Equivalence Paraphrases (Score 4-5)"
        tf_col = METRIC_COLS_SHORT[0]
        content += "### Paraphrase TF Score by Model:\n" + df_high_eq.groupby('model')[tf_col].describe().to_string() + "\n\n"
        content += "### Paraphrase TF Score by Dataset:\n" + df_high_eq.groupby('dataset')[tf_col].describe().to_string() + "\n\n"
        
        # ROBUST FIX for Individual Tag Statistics
        logging.info("Calculating statistics for individual tags...")
        exploded_records = []
        # Use itertuples for memory-efficient iteration
        for row in df_high_eq[['tags', tf_col]].itertuples():
            # row[1] is 'tags', row[2] is the tf_col
            if isinstance(row[1], tuple):
                for tag in row[1]:
                    exploded_records.append({'tag': tag, 'tf_score': row[2]})
        
        if exploded_records:
            df_tags_exploded = pd.DataFrame(exploded_records)
            tag_stats = df_tags_exploded.groupby('tag')['tf_score'].describe().sort_values('mean', ascending=False)
            content += "### Paraphrase TF Score by Individual Tag:\n" + tag_stats.to_string() + "\n\n"
        else:
            content += "### Paraphrase TF Score by Individual Tag:\nNo tags found to analyze.\n\n"

        self._add_result(title, content)
        del df_high_eq, exploded_records
        if 'df_tags_exploded' in locals():
            del df_tags_exploded
        gc.collect()

    def run_ft_comparison(self):
        logging.info("Running Fine-Tuning Comparison for high-equivalence paraphrases...")
        df_high_eq = self.df[self.df['p_content_score'].isin([4, 5])].copy()
        
        baseline_df = df_high_eq[df_high_eq['model'].str.contains('baseline', case=False, na=False)].copy()
        ft_df = df_high_eq[~df_high_eq['model'].str.contains('baseline', case=False, na=False)].copy()

        if baseline_df.empty or ft_df.empty:
            logging.warning("Not enough baseline or fine-tuning data for FT comparison. Skipping.")
            return

        merge_cols = ['prompt_count', 'instruct_type', 'dataset']
        baseline_subset = baseline_df[merge_cols + [METRIC_COLS_SHORT[0], 'perplexity']].rename(columns={
            METRIC_COLS_SHORT[0]: 'tf_baseline', 'perplexity': 'ppl_baseline'})

        comparison_df = pd.merge(ft_df, baseline_subset, on=merge_cols, how='inner')
        if comparison_df.empty:
            logging.warning("Merge between baseline and FT data yielded no results. Skipping FT comparison.")
            return

        comparison_df['delta_tf'] = comparison_df[METRIC_COLS_SHORT[0]] - comparison_df['tf_baseline']
        comparison_df['delta_ppl'] = comparison_df['perplexity'] - comparison_df['ppl_baseline']

        summary = comparison_df.groupby('model')[['delta_tf', 'delta_ppl']].agg(['mean', 'std', 'count'])
        self._add_result("Fine-Tuning Performance Delta (Equiv. 4-5)", "Change in TF and Perplexity vs. Baseline.\n\n" + summary.to_string())

        # Bar chart for delta_tf
        fig, ax = plt.subplots(figsize=(14, 8))
        summary_means = summary['delta_tf'].sort_values('mean', ascending=False)
        ax.barh(summary_means.index, summary_means['mean'], xerr=summary_means['std'], capsize=4)
        ax.axvline(0, color='grey', linestyle='--')
        ax.set_title('Mean Change in Task Fulfilment after FT (vs. Baseline, Equiv. 4-5)')
        ax.set_xlabel('Δ TF Score (FT - Baseline)')
        self._save_plot(fig, "ft_delta_tf_by_model", "Bar chart of average change in TF score for each FT model.")

        # Scatter plot for delta_ppl vs delta_tf
        fig, ax = plt.subplots(figsize=(12, 8))
        sample_df = comparison_df.sample(n=min(5000, len(comparison_df)), random_state=42)
        sns.scatterplot(data=sample_df, x='delta_ppl', y='delta_tf', hue='model', alpha=0.7, ax=ax)
        ax.axhline(0, color='grey', linestyle='--'); ax.axvline(0, color='grey', linestyle='--')
        ax.set_title('Δ Perplexity vs. Δ TF Score after Fine-Tuning (Sampled)')
        ax.set_xlabel('Δ Perplexity (FT - Baseline)'); ax.set_ylabel('Δ TF Score (FT - Baseline)')
        self._save_plot(fig, "ft_delta_ppl_vs_delta_tf", "How change in perplexity relates to change in TF score.")

    def run_perplexity_analysis(self):
        logging.info("Running Perplexity Analysis...")
        df_ppl = self.df.dropna(subset=['perplexity', METRIC_COLS_SHORT[0]]).copy()
        corr_cols = ['perplexity', METRIC_COLS_SHORT[0], 'p_content_score', 'p_len']
        corr_matrix = df_ppl[corr_cols].corr(method='spearman')
        self._add_result("Perplexity Spearman Correlations", corr_matrix.to_string())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='vlag', center=0, ax=ax)
        ax.set_title('Spearman Correlation Heatmap with Perplexity')
        self._save_plot(fig, "ppl_correlation_heatmap", "Heatmap of Spearman correlations.")
        del df_ppl, corr_matrix
        gc.collect()

    def run_advanced_stats(self):
        logging.info("Running Advanced Statistics...")
        tf_col = METRIC_COLS_SHORT[0]
        model = ols(f'{tf_col} ~ C(bucket)', data=self.df.dropna(subset=[tf_col, 'bucket'])).fit()
        self._add_result("ANOVA: Task Fulfilment vs. Bucket", str(model.summary()))

    def save_summary_report(self):
        report_path = self.output_dir / "results_summary.txt"
        logging.info(f"Saving summary report to: {report_path}")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            for result in self.results_log: f.write(result)
            f.write(f"\n{'='*80}\n## Generated Graphics\n{'='*80}\n\n")
            for filename, description in self.graphics_log.items():
                f.write(f"- **File:** `{filename}`\n  **Description:** {description}\n\n")


# MAIN EXECUTION

def main():
    parser = argparse.ArgumentParser(description="Memory-efficient analysis script for LLM prompt robustness.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with main data JSONs.")
    parser.add_argument("--ft_dir", type=str, default=None, help="Optional root directory for FT outputs.")
    parser.add_argument("--original_scores_dir", type=str, default="e_eval/data/scores", help="Dir with original instruction scores.")
    parser.add_argument("--output_dir", type=str, default="e_eval/output", help="Directory for results.")
    
    parser.add_argument("--run_all", action='store_true', help="Run all analysis modules.")
    parser.add_argument("--do_descriptive", action='store_true', help="Run descriptive statistics.")
    parser.add_argument("--do_ft_compare", action='store_true', help="Run baseline vs fine-tuning comparison.")
    parser.add_argument("--do_perplexity", action='store_true', help="Run perplexity analysis.")
    parser.add_argument("--do_advanced", action='store_true', help="Run advanced stats (ANOVA, etc.).")
    args = parser.parse_args()

    if args.run_all:
        args.do_descriptive = args.do_ft_compare = args.do_perplexity = args.do_advanced = True

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path / "logs")

    try:
        full_df = load_and_prepare_data(args)
        analyzer = Analyzer(full_df, output_path, args)
        analyzer.run_all()
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
        
    logging.info("Analysis script finished.")

if __name__ == "__main__":
    main()