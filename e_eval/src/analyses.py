"""
python e_eval/src/analyses.py --run_all --data_dir e_eval/data --output_dir e_eval/output
python e_eval/src/analyses.py --do_descriptive --do_perplexity --output_dir e_eval/output
python e_eval/src/analyses.py --run_all --data_dir path/to/data --ft_dir path/to/ft_outputs --output_dir e_eval/output


python e_eval/src/analyses.py --run_all --data_dir e_eval/data --output_dir e_eval/output
python e_eval/src/analyses.py --do_descriptive --do_perplexity --data_dir e_eval/data --output_dir e_eval/output
python e_eval/src/analyses.py --run_all --data_dir e_eval/data --ft_dir f_finetune/outputs --output_dir e_eval/output

python e_eval/src/analyses.py --run_all --output_dir e_eval/output
python e_eval/src/analyses.py --do_descriptive --do_perplexity --output_dir e_eval/output
python e_eval/src/analyses.py --run_all --data_dir path/to/your/data --ft_dir path/to/your/ft_outputs --output_dir e_eval/output
"""

import argparse
import logging
import os
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
from itertools import combinations
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# Import advanced stats libraries safely
try:
    from scipy import stats
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    ADVANCED_LIBS_AVAILABLE = True
except ImportError:
    ADVANCED_LIBS_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# GLOBAL CONFIGURATIONS
BASE_DIR = Path(__file__).resolve().parent.parent
METRIC_NAMES = [
    "Task Fulfilment / Relevance", "Usefulness & Actionability", "Factual Accuracy & Verifiabiliy",
    "Efficiency / Depth & Completeness", "Reasoning Quality / Transparency", "Tone & Likeability",
    "Adaptation to Context", "Safety & Bias Avoidance", "Structure & Formatting & UX Extras", "Creativity"
]
METRIC_COLS = [f"metric_{i}" for i in range(1, 11)]

# LOGGING SETUP
def setup_logging(log_dir: Path):
    """Sets up a timestamped log file."""
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"analysis_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")

# DATA LOADING & PREPROCESSING
def load_main_data(data_path: Path) -> pd.DataFrame:
    """Loads and unnests the main data JSONs."""
    records = []
    logging.info(f"Loading main data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

<<<<<<< HEAD
        # TMP !!!!!!!
        data = data[:500]

=======
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
    dataset_name = data_path.stem.split('_')[1] # e.g., 'alpaca' from 'all_alpaca_gemma-2-2b-it'
    model_name = data_path.stem.split('_')[-1] # e.g., 'gemma-2-2b-it'

    for prompt_obj in data:
        for para_obj in prompt_obj.get("paraphrases", []):
            record = {
                "prompt_count": prompt_obj["prompt_count"],
                "dataset": dataset_name,
                "model": f"{model_name}_baseline",
                "instruction_original": prompt_obj["instruction_original"],
                "instruct_type": para_obj["instruct_type"],
                "paraphrase": para_obj["paraphrase"],
                "task_score": para_obj.get("task_score"),
                "paraphrase_content_score": para_obj.get("paraphrase_content_score"),
                "bucket": para_obj.get("bucket"),
                "tags": tuple(para_obj.get("tags", [])), # Use tuple for hashability
                "perplexity": para_obj.get("perplexity"),
                "prompt_length": len(para_obj["paraphrase"].split())
            }
            scores = para_obj.get("answer_scores", [np.nan] * 10)
            for i, score in enumerate(scores):
                record[f"metric_{i+1}"] = score
            records.append(record)
    return pd.DataFrame(records)

def load_ft_data(ft_path: Path) -> pd.DataFrame:
    """Loads and transforms a fine-tuning scores JSON."""
    records = []
    logging.info(f"Loading FT scores from: {ft_path}")
    
    # Extract metadata from path: e.g., f_finetune/outputs/alpaca/all_layers/ft_inf_scores/buckets_1-2.json
    parts = ft_path.parts
    dataset_group = parts[-4] # 'alpaca' or 'all_data'
    layers_group = parts[-3] # 'all_layers' or 'specific_layers'
    bucket_file = parts[-1]
    bucket_num = int(bucket_file.split('.')[0].split('-')[-1])
    
    model_config_name = f"ft_{dataset_group}_{layers_group}_b{bucket_num}"
    
    with open(ft_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        prompt_id = item.pop("prompt_count")
        for instruct_type, scores in item.items():
            record = {
                "prompt_count": prompt_id,
                "instruct_type": instruct_type,
                "model": model_config_name,
            }
            for i, score in enumerate(scores):
                record[f"metric_{i+1}"] = score
            record["task_score"] = scores[0] if scores else np.nan
            records.append(record)
    return pd.DataFrame(records)


def load_and_prepare_data(args: argparse.Namespace) -> pd.DataFrame:
    """Loads all data sources and merges them into a single tidy DataFrame."""
    # Load all baseline data
<<<<<<< HEAD
    main_files = list(Path(args.data_dir).glob("*.json"))
=======
    main_files = list(Path(args.data_dir).glob("all_*.json"))
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
    if not main_files:
        raise FileNotFoundError(f"No main data files found in {args.data_dir}")
    baseline_df = pd.concat([load_main_data(f) for f in main_files], ignore_index=True)
    
    # Load all fine-tuning data
    ft_files = list(Path(args.ft_dir).glob("**/*buckets*.json"))
    if not ft_files:
        logging.warning(f"No fine-tuning score files found in {args.ft_dir}. Continuing with baseline data only.")
        ft_df_list = [pd.DataFrame()]
    else:
        ft_df_list = [load_ft_data(f) for f in ft_files]

    ft_df = pd.concat(ft_df_list, ignore_index=True)

    # Merge FT scores with baseline metadata
    # We keep metadata (tags, content_score, etc.) from the baseline runs
    metadata_cols = [
        "prompt_count", "instruct_type", "dataset", "paraphrase_content_score", 
        "bucket", "tags", "perplexity", "prompt_length"
    ]
    # Use only the first dataset's metadata for merging
    merged_df = pd.merge(
        ft_df,
        baseline_df[metadata_cols].drop_duplicates(subset=["prompt_count", "instruct_type"]),
        on=["prompt_count", "instruct_type"],
        how="left"
    )

    # Combine baseline and merged FT data
    full_df = pd.concat([baseline_df, merged_df], ignore_index=True)

    # Clean up column names for metrics
    full_df = full_df.rename(columns={f"metric_{i+1}": col for i, col in enumerate(METRIC_COLS)})
    
    logging.info(f"Data loading complete. Final DataFrame shape: {full_df.shape}")
    logging.info(f"Models found: {full_df['model'].unique()}")
    return full_df

<<<<<<< HEAD
# ANALYSIS & VISUALISATION
=======
# ANALYSIS & VISUALIZATION
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
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
        
        if ADVANCED_LIBS_AVAILABLE:
            if self.args.do_advanced: self.run_advanced_stats()
        elif self.args.do_advanced:
            logging.error("Cannot run advanced stats. `scikit-learn` or `statsmodels` not found.")

        self.save_summary_report()

    def _add_result(self, title, content):
        self.results_log.append(f"\n{'='*80}\n## {title}\n{'='*80}\n\n{content}\n")

    def _add_graphic(self, filename, description):
        self.graphics_log[filename] = description

    def _save_plot(self, fig, name, description):
        """Saves a matplotlib figure and logs it."""
        filepath = self.output_dir / "plots" / f"{name}.png"
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        self._add_graphic(f"plots/{name}.png", description)
        logging.info(f"Saved plot: {filepath}")

    def run_descriptive_stats(self):
        """Calculates basic stats for paraphrases with content scores 4 and 5."""
        logging.info("Running Descriptive Statistics for high-equivalence paraphrases...")
        df_high_eq = self.df[self.df['paraphrase_content_score'].isin([4, 5])].copy()
        if df_high_eq.empty:
            logging.warning("No paraphrases with content score 4 or 5 found. Skipping descriptive stats.")
            return

        # Explode tags for tag-level analysis
        df_tags = df_high_eq.explode('tags')

        # Calculations
        title = "Descriptive Stats (Equivalence Score 4-5)"
        content = "TF stands for 'Task Fulfilment / Relevance'.\n\n"

        # Overall TF
        content += "### Overall TF Score:\n"
        content += df_high_eq['Task Fulfilment / Relevance'].describe().to_string() + "\n\n"

        # By Model
        content += "### TF Score by Model:\n"
        content += df_high_eq.groupby('model')['Task Fulfilment / Relevance'].describe().to_string() + "\n\n"
        
        # By Dataset
        content += "### TF Score by Dataset:\n"
        content += df_high_eq.groupby('dataset')['Task Fulfilment / Relevance'].describe().to_string() + "\n\n"

        # By Tag
        content += "### TF Score by Tag:\n"
        content += df_tags.groupby('tags')['Task Fulfilment / Relevance'].describe().sort_values('mean', ascending=False).to_string() + "\n\n"

        self._add_result(title, content)

        # Visualizations
        # Histogram of TF scores by tag
        g = sns.FacetGrid(df_tags.dropna(subset=['tags']), col="tags", col_wrap=5, sharex=True, sharey=False)
        g.map(sns.histplot, "Task Fulfilment / Relevance", bins=11)
        g.fig.suptitle("TF Score Distribution by Tag (Equiv. 4-5)", y=1.02)
        self._save_plot(g.fig, "tf_hist_by_tag", "Distribution of Task Fulfilment scores for each tag, for high-equivalence paraphrases.")

        # Boxplot of TF scores by model
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=df_high_eq, x='Task Fulfilment / Relevance', y='model', ax=ax)
        ax.set_title("TF Score Distribution by Model (Equiv. 4-5)")
        self._save_plot(fig, "tf_boxplot_by_model", "Box plots showing the distribution of Task Fulfilment scores for each model.")
        
    def run_ft_comparison(self):
        """Compares baseline model performance to fine-tuned models."""
        logging.info("Running Fine-Tuning Comparison...")
        
        baseline_model = [m for m in self.df['model'].unique() if 'baseline' in m]
        if not baseline_model:
            logging.warning("No baseline model found. Skipping FT comparison.")
            return
        baseline_model = baseline_model[0]

        df_baseline = self.df[self.df['model'] == baseline_model].set_index(['prompt_count', 'instruct_type'])
        df_ft = self.df[self.df['model'] != baseline_model]
        
        if df_ft.empty:
            logging.warning("No fine-tuning data found. Skipping FT comparison.")
            return
            
        comparison_data = []
        for _, row in df_ft.iterrows():
            idx = (row['prompt_count'], row['instruct_type'])
            if idx in df_baseline.index:
                baseline_score = df_baseline.loc[idx, 'Task Fulfilment / Relevance']
                comparison_data.append({
                    'model': row['model'],
                    'instruct_type': row['instruct_type'],
                    'bucket': row['bucket'],
                    'baseline_tf': baseline_score,
                    'ft_tf': row['Task Fulfilment / Relevance'],
                    'delta_tf': row['Task Fulfilment / Relevance'] - baseline_score
                })
        
        comp_df = pd.DataFrame(comparison_data)
        if comp_df.empty:
            logging.warning("Could not create comparison data. Skipping FT comparison.")
            return

        # Calculations
        avg_delta = comp_df.groupby('model')['delta_tf'].mean().sort_values(ascending=False)
        content = "Average change in Task Fulfilment score after fine-tuning (Positive = Improvement).\n\n"
        content += avg_delta.to_string()
        self._add_result("Fine-Tuning Performance Delta", content)

<<<<<<< HEAD
        # Visualisations
=======
        # Visualizations
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=comp_df, x='delta_tf', y='model', ax=ax, estimator=np.mean)
        ax.set_title('Average Change in Task Fulfilment Score vs. Baseline')
        ax.set_xlabel('Mean Δ TF (FT Score - Baseline Score)')
        ax.axvline(0, color='r', linestyle='--')
        self._save_plot(fig, "ft_delta_tf_by_model", "Bar chart showing the average improvement or degradation in TF score for each fine-tuning configuration.")

    def run_perplexity_analysis(self):
        """Runs all analyses related to perplexity."""
        logging.info("Running Perplexity Analysis...")
        df_ppl = self.df.dropna(subset=['perplexity', 'Task Fulfilment / Relevance']).copy()

        # Calculations
        corr_matrix = df_ppl[['perplexity', 'Task Fulfilment / Relevance', 'paraphrase_content_score', 'prompt_length']].corr(method='spearman')
        content = "Spearman Correlation Matrix involving Perplexity:\n\n"
        content += corr_matrix.to_string()
        self._add_result("Perplexity Correlations", content)

<<<<<<< HEAD
        # Visualisations
=======
        # Visualizations
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
        # Heatmap of correlations
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='vlag', center=0, ax=ax)
        ax.set_title('Spearman Correlation Heatmap with Perplexity')
        self._save_plot(fig, "ppl_correlation_heatmap", "Heatmap showing the Spearman correlation between perplexity, TF score, content score, and prompt length.")

        # Scatter plot: Perplexity vs. Task Fulfilment
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.scatterplot(data=df_ppl, x='perplexity', y='Task Fulfilment / Relevance', hue='bucket', palette='viridis', alpha=0.6, ax=ax)
        ax.set_title('Perplexity vs. Task Fulfilment (Colored by Bucket)')
        ax.set_xscale('log') # Perplexity often has a long tail
        self._save_plot(fig, "ppl_vs_tf_scatter", "Scatter plot of Perplexity vs. Task Fulfilment. Shows if higher perplexity (more surprising) prompts lead to lower scores.")
        
        # Box plot: Perplexity by bucket
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.boxplot(data=df_ppl, x='bucket', y='perplexity', ax=ax)
        ax.set_title('Perplexity Distribution by Performance Bucket')
        ax.set_yscale('log')
        self._save_plot(fig, "ppl_by_bucket_boxplot", "Box plots showing the distribution of perplexity for each performance bucket.")

    def run_advanced_stats(self):
        """Runs advanced analyses: statistical testing, PCA, clustering."""
        logging.info("Running Advanced Statistics...")
        
<<<<<<< HEAD
        # Deeper descriptive
=======
        # A. Deeper descriptive
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
        content = ""
        desc_df = self.df.groupby('bucket')['Task Fulfilment / Relevance'].describe(percentiles=[.1, .25, .5, .75, .9])
        content += "### TF Score Percentiles by Bucket:\n" + desc_df.to_string() + "\n\n"

<<<<<<< HEAD
        # Inferential Testing (Example: ANOVA on buckets)
=======
        # B. Inferential Testing (Example: ANOVA on buckets)
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
        model = ols('Q("Task Fulfilment / Relevance") ~ C(bucket)', data=self.df.dropna()).fit()
        content += "### ANOVA: Task Fulfilment vs. Bucket\n"
        content += str(model.summary()) + "\n\n"
        self._add_result("Advanced Stats: Descriptive & Inferential", content)

<<<<<<< HEAD
        # Predictive Modeling (Example: Feature Importance)
=======
        # C. Predictive Modeling (Example: Feature Importance)
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
        df_model = self.df.dropna(subset=METRIC_COLS).copy()
        X = df_model[[col for col in METRIC_COLS if col != "Task Fulfilment / Relevance"]]
        y = df_model["Task Fulfilment / Relevance"]
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        feature_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        content = "### Feature Importance (Random Forest)\nPredicting Task Fulfilment from other 9 metrics:\n\n" + feature_imp.to_string()
        self._add_result("Advanced Stats: Predictive Modeling", content)

<<<<<<< HEAD
        # Dimensionality Reduction (PCA)
=======
        # D. Dimensionality Reduction (PCA)
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
        df_pca = self.df.dropna(subset=METRIC_COLS).copy()
        X_scaled = StandardScaler().fit_transform(df_pca[METRIC_COLS])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_pca[['PC1', 'PC2']] = X_pca
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='bucket', palette='Set2', alpha=0.7, ax=ax)
        ax.set_title('PCA of 10 Performance Metrics (Colored by Bucket)')
        ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        self._save_plot(fig, "pca_of_metrics", "2D PCA plot of all 10 performance metrics, showing how paraphrases cluster based on their performance profile.")

    def save_summary_report(self):
        """Saves all collected results and the graphics log to a text file."""
        report_path = self.output_dir / "results_summary.txt"
        logging.info(f"Saving summary report to: {report_path}")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("This report summarizes the statistical analysis and visualizations generated by the script.\n")
            
            for result in self.results_log:
                f.write(result)
            
            f.write(f"\n{'='*80}\n## Generated Graphics\n{'='*80}\n\n")
            if not self.graphics_log:
                f.write("No graphics were generated in this run.\n")
            else:
                for filename, description in self.graphics_log.items():
                    f.write(f"- **File:** `{filename}`\n")
                    f.write(f"  **Description:** {description}\n\n")

# MAIN EXECUTION
def main():
    parser = argparse.ArgumentParser(description="Comprehensive analysis script for LLM prompt robustness research.")
    
    # Path arguments
    parser.add_argument("--data_dir", type=str, default="f_finetune/data", help="Directory containing main data JSONs (e.g., all_alpaca_...).")
    parser.add_argument("--ft_dir", type=str, default="f_finetune/outputs", help="Root directory for fine-tuning outputs containing score JSONs.")
    parser.add_argument("--output_dir", type=str, default="e_eval/output", help="Directory to save all analysis results and plots.")
    
    # Module selection arguments
    parser.add_argument("--run_all", action='store_true', help="Run all analysis modules.")
    parser.add_argument("--do_descriptive", action='store_true', help="Run descriptive statistics for high-equivalence paraphrases.")
    parser.add_argument("--do_ft_compare", action='store_true', help="Run baseline vs. fine-tuning comparison.")
    parser.add_argument("--do_perplexity", action='store_true', help="Run all perplexity-related analyses.")
    parser.add_argument("--do_advanced", action='store_true', help="Run advanced stats (ANOVA, PCA, Feature Importance, etc.). Requires scikit-learn and statsmodels.")

    args = parser.parse_args()

    # If --run_all is specified, set all module flags to True
    if args.run_all:
        args.do_descriptive = True
        args.do_ft_compare = True
        args.do_perplexity = True
        args.do_advanced = True

    # Setup
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path / "logs")

    try:
        # Data Loading
        full_df = load_and_prepare_data(args)
        
        # Analysis
        analyzer = Analyzer(full_df, output_path, args)
        analyzer.run_all()

    except FileNotFoundError as e:
        logging.error(f"A required file or directory was not found: {e}")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
        
    logging.info("Analysis script finished.")

if __name__ == "__main__":
    main()
