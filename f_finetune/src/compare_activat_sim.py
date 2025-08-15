#!/usr/bin/env python
"""
pip install torch transformers accelerate peft pandas seaborn matplotlib scipy

Advanced Robustness Analyzer for LLM Activations and Weights.

This script performs an in-depth comparison between a base and a fine-tuned LLM
to understand the mechanisms behind improved robustness to paraphrasing.

It now includes four distinct analysis modules:
1.  **Weight Delta Analysis**: (Aggregate mode only) Visualizes which model layers
    were most changed during fine-tuning by plotting the norm of weight differences.
2.  **Representational Similarity**: The original analysis, which computes the cosine
    similarity of hidden states between original and paraphrased prompts.
3.  **Embedding Space Analysis**: Measures if fine-tuning brings the initial embeddings
    of paraphrases closer to their original counterparts.
4.  **Activation Norm Analysis**: Investigates if the fine-tuned model uses higher-norm
    activations, potentially indicating more salient features.

MODES:
1.  case_study: Generates multi-panel plots and detailed numerical reports
    for specific prompt/paraphrase pairs.
2.  aggregate: Runs on a large number of prompts (or a specific data split)
    to compute and visualize average trends, showing systematic changes from fine-tuning.


# For a detailed look at specific prompts (using --prompt_ids)
srun python robustness_analyzer.py \
  --run_mode case_study \
  --base_model_path "f_finetune/model" \
  --ft_model_path "f_finetune/outputs/l9x_a1_notarg_50k_ft/final" \
  --prompts_json_path "/path/to/your/prompts.json" \
  --output_dir "analysis_results/case_studies" \
  --prompt_ids 1 3 \
  --paraphrase_keys "instruct_polite_request" "instruct_sardonic"

# For a detailed look at the first 5 prompts in the validation set (using --limit)
srun python robustness_analyzer.py \
  --run_mode case_study \
  --data_split test \
  --limit 5 \
  --base_model_path "f_finetune/model" \
  --ft_model_path "f_finetune/outputs/l9x_a1_notarg_50k_ft/final" \
  --prompts_json_path "/path/to/your/prompts.json" \
  --output_dir "analysis_results/case_studies_limited"

# For a broad, statistical analysis on 100 prompts from the test set
srun python robustness_analyzer.py \
  --run_mode aggregate \
  --data_split test \
  --limit 100 \
  --base_model_path "f_finetune/model" \
  --ft_model_path "f_finetune/outputs/l9x_a1_notarg_50k_ft/final" \
  --prompts_json_path "/path/to/your/prompts.json" \
  --output_dir "analysis_results/aggregate_test_set"

python3 f_finetune/src/compare_activat_sim.py \
  --run_mode aggregate \
  --data_split test \
  --limit 5 \
  --base_model_path "f_finetune/model" \
  --ft_model_path "f_finetune/outputs/l9x_a1_notarg_50k_ft/final" \
  --prompts_json_path "a_data/alpaca/50k_phrxed.json" \
  --output_dir "f_finetune/outputs/l9x_a1_notarg_50k_ft/aggregate_val_set_test5"
"""
import argparse
import json
import logging
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from peft import PeftModel
from scipy import stats
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)


# Logging Setup
class SafeFormatter(logging.Formatter):
    def format(self, record):
        record_copy = logging.makeLogRecord(record.__dict__)
        try:
            return super().format(record_copy)
        except (TypeError, ValueError):
            record_copy.msg = f"MALFORMED LOG: {record_copy.msg} | ARGS: {record_copy.args}"
            record_copy.args = ()
            return logging.Formatter('[%(levelname)s] %(message)s').format(record_copy)

safe_handler = logging.StreamHandler(sys.stdout)
safe_handler.setFormatter(SafeFormatter("%(asctime)s [%(levelname)s] - %(message)s"))
script_logger = logging.getLogger(__name__)
script_logger.setLevel(logging.INFO)
script_logger.handlers.clear()
script_logger.addHandler(safe_handler)
script_logger.propagate = False
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
transformers_logger.handlers.clear()
transformers_logger.addHandler(safe_handler)
transformers_logger.propagate = False

# Matplotlib and Style Setup
plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8-whitegrid')


# Standalone Model Analysis
def analyze_and_plot_weight_deltas(base_model: PreTrainedModel, ft_model: PreTrainedModel, output_path: Path):
    script_logger.info("Comparing model weights layer by layer...")
    deltas = {}
    base_params = dict(base_model.named_parameters())
    ft_params = dict(ft_model.named_parameters())
    for name, ft_param in ft_params.items():
        if name in base_params and ft_param.shape == base_params[name].shape:
            with torch.no_grad():
                delta = torch.linalg.norm(ft_param.to(base_params[name].device) - base_params[name]).item()
                deltas[name] = delta
    if not deltas:
        script_logger.warning("Could not compute any weight deltas.")
        return
    layer_deltas = defaultdict(float)
    for name, delta in deltas.items():
        if 'layers.' in name:
            try:
                layer_num = int(name.split('layers.')[1].split('.')[0])
                layer_deltas[layer_num] += delta
            except (IndexError, ValueError): pass
    if not layer_deltas:
        script_logger.warning("Weight delta analysis found no standard 'layers.X' parameters to aggregate.")
        return
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(sorted(layer_deltas.keys()), [layer_deltas[i] for i in sorted(layer_deltas.keys())], color='indigo', alpha=0.8)
    ax.set_title('Fine-Tuning Impact: L2 Norm of Weight Deltas per Layer', fontsize=16, weight='bold')
    ax.set_xlabel('Decoder Layer'); ax.set_ylabel('Sum of L2 Norms of Parameter Deltas')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(output_path / "analysis_weight_deltas.png")
    script_logger.info(f"Saved weight delta analysis plot to {output_path / 'analysis_weight_deltas.png'}")
    plt.close(fig)


# Core Analysis Logic
class ActivationExtractor:
    def __init__(self, model: PreTrainedModel):
        self._model = model
        self.hidden_states = {}
        self._hook_handles = []
    def _hook_fn(self, layer_idx: int):
        def hook(module, input, output):
            self.hidden_states[layer_idx] = output[0].detach()
        return hook
    def __enter__(self):
        self.hidden_states.clear()
        if not hasattr(self._model, 'model') or not hasattr(self._model.model, 'layers'):
            raise TypeError("Model does not have the expected 'model.layers' structure.")
        for i, layer in enumerate(self._model.model.layers):
            self._hook_handles.append(layer.register_forward_hook(self._hook_fn(i)))
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self._hook_handles: handle.remove()
        self._hook_handles.clear()

@torch.no_grad()
def run_and_analyze_pair(base_model: PreTrainedModel, ft_model: PreTrainedModel, tokenizer: AutoTokenizer, original_text: str, paraphrase_text: str) -> Dict:
    inputs = tokenizer([original_text, paraphrase_text], return_tensors='pt', padding='longest', truncation=True, max_length=512).to(base_model.device)
    input_ids_orig, input_ids_para = inputs.input_ids[0:1], inputs.input_ids[1:2]
    results = {}
    base_extractor, ft_extractor = ActivationExtractor(base_model), ActivationExtractor(ft_model)
    with base_extractor: base_model(input_ids_orig); base_activations_orig = dict(sorted(base_extractor.hidden_states.items()))
    with base_extractor: base_model(input_ids_para); base_activations_para = dict(sorted(base_extractor.hidden_states.items()))
    with ft_extractor: ft_model(input_ids_orig); ft_activations_orig = dict(sorted(ft_extractor.hidden_states.items()))
    with ft_extractor: ft_model(input_ids_para); ft_activations_para = dict(sorted(ft_extractor.hidden_states.items()))

    base_emb_orig, base_emb_para = base_model.get_input_embeddings()(input_ids_orig), base_model.get_input_embeddings()(input_ids_para)
    ft_emb_orig, ft_emb_para = ft_model.get_input_embeddings()(input_ids_orig), ft_model.get_input_embeddings()(input_ids_para)
    results['base_emb_sim'] = torch.nn.functional.cosine_similarity(base_emb_orig.mean(dim=1), base_emb_para.mean(dim=1)).item()
    results['ft_emb_sim'] = torch.nn.functional.cosine_similarity(ft_emb_orig.mean(dim=1), ft_emb_para.mean(dim=1)).item()
    
    results.update({
        'base_cos_sims': {}, 'ft_cos_sims': {},
        'base_norms_para': {}, 'ft_norms_para': {}
    })
    
    for layer_idx in base_activations_orig.keys():
        results['base_cos_sims'][layer_idx] = torch.nn.functional.cosine_similarity(base_activations_orig[layer_idx].view(1,-1), base_activations_para[layer_idx].view(1,-1)).item()
        results['ft_cos_sims'][layer_idx] = torch.nn.functional.cosine_similarity(ft_activations_orig[layer_idx].view(1,-1), ft_activations_para[layer_idx].view(1,-1)).item()
        results['base_norms_para'][layer_idx] = torch.linalg.norm(base_activations_para[layer_idx].squeeze().float(), dim=1).mean().item()
        results['ft_norms_para'][layer_idx] = torch.linalg.norm(ft_activations_para[layer_idx].squeeze().float(), dim=1).mean().item()
        
    return results


# Model Loading and Data Splitting
def load_model_and_tokenizer(model_path: str, base_model_path: str, device: str) -> tuple[PreTrainedModel, AutoTokenizer]:
    path, base_path = Path(model_path), Path(base_model_path)
    if not base_path.exists(): script_logger.error(f"Base model path not found: {base_path}"); sys.exit(1)
    script_logger.info(f"Loading tokenizer from: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if (path / "adapter_config.json").exists():
        script_logger.info("Detected LoRA adapters. Loading base model and applying adapters.")
        model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map={"": device})
        model = PeftModel.from_pretrained(model, model_path)
    else:
        script_logger.info("Detected a full model. Loading directly.")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map={"": device})
    if model is None: script_logger.error(f"Could not load model from {model_path}"); sys.exit(1)
    return model, tokenizer

def get_group_wise_split_ids(all_prompt_ids: List[int], val_size: float, test_size: float, seed: int) -> Tuple[Set[int], Set[int], Set[int]]:
    prompt_ids = sorted(list(set(all_prompt_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(prompt_ids)
    n_test, n_val = int(len(prompt_ids) * test_size), int(len(prompt_ids) * val_size)
    test_ids, val_ids = set(prompt_ids[:n_test]), set(prompt_ids[n_test : n_test + n_val])
    train_ids = set(prompt_ids[n_test + n_val:])
    script_logger.info(f"Group-wise split complete. Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    return train_ids, val_ids, test_ids


# Visualisation and Reporting Functions
def plot_case_study(results: Dict, prompt_id: int, p_key: str, output_path: Path):
    layers = list(results['base_cos_sims'].keys())
    base_sims, ft_sims = np.array(list(results['base_cos_sims'].values())), np.array(list(results['ft_cos_sims'].values()))
    sim_delta = ft_sims - base_sims
    base_norms, ft_norms = np.array(list(results['base_norms_para'].values())), np.array(list(results['ft_norms_para'].values()))
    norm_delta = ft_norms - base_norms

    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Comprehensive Case Study Analysis\nPrompt ID: {prompt_id} | Paraphrase: "{p_key}"', fontsize=20, weight='bold')
    axs[0, 0].plot(layers, base_sims, 'o-', label='Base Model Similarity', color='cornflowerblue', lw=2)
    axs[0, 0].plot(layers, ft_sims, 's-', label='Fine-Tuned Model Similarity', color='firebrick', lw=2)
    axs[0, 0].fill_between(layers, base_sims, ft_sims, where=(ft_sims > base_sims), color='mediumseagreen', alpha=0.3, label='Improvement')
    axs[0, 0].set_title('1. Representational Similarity Trajectory', fontsize=14); axs[0, 0].set_ylabel('Cosine Similarity'); axs[0, 0].legend()
    colors_sim = ['mediumseagreen' if d >= 0 else 'salmon' for d in sim_delta]
    axs[0, 1].bar(layers, sim_delta, color=colors_sim, alpha=0.8); axs[0, 1].axhline(0, color='black', lw=1, linestyle='--')
    axs[0, 1].set_title('2. Change in Similarity (Δ)', fontsize=14); axs[0, 1].set_ylabel('Δ Cosine Similarity (FT - Base)')
    axs[1, 0].plot(layers, base_norms, 'o-', label='Base Model Norms', color='darkorange', lw=2)
    axs[1, 0].plot(layers, ft_norms, 's-', label='Fine-Tuned Model Norms', color='purple', lw=2)
    axs[1, 0].set_title('3. Mean Activation Norms (Paraphrase Input)', fontsize=14); axs[1, 0].set_ylabel('Mean L2 Norm'); axs[1, 0].legend()
    colors_norm = ['mediumseagreen' if d >= 0 else 'salmon' for d in norm_delta]
    axs[1, 1].bar(layers, norm_delta, color=colors_norm, alpha=0.8); axs[1, 1].axhline(0, color='black', lw=1, linestyle='--')
    axs[1, 1].set_title('4. Change in Activation Norm (Δ)', fontsize=14); axs[1, 1].set_ylabel('Δ Mean L2 Norm (FT - Base)')
    for ax in axs.flat: ax.grid(True, which='both', linestyle='--', linewidth=0.5); ax.set_xlabel('Decoder Layer')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]); fig.savefig(output_path / f"case_study_prompt_{prompt_id}_{p_key}.png"); plt.close(fig)

def plot_aggregate_results(agg_data: Dict, num_samples: int, output_path: Path):
    if not any(agg_data.values()):
        script_logger.warning("No data for aggregation plot. Skipping.")
        return
    
    sim_base_records = [(pkey, layer, val) for pkey, layers in agg_data['base_sims'].items() for layer, vals in layers.items() for val in vals]
    df_sim_base = pd.DataFrame(sim_base_records, columns=['paraphrase', 'layer', 'value'])
    sim_ft_records = [(pkey, layer, val) for pkey, layers in agg_data['ft_sims'].items() for layer, vals in layers.items() for val in vals]
    df_sim_ft = pd.DataFrame(sim_ft_records, columns=['paraphrase', 'layer', 'value'])
    df_sim_delta = df_sim_ft['value'] - df_sim_base['value']
    df_sim_delta = pd.concat([df_sim_base[['paraphrase', 'layer']], df_sim_delta.rename('delta')], axis=1)

    norm_base_records = [(pkey, layer, val) for pkey, layers in agg_data['base_norms'].items() for layer, vals in layers.items() for val in vals]
    df_norm_base = pd.DataFrame(norm_base_records, columns=['paraphrase', 'layer', 'value'])
    norm_ft_records = [(pkey, layer, val) for pkey, layers in agg_data['ft_norms'].items() for layer, vals in layers.items() for val in vals]
    df_norm_ft = pd.DataFrame(norm_ft_records, columns=['paraphrase', 'layer', 'value'])
    df_norm_delta = df_norm_ft['value'] - df_norm_base['value']
    df_norm_delta = pd.concat([df_norm_base[['paraphrase', 'layer']], df_norm_delta.rename('delta')], axis=1)

    emb_base_records = [(pkey, val) for pkey, vals in agg_data['base_emb_sims'].items() for val in vals]
    df_emb_base = pd.DataFrame(emb_base_records, columns=['paraphrase', 'value'])
    emb_ft_records = [(pkey, val) for pkey, vals in agg_data['ft_emb_sims'].items() for val in vals]
    df_emb_ft = pd.DataFrame(emb_ft_records, columns=['paraphrase', 'value'])
    df_emb_delta = df_emb_ft['value'] - df_emb_base['value']
    df_emb_delta = pd.concat([df_emb_base[['paraphrase']], df_emb_delta.rename('delta')], axis=1)

    if not df_sim_delta.empty:
        mean_by_layer = df_sim_delta.groupby('layer')['delta'].mean()
        sem_by_layer = df_sim_delta.groupby('layer')['delta'].sem()
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(mean_by_layer.index, mean_by_layer.values, 'o-', color='mediumseagreen', label='Mean Δ Similarity')
        ax.fill_between(mean_by_layer.index, mean_by_layer.values - 1.96 * sem_by_layer, mean_by_layer.values + 1.96 * sem_by_layer, color='mediumseagreen', alpha=0.2, label='95% Confidence Interval')
        ax.axhline(0, color='black', lw=1, linestyle='--'); ax.legend(); ax.grid(True)
        ax.set_title(f'Overall Average Change in Representational Similarity (N={num_samples})', fontsize=16, weight='bold')
        ax.set_xlabel('Decoder Layer'); ax.set_ylabel('Average Δ Cosine Similarity (FT - Base)')
        plt.tight_layout(); fig.savefig(output_path / "aggregate_lineplot_sim_delta.png"); plt.close(fig)

    if not df_sim_delta.empty:
        heatmap_data = df_sim_delta.groupby(['paraphrase', 'layer'])['delta'].mean().unstack(level='layer')
        fig, ax = plt.subplots(figsize=(16, max(8, len(heatmap_data.index) * 0.5)))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar_kws={'label': 'Average Δ Cosine Similarity (FT - Base)'})
        ax.set_title(f'Aggregate Analysis: Mean Change in Similarity by Paraphrase (N={num_samples})', fontsize=16, weight='bold')
        ax.set_xlabel('Decoder Layer'); ax.set_ylabel('Paraphrase Type')
        plt.xticks(rotation=45); plt.tight_layout(); fig.savefig(output_path / "aggregate_heatmap_sim_delta.png"); plt.close(fig)

    if not df_norm_delta.empty:
        mean_by_layer = df_norm_delta.groupby('layer')['delta'].mean()
        sem_by_layer = df_norm_delta.groupby('layer')['delta'].sem()
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(mean_by_layer.index, mean_by_layer.values, 'o-', color='darkslateblue', label='Mean Δ Activation Norm')
        ax.fill_between(mean_by_layer.index, mean_by_layer.values - 1.96 * sem_by_layer, mean_by_layer.values + 1.96 * sem_by_layer, color='darkslateblue', alpha=0.2, label='95% CI')
        ax.axhline(0, color='black', lw=1, linestyle='--'); ax.legend(); ax.grid(True)
        ax.set_title(f'Overall Average Change in Activation Norm (N={num_samples})', fontsize=16, weight='bold')
        ax.set_xlabel('Decoder Layer'); ax.set_ylabel('Average Δ Mean L2 Norm (FT - Base)')
        plt.tight_layout(); fig.savefig(output_path / "aggregate_lineplot_norm_delta.png"); plt.close(fig)

    if not df_emb_delta.empty:
        fig, ax = plt.subplots(figsize=(14, max(8, len(df_emb_delta['paraphrase'].unique()) * 0.6)))
        sns.boxplot(data=df_emb_delta, x='delta', y='paraphrase', orient='h', ax=ax, palette='coolwarm')
        ax.axvline(0, color='black', lw=1, linestyle='--')
        ax.set_title(f'Change in Embedding Space Similarity After Fine-Tuning (N={num_samples})', fontsize=16, weight='bold')
        ax.set_xlabel('Δ Cosine Similarity (FT - Base)'); ax.set_ylabel('Paraphrase Type')
        ax.grid(True, axis='x'); plt.tight_layout(); fig.savefig(output_path / "aggregate_boxplot_embedding_delta.png"); plt.close(fig)
    script_logger.info("Saved all aggregate plots.")

def report_case_study_numerics(results: Dict, prompt_id: int, p_key: str, output_path: Path):
    base_sims, ft_sims = np.array(list(results['base_cos_sims'].values())), np.array(list(results['ft_cos_sims'].values()))
    sim_delta, sim_ttest = ft_sims - base_sims, stats.ttest_rel(ft_sims, base_sims)
    base_norms, ft_norms = np.array(list(results['base_norms_para'].values())), np.array(list(results['ft_norms_para'].values()))
    norm_delta, norm_ttest = ft_norms - base_norms, stats.ttest_rel(ft_norms, base_norms)
    emb_sim_delta = results['ft_emb_sim'] - results['base_emb_sim']
    report = [f"# Comprehensive Analysis Report: Case Study", f"- **Prompt ID:** `{prompt_id}`", f"- **Paraphrase Type:** `{p_key}`", "---",
              "## 1. Embedding Space Analysis", "| Statistic | Value |", "|---|---|", f"| Base Model Embedding Similarity | `{results['base_emb_sim']:.6f}` |", f"| Fine-Tuned Model Embedding Similarity | `{results['ft_emb_sim']:.6f}` |", f"| **Delta (FT - Base)** | **`{emb_sim_delta:+.6f}`** |", "---",
              "## 2. Representational Similarity (Hidden States)", "| Statistic | Value |", "|---|---|", f"| Mean Delta | `{sim_delta.mean():.6f}` |", f"| Paired T-test p-value | `{sim_ttest.pvalue:.6f}` |", f"| **Significant?** | **{'YES' if sim_ttest.pvalue < 0.05 else 'NO'}** |", "---",
              "## 3. Activation Norm Analysis (Hidden States)", "| Statistic | Value |", "|---|---|", f"| Mean Delta | `{norm_delta.mean():.6f}` |", f"| Paired T-test p-value | `{norm_ttest.pvalue:.6f}` |", f"| **Significant?** | **{'YES' if norm_ttest.pvalue < 0.05 else 'NO'}** |", "---",
              "## 4. Raw Data by Layer", "| Layer | Base Sim. | FT Sim. | Sim Delta | Base Norm | FT Norm | Norm Delta |", "|---|---|---|---|---|---|---|"]
    for i in results['base_cos_sims'].keys():
        report.append(f"| {i:<5} | {results['base_cos_sims'][i]:.6f} | {results['ft_cos_sims'][i]:.6f} | {sim_delta[i]:+.6f} | {results['base_norms_para'][i]:.6f} | {results['ft_norms_para'][i]:.6f} | {norm_delta[i]:+.6f} |")
    (output_path / f"report_case_study_prompt_{prompt_id}_{p_key}.md").write_text("\n".join(report), encoding="utf-8")

def report_aggregate_numerics(agg_data: Dict, num_samples: int, output_path: Path):
    if not any(agg_data.values()): return
    all_dfs = {
        "Embedding Similarity": pd.DataFrame([(p, v, 'base') for p, V in agg_data['base_emb_sims'].items() for v in V] + 
                                            [(p, v, 'ft') for p, V in agg_data['ft_emb_sims'].items() for v in V], 
                                            columns=['paraphrase', 'value', 'model']),
        "Representational Similarity": pd.DataFrame([(p, l, v, 'base') for p, L in agg_data['base_sims'].items() for l, V in L.items() for v in V] +
                                                    [(p, l, v, 'ft') for p, L in agg_data['ft_sims'].items() for l, V in L.items() for v in V],
                                                    columns=['paraphrase', 'layer', 'value', 'model']),
        "Activation Norm": pd.DataFrame([(p, l, v, 'base') for p, L in agg_data['base_norms'].items() for l, V in L.items() for v in V] +
                                        [(p, l, v, 'ft') for p, L in agg_data['ft_norms'].items() for l, V in L.items() for v in V],
                                        columns=['paraphrase', 'layer', 'value', 'model'])
    }
    report_md = ["# Aggregate Numerical Analysis Report", f"Based on **{num_samples}** processed prompts.", "---"]
    report_json = {"num_samples": num_samples, "analyses": {}}
    for analysis_name, df in all_dfs.items():
        if df.empty: continue
        report_md.append(f"## Analysis: `{analysis_name}`")
        report_md.append("| Paraphrase Type | Mean Base | Mean FT | Mean Delta | Median Delta | Std Dev Delta | T-test p-value (on Delta) | Significant? |")
        report_md.append("|---|---|---|---|---|---|---|---|")
        analysis_stats = {}
        all_base_vals, all_ft_vals = [], []
        p_keys = df['paraphrase'].unique()
        for p_key in sorted(p_keys):
            base_series = df[(df['paraphrase'] == p_key) & (df['model'] == 'base')]['value']
            ft_series = df[(df['paraphrase'] == p_key) & (df['model'] == 'ft')]['value']
            if len(base_series) < 2 or len(ft_series) < 2: continue
            all_base_vals.extend(base_series)
            all_ft_vals.extend(ft_series)
            delta_series = ft_series.values - base_series.values
            mean_base, mean_ft = base_series.mean(), ft_series.mean()
            mean_delta, median_delta, std_delta = delta_series.mean(), np.median(delta_series), delta_series.std()
            ttest_res = stats.ttest_1samp(delta_series, 0)
            is_sig = ttest_res.pvalue < 0.05
            report_md.append(f"| `{p_key}` | `{mean_base:.6f}` | `{mean_ft:.6f}` | `{mean_delta:+.6f}` | `{median_delta:+.6f}` | `{std_delta:.6f}` | `{ttest_res.pvalue:.6f}` | **{'YES' if is_sig else 'NO'}** |")
            analysis_stats[p_key] = {"mean_base": mean_base, "mean_ft": mean_ft, "mean_delta": mean_delta, "median_delta": median_delta, "std_dev_delta": std_delta, "p_value_delta": ttest_res.pvalue}
        if len(all_base_vals) >= 2:
            g_delta = np.array(all_ft_vals) - np.array(all_base_vals)
            g_mean_base, g_mean_ft = np.mean(all_base_vals), np.mean(all_ft_vals)
            g_mean_delta, g_median_delta, g_std_delta = g_delta.mean(), np.median(g_delta), g_delta.std()
            g_ttest = stats.ttest_1samp(g_delta, 0)
            g_is_sig = g_ttest.pvalue < 0.05
            report_md.append(f"| **_GRAND TOTAL_** | `{g_mean_base:.6f}` | `{g_mean_ft:.6f}` | `{g_mean_delta:+.6f}` | `{g_median_delta:+.6f}` | `{g_std_delta:.6f}` | `{g_ttest.pvalue:.6f}` | **{'YES' if g_is_sig else 'NO'}** |")
            analysis_stats["__grand_total__"] = {"mean_base": g_mean_base, "mean_ft": g_mean_ft, "mean_delta": g_mean_delta, "median_delta": g_median_delta, "std_dev_delta": g_std_delta, "p_value_delta": g_ttest.pvalue}
        report_md.append("\n---\n")
        report_json["analyses"][analysis_name] = analysis_stats
    (output_path / "aggregate_summary.md").write_text("\n".join(report_md), encoding="utf-8")
    (output_path / "aggregate_summary.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    script_logger.info("Saved aggregate numerical reports (MD and JSON).")


# Main impl
def main():
    parser = argparse.ArgumentParser(description="Advanced Robustness Analyzer for LLM Activations.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--ft_model_path", type=str, required=True)
    parser.add_argument("--prompts_json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="robustness_analysis_results")
    parser.add_argument("--run_mode", type=str, choices=['case_study', 'aggregate'], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--paraphrase_keys", type=str, nargs='+')
    parser.add_argument("--prompt_ids", type=int, nargs='+')
    parser.add_argument("--data_split", type=str, choices=['all', 'train', 'val', 'test'], default='val')
    parser.add_argument("--val_size", type=float, default=0.05)
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output_dir); output_path.mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": script_logger.warning("CUDA not available. Running on CPU will be extremely slow.")

    script_logger.info("--- Model and Data Loading ---")
    base_model, tokenizer = load_model_and_tokenizer(args.base_model_path, args.base_model_path, device)
    ft_model, _ = load_model_and_tokenizer(args.ft_model_path, args.base_model_path, device)
    ft_model.eval()

    with open(args.prompts_json_path, 'r', encoding='utf-8') as f: prompts_data_all = json.load(f)

    if args.data_split != 'all':
        all_pids = [item['prompt_count'] for item in prompts_data_all if 'prompt_count' in item]
        train_ids, val_ids, test_ids = get_group_wise_split_ids(all_pids, args.val_size, args.test_size, args.seed)
        target_ids = {'train': train_ids, 'val': val_ids, 'test': test_ids}[args.data_split]
        prompts_data = [item for item in prompts_data_all if item.get('prompt_count') in target_ids]
    else:
        prompts_data = prompts_data_all
    
    if args.limit > 0 and args.run_mode == 'aggregate':
        prompts_data = prompts_data[:args.limit]
        script_logger.info(f"Limited aggregate run to the first {len(prompts_data)} items.")
    prompts_map = {item['prompt_count']: item for item in prompts_data}

    if args.run_mode == 'aggregate':
        analyze_and_plot_weight_deltas(base_model, ft_model, output_path)
        aggregate_data = {
            'base_emb_sims': defaultdict(list), 'ft_emb_sims': defaultdict(list),
            'base_sims': defaultdict(lambda: defaultdict(list)), 'ft_sims': defaultdict(lambda: defaultdict(list)),
            'base_norms': defaultdict(lambda: defaultdict(list)), 'ft_norms': defaultdict(lambda: defaultdict(list))
        }
        # List to store outlier data for the report
        outlier_records = []
        
        num_processed = 0
        for i, prompt_item in enumerate(prompts_data):
            pid, original_text = prompt_item.get('prompt_count'), prompt_item.get('instruction_original')
            if not all([pid, original_text]): continue
            keys_to_process = args.paraphrase_keys if args.paraphrase_keys else [k for k in prompt_item if k.startswith('instruct_') and prompt_item[k]]
            for p_key in keys_to_process:
                paraphrase_text = prompt_item.get(p_key)
                if not paraphrase_text: continue
                script_logger.info(f"Processing... [Prompt {i+1}/{len(prompts_data)}] [ID: {pid}] [{p_key}]")
                try:
                    results = run_and_analyze_pair(base_model, ft_model, tokenizer, original_text, paraphrase_text)
                    
                    # Outlier tracking - stores data for the report
                    emb_sim_delta = results['ft_emb_sim'] - results['base_emb_sim']
                    OUTLIER_THRESHOLD = 0.001 
                    if abs(emb_sim_delta) > OUTLIER_THRESHOLD:
                        outlier_info = {
                            "pid": pid, "p_key": p_key, "delta": emb_sim_delta,
                            "original": original_text, "paraphrase": paraphrase_text
                        }
                        outlier_records.append(outlier_info)
                        script_logger.warning(f"Outlier detected for Prompt ID {pid} ({p_key}). Delta: {emb_sim_delta:+.6f}")
                    
                    aggregate_data['base_emb_sims'][p_key].append(results['base_emb_sim'])
                    aggregate_data['ft_emb_sims'][p_key].append(results['ft_emb_sim'])
                    for layer_idx in results['base_cos_sims']:
                        aggregate_data['base_sims'][p_key][layer_idx].append(results['base_cos_sims'][layer_idx])
                        aggregate_data['ft_sims'][p_key][layer_idx].append(results['ft_cos_sims'][layer_idx])
                        aggregate_data['base_norms'][p_key][layer_idx].append(results['base_norms_para'][layer_idx])
                        aggregate_data['ft_norms'][p_key][layer_idx].append(results['ft_norms_para'][layer_idx])
                except Exception as e:
                    script_logger.error(f"Failed on Prompt ID {pid}, Paraphrase '{p_key}'. Error: {e}")
            num_processed += 1
            
        if num_processed > 0:
            script_logger.info(f"Aggregation complete. Processed {num_processed} prompts.")
            
            # Write the outlier report file
            if outlier_records:
                report_path = output_path / "embedding_similarity_outliers.md"
                script_logger.info(f"Writing {len(outlier_records)} outliers to {report_path}")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("# Embedding Similarity Outlier Report\n\n")
                    f.write("This report details prompts where the change in embedding-level cosine similarity between the original and paraphrased text exceeded the defined threshold.\n\n")
                    for record in sorted(outlier_records, key=lambda x: x['pid']):
                        f.write(f"## Prompt ID: {record['pid']}\n\n")
                        f.write(f"- **Paraphrase Type:** `{record['p_key']}`\n")
                        f.write(f"- **Similarity Delta (FT - Base):** `{record['delta']:+.6f}`\n\n")
                        f.write(f"**Original Text:**\n```\n{record['original']}\n```\n\n")
                        f.write(f"**Paraphrase Text:**\n```\n{record['paraphrase']}\n```\n")
                        f.write("\n---\n\n")
            
            plot_aggregate_results(aggregate_data, num_processed, output_path)
            report_aggregate_numerics(aggregate_data, num_processed, output_path)
        else:
            script_logger.warning("No prompts were successfully processed.")

    elif args.run_mode == 'case_study':
        pids_to_process = []
        if args.prompt_ids: pids_to_process = [pid for pid in args.prompt_ids if pid in prompts_map]
        elif args.limit > 0: pids_to_process = [item['prompt_count'] for item in prompts_data[:args.limit]]
        for pid in pids_to_process:
            item = prompts_map[pid]
            keys_to_process = args.paraphrase_keys if args.paraphrase_keys else [k for k in item if k.startswith('instruct_')]
            for p_key in keys_to_process:
                original_text, paraphrase_text = item.get('instruction_original'), item.get(p_key)
                if not all([original_text, paraphrase_text]): continue
                script_logger.info(f"Analyzing Prompt ID: {pid}, Paraphrase: {p_key}")
                try:
                    results = run_and_analyze_pair(base_model, ft_model, tokenizer, original_text, paraphrase_text)
                    plot_case_study(results, pid, p_key, output_path)
                    report_case_study_numerics(results, pid, p_key, output_path)
                except Exception as e:
                    script_logger.error(f"Failed on Prompt ID {pid}, Paraphrase '{p_key}'. Error: {e}"); traceback.print_exc()

    script_logger.info(f"Analysis complete. All outputs saved to: {output_path}")

if __name__ == "__main__":
    main()
