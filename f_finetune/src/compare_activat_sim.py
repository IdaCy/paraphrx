#!/usr/bin/env python
"""
comparing model activation similarity

analyses the internal representational similarity between original instructions
and their paraphrases across a base and a fine-tuned model; modes:

1.  case_study: multi-panel plots for specific prompt/paraphrase pairs
2.  aggregate: runs on a large number of prompts to compute and visualise
    - average trends, for systematic changes from fine-tuning

detects and handles both fully fine-tuned models and LoRA adapters

srun python robustness_analyzer.py \
  --run_mode case_study \
  --base_model_path "f_finetune/model" \
  --ft_model_path "f_finetune/outputs/l9x_a1_notarg_50k_ft/final" \
  --prompts_json_path "/path/to/your/prompts.json" \
  --output_dir "analysis_results/case_studies" \
  --prompt_ids 1 3 \
  --paraphrase_keys "instruct_polite_request" "instruct_sardonic"

srun python robustness_analyzer.py \
  --run_mode aggregate \
  --base_model_path "f_finetune/model" \
  --ft_model_path "f_finetune/outputs/l9x_a1_notarg_50k_ft/final" \
  --prompts_json_path "/path/to/your/prompts.json" \
  --output_dir "analysis_results/aggregate" \
  --limit 100  # Optional: Process the first 100 prompts for a quick test run
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
plt.style.use('seaborn-v0_8-whitegrid')
# backend that doesn't require a GUI, essential for SLURM
plt.switch_backend('Agg')


class ActivationComparator:
    """
    stateful class to compare activations between two forward passes using hooks
    Designed for a two-step process: run a base input, then a comparison input
    """
    def __init__(self, model: PreTrainedModel):
        self._model = model
        self._device = model.device
        self._layer_outputs: Dict[str, torch.Tensor] = {}
        self.similarities: Dict[int, float] = {}
        self._hook_handles = []

    def _hook_fn(self, layer_name: str):
        def hook(module, input, output):
            hidden_state = output[0].detach()
            if layer_name not in self._layer_outputs:
                self._layer_outputs[layer_name] = hidden_state
            else:
                base_activation = self._layer_outputs.pop(layer_name)
                sim = torch.nn.functional.cosine_similarity(
                    base_activation.view(1, -1),
                    hidden_state.to(self._device).view(1, -1),
                    dim=1
                ).item()
                layer_idx = int(layer_name.split('.')[-1])
                self.similarities[layer_idx] = sim
        return hook

    def attach_hooks(self):
        if not hasattr(self._model, 'model') or not hasattr(self._model.model, 'layers'):
            raise TypeError("Model does not have the expected 'model.layers' structure.")
        for i, layer in enumerate(self._model.model.layers):
            layer_name = f"model.layers.{i}"
            handle = layer.register_forward_hook(self._hook_fn(layer_name))
            self._hook_handles.append(handle)

    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()

    def __enter__(self):
        self.attach_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


def run_and_compare_activations(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    original_text: str,
    paraphrase_text: str
) -> Dict[int, float]:
    """Runs a pair of prompts and returns layer-wise cosine similarities"""
    model.eval()
    with ActivationComparator(model) as comparator:
        inputs = tokenizer(
            [original_text, paraphrase_text],
            return_tensors="pt", padding="longest", truncation=True, max_length=512
        ).to(model.device)
        with torch.no_grad():
            _ = model(input_ids=inputs.input_ids[0].unsqueeze(0)) # Run 1: Store activations
            _ = model(input_ids=inputs.input_ids[1].unsqueeze(0)) # Run 2: Compare
    return dict(sorted(comparator.similarities.items()))


# model loader

def load_model_and_tokenizer(
    model_path: str, base_model_path: str, device: str
) -> Tuple[PreTrainedModel, AutoTokenizer]:
    """
    Loads a model and tokenizer, automatically handling full models vs. LoRA
    """
    path = Path(model_path)
    base_path = Path(base_model_path)
    
    if not base_path.exists():
        logging.error(f"Base model path not found at: {base_path}")
        sys.exit(1)
        
    logging.info(f"Loading tokenizer from base path: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_to_load = None
    if (path / "adapter_config.json").exists():
        logging.info("Detected LoRA adapters. Loading base model and applying adapters.")
        model_to_load = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map={"": device}
        )
        model_to_load = PeftModel.from_pretrained(model_to_load, model_path)
        logging.info("Successfully merged LoRA adapters.")
    else:
        logging.info("Detected a full model. Loading directly.")
        model_to_load = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device}
        )

    if model_to_load is None:
        logging.error(f"Could not determine how to load model from {model_path}")
        sys.exit(1)
        
    return model_to_load, tokenizer


# visualisation functions

def plot_case_study(
    base_sims: Dict, ft_sims: Dict, prompt_id: int, paraphrase_key: str, output_path: Path
):
    """Creates a detailed, multi-panel plot for a single analysis case"""
    layers = list(base_sims.keys())
    base_scores = np.array(list(base_sims.values()))
    ft_scores = np.array(list(ft_sims.values()))
    delta = ft_scores - base_scores

    fig, axs = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'Case Study: Representational Similarity\nPrompt ID: {prompt_id} | Paraphrase: "{paraphrase_key}"', fontsize=18, weight='bold')

    # Panel 1: Similarity Trajectory
    ax1 = axs[0]
    ax1.plot(layers, base_scores, 'o-', label='Base Model Similarity', color='cornflowerblue', lw=2)
    ax1.plot(layers, ft_scores, 's-', label='Fine-Tuned Model Similarity', color='firebrick', lw=2)
    ax1.fill_between(layers, base_scores, ft_scores, where=(ft_scores > base_scores), color='mediumseagreen', alpha=0.3, label='Improvement')
    ax1.fill_between(layers, base_scores, ft_scores, where=(ft_scores <= base_scores), color='salmon', alpha=0.3, label='Regression')
    ax1.set_title('Similarity Trajectory Across Layers', fontsize=14)
    ax1.set_ylabel('Cosine Similarity', fontsize=12)
    ax1.set_xlabel('Decoder Layer', fontsize=12)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Panel 2: Change in Similarity (Delta)
    ax2 = axs[1]
    colors = ['mediumseagreen' if d >= 0 else 'salmon' for d in delta]
    ax2.bar(layers, delta, color=colors, alpha=0.8, label='Similarity Delta (FT - Base)')
    ax2.axhline(0, color='black', lw=1, linestyle='--')
    ax2.set_title('Change in Similarity After Fine-Tuning', fontsize=14)
    ax2.set_ylabel('Δ Cosine Similarity', fontsize=12)
    ax2.set_xlabel('Decoder Layer', fontsize=12)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"case_study_prompt_{prompt_id}_{paraphrase_key}.png"
    fig.savefig(output_path / filename)
    logging.info(f"Saved case study plot to {output_path / filename}")
    plt.close(fig)

def plot_aggregate_results(
    agg_data: Dict[str, Dict[int, List[float]]], num_samples: int, output_path: Path
):
    """Creates summary plots (heatmap, line plot) from aggregated data"""
    if not agg_data:
        logging.warning("No data for aggregation plot. Skipping.")
        return

    # Plot 1: Heatmap of Average Delta
    paraphrase_types = sorted(agg_data.keys())
    first_key = paraphrase_types[0]
    layers = sorted(agg_data[first_key].keys())
    heatmap_data = np.array([[np.mean(agg_data[p_key][layer]) for layer in layers] for p_key in paraphrase_types])

    fig, ax = plt.subplots(figsize=(16, max(8, len(paraphrase_types) * 0.5)))
    sns.heatmap(
        heatmap_data, yticklabels=paraphrase_types, xticklabels=layers,
        annot=True, fmt=".3f", cmap="viridis", ax=ax,
        cbar_kws={'label': 'Average Δ Cosine Similarity (FT - Base)'}
    )
    ax.set_title(f'Aggregate Analysis: Mean Change in Similarity per Paraphrase Type (N={num_samples})', fontsize=16, weight='bold')
    ax.set_xlabel('Decoder Layer', fontsize=12)
    ax.set_ylabel('Paraphrase Type', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(output_path / "aggregate_heatmap_delta_by_type.png")
    logging.info(f"Saved aggregate heatmap to {output_path / 'aggregate_heatmap_delta_by_type.png'}")
    plt.close(fig)

    # Plot 2: Line plot of overall average delta
    all_deltas_by_layer = defaultdict(list)
    for p_key in paraphrase_types:
        for layer in layers:
            all_deltas_by_layer[layer].extend(agg_data[p_key][layer])

    mean_deltas = np.array([np.mean(all_deltas_by_layer[layer]) for layer in layers])
    std_devs = np.array([np.std(all_deltas_by_layer[layer]) for layer in layers])
    # Standard error of the mean for confidence interval
    sem = np.array([np.std(all_deltas_by_layer[layer]) / np.sqrt(len(all_deltas_by_layer[layer])) for layer in layers])

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(layers, mean_deltas, 'o-', color='darkslateblue', label='Mean Δ Similarity')
    ax.fill_between(layers, mean_deltas - 1.96 * sem, mean_deltas + 1.96 * sem, color='darkslateblue', alpha=0.2, label='95% Confidence Interval')
    ax.axhline(0, color='black', lw=1, linestyle='--')
    ax.set_title(f'Overall Average Change in Similarity Across All Paraphrases (N={num_samples})', fontsize=16, weight='bold')
    ax.set_xlabel('Decoder Layer', fontsize=12)
    ax.set_ylabel('Average Δ Cosine Similarity (FT - Base)', fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(output_path / "aggregate_lineplot_mean_delta.png")
    logging.info(f"Saved aggregate line plot to {output_path / 'aggregate_lineplot_mean_delta.png'}")
    plt.close(fig)

# Main impl

def main():
    parser = argparse.ArgumentParser(description="Robustness Analyzer for LLM Activations.")
    # Model and Data Paths
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--ft_model_path", type=str, required=True, help="Path to the fine-tuned model or LoRA adapters.")
    parser.add_argument("--prompts_json_path", type=str, required=True, help="Path to the JSON file with prompts.")
    parser.add_argument("--output_dir", type=str, default="robustness_analysis_results", help="Directory to save plots.")
    # Mode Selection
    parser.add_argument("--run_mode", type=str, choices=['case_study', 'aggregate'], default='case_study', help="Choose analysis mode.")
    # Mode-specific arguments
    parser.add_argument("--prompt_ids", type=int, nargs='+', help="[Case Study] Space-separated prompt_count IDs to analyze.")
    parser.add_argument("--paraphrase_keys", type=str, nargs='+', help="[Case Study] Space-separated instruct_* keys to analyze.")
    parser.add_argument("--limit", type=int, default=0, help="[Aggregate] Limit the number of prompts to process (0 for all).")

    args = parser.parse_args()

    # Validate Mode-specific Arguments
    if args.run_mode == 'case_study' and (not args.prompt_ids or not args.paraphrase_keys):
        parser.error("--prompt_ids and --paraphrase_keys are required for --run_mode case_study.")

    # Setup
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.warning("CUDA not available. Running on CPU will be extremely slow.")

    # Load Models
    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model_path, args.base_model_path, device)
    ft_model, ft_tokenizer = load_model_and_tokenizer(args.ft_model_path, args.base_model_path, device)

    # Load Data
    with open(args.prompts_json_path, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    prompts_map = {item['prompt_count']: item for item in prompts_data}

    # Execute Selected Mode
    if args.run_mode == 'case_study':
        logging.info("--- Running in Case Study Mode ---")
        for pid in args.prompt_ids:
            if pid not in prompts_map:
                logging.warning(f"Prompt ID {pid} not found. Skipping.")
                continue
            prompt_item = prompts_map[pid]
            for p_key in args.paraphrase_keys:
                if p_key not in prompt_item or not prompt_item[p_key]:
                    logging.warning(f"Paraphrase key '{p_key}' not found for prompt {pid}. Skipping.")
                    continue
                
                logging.info(f"Analyzing Prompt ID: {pid}, Paraphrase: {p_key}")
                base_sims = run_and_compare_activations(base_model, base_tokenizer, prompt_item['instruction_original'], prompt_item[p_key])
                ft_sims = run_and_compare_activations(ft_model, ft_tokenizer, prompt_item['instruction_original'], prompt_item[p_key])
                
                plot_case_study(base_sims, ft_sims, pid, p_key, output_path)

    elif args.run_mode == 'aggregate':
        logging.info("--- Running in Aggregate Mode ---")
        aggregate_data = defaultdict(lambda: defaultdict(list))
        
        prompts_to_process = prompts_data[:args.limit] if args.limit > 0 else prompts_data
        num_processed = 0

        for i, prompt_item in enumerate(prompts_to_process):
            pid = prompt_item['prompt_count']
            original_instruction = prompt_item.get('instruction_original')
            if not original_instruction: continue

            paraphrase_keys = [k for k in prompt_item if k.startswith('instruct_') and prompt_item[k]]
            
            for p_key in paraphrase_keys:
                logging.info(f"Processing... [Prompt {i+1}/{len(prompts_to_process)}] [ID: {pid}] [{p_key}]")
                base_sims = run_and_compare_activations(base_model, base_tokenizer, original_instruction, prompt_item[p_key])
                ft_sims = run_and_compare_activations(ft_model, ft_tokenizer, original_instruction, prompt_item[p_key])
                
                for layer_idx in base_sims.keys():
                    delta = ft_sims[layer_idx] - base_sims[layer_idx]
                    aggregate_data[p_key][layer_idx].append(delta)

            num_processed += 1
        
        if num_processed > 0:
            logging.info(f"Aggregation complete. Processed {num_processed} prompts.")
            plot_aggregate_results(aggregate_data, num_processed, output_path)
        else:
            logging.warning("No prompts were processed in aggregate mode.")

    logging.info(f"Analysis complete. All outputs saved to: {output_path}")


if __name__ == "__main__":
    main()
