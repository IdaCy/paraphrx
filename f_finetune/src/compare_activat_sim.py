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
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)


class SafeFormatter(logging.Formatter):
    """A custom formatter that prevents crashes from malformed log messages."""
    def format(self, record):
        record_copy = logging.makeLogRecord(record.__dict__)
        try:
            return super().format(record_copy)
        except (TypeError, ValueError):
            record_copy.msg = f"MALFORMED LOG: {record_copy.msg} | ARGS: {record_copy.args}"
            record_copy.args = ()
            return logging.Formatter('[%(levelname)s] %(message)s').format(record_copy)

# Create a single, safe handler
safe_handler = logging.StreamHandler(sys.stdout)
safe_handler.setFormatter(SafeFormatter("%(asctime)s [%(levelname)s] - %(message)s"))

# Configure the logger for THIS script's messages
script_logger = logging.getLogger(__name__)
script_logger.setLevel(logging.INFO)
script_logger.handlers.clear()
script_logger.addHandler(safe_handler)
script_logger.propagate = False # vent messages from being sent to the root logger

# Surgically target and fix the 'transformers' library's logger
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING) # only want to see warnings/errors from it
transformers_logger.handlers.clear()
transformers_logger.addHandler(safe_handler)
transformers_logger.propagate = False # stop it from propagating up

# Matplotlib and Style Setup
plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8-whitegrid')


# Core Comparison Logic using Hooks (No changes needed here)

class ActivationComparator:
    """A stateful class to compare activations between two forward passes using hooks."""
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
    model.eval()
    with ActivationComparator(model) as comparator:
        inputs = tokenizer(
            [original_text, paraphrase_text],
            return_tensors="pt", padding="longest", truncation=True, max_length=512
        ).to(model.device)
        with torch.no_grad():
            _ = model(input_ids=inputs.input_ids[0].unsqueeze(0))
            _ = model(input_ids=inputs.input_ids[1].unsqueeze(0))
    return dict(sorted(comparator.similarities.items()))


# model loader

def load_model_and_tokenizer(
    model_path: str, base_model_path: str, device: str
) -> tuple[PreTrainedModel, AutoTokenizer]:
    path = Path(model_path)
    base_path = Path(base_model_path)

    if not base_path.exists():
        script_logger.error(f"Base model path not found at: {base_path}")
        sys.exit(1)

    script_logger.info(f"Loading tokenizer from base path: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_to_load = None
    if (path / "adapter_config.json").exists():
        script_logger.info("Detected LoRA adapters. Loading base model and applying adapters.")
        model_to_load = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map={"": device}
        )
        model_to_load = PeftModel.from_pretrained(model_to_load, model_path)
        script_logger.info("Successfully merged LoRA adapters.")
    else:
        script_logger.info("Detected a full model or fine-tune without adapters. Loading directly.")
        model_to_load = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device}
        )

    if model_to_load is None:
        script_logger.error(f"Could not determine how to load model from {model_path}")
        sys.exit(1)

    return model_to_load, tokenizer


# Visualistion functions

def plot_case_study(base_sims, ft_sims, prompt_id, p_key, output_path):
    layers, base_scores, ft_scores = list(base_sims.keys()), np.array(list(base_sims.values())), np.array(list(ft_sims.values()))
    delta = ft_scores - base_scores
    fig, axs = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'Case Study: Representational Similarity\nPrompt ID: {prompt_id} | Paraphrase: "{p_key}"', fontsize=18, weight='bold')
    ax1 = axs[0]
    ax1.plot(layers, base_scores, 'o-', label='Base Model Similarity', color='cornflowerblue', lw=2)
    ax1.plot(layers, ft_scores, 's-', label='Fine-Tuned Model Similarity', color='firebrick', lw=2)
    ax1.fill_between(layers, base_scores, ft_scores, where=(ft_scores > base_scores), color='mediumseagreen', alpha=0.3, label='Improvement')
    ax1.fill_between(layers, base_scores, ft_scores, where=(ft_scores <= base_scores), color='salmon', alpha=0.3, label='Regression')
    ax1.set_title('Similarity Trajectory Across Layers', fontsize=14); ax1.set_ylabel('Cosine Similarity', fontsize=12); ax1.set_xlabel('Decoder Layer', fontsize=12); ax1.legend(); ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2 = axs[1]
    colors = ['mediumseagreen' if d >= 0 else 'salmon' for d in delta]
    ax2.bar(layers, delta, color=colors, alpha=0.8, label='Similarity Delta (FT - Base)'); ax2.axhline(0, color='black', lw=1, linestyle='--')
    ax2.set_title('Change in Similarity After Fine-Tuning', fontsize=14); ax2.set_ylabel('Δ Cosine Similarity', fontsize=12); ax2.set_xlabel('Decoder Layer', fontsize=12); ax2.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig.savefig(output_path / f"case_study_prompt_{prompt_id}_{p_key}.png")
    script_logger.info(f"Saved case study plot to {output_path / f'case_study_prompt_{prompt_id}_{p_key}.png'}")
    plt.close(fig)

def plot_aggregate_results(agg_data, num_samples, output_path):
    if not agg_data: script_logger.warning("No data for aggregation plot. Skipping."); return
    p_keys = sorted(agg_data.keys()); layers = sorted(agg_data[p_keys[0]].keys())
    heatmap_data = np.array([[np.mean(agg_data[pk][layer]) for layer in layers] for pk in p_keys])
    fig, ax = plt.subplots(figsize=(16, max(8, len(p_keys) * 0.5)))
    sns.heatmap(heatmap_data, yticklabels=p_keys, xticklabels=layers, annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar_kws={'label': 'Average Δ Cosine Similarity (FT - Base)'})
    ax.set_title(f'Aggregate Analysis: Mean Change in Similarity (N={num_samples})', fontsize=16, weight='bold'); ax.set_xlabel('Decoder Layer', fontsize=12); ax.set_ylabel('Paraphrase Type', fontsize=12)
    plt.xticks(rotation=45); plt.tight_layout(); fig.savefig(output_path / "aggregate_heatmap_delta_by_type.png")
    script_logger.info(f"Saved aggregate heatmap to {output_path / 'aggregate_heatmap_delta_by_type.png'}"); plt.close(fig)
    all_deltas = defaultdict(list)
    for pk in p_keys:
        for layer in layers: all_deltas[layer].extend(agg_data[pk][layer])
    mean_deltas = np.array([np.mean(all_deltas[layer]) for layer in layers]); sem = np.array([np.std(all_deltas[layer]) / np.sqrt(len(all_deltas[layer])) for layer in layers])
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(layers, mean_deltas, 'o-', color='darkslateblue', label='Mean Δ Similarity'); ax.fill_between(layers, mean_deltas - 1.96 * sem, mean_deltas + 1.96 * sem, color='darkslateblue', alpha=0.2, label='95% Confidence Interval')
    ax.axhline(0, color='black', lw=1, linestyle='--'); ax.set_title(f'Overall Average Change in Similarity (N={num_samples})', fontsize=16, weight='bold')
    ax.set_xlabel('Decoder Layer', fontsize=12); ax.set_ylabel('Average Δ Cosine Similarity (FT - Base)', fontsize=12); ax.legend(); ax.grid(True); plt.tight_layout()
    fig.savefig(output_path / "aggregate_lineplot_mean_delta.png")
    script_logger.info(f"Saved aggregate line plot to {output_path / 'aggregate_lineplot_mean_delta.png'}"); plt.close(fig)


# Main impl

def main():
    parser = argparse.ArgumentParser(description="Robustness Analyzer for LLM Activations.")
    parser.add_argument("--base_model_path", type=str, required=True); parser.add_argument("--ft_model_path", type=str, required=True)
    parser.add_argument("--prompts_json_path", type=str, required=True); parser.add_argument("--output_dir", type=str, default="robustness_analysis_results")
    parser.add_argument("--run_mode", type=str, choices=['case_study', 'aggregate'], default='case_study'); parser.add_argument("--prompt_ids", type=int, nargs='+')
    parser.add_argument("--paraphrase_keys", type=str, nargs='+'); parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.run_mode == 'case_study' and (not args.prompt_ids or not args.paraphrase_keys):
        parser.error("--prompt_ids and --paraphrase_keys are required for --run_mode case_study.")

    output_path = Path(args.output_dir); output_path.mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": script_logger.warning("CUDA not available. Running on CPU will be extremely slow.")

    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model_path, args.base_model_path, device)
    ft_model, ft_tokenizer = load_model_and_tokenizer(args.ft_model_path, args.base_model_path, device)

    with open(args.prompts_json_path, 'r', encoding='utf-8') as f: prompts_data = json.load(f)
    prompts_map = {item['prompt_count']: item for item in prompts_data}

    if args.run_mode == 'case_study':
        script_logger.info("--- Running in Case Study Mode ---")
        for pid in args.prompt_ids:
            if pid not in prompts_map: continue
            for p_key in args.paraphrase_keys:
                if p_key not in prompts_map[pid] or not prompts_map[pid][p_key]: continue
                script_logger.info(f"Analyzing Prompt ID: {pid}, Paraphrase: {p_key}")
                base_sims = run_and_compare_activations(base_model, base_tokenizer, prompts_map[pid]['instruction_original'], prompts_map[pid][p_key])
                ft_sims = run_and_compare_activations(ft_model, ft_tokenizer, prompts_map[pid]['instruction_original'], prompts_map[pid][p_key])
                plot_case_study(base_sims, ft_sims, pid, p_key, output_path)

    elif args.run_mode == 'aggregate':
        script_logger.info("--- Running in Aggregate Mode ---")
        aggregate_data = defaultdict(lambda: defaultdict(list))
        prompts_to_process = prompts_data[:args.limit] if args.limit > 0 else prompts_data
        num_processed = 0
        for i, prompt_item in enumerate(prompts_to_process):
            pid, original_instruction = prompt_item.get('prompt_count'), prompt_item.get('instruction_original')
            if not all([pid, original_instruction]): continue
            paraphrase_keys = [k for k in prompt_item if k.startswith('instruct_') and prompt_item[k]]
            for p_key in paraphrase_keys:
                script_logger.info(f"Processing... [Prompt {i+1}/{len(prompts_to_process)}] [ID: {pid}] [{p_key}]")
                base_sims = run_and_compare_activations(base_model, base_tokenizer, original_instruction, prompt_item[p_key])
                ft_sims = run_and_compare_activations(ft_model, ft_tokenizer, original_instruction, prompt_item[p_key])
                for layer_idx in base_sims.keys(): aggregate_data[p_key][layer_idx].append(ft_sims[layer_idx] - base_sims[layer_idx])
            num_processed += 1
        if num_processed > 0: plot_aggregate_results(aggregate_data, num_processed, output_path)

    script_logger.info(f"Analysis complete. All outputs saved to: {output_path}")

if __name__ == "__main__":
    main()
