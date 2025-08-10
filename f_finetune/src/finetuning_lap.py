#!/usr/bin/env python
"""
fine-tuning script with Latent Adversarial Paraphrasing (LAP)

Full-parameter fine-tune (2 B params, BF16)

srun python finetuning_lap_final.py \
  --target_modules none \
  --batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --bf16 \
  --run_name gemma2b_fullft_safe \
  --output_dir f_finetune/runs/gemma2b_fullft_safe

LoRA-NF4 4-bit adapters with LAP

srun python finetuning_lap_final.py \
  --data_paths ./path/to/data.json \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --bnb_8bit_optim --bf16 \
  --num_epochs 3 \
  --use_lap \
  --lap_layer 10 \
  --lap_t_inner 5 \
  --run_name gemma2b_lap_safe \
  --output_dir f_finetune/runs/gemma2b_lap_safe

Definitive, robust fine-tuning script with Latent Adversarial Paraphrasing (LAP).

This version includes two major improvements for stability and speed:
1.  **Persistent Caching:** Tokenizes the dataset only ONCE and saves it to disk.
    Subsequent runs load the pre-processed data in seconds, avoiding wasted time.
2.  **AttributeError Fix:** The tokenization function is now correctly structured
    to work with the `datasets` library's multiprocessing.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import numpy as np

import torch
import wandb
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    logging as hf_logging,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Global Configuration
os.environ.setdefault("TQDM_MININTERVAL", "30")
hf_logging.set_verbosity_warning()
tokenizer = None

# Data Structures and Utilities
@dataclasses.dataclass
class Example:
    prompt_count: int
    instruction: str
    inp: str
    answer: str
    style: str

def build_chat_prompt(instruction: str, inp: str | None = "") -> str:
    user_msg = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    messages = [{"role": "user", "content": user_msg}]
    if tokenizer:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"

def load_examples(paths: List[str], instruct_types: List[str]) -> List[Example]:
    examples: List[Example] = []
    load_all = not instruct_types
    for p in paths:
        logging.info(f"Loading raw data from {p}")
        with open(p, "r", encoding="utf-8") as fh: data = json.load(fh)
        for item in data:
            pc_id = item["prompt_count"]
            inp = item.get("input", "")
            base_ans = item.get("output", "")
            if "instruction_original" in item:
                examples.append(Example(pc_id, item["instruction_original"], inp, base_ans, "instruction_original"))
            
            keys_to_process = [k for k in item if k.startswith("instruct_")] if load_all else instruct_types
            for k in keys_to_process:
                if k in item and item[k] and k != "instruction_original":
                    examples.append(Example(pc_id, item[k], inp, base_ans, k))
    random.shuffle(examples)
    logging.info(f"Loaded a total of {len(examples)} examples from JSON.")
    return examples

def tokenise_example(example: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    This function is now designed to accept a dictionary, which is what the
    .map() function provides for each row when using multiprocessing.
    """
    MAX_TOTAL_LENGTH = 1024
    prompt_ids = tokenizer(build_chat_prompt(example['instruction'], example['inp']), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(example['answer'], add_special_tokens=False)["input_ids"]

    if len(prompt_ids) + len(answer_ids) + 1 > MAX_TOTAL_LENGTH:
        answer_ids = answer_ids[:MAX_TOTAL_LENGTH - len(prompt_ids) - 1]

    answer_ids.append(tokenizer.eos_token_id)
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "labels": labels}

class DataCollatorWithPositionIds(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors)
        input_ids = batch["input_ids"]
        shape = input_ids.shape
        batch["position_ids"] = torch.arange(0, shape[1], dtype=torch.long, device=input_ids.device).repeat(shape[0], 1)
        return batch

class LAPTtrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.lap_kwargs = kwargs.pop('lap_kwargs', {})
        super().__init__(*args, **kwargs)
        if self.lap_kwargs.get('use_lap', False):
            logging.info(f"LAP Trainer initialized with: {json.dumps(self.lap_kwargs, indent=2)}")

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self._prepare_inputs(inputs)
        use_lap = self.lap_kwargs.get('use_lap', False) and random.random() < self.lap_kwargs.get('lap_p_sample', 0.5)

        if not use_lap:
            model.train()
            loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            return loss

        model.eval()
        t_inner, epsilon, delta_lr, lambda_lr, layer_idx = (
            self.lap_kwargs['lap_t_inner'], self.lap_kwargs['lap_epsilon'],
            self.lap_kwargs['lap_delta_lr'], self.lap_kwargs['lap_lambda_lr'],
            self.lap_kwargs['lap_layer']
        )
        
        try:
            with torch.no_grad():
                # Correctly call the base model to get hidden states
                outputs_orig = model.get_base_model().model(**inputs, output_hidden_states=True)
                J0 = self.compute_loss(model, inputs).detach()
                hidden_states_l = outputs_orig.hidden_states[layer_idx]

            delta = torch.zeros_like(hidden_states_l, requires_grad=True)
            log_lambda = torch.tensor(0.0, device=model.device, requires_grad=True)
            target_layer = model.get_base_model().model.layers[layer_idx]

            for _ in range(t_inner):
                def add_delta_hook(module, p_inputs, kwargs): return (p_inputs[0] + delta,), kwargs
                hook_handle = target_layer.register_forward_pre_hook(add_delta_hook, with_kwargs=True)
                J_delta = self.compute_loss(model, inputs)
                hook_handle.remove()
                
                lagrangian_delta = -torch.norm(delta, p=2) + torch.exp(log_lambda).detach() * (J_delta - J0 - epsilon)
                delta_grad, = torch.autograd.grad(lagrangian_delta, delta, grad_outputs=-torch.ones_like(lagrangian_delta))
                delta.data.add_(delta_lr * delta_grad)
                delta.grad = None

                def add_updated_delta_hook(module, p_inputs, kwargs): return (p_inputs[0] + delta.detach(),), kwargs
                hook_handle = target_layer.register_forward_pre_hook(add_updated_delta_hook, with_kwargs=True)
                J_delta_updated = self.compute_loss(model, inputs)
                hook_handle.remove()

                lagrangian_lambda = torch.exp(log_lambda) * (J_delta_updated.detach() - J0 - epsilon)
                log_lambda_grad, = torch.autograd.grad(lagrangian_lambda, log_lambda)
                log_lambda.data.add_(lambda_lr * log_lambda_grad)
                log_lambda.grad = None

            final_delta = delta.detach()
        except Exception as e:
            logging.error(f"LAP inner loop failed, falling back to SFT: {e}", exc_info=True)
            torch.cuda.empty_cache()
            model.train()
            return self.compute_loss(model, inputs)

        model.train()
        def add_final_delta_hook(module, p_inputs, kwargs): return (p_inputs[0] + final_delta,), kwargs
        hook_handle = target_layer.register_forward_pre_hook(add_final_delta_hook, with_kwargs=True)
        loss = self.compute_loss(model, inputs)
        hook_handle.remove()
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        return loss

def make_arg_parser():
    p = argparse.ArgumentParser(description="Robust fine-tuning with LAP")
    p.add_argument("--data_paths", nargs="+", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--instruct_types", nargs="+", default=[])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--bnb_8bit_optim", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="paraphrx_ft_lap")
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--use_lap", action="store_true")
    p.add_argument("--lap_layer", type=int, default=10)
    p.add_argument("--lap_t_inner", type=int, default=2)
    p.add_argument("--lap_p_sample", type=float, default=0.5)
    p.add_argument("--lap_epsilon", type=float, default=0.05)
    p.add_argument("--lap_delta_lr", type=float, default=1e-2)
    p.add_argument("--lap_lambda_lr", type=float, default=1e-3)
    return p

def main():
    global tokenizer
    args = make_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.FileHandler(output_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    logging.info(f"Starting run {args.run_name} with config:\n{json.dumps(vars(args), indent=2)}")

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # PERSISTENT CACHING LOGIC
    # Create a unique path for the tokenized data based on the model path
    model_name_safe = Path(args.model_path).name
    tokenized_data_path = output_dir / f"tokenized_data_{model_name_safe}"

    if os.path.exists(tokenized_data_path):
        logging.info(f"Found cached tokenized data at {tokenized_data_path}. Loading from disk.")
        tokenized_datasets = load_from_disk(str(tokenized_data_path))
        train_ds = tokenized_datasets["train"]
        val_ds = tokenized_datasets["validation"]
    else:
        logging.info("No cached data found. Starting full tokenization process.")
        examples = load_examples(args.data_paths, args.instruct_types)
        raw_ds = Dataset.from_list([dataclasses.asdict(e) for e in examples])
        
        split_ds = raw_ds.train_test_split(test_size=0.1, seed=args.seed)
        train_raw, val_raw = split_ds["train"], split_ds["test"]

        num_workers = min(8, os.cpu_count())
        logging.info(f"Using {num_workers} processes for tokenization.")
        
        train_ds = train_raw.map(tokenise_example, remove_columns=train_raw.column_names, num_proc=num_workers, desc="Tokenizing train set")
        val_ds = val_raw.map(tokenise_example, remove_columns=val_raw.column_names, num_proc=num_workers, desc="Tokenizing validation set")

        logging.info(f"Tokenization complete. Saving to disk at {tokenized_data_path} for future runs.")
        full_tokenized_dataset = DatasetDict({"train": train_ds, "validation": val_ds})
        full_tokenized_dataset.save_to_disk(str(tokenized_data_path))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    lcfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=args.target_modules.split(","), lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(output_dir), run_name=args.run_name,
        num_train_epochs=args.num_epochs, optim="paged_adamw_8bit" if args.bnb_8bit_optim else "adamw_torch",
        bf16=args.bf16, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps", eval_steps=200,
        save_strategy="steps", save_steps=200,
        save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        learning_rate=args.learning_rate, weight_decay=0.01,
        lr_scheduler_type="cosine", warmup_ratio=0.1,
        logging_steps=10, report_to=['wandb'] if run else [], seed=args.seed
    )
    
    trainer = LAPTtrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPositionIds(tokenizer=tokenizer, label_pad_token_id=-100),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        lap_kwargs={k: v for k, v in vars(args).items() if k.startswith('lap_')}
    )
    
    trainer.train()
    trainer.save_model(str(output_dir / 'final'))
    tokenizer.save_pretrained(str(output_dir / 'final'))
    if run: run.finish()

if __name__ == '__main__':
    main()
