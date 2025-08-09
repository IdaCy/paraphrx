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
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import logging
import math
import os
import random
import sys
import importlib
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import numpy as np

import torch
import wandb
from datasets import Dataset
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# Global Configuration & Logger Setup
os.environ.setdefault("TQDM_MININTERVAL", "60")
os.environ.setdefault("TQDM_MINITER", "200")
DEBUG_PROMPT_IDS = {1, 42, 321}

# Suppress excessive warnings
hf_logging.set_verbosity_warning()

@dataclasses.dataclass
class Example:
    """Lightweight container used before tokenisation"""
    prompt_count: int
    instruction: str
    inp: str
    answer: str
    style: str

def build_chat_prompt(instruction: str, inp: str | None = "") -> str:
    """Return a single-turn chat prompt in the format Gemma-2-IT was trained on"""
    user_msg = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def tokenise_example(ex: Example):
    """
    Correctly tokenizes an example, masking the prompt part of the labels with -100
    Enforces a total maximum sequence length to prevent OOM errors and logs when truncation occurs
    """
    MAX_TOTAL_LENGTH = 1024

    prompt_txt = build_chat_prompt(ex.instruction, ex.inp)
    prompt_ids_untruncated = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
    answer_ids_untruncated = tokenizer(ex.answer, add_special_tokens=False)["input_ids"]
    original_total_length = len(prompt_ids_untruncated) + len(answer_ids_untruncated) + 1

    full_tokenized = tokenizer(
        prompt_txt + ex.answer + tokenizer.eos_token,
        truncation=True,
        max_length=MAX_TOTAL_LENGTH,
        add_special_tokens=False
    )
    input_ids = full_tokenized["input_ids"]

    if original_total_length > MAX_TOTAL_LENGTH:
        logging.warning(
            f"Sequence truncated for prompt_count {ex.prompt_count}. "
            f"Original length ({original_total_length}) > MAX_TOTAL_LENGTH ({MAX_TOTAL_LENGTH}). "
            f"Truncated to {len(input_ids)} tokens."
        )

    prompt_len_in_sequence = len(prompt_ids_untruncated)
    if prompt_len_in_sequence >= len(input_ids):
        prompt_len_in_sequence = len(input_ids)

    labels = [-100] * prompt_len_in_sequence + input_ids[prompt_len_in_sequence:]
    if len(labels) != len(input_ids):
       return {"input_ids": input_ids, "labels": [-100] * len(input_ids)}

    return {"input_ids": input_ids, "labels": labels}

def three_way_split(ds: Dataset, *, val_pct: float = 0.05, test_pct: float = 0.05, seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """Group-wise split guaranteeing each prompt_count appears in one split"""
    rng = np.random.default_rng(seed)
    pcs = list({ex["prompt_count"] for ex in ds})
    rng.shuffle(pcs)
    n = len(pcs)
    n_val, n_test = int(n * val_pct), int(n * test_pct)
    val_ids, test_ids = set(pcs[:n_val]), set(pcs[n_val : n_val + n_test])
    train_ids = set(pcs[n_val + n_test :])
    train = ds.filter(lambda ex: ex["prompt_count"] in train_ids)
    val = ds.filter(lambda ex: ex["prompt_count"] in val_ids)
    test = ds.filter(lambda ex: ex["prompt_count"] in test_ids)
    return train, val, test

def load_examples(paths: List[str], instruct_types: List[str], use_para_ans: bool) -> List[Example]:
    """Loads and parses the prompt data from JSON files"""
    examples: List[Example] = []
    for p in paths:
        logging.info(f"Loading {p}")
        with open(p, "r", encoding="utf-8") as fh: data = json.load(fh)
        for item in data:
            pc_id = item["prompt_count"]
            base_ans = item.get("output", "")
            inp = item.get("input", "")
            if "instruction_original" in item:
                examples.append(Example(pc_id, item["instruction_original"], inp, base_ans, "instruction_original"))
            keys_to_process = instruct_types or [k for k in item if k.startswith("instruct_")]
            for k in keys_to_process:
                if k in item and item[k] and k != "instruction_original":
                    examples.append(Example(pc_id, item[k], inp, base_ans, k))
    random.shuffle(examples)
    logging.info(f"Loaded a total of {len(examples)} examples.")
    return examples

class LAPTtrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.lap_kwargs = kwargs.pop('lap_kwargs', {})
        super().__init__(*args, **kwargs)
        if self.lap_kwargs.get('use_lap', False):
            logging.info(f"LAP Trainer initialized with robust hook-based method: {json.dumps(self.lap_kwargs, indent=2)}")

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        use_lap = self.lap_kwargs.get('use_lap', False) and random.random() < self.lap_kwargs.get('lap_p_sample', 0.5)
        if not use_lap:
            return self.compute_loss(model, inputs)

        model.eval()
        t_inner = self.lap_kwargs['lap_t_inner']
        epsilon = self.lap_kwargs['lap_epsilon']
        delta_lr = self.lap_kwargs['lap_delta_lr']
        lambda_lr = self.lap_kwargs['lap_lambda_lr']
        layer_idx = self.lap_kwargs['lap_layer']
        inputs = self._prepare_inputs(inputs)

        try:
            with torch.no_grad():
                outputs_orig = model(**inputs, output_hidden_states=True)
                J0 = outputs_orig.loss.detach()
                hidden_states_l_shape = outputs_orig.hidden_states[layer_idx].shape
                hidden_states_l_device = outputs_orig.hidden_states[layer_idx].device
            
            del outputs_orig
            torch.cuda.empty_cache()

            delta = torch.zeros(hidden_states_l_shape, device=hidden_states_l_device, requires_grad=True)
            delta_optimizer = torch.optim.SGD([delta], lr=delta_lr)
            log_lambda = torch.tensor(0.0, device=model.device, requires_grad=True)
            lambda_optimizer = torch.optim.SGD([log_lambda], lr=lambda_lr)

            layer_list = model.model.model.layers
            target_layer = layer_list[layer_idx]

            for _ in range(t_inner):
                delta_optimizer.zero_grad()
                def add_delta_hook(module, p_inputs):
                    return (p_inputs[0] + delta,) + p_inputs[1:]
                hook_handle = target_layer.register_forward_pre_hook(add_delta_hook)
                J_delta = model(**inputs).loss
                hook_handle.remove()
                lagrangian_for_delta = -torch.norm(delta, p=2) + torch.exp(log_lambda).detach() * (J_delta - J0 - epsilon)
                (-lagrangian_for_delta).backward()
                delta_optimizer.step()
                delta.grad.zero_()

                lambda_optimizer.zero_grad()
                def add_updated_delta_hook(module, p_inputs):
                    return (p_inputs[0] + delta.detach(),) + p_inputs[1:]
                hook_handle = target_layer.register_forward_pre_hook(add_updated_delta_hook)
                J_delta_updated = model(**inputs).loss
                hook_handle.remove()
                lagrangian_for_lambda = torch.exp(log_lambda) * (J_delta_updated - J0 - epsilon)
                lagrangian_for_lambda.backward()
                lambda_optimizer.step()
                
                del J_delta, J_delta_updated, lagrangian_for_delta, lagrangian_for_lambda
                torch.cuda.empty_cache()

            final_delta = delta.detach()

        except Exception as e:
            logging.error(f"Error in LAP inner loop: {e}", exc_info=True)
            model.train()
            return self.compute_loss(model, inputs)

        model.train()
        def add_final_delta_hook(module, p_inputs):
            return (p_inputs[0] + final_delta,) + p_inputs[1:]
        hook_handle = target_layer.register_forward_pre_hook(add_final_delta_hook)
        loss = model(**inputs).loss
        hook_handle.remove()
        return loss

def make_arg_parser():
    p = argparse.ArgumentParser(description="Fine-tuning with optional Latent Adversarial Paraphrasing")
    p.add_argument("--data_paths", nargs="+", required=True);
    p.add_argument("--model_path", default="f_finetune/model"); p.add_argument("--output_dir", required=True); p.add_argument("--run_name", default="gemma_paraphrx"); p.add_argument("--instruct_types", nargs="+", default=[]); p.add_argument("--use_paraphrase_answer", action="store_true"); p.add_argument("--val_pct", type=float, default=0.05); p.add_argument("--test_pct", type=float, default=0.05); p.add_argument("--batch_size", type=int, default=4); p.add_argument("--gradient_accumulation_steps", type=int, default=4); p.add_argument("--num_epochs", type=int, default=3); p.add_argument("--learning_rate", type=float, default=3e-5); p.add_argument("--warmup_ratio", type=float, default=0.03); p.add_argument("--lr_scheduler_type", default="cosine"); p.add_argument("--weight_decay", type=float, default=0.0); p.add_argument("--lora_rank", type=int, default=16); p.add_argument("--lora_alpha", type=int, default=32); p.add_argument("--lora_dropout", type=float, default=0.05); p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"); p.add_argument("--use_deepspeed", action="store_true"); p.add_argument("--deepspeed_config", default="ds_zero2.json"); p.add_argument("--bnb_8bit_optim", action="store_true"); p.add_argument("--bf16", action="store_true"); p.add_argument("--eval_steps", type=int, default=800); p.add_argument("--save_steps", type=int, default=800); p.add_argument("--logging_steps", type=int, default=100); p.add_argument("--seed", type=int, default=42); p.add_argument("--wandb_project", default="paraphrx_ft_50k"); p.add_argument("--early_stopping_patience", type=int, default=9); p.add_argument("--use_lap", action="store_true"); p.add_argument("--lap_layer", type=int, default=10); p.add_argument("--lap_t_inner", type=int, default=5); p.add_argument("--lap_p_sample", type=float, default=0.5); p.add_argument("--lap_epsilon", type=float, default=0.05); p.add_argument("--lap_delta_lr", type=float, default=1e-2); p.add_argument("--lap_lambda_lr", type=float, default=1e-3); p.add_argument("--debug_n_samples", type=int, default=0); p.add_argument("--debug_seed", type=int, default=123);
    return p

def summarise_training(trainer: Trainer, out_dir: Path):
    hist = trainer.state.log_history
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_history.json").write_text(json.dumps(hist, indent=2))
    tr_steps, tr_loss, ev_steps, ev_loss = [], [], [], []
    for h in hist:
        if "loss" in h: tr_steps.append(h["step"]); tr_loss.append(h["loss"])
        if "eval_loss" in h: ev_steps.append(h["step"]); ev_loss.append(h["eval_loss"])
    try:
        import matplotlib.pyplot as plt
        plt.figure();
        if tr_loss: plt.plot(tr_steps, tr_loss, label="train")
        if ev_loss: plt.plot(ev_steps, ev_loss, label="val")
        plt.legend(); plt.grid(True); plt.xlabel("Step"); plt.ylabel("Loss")
        plt.tight_layout(); plt.savefig(out_dir / "loss_curve.png"); plt.close()
    except Exception as e: logging.warning(f"Plot failed: {e}")
    summary = {"train_examples": len(trainer.train_dataset), "val_examples": len(trainer.eval_dataset) if trainer.eval_dataset else 0, "trainable_params": trainer.model.num_parameters(only_trainable=True), "total_params": trainer.model.num_parameters()}
    if ev_loss: summary["final_val_loss"], summary["final_val_ppl"] = float(ev_loss[-1]), float(math.exp(ev_loss[-1]))
    with open(out_dir / "summary.txt", "w") as fh: [fh.write(f"{k}: {v}\n") for k, v in summary.items()]
    if wandb.run: wandb.save(str(out_dir / "summary.txt")); wandb.save(str(out_dir / "loss_curve.png"))

def main(argv=None):
    args = make_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "run.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    logging.info(f"Starting run {args.run_name} -> {args.output_dir}")
    logging.info(f"Args: {json.dumps(vars(args), indent=2)}")

    torch.manual_seed(args.seed); random.seed(args.seed); torch.backends.cuda.matmul.allow_tf32 = True

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    )
    model = prepare_model_for_kbit_training(model)
    lcfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lcfg)
    logging.info(f"LoRA on modules: {args.target_modules.split(',')}")
    
    model.gradient_checkpointing_enable(); model.config.use_cache = False

    examples = load_examples(args.data_paths, args.instruct_types, args.use_paraphrase_answer)
    raw_ds = Dataset.from_list([dataclasses.asdict(e) for e in examples])
    train_raw, val_raw, test_raw = three_way_split(raw_ds, val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed)
    
    def batch_tokenise(batch):
        processed_batch = defaultdict(list)
        for i in range(len(batch["prompt_count"])):
            ex = Example(
                prompt_count=batch["prompt_count"][i],
                instruction=batch["instruction"][i],
                inp=batch["inp"][i],
                answer=batch["answer"][i],
                style=batch["style"][i],
            )
            tokenized = tokenise_example(ex)
            processed_batch["input_ids"].append(tokenized["input_ids"])
            processed_batch["labels"].append(tokenized["labels"])
        return processed_batch
    
    logging.info("Starting data tokenization... This may take a few minutes with multiprocessing disabled.")
    train_ds = train_raw.map(batch_tokenise, batched=True, remove_columns=train_raw.column_names)
    val_ds = val_raw.map(batch_tokenise, batched=True, remove_columns=val_raw.column_names)
    logging.info("Data tokenization complete.")

    targs = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        optim=("adamw_bnb_8bit" if args.bnb_8bit_optim else "adamw_torch"),
        bf16=args.bf16,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        report_to=['wandb'],
        seed=args.seed
    )
    
    class StepDigest(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs: logging.info(f"step {state.global_step:4d} | train_loss {logs['loss']:.4f} | lr {logs.get('learning_rate', 0):.3g}")
            if logs and "eval_loss" in logs: logging.info(f"step {state.global_step:4d} | eval_loss  {logs['eval_loss']:.4f} | perplexity {math.exp(logs['eval_loss']):.2f}")

    lap_kwargs = {k: v for k, v in vars(args).items() if k.startswith('lap_')}
    lap_kwargs['use_lap'] = args.use_lap

    trainer_class = LAPTtrainer if args.use_lap else Trainer
    trainer = trainer_class(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8),
        callbacks=[StepDigest(), EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    
    trainer.train()
    trainer.save_model(str(output_dir / 'final'))
    tokenizer.save_pretrained(str(output_dir / 'final'))
    summarise_training(trainer, output_dir)
    if run: run.finish()

if __name__ == '__main__':
    main()
