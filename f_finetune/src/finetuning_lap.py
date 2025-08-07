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
UNEXPECTED_EVENTS = []

# Suppress excessive warnings
hf_logging.set_verbosity_warning()
logger = logging.getLogger(__name__)

# UTILITIES
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
    Correctly tokenizes an example, masking the prompt part of the labels.
    """
    prompt_txt = build_chat_prompt(ex.instruction, ex.inp)
    prompt_ids = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]

    answer_ids = tokenizer(ex.answer, add_special_tokens=False, truncation=True, max_length=1024)["input_ids"]
    answer_ids.append(tokenizer.eos_token_id)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids  # Mask prompt tokens
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
    examples: List[Example] = []
    for p in paths:
        logging.info(f"Loading {p}")
        with open(p, "r", encoding="utf-8") as fh: data = json.load(fh)
        for item in data:
            pc_id, base_ans, inp = item["prompt_count"], item.get("output", ""), item.get("input", "")
            examples.append(Example(pc_id, item["instruction_original"], inp, base_ans, "instruction_original"))
            keys = instruct_types or [k for k in item if k.startswith("instruct_")]
            for k in keys:
                if k in item and item[k]: examples.append(Example(pc_id, item[k], inp, base_ans, k))
    random.shuffle(examples)
    return examples

# LAP LOGIC INJECTION
class LAPTtrainer(Trainer):
    """Trainer subclass to implement Latent Adversarial Paraphrasing (LAP)"""
    def __init__(self, *args, **kwargs):
        self.lap_kwargs = kwargs.pop('lap_kwargs', {})
        super().__init__(*args, **kwargs)
        if self.lap_kwargs.get('use_lap', False):
            logger.info("LAP Trainer initialized: %s", json.dumps(self.lap_kwargs, indent=2))

    def _forward_with_perturbation(self, model: torch.nn.Module, inputs: Dict[str, Any], delta: torch.Tensor, layer_idx: int) -> Dict[str, torch.Tensor]:
        causal_lm_model = model.model if isinstance(model, PeftModel) else model
        base_transformer_model = causal_lm_model.model

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        if attention_mask is not None and hasattr(base_transformer_model, "_prepare_4d_causal_attention_mask"):
             attention_mask = base_transformer_model._prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), 0)

        hidden_states = base_transformer_model.embed_tokens(input_ids)

        for i in range(layer_idx):
            hidden_states, _, _ = base_transformer_model.layers[i](hidden_states, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
        hidden_states = hidden_states + delta
        for i in range(layer_idx, len(base_transformer_model.layers)):
            hidden_states, _, _ = base_transformer_model.layers[i](hidden_states, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
        
        hidden_states = base_transformer_model.norm(hidden_states)
        logits = causal_lm_model.lm_head(hidden_states)

        loss = None
        if 'labels' in inputs:
            loss_fct = torch.nn.CrossEntropyLoss()
            logits_for_loss = logits[..., :-1, :].contiguous()
            labels = inputs['labels'][..., 1:].contiguous()
            loss = loss_fct(logits_for_loss.view(-1, model.config.vocab_size), labels.view(-1))
        return {"loss": loss, "logits": logits}

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        use_lap = self.lap_kwargs.get('use_lap', False) and random.random() < self.lap_kwargs.get('lap_p_sample', 0.5)
        if not use_lap:
            return super().training_step(model, inputs)

        model.eval()
        t_inner = self.lap_kwargs['lap_t_inner']
        epsilon, delta_lr, lambda_lr = self.lap_kwargs['lap_epsilon'], self.lap_kwargs['lap_delta_lr'], self.lap_kwargs['lap_lambda_lr']
        layer_idx = self.lap_kwargs['lap_layer']
        inputs = self._prepare_inputs(inputs)

        try:
            with torch.no_grad():
                outputs_orig = model(**inputs, output_hidden_states=True)
                J0 = outputs_orig.loss.detach()
                hidden_states_l = outputs_orig.hidden_states[layer_idx].detach()

            delta = torch.zeros_like(hidden_states_l, requires_grad=True)
            delta_optimizer = torch.optim.SGD([delta], lr=delta_lr)
            log_lambda = torch.tensor(0.0, device=model.device, requires_grad=True)
            lambda_optimizer = torch.optim.SGD([log_lambda], lr=-lambda_lr)

            for i in range(t_inner):
                delta_optimizer.zero_grad()
                outputs_perturbed = self._forward_with_perturbation(model, inputs, delta, layer_idx)
                J_delta = outputs_perturbed['loss']
                norm_delta = torch.norm(delta, p=2)
                lagrangian_for_delta = -norm_delta + torch.exp(log_lambda).detach() * (J_delta - J0 - epsilon)
                (-lagrangian_for_delta).backward()
                delta_optimizer.step()

                lambda_optimizer.zero_grad()
                J_delta_updated = self._forward_with_perturbation(model, inputs, delta.detach(), layer_idx)['loss']
                lagrangian_for_lambda = torch.exp(log_lambda) * (J_delta_updated - J0 - epsilon)
                lagrangian_for_lambda.backward()
                lambda_optimizer.step()
            final_delta = delta.detach()
        except Exception as e:
            logger.error(f"Error in LAP inner loop: {e}", exc_info=True)
            return super().training_step(model, inputs)
            
        model.train()
        loss = self._forward_with_perturbation(model, inputs, final_delta, layer_idx)['loss']
        
        if self.do_grad_scaling: self.scaler.scale(loss).backward()
        else: self.accelerator.backward(loss)
        
        return loss.detach() / self.args.gradient_accumulation_steps

# CLI & Main Execution Logic from Original Script (B)
def make_arg_parser():
    p = argparse.ArgumentParser(description="Fine-tuning with optional Latent Adversarial Paraphrasing")
    p.add_argument("--data_paths", nargs="+", required=True)
    p.add_argument("--model_path", default="f_finetune/model")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default="gemma_paraphrx")
    p.add_argument("--instruct_types", nargs="+", default=[], help="Space-separated list of instruct_* keys to include; empty for all.")
    p.add_argument("--use_paraphrase_answer", action="store_true")
    p.add_argument("--val_pct", type=float, default=0.05)
    p.add_argument("--test_pct", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--use_deepspeed", action="store_true")
    p.add_argument("--deepspeed_config", default="ds_zero2.json")
    p.add_argument("--bnb_8bit_optim", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--eval_steps", type=int, default=800)
    p.add_argument("--save_steps", type=int, default=800)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="paraphrx_ft_lap")
    p.add_argument("--early_stopping_patience", type=int, default=3)
    # LAP Arguments Added
    p.add_argument("--use_lap", action="store_true", help="Enable Latent Adversarial Paraphrasing")
    p.add_argument("--lap_layer", type=int, default=10, help="Layer to inject perturbation")
    p.add_argument("--lap_t_inner", type=int, default=5, help="Number of inner loop iterations for LAP")
    p.add_argument("--lap_p_sample", type=float, default=0.5, help="Probability of applying LAP to a batch")
    p.add_argument("--lap_epsilon", type=float, default=0.05, help="Loss constraint margin for LAP")
    p.add_argument("--lap_delta_lr", type=float, default=1e-2, help="Learning rate for perturbation delta")
    p.add_argument("--lap_lambda_lr", type=float, default=1e-3, help="Learning rate for Lagrange multiplier")
    # Debugging Arguments from (B)
    p.add_argument("--debug_n_samples", type=int, default=0, help="If >0, print N samples and exit")
    p.add_argument("--debug_seed", type=int, default=123)
    return p

def summarise_training(trainer: Trainer, out_dir: Path):
    """Generates and saves training summary plots and text"""
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
    except Exception as e:
        logger.warning(f"Plot failed: {e}")
    summary = {
        "train_examples": len(trainer.train_dataset),
        "val_examples": len(trainer.eval_dataset) if trainer.eval_dataset else 0,
        "trainable_params": trainer.model.num_parameters(only_trainable=True),
        "total_params": trainer.model.num_parameters(),
    }
    if ev_loss:
        summary["final_val_loss"] = float(ev_loss[-1])
        summary["final_val_ppl"] = float(math.exp(ev_loss[-1]))
    with open(out_dir / "summary.txt", "w") as fh:
        for k, v in summary.items(): fh.write(f"{k}: {v}\n")
    if wandb.run:
        wandb.save(str(out_dir / "summary.txt"))
        wandb.save(str(out_dir / "loss_curve.png"))

def main(argv=None):
    args = make_arg_parser().parse_args(argv)
    run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info(f"Starting run {args.run_name}")
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")

    torch.manual_seed(args.seed); random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Model loading logic from (B)
    if not args.target_modules or args.target_modules.lower() in {"all", "none"}:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16 if args.bf16 else torch.float32)
    else:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", quantization_config=bnb_cfg)
        model = prepare_model_for_kbit_training(model)
        lcfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=args.target_modules.split(","), lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
        logger.info(f"LoRA on modules: {args.target_modules.split(',')}")
    
    model.gradient_checkpointing_enable(); model.config.use_cache = False

    # Data pipeline from (B)
    examples = load_examples(args.data_paths, args.instruct_types, args.use_paraphrase_answer)
    if args.debug_n_samples > 0:
        # Debug logic from (B)
        random.seed(args.debug_seed)
        for ex in random.sample(examples, min(len(examples), args.debug_n_samples)):
            toks = tokenise_example(ex)
            logging.info(f"DEBUG SAMPLE:\nPROMPT:\n{build_chat_prompt(ex.instruction, ex.inp)}\nANSWER:\n{ex.answer}\nLABELS:\n{toks['labels']}")
        return
        
    raw_ds = Dataset.from_list([dataclasses.asdict(e) for e in examples])
    train_raw, val_raw, test_raw = three_way_split(raw_ds, val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed)
    logger.info(f"Data sizes – train: {len(train_raw)} | val: {len(val_raw)} | test: {len(test_raw)}")

    # Correct batch tokenization from (B)
    def batch_tokenise(batch):
        iids, lbls, styles = [], [], []
        for pc, ins, inp, ans, sty in zip(batch["prompt_count"], batch["instruction"], batch["inp"], batch["answer"], batch["style"]):
            tok = tokenise_example(Example(pc, ins, inp, ans, sty))
            iids.append(tok["input_ids"]); lbls.append(tok["labels"]); styles.append(sty)
        return {"input_ids": iids, "labels": lbls, "style": styles}

    train_ds = train_raw.map(batch_tokenise, batched=True, remove_columns=train_raw.column_names)
    val_ds = val_raw.map(batch_tokenise, batched=True, remove_columns=val_raw.column_names)
    test_ds = test_raw.map(batch_tokenise, batched=True, remove_columns=test_raw.column_names)
    logger.info(f"Tokenised sizes – train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
    
    # Training Arguments from (B)
    targs = TrainingArguments(
        output_dir=args.output_dir, run_name=args.run_name, num_train_epochs=args.num_epochs,
        optim=("adamw_bnb_8bit" if args.bnb_8bit_optim else "adamw_torch"), bf16=args.bf16,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio, logging_steps=args.logging_steps, report_to=['wandb'], seed=args.seed
    )

    # StepDigest callback from (B)
    class StepDigest(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                logging.info(f"step {state.global_step:4d} | train_loss {logs['loss']:.4f} | lr {logs.get('learning_rate', 0):.3g}")
            if logs and "eval_loss" in logs:
                logging.info(f"step {state.global_step:4d} | eval_loss  {logs['eval_loss']:.4f} | perplexity {math.exp(logs['eval_loss']):.2f}")

    # Conditional Trainer Selection
    trainer_class = LAPTtrainer if args.use_lap else Trainer
    lap_kwargs = {k.replace('lap_', ''): v for k, v in vars(args).items() if k.startswith('lap_')}
    lap_kwargs['use_lap'] = args.use_lap

    trainer = trainer_class(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8),
        callbacks=[StepDigest(), EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        lap_kwargs=lap_kwargs
    )

    trainer.train()
    logger.info(f"Total training steps: {trainer.state.max_steps}")
    trainer.save_model(Path(args.output_dir)/'final')
    tokenizer.save_pretrained(Path(args.output_dir)/'final')
    logger.info(f"Training done: model & tokenizer saved to {args.output_dir}/final")

    # Per-style validation loss calculation from (B)
    val_loader = torch.utils.data.DataLoader(val_ds.remove_columns(["style"]), batch_size=args.batch_size*2, collate_fn=trainer.data_collator)
    styles = val_ds["style"]
    model.eval(); losses_by_style = defaultdict(list)
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            out = model(**{k: v.to(model.device) for k, v in batch.items()})
            bsz = batch["input_ids"].size(0)
            sl = styles[idx * bsz : (idx + 1) * bsz]
            for s in sl: losses_by_style[s].append(out.loss.item())
    for sty, lv in sorted({k: float(np.mean(v)) for k, v in losses_by_style.items()}.items(), key=lambda x:x[1], reverse=True):
        logger.info(f"VAL-loss {sty:<25} {lv:.4f}")
        if wandb.run: wandb.log({f"val_loss/{sty}": lv})
        
    summarise_training(trainer, Path(args.output_dir))

    if run: run.finish()

if __name__ == '__main__':
    main()
