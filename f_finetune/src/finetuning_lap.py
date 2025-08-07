#!/usr/bin/env python
"""
fine-tuning script with Latent Adversarial Paraphrasing (LAP)

Original LoRA-NF4 4-bit adapters (recommended first try)

srun python finetuning_50k_lap.py \
  --data_paths ./path/to/your/data.json \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --bnb_8bit_optim --bf16 \
  --num_epochs 3 \
  --run_name gemma2b_lora_safe \
  --output_dir f_finetune/runs/gemma2b_lora_safe

To run with Latent Adversarial Paraphrasing (LAP):

srun python finetuning_50k_lap.py \
  --data_paths ./path/to/your/data.json \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --bnb_8bit_optim --bf16 \
  --num_epochs 3 \
  --run_name gemma2b_lap_safe \
  --output_dir f_finetune/runs/gemma2b_lap_safe \
  --use_lap \
  --lap_layer 10 \
  --lap_t_inner 5 \
  --lap_epsilon 0.05 \
  --lap_delta_lr 1e-2 \
  --lap_lambda_lr 1e-3
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

# Global Configuration
os.environ.setdefault("TQDM_MININTERVAL", "60")
os.environ.setdefault("TQDM_MINITER", "200")
DEBUG_PROMPT_IDS = {1, 42, 321}
UNEXPECTED_EVENTS = [] # Global list for tracking issues

# Suppress verbose warnings from transformers
hf_logging.set_verbosity_warning()

# Custom Logger Setup
logger = logging.getLogger(__name__)

class UnhandledExceptionsTracker(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.WARNING:
            UNEXPECTED_EVENTS.append(f"[{record.levelname}] {record.getMessage()}")

def setup_logging(run_name: str, output_dir: str):
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_name}_{ts}.log"

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logging.basicConfig(level=logging.INFO)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    tracker_handler = UnhandledExceptionsTracker()
    tracker_handler.setLevel(logging.WARNING)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.addHandler(tracker_handler)

@dataclasses.dataclass
class Example:
    prompt_count: int
    instruction: str
    inp: str
    answer: str
    style: str

    def to_prompt(self, with_answer: bool = False, add_eos: bool = True) -> str:
        if self.inp:
            prompt = (f"### Instruction:\n{self.instruction}\n\n### Input:\n{self.inp}\n\n### Response:\n")
        else:
            prompt = f"### Instruction:\n{self.instruction}\n\n### Response:\n"
        if with_answer: prompt += self.answer
        if add_eos: prompt += tokenizer.eos_token
        return prompt

def build_chat_prompt(instruction: str, inp: str | None = "") -> str:
    user_msg = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def tokenise_example(ex: Example):
    prompt_txt = build_chat_prompt(ex.instruction, ex.inp)
    prompt_ids = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(ex.answer, add_special_tokens=False, truncation=True, max_length=1024)["input_ids"]
    answer_ids.append(tokenizer.eos_token_id)
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "labels": labels}

def three_way_split(ds: Dataset, *, val_pct: float = 0.05, test_pct: float = 0.05, seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    pcs = list({ex["prompt_count"] for ex in ds})
    rng.shuffle(pcs)
    n = len(pcs)
    n_val, n_test = int(n * val_pct), int(n * test_pct)
    val_ids, test_ids = set(pcs[:n_val]), set(pcs[n_val : n_val + n_test])
    train_ids = set(pcs[n_val + n_test :])
    train = ds.filter(lambda ex: ex["prompt_count"] in train_ids, num_proc=4)
    val = ds.filter(lambda ex: ex["prompt_count"] in val_ids, num_proc=4)
    test = ds.filter(lambda ex: ex["prompt_count"] in test_ids, num_proc=4)
    return train, val, test

def load_examples(paths: List[str], instruct_types: List[str], use_para_ans: bool) -> List[Example]:
    examples: List[Example] = []
    for p in paths:
        logger.info("Loading %s", p)
        with open(p, "r", encoding="utf-8") as fh: data = json.load(fh)
        for item in data:
            pc_id = item["prompt_count"]
            base_ans = item.get("output", "")
            inp = item.get("input", "")
            examples.append(Example(pc_id, item["instruction_original"], inp, base_ans, "instruction_original"))
            keep_keys = instruct_types if instruct_types else [k for k in item.keys() if k.startswith("instruct_")]
            for k in keep_keys:
                if k not in item or not item[k]: continue
                ans = item.get("output_paraphrase", base_ans) if use_para_ans else base_ans
                examples.append(Example(pc_id, item[k], inp, ans, k))
                if pc_id in DEBUG_PROMPT_IDS: logger.debug("[DBG %s-%s] %s", pc_id, k, item[k][:80])
    random.shuffle(examples)
    return examples

class LAPTtrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.lap_kwargs = kwargs.pop('lap_kwargs', {})
        super().__init__(*args, **kwargs)
        self.lap_enabled = self.lap_kwargs.get('use_lap', False)
        if self.lap_enabled: logger.info("LAP Trainer initialized with parameters: %s", json.dumps(self.lap_kwargs, indent=2))

    def _forward_with_perturbation(self, model: torch.nn.Module, inputs: Dict[str, Any], delta: torch.Tensor, layer_idx: int) -> Dict[str, torch.Tensor]:
        # Get the top-level CausalLM model (e.g., GemmaForCausalLM), unwrapping PEFT if necessary
        causal_lm_model = model.model if isinstance(model, PeftModel) else model
        
        # Get the actual base transformer model (e.g., GemmaModel), which contains layers and embeddings
        base_transformer_model = causal_lm_model.model
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        
        if attention_mask is not None and hasattr(base_transformer_model, "_prepare_4d_causal_attention_mask"):
            attention_mask = base_transformer_model._prepare_4d_causal_attention_mask(
                attention_mask, (input_ids.shape[0], input_ids.shape[1]), 0, 0
            )

        # Use the correct object (`base_transformer_model`) for the forward pass components
        hidden_states = base_transformer_model.embed_tokens(input_ids)

        for i in range(layer_idx):
            hidden_states, _, _ = base_transformer_model.layers[i](hidden_states, attention_mask=attention_mask, position_ids=None, use_cache=False)
        
        hidden_states = hidden_states + delta

        for i in range(layer_idx, len(base_transformer_model.layers)):
            hidden_states, _, _ = base_transformer_model.layers[i](hidden_states, attention_mask=attention_mask, position_ids=None, use_cache=False)
        
        hidden_states = base_transformer_model.norm(hidden_states)
        
        # The lm_head is on the top-level causal_lm_model
        logits = causal_lm_model.lm_head(hidden_states)

        loss = None
        if 'labels' in inputs:
            loss_fct = torch.nn.CrossEntropyLoss()
            logits_for_loss = logits[..., :-1, :].contiguous()
            labels = inputs['labels'][..., 1:].contiguous()
            loss = loss_fct(logits_for_loss.view(-1, model.config.vocab_size), labels.view(-1))

        return {"loss": loss, "logits": logits}

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        use_lap_for_batch = self.lap_enabled and (random.random() < self.lap_kwargs['lap_p_sample'])
        if not use_lap_for_batch:
            return super().training_step(model, inputs)

        model.eval()
        t_inner, epsilon = self.lap_kwargs['lap_t_inner'], self.lap_kwargs['lap_epsilon']
        delta_lr, lambda_lr = self.lap_kwargs['lap_delta_lr'], self.lap_kwargs['lap_lambda_lr']
        layer_idx = self.lap_kwargs['lap_layer']
        
        inputs = self._prepare_inputs(inputs)

        try:
            with torch.no_grad():
                outputs_orig = model(**inputs, output_hidden_states=True)
                J0 = outputs_orig.loss.detach()
                hidden_states_l = outputs_orig.hidden_states[layer_idx].detach()

            delta = torch.zeros_like(hidden_states_l, requires_grad=True)
            delta_optimizer = torch.optim.SGD([delta], lr=delta_lr)
            log_lambda = torch.tensor(math.log(1e-4), device=model.device, requires_grad=False)
            
            for i in range(t_inner):
                delta_optimizer.zero_grad()
                outputs_perturbed = self._forward_with_perturbation(model, inputs, delta, layer_idx)
                J_delta = outputs_perturbed['loss']
                lambda_val = torch.exp(log_lambda)
                norm_delta = torch.norm(delta, p=2)
                loss_constraint = J_delta - J0 - epsilon
                lagrangian_for_delta = -norm_delta + lambda_val.detach() * loss_constraint
                (-lagrangian_for_delta).backward()
                delta_optimizer.step()

                with torch.no_grad():
                    J_delta_new = self._forward_with_perturbation(model, inputs, delta, layer_idx)['loss']
                    log_lambda += lambda_lr * (J_delta_new - J0 - epsilon)

                if self.state.global_step % self.args.logging_steps == 0 and i == t_inner - 1:
                    logger.info("LAP Inner (step %d, iter %d): J0=%.4f, J_delta=%.4f, ||d||=%.4f, lambda=%.4f",
                        self.state.global_step, i + 1, J0.item(), J_delta.item(), norm_delta.item(), torch.exp(log_lambda).item())

            final_delta = delta.detach()
        except Exception as e:
            logger.error(f"Error in LAP inner loop: {e}", exc_info=True)
            return super().training_step(model, inputs)
            
        model.train()
        outputs_final = self._forward_with_perturbation(model, inputs, final_delta, layer_idx)
        loss = outputs_final['loss']
        
        if self.do_grad_scaling: self.scaler.scale(loss).backward()
        else: self.accelerator.backward(loss)
        
        return loss.detach() / self.args.gradient_accumulation_steps

def make_arg_parser():
    p = argparse.ArgumentParser(description="Fine‑tuning with optional Latent Adversarial Paraphrasing")
    p.add_argument("--data_paths", nargs="+", required=True)
    p.add_argument("--model_path", default="f_finetune/model")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default="gemma_paraphrx")
    p.add_argument("--instruct_types", nargs="+", default=[], help="Space‑separated list of instruct_* keys to include; empty for all.")
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_steps", type=int, default=800)
    p.add_argument("--save_steps", type=int, default=800)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--wandb_project", default="paraphrx_ft_lap")
    p.add_argument("--early_stopping_patience", type=int, default=9)
    p.add_argument("--use_lap", action="store_true", help="Enable Latent Adversarial Paraphrasing")
    p.add_argument("--lap_layer", type=int, default=10, help="Layer to inject perturbation (for Gemma 2B, ~10 is mid-model)")
    p.add_argument("--lap_t_inner", type=int, default=5, help="Number of inner loop iterations for LAP")
    p.add_argument("--lap_p_sample", type=float, default=0.5, help="Probability of applying LAP to a batch")
    p.add_argument("--lap_epsilon", type=float, default=0.05, help="Loss constraint margin for LAP")
    p.add_argument("--lap_delta_lr", type=float, default=1e-2, help="Learning rate for perturbation delta")
    p.add_argument("--lap_lambda_lr", type=float, default=1e-3, help="Learning rate for Lagrange multiplier")
    p.add_argument("--debug_n_samples", type=int, default=0)
    p.add_argument("--debug_seed", type=int, default=123)
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
        plt.figure()
        if tr_loss: plt.plot(tr_steps, tr_loss, label="train")
        if ev_loss: plt.plot(ev_steps, ev_loss, label="val")
        plt.legend(); plt.grid(True); plt.xlabel("Step"); plt.ylabel("Loss")
        plt.tight_layout(); plt.savefig(out_dir / "loss_curve.png"); plt.close()
    except Exception as e: logger.warning("Plot failed: %s", e)

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
        wandb.save(str(out_dir / "metrics_history.json"))

def main(argv=None):
    args = make_arg_parser().parse_args(argv)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(args.run_name, args.output_dir)
    run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    logger.info("Starting run %s", args.run_name)
    logger.info("Command-line args:\n%s", json.dumps(vars(args), indent=2))
    torch.manual_seed(args.seed); random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset_paths = []
    for spec in args.data_paths:
        if os.path.exists(spec): dataset_paths.append(spec)
        else:
            logger.info("Downloading Artifact %s", spec)
            art = run.use_artifact(f"{args.wandb_project}/{spec}:latest", type="dataset")
            ddir, files = Path(art.download()), list(ddir.glob("*.json"))
            if not files: raise FileNotFoundError(f"No JSON in artifact {spec}")
            dataset_paths.append(str(files[0]))
    logger.info("Datasets: %s", dataset_paths)

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    full_ft = not args.target_modules or args.target_modules.lower() in {"all", "none"}

    if full_ft:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    else:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", quantization_config=bnb_cfg)
        model = prepare_model_for_kbit_training(model)
        mods = [m.strip() for m in args.target_modules.split(",")]
        lcfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=mods, lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
        logger.info("LoRA on modules: %s", mods)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    examples = load_examples(dataset_paths, instruct_types=args.instruct_types, use_para_ans=args.use_paraphrase_answer)

    if args.debug_n_samples > 0: return
    raw_ds = Dataset.from_list([dataclasses.asdict(e) for e in examples])
    train_raw, val_raw, test_raw = three_way_split(raw_ds, val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed)
    logger.info("raw sizes – train: %d | val: %d | test: %d", len(train_raw), len(val_raw), len(test_raw))

    def batch_tokenise(batch):
        iids, lbls, styles = [], [], []
        for pc, ins, inp, ans, sty in zip(batch["prompt_count"], batch["instruction"], batch["inp"], batch["answer"], batch["style"]):
            tok = tokenise_example(Example(pc, ins, inp, ans, sty))
            iids.append(tok["input_ids"]); lbls.append(tok["labels"]); styles.append(sty)
        return {"input_ids": iids, "labels": lbls, "style": styles}

    train_ds = train_raw.map(batch_tokenise, batched=True, remove_columns=train_raw.column_names, num_proc=4)
    val_ds = val_raw.map(batch_tokenise, batched=True, remove_columns=val_raw.column_names, num_proc=4)
    test_ds = test_raw.map(batch_tokenise, batched=True, remove_columns=test_raw.column_names, num_proc=4)
    logger.info("tokenised – train: %d | val: %d | test: %d", len(train_ds), len(val_ds), len(test_ds))
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
    ds_cfg = args.deepspeed_config if args.use_deepspeed else None

    if args.use_deepspeed:
        try: importlib.import_module("deepspeed")
        except ImportError: logger.warning("DeepSpeed not available, disabling"); ds_cfg = None

    targs = TrainingArguments(output_dir=args.output_dir, run_name=args.run_name, num_train_epochs=args.num_epochs, optim=("adamw_bnb_8bit" if args.bnb_8bit_optim else "adamw_torch"), bf16=args.bf16, fp16=not (args.bf16 or args.bnb_8bit_optim), per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2, gradient_accumulation_steps=args.gradient_accumulation_steps, eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.save_steps, save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, group_by_length=True, learning_rate=args.learning_rate, weight_decay=args.weight_decay, lr_scheduler_type=args.lr_scheduler_type, warmup_ratio=args.warmup_ratio, max_grad_norm=0.3, logging_steps=args.logging_steps, logging_first_step=True, report_to=['wandb'], deepspeed=ds_cfg, seed=args.seed)
    class StepDigest(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs: return
            logger.info("step %4d | train_loss %.4f | lr %.3g", state.global_step, logs["loss"], logs.get("learning_rate", float("nan")))
            if "eval_loss" in logs: logger.info("step %4d | eval_loss  %.4f | perplexity %.2f", state.global_step, logs["eval_loss"], math.exp(logs["eval_loss"]))

    trainer_class = LAPTtrainer if args.use_lap else Trainer
    lap_kwargs = {"use_lap": args.use_lap, "lap_layer": args.lap_layer, "lap_t_inner": args.lap_t_inner, "lap_p_sample": args.lap_p_sample, "lap_epsilon": args.lap_epsilon, "lap_delta_lr": args.lap_delta_lr, "lap_lambda_lr": args.lap_lambda_lr}
    trainer = trainer_class(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator, callbacks=[StepDigest(), EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)], lap_kwargs=lap_kwargs)
    logger.info("Step-0 evaluation (no fine-tuning)")
    init_metrics = trainer.evaluate()
    logger.info("step 0 | eval_loss %.4f | ppl %.2f", init_metrics["eval_loss"], math.exp(init_metrics["eval_loss"]))

    if wandb.run: wandb.log({"eval_loss/step0": init_metrics["eval_loss"]})
    trainer.train()
    logger.info("Total training steps: %d", trainer.state.max_steps)
    trainer.save_model(Path(args.output_dir) / 'final')
    tokenizer.save_pretrained(Path(args.output_dir) / 'final')
    logger.info("Training done: model & tokenizer in %s/final", args.output_dir)
    val_loader = torch.utils.data.DataLoader(val_ds.remove_columns(["style"]), batch_size=args.batch_size * 2, shuffle=False, collate_fn=collator)
    styles = val_ds["style"]
    model.eval(); losses_by_style = defaultdict(list)

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            bsz = batch["input_ids"].size(0)
            sl = styles[idx * bsz : (idx + 1) * bsz]
            loss_item = out.loss.item()
            for s in sl: losses_by_style[s].append(loss_item)

    style_avg = {k: float(np.mean(v)) for k, v in losses_by_style.items()}
    for sty, lv in sorted(style_avg.items(), key=lambda x: x[1], reverse=True): logger.info("VAL-loss %-25s %.4f", sty, lv)
    if wandb.run: wandb.log({f"val_loss/{sty}": lv for sty, lv in style_avg.items()})
    summarise_training(trainer, Path(args.output_dir))
    test_metrics = trainer.evaluate(test_ds)
    logger.info("TEST loss %.4f | ppl %.2f", test_metrics["eval_loss"], math.exp(test_metrics["eval_loss"]))
    logger.info("--- Run Finished ---")

    if UNEXPECTED_EVENTS:
        logger.warning("Encountered %d unexpected events during the run:", len(UNEXPECTED_EVENTS))
        for event in UNEXPECTED_EVENTS: logger.warning(" - %s", event)
    else: logger.info("No unexpected events were logged.")
    
    if wandb.run is not None: wandb.finish()

if __name__ == '__main__':
    main()
