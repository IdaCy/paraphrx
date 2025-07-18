#!/usr/bin/env python
"""
LoRA fine‑tune on the paraphrx paraphrase‑robustness corpora.

Highlights
----------
* Accept **local JSON** *or* **Weights & Biases Artifacts** via `--data_paths`.
* Early `wandb.init` so dataset artifacts can be resolved.
* Optional **debug mode** – `--debug_n_samples 5` prints tokenised samples and exits.
* Robust Bits‑and‑Bytes handling: silently falls back to fp16 if 4‑bit load fails.
* Writes `summary.txt`, `metrics_history.json`, `loss_curve.png` into `output_dir` and logs them to W&B.

Run example
~~~~~~~~~~~
```bash
srun python finetune_lora_paraphrx.py \
  --data_paths alpaca_gemma-2-2b-it gsm8k_gemma-2-2b-it mmlu_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir f_finetune/outputs_buckets_1-5 \
  --run_name buckets_1-5 \
  --buckets 1-5 \
  --bf16 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --save_steps 200 \
  --logging_steps 100
```  
"""

from __future__ import annotations
import argparse
import dataclasses
import datetime as _dt
import importlib
import json
import logging
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
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
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Environment defaults & constants
os.environ.setdefault("TQDM_MININTERVAL", "60")
os.environ.setdefault("TQDM_MINITER", "200")
DEBUG_PROMPT_IDS = {1, 42, 321}

# Example wrapper
@dataclasses.dataclass
class Example:
    instruction: str
    inp: str
    answer: str

    def to_prompt(self, with_answer: bool = False, add_eos: bool = True) -> str:
        if self.inp:
            prompt = (
                f"### Instruction:\n{self.instruction}\n\n"
                f"### Input:\n{self.inp}\n\n"
                "### Response:\n"
            )
        else:
            prompt = f"### Instruction:\n{self.instruction}\n\n### Response:\n"
        if with_answer:
            prompt += self.answer
        if add_eos:
            prompt += tokenizer.eos_token
        return prompt

# Tokenisation helper
def tokenise_example(ex: Example):
    prefix = ex.to_prompt(with_answer=False, add_eos=False)
    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
        truncation=True,
        max_length=2048
    )["input_ids"]

    answer_ids = tokenizer(
        ex.answer,
        add_special_tokens=False,
        truncation=True,
        max_length=1024
    )["input_ids"]
    if answer_ids and answer_ids[-1] == tokenizer.eos_token_id:
        answer_ids = answer_ids[:-1]
    answer_ids += [tokenizer.eos_token_id]

    bos = [tokenizer.bos_token_id]
    input_ids = bos + prefix_ids + answer_ids
    labels = [-100] + [-100] * len(prefix_ids) + answer_ids

    max_len = tokenizer.model_max_length or 4096
    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        labels = labels[-max_len:]
    return {"input_ids": input_ids, "labels": labels}

# Data loading utilities
def parse_bucket_spec(spec: str) -> List[int]:
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    result = set()
    for part in parts:
        if '-' in part:
            a, b = part.split('-')
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(part))
    allowed = sorted([i for i in result if 1 <= i <= 5])
    if not allowed:
        raise ValueError("Bucket spec must select at least one bucket between 1 and 5")
    return allowed

def load_examples(paths: List[str], buckets: List[int], use_para_ans: bool):
    examples: List[Example] = []
    counter: Counter = Counter()
    for p in paths:
        logging.info("Loading %s", p)
        with open(p, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        for item in data:
            pc_id = item.get("prompt_count")
            base_ans = item.get("output", "")
            inp = item.get("scenarios") or item.get("input", "")

            if 1 in buckets:
                examples.append(Example(item["instruction_original"], inp, base_ans))
                if pc_id in DEBUG_PROMPT_IDS:
                    logging.debug("[DBG %s-orig] %s",
                                  pc_id, item["instruction_original"][:80])

            for para in item.get("paraphrases", []):
                if int(para.get("bucket", 0)) not in buckets:
                    continue
                inst = para.get("paraphrase") or item["instruction_original"]
                ans = para.get("answer") if use_para_ans else base_ans
                if not (inst and ans):
                    continue
                examples.append(Example(inst, inp, ans))
                counter[int(para["bucket"])]+=1
                if pc_id in DEBUG_PROMPT_IDS:
                    logging.debug("[DBG %s-%s] %s",
                                  pc_id, para.get("instruct_type"), inst[:80])

    random.shuffle(examples)
    return examples, counter

# CLI args
def make_arg_parser():
    p = argparse.ArgumentParser(description="LoRA fine-tuning on paraphrx data")
    p.add_argument('--data_paths', nargs='+', required=True,
                   help='Local JSON paths or W&B artifact names')
    p.add_argument('--model_path', default='f_finetune/model')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--run_name', default='gemma_paraphrx')
    p.add_argument('--buckets', default='1-5')
    p.add_argument('--use_paraphrase_answer', action='store_true')

    # Training params
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--gradient_accumulation_steps', type=int, default=4)
    p.add_argument('--num_epochs', type=int, default=3)
    p.add_argument('--learning_rate', type=float, default=2e-4)
    p.add_argument('--warmup_ratio', type=float, default=0.03)
    p.add_argument('--lr_scheduler_type', default='cosine')

    # LoRA specifics (specific layers)
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--target_modules', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
                   help='Comma-separated list of modules to apply LoRA to')

    # Bits & Bytes / DeepSpeed options
    p.add_argument('--bnb_8bit_optim', action='store_true')
    p.add_argument('--use_deepspeed', action='store_true')
    p.add_argument('--deepspeed_config', default='ds_zero2.json')
    p.add_argument('--bf16', action='store_true')

    p.add_argument('--save_steps', type=int, default=200)
    p.add_argument('--logging_steps', type=int, default=100,
                   help='Log training loss every N steps')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--wandb_project', default='paraphrx_lora')

    p.add_argument('--debug_n_samples', type=int, default=0,
                   help='If >0, print N random tokenised samples and exit')
    p.add_argument('--debug_seed', type=int, default=123)
    return p

# Summary writer
def summarise_training(trainer: Trainer, bucket_counter: Counter, out_dir: Path):
    hist = trainer.state.log_history
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics_history.json').write_text(json.dumps(hist, indent=2))

    tr_steps, tr_loss, ev_steps, ev_loss = [], [], [], []
    for h in hist:
        if 'loss' in h:
            tr_steps.append(h['step']); tr_loss.append(h['loss'])
        if 'eval_loss' in h:
            ev_steps.append(h['step']); ev_loss.append(h['eval_loss'])

    try:
        plt.figure()
        if tr_loss: plt.plot(tr_steps, tr_loss, label='train')
        if ev_loss: plt.plot(ev_steps, ev_loss, label='eval')
        plt.legend(); plt.grid(True)
        plt.xlabel('Step'); plt.ylabel('Loss'); plt.tight_layout()
        plt.savefig(out_dir / 'loss_curve.png'); plt.close()
    except Exception as e:
        logging.warning("Plot failed: %s", e)

    summary = {
        'train_examples': trainer.num_examples,
        'val_examples':   trainer.eval_dataset.num_rows if trainer.eval_dataset else 0,
        'bucket_hist':    dict(bucket_counter),
        'trainable_params': trainer.model.num_parameters(only_trainable=True),
        'total_params':     trainer.model.num_parameters(),
    }
    if tr_loss:
        summary['final_train_loss'] = float(tr_loss[-1])
    if ev_loss:
        summary['final_eval_loss'] = float(ev_loss[-1])
        summary['best_eval_loss']  = float(min(ev_loss))
        summary['final_ppl']       = float(math.exp(ev_loss[-1]))
        summary['best_ppl']        = float(math.exp(min(ev_loss)))

    with open(out_dir / 'summary.txt', 'w') as fh:
        for k, v in summary.items(): fh.write(f"{k}: {v}\n")

    if wandb.run:
        wandb.save(str(out_dir / 'summary.txt'))
        wandb.save(str(out_dir / 'loss_curve.png'))
        wandb.save(str(out_dir / 'metrics_history.json'))

# Main execution
def main(argv=None):
    args = make_arg_parser().parse_args(argv)

    # Early W&B init
    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        job_type='finetune',
        config=vars(args)
    )

    # Logging setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{args.run_name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Starting run %s", args.run_name)

    # Seed & TF32
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Resolve dataset paths
    dataset_paths: List[str] = []
    for spec in args.data_paths:
        if os.path.exists(spec):
            dataset_paths.append(spec)
        else:
            logging.info("Downloading Artifact %s", spec)
            art = run.use_artifact(f"{spec}:latest", type='dataset')
            ddir = Path(art.download())
            files = list(ddir.glob('*.json'))
            if not files:
                raise FileNotFoundError(f"No JSON in artifact {spec}")
            dataset_paths.append(str(files[0]))
    logging.info("Datasets: %s", dataset_paths)

    # Tokenizer & model load with 4-bit fallback
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        quant_kwargs = {'quantization_config': bnb_cfg}
    except Exception as e:
        logging.warning("4-bit quant load failed, using default precision: %s", e)
        quant_kwargs = {}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map='auto',
        **quant_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # LoRA adapter (specific layers)
    mods = [m.strip() for m in args.target_modules.split(',') if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=mods,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_cfg)
    logging.info(
        "LoRA params: %s trainable / %s total",
        model.num_parameters(only_trainable=True),
        model.num_parameters()
    )

    # Load examples & optional debug
    buckets      = parse_bucket_spec(args.buckets)
    examples, bucket_counter = load_examples(dataset_paths, buckets, args.use_paraphrase_answer)

    if args.debug_n_samples > 0:
        random.seed(args.debug_seed)
        for ex in random.sample(examples, min(len(examples), args.debug_n_samples)):
            toks = tokenise_example(ex)
            logging.info(
                "DEBUG SAMPLE:\nPROMPT:\n%s\nANSWER:\n%s\nTOKENS:%s\nLABELS:%s",
                ex.to_prompt(False), ex.answer,
                toks['input_ids'][:40], toks['labels'][:40]
            )
        return

    # Tokenise dataset
    ds = Dataset.from_list([dataclasses.asdict(e) for e in examples])
    def batch_tokenise(batch):
        ins, inp, ans = batch['instruction'], batch['inp'], batch['answer']
        iids, lbls = [], []
        for i, p, a in zip(ins, inp, ans):
            tok = tokenise_example(Example(i, p, a))
            iids.append(tok['input_ids']); lbls.append(tok['labels'])
        return {'input_ids': iids, 'labels': lbls}

    tokenised = ds.map(batch_tokenise, batched=True, remove_columns=ds.column_names)
    split    = tokenised.train_test_split(test_size=0.05, seed=args.seed)
    train_ds, val_ds = split['train'], split['test']
    logging.info("DS sizes – train: %d | val: %d", len(train_ds), len(val_ds))

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # Trainer setup
    ds_cfg = args.deepspeed_config if args.use_deepspeed else None
    if args.use_deepspeed:
        try: importlib.import_module('deepspeed')
        except ImportError:
            logging.warning("DeepSpeed not available, skipping")
            ds_cfg = None

    targs = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        optim=('adamw_bnb_8bit' if args.bnb_8bit_optim else None),
        bf16=args.bf16,
        fp16=not (args.bf16 or args.bnb_8bit_optim),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=args.save_steps,
        group_by_length=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=0.3,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_total_limit=3,
        report_to=['wandb'],
        deepspeed=ds_cfg,
        seed=args.seed,
    )

    class StepDigest(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.global_step % args.logging_steps == 0:
                logging.info(
                    "step %d | loss %.4f | lr %.3g",
                    state.global_step,
                    logs.get('loss', float('nan')),
                    logs.get('learning_rate', float('nan'))
                )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[StepDigest()],
    )

    # Train
    trainer.train()
    trainer.save_model(Path(args.output_dir) / 'final')
    tokenizer.save_pretrained(Path(args.output_dir) / 'final')
    logging.info("Training complete: adapters & tokenizer saved to %s/final", args.output_dir)

    summarise_training(trainer, bucket_counter, Path(args.output_dir))

    if wandb.run is not None:
        wandb.finish()

if __name__ == '__main__':
    main()
