#!/usr/bin/env python
"""
python finetune_paraphrx.py \
  --data_paths f_finetune/data/output_splits/buckets_1-3_train.json \
  --output_dir f_finetune/outputs/alpaca/ft_inf_results/bucket3.json \
  --run_name gemma_bkt1_3 \
  --model_path f_finetune/model
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["TQDM_MININTERVAL"] = "60"    # seconds
os.environ["TQDM_MINITER"]     = "200"

from transformers import TrainerCallback
import datetime as dt

# Helper dataclass for one formatted example

@dataclasses.dataclass
class Example:
    instruction: str
    inp: str
    answer: str

    def to_prompt(self, add_eos: bool = True) -> str:
        """Construct Alpaca-style prompt"""
        if self.inp:
            prompt = (
                f"### Instruction:\n{self.instruction}\n\n"
                f"### Input:\n{self.inp}\n\n"
                "### Response:\n"
            )
        else:
            prompt = (
                f"### Instruction:\n{self.instruction}\n\n"
                "### Response:\n"
            )
        if add_eos:
            prompt += self.answer + tokenizer.eos_token  # noqa: F821 - tokenizer is injected later
        else:
            prompt += self.answer
        return prompt


# Data loading utilities

def parse_bucket_spec(spec: str) -> List[int]:
    """Convert a bucket spec like "1-3,5" to sorted unique list [1,2,3,5]"""
    result = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-')
            result.update(range(int(start), int(end) + 1))
        else:
            result.add(int(part))
    allowed = [i for i in sorted(result) if 1 <= i <= 5]
    if not allowed:
        raise ValueError("Bucket specification must select at least one bucket between 1 and 5")
    return allowed


def load_examples(paths: List[str], buckets: List[int], use_paraphrase_answer: bool) -> List[Example]:
    """Load and filter JSON files, returning a flat list of Examples"""
    examples: List[Example] = []
    for p in paths:
        logging.info("Loading %s", p)
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            base_output = item.get('output', '')
            inp = item.get('input', '')
            # Include original instruction if bucket passes; it's stored inside paraphrases too, but guard anyway.
            for para in item.get('paraphrases', []):
                if int(para.get('bucket', 0)) in buckets:
                    ins = para.get('paraphrase') or item.get('instruction_original', '')
                    ans = para.get('answer') if use_paraphrase_answer else base_output
                    if not (ins and ans):
                        continue  # Skip malformed rows
                    examples.append(Example(ins, inp, ans))
    random.shuffle(examples)
    logging.info("Loaded %d filtered examples (buckets=%s).", len(examples), buckets)
    return examples


# Argument parsing

def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for paraphrase robustness on GEMMA-2-2B-IT")
    p.add_argument('--data_paths', type=str, nargs='+', required=True, help='Paths to JSON dataset files')
    p.add_argument('--model_path', type=str, default='f_finetune/model', help='Directory with base GEMMA model')
    p.add_argument('--output_dir', type=str, required=True, help='Where to save LoRA adapters & checkpoints')
    p.add_argument('--run_name', type=str, default='gemma_paraphrx', help='Name for WandB & logs')
    p.add_argument('--buckets', type=str, default='1', help='Bucket spec, e.g. "1", "1-3", "1,2,3"')
    p.add_argument('--use_paraphrase_answer', action='store_true', help='Train on paraphrase answer instead of original output')

    # Training hyper-params
    p.add_argument('--batch_size', type=int, default=4, help='Per-device micro batch size')
    p.add_argument('--gradient_accumulation_steps', type=int, default=4)
    p.add_argument('--num_epochs', type=int, default=3)
    p.add_argument('--learning_rate', type=float, default=2e-4)
    p.add_argument('--warmup_ratio', type=float, default=0.05)
    p.add_argument('--lr_scheduler_type', type=str, default='cosine')

    # LoRA specific
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--target_modules', type=str,
                   default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
                   help='Comma-separated list of modules to LoRA-ise')

    # Misc.
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bf16', action='store_true', help='Use bfloat16 instead of fp16 where supported')
    p.add_argument('--wandb_project', type=str, default='paraphrx_lora', help='Weights & Biases project')
    p.add_argument('--save_steps', type=int, default=200)

    return p


# Main execution

def main(argv: List[str] | None = None) -> None:
    args = make_arg_parser().parse_args(argv)

    # Logging setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{args.run_name}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Starting run %s", args.run_name)

    # Seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Tokeniser & model
    global tokenizer  # Required inside Example.to_prompt

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # GEMMA has no explicit PAD

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map='auto',
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[m.strip() for m in args.target_modules.split(',') if m.strip()],
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )

    model = get_peft_model(model, lora_cfg)
    logging.info("LoRA params: %s trainable / %s total", model.num_parameters(only_trainable=True), model.num_parameters())

    # Dataset preparation
    buckets = parse_bucket_spec(args.buckets)
    examples = load_examples(args.data_paths, buckets, args.use_paraphrase_answer)

    def to_tokenised_dict(ex: Example):
        # build prefix (everything up to “### Response:\n”)
        prefix  = ex.to_prompt(add_eos=False)          # no answer, no <EOS>
        answer  = ex.answer + tokenizer.eos_token

        full_txt = prefix + answer
        enc_full = tokenizer(full_txt,
                             truncation=True,
                             max_length=1024,
                             padding=False)

        prefix_len = len(tokenizer(prefix,
                                   add_special_tokens=False)["input_ids"])

        labels = [-100] * prefix_len + enc_full["input_ids"][prefix_len:]
        enc_full["labels"] = labels                     # ← important
        return enc_full

    tokenised_ds = Dataset.from_list([dataclasses.asdict(e) for e in examples])
    tokenised_ds = tokenised_ds.map(lambda record: to_tokenised_dict(Example(record['instruction'], record['inp'], record['answer'])),
                                    remove_columns=list(tokenised_ds.column_names))

    logging.info("Tokenisation finished - %d rows", len(tokenised_ds))

    train_ds = tokenised_ds
    logging.info("Dataset size - train: %d (no validation split)", len(train_ds))

    #data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    #data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    # TrainingArguments & Trainer
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps      = 500,
        logging_first_step = True,
        eval_strategy = "no",
        save_strategy       = "epoch",
        #eval_steps=100,
        #save_steps=args.save_steps,
        save_total_limit=3,
        report_to=['wandb'],
        bf16=args.bf16,
        fp16=not args.bf16,
        seed=args.seed,
    )

    os.environ['WANDB_PROJECT'] = args.wandb_project

    class StepDigest(TrainerCallback):
        """Log one terse line every N steps (matches logging_steps)"""
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.global_step % args.logging_steps == 0:
                logging.info("step %d | loss %.4f | lr %.3g | %s",
                            state.global_step,
                            logs.get("loss", float("nan")),
                            logs.get("learning_rate", float("nan")),
                            dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        callbacks=[StepDigest()],
    )


    # Train
    trainer.train()
    trainer.save_model(Path(args.output_dir) / 'final')
    tokenizer.save_pretrained(Path(args.output_dir) / 'final')
    logging.info("Training completed - adapters & tokenizer saved to %s", Path(args.output_dir) / 'final')


if __name__ == '__main__':
    main()
