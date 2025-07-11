#!/usr/bin/env python
"""
srun python "$RUN_SCRIPT" \
  --data_paths "$INPUT_JSON" \
  --output_dir f_finetune/outputs_buckets_1-5 \
  --run_name buckets_1-5 \
  --buckets 1-5 \
  --bf16 \
  $WANDB_FLAG
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
from collections import Counter

import torch
from datasets import Dataset
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
DEBUG_PROMPT_IDS = {1, 42, 321}

from transformers import TrainerCallback
import datetime as dt

# Helper dataclass for one formatted example

@dataclasses.dataclass
class Example:
    instruction: str
    inp: str
    answer: str

    def to_prompt(self, with_answer: bool = False, add_eos: bool = True) -> str:
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
        if with_answer:
            prompt += self.answer
        if add_eos:
            prompt += tokenizer.eos_token

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


def load_examples(paths: List[str],
                  buckets: List[int],
                  use_paraphrase_answer: bool) -> tuple[list[Example], Counter]:
    """Load and filter JSON files, returning a flat list of Examples"""
    examples: List[Example] = []
    bucket_counter: Counter = Counter()
    for p in paths:
        logging.info("Loading %s", p)
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            pc_id = item.get("prompt_count")
            base_output = item.get('output', '')
            # non-null scenarios or input or nothing
            raw_scen = item.get("scenarios")
            if raw_scen is not None:
                inp = raw_scen
            else:
                inp = item.get("input", "")

            # include the *original* instruction---
            if 1 in buckets:            # original always belongs to bucket 1
                ex = Example(item["instruction_original"], inp, base_output)
                examples.append(ex)
                if pc_id in DEBUG_PROMPT_IDS:
                    logging.info("[DEBUG %d-orig] prompt=%r | answer=%r",
                                 pc_id, ex.instruction[:140], ex.answer[:140])

            # Include paraphrases
            for para in item.get('paraphrases', []):
                if int(para.get('bucket', 0)) in buckets:
                    ins = para.get('paraphrase') or item.get('instruction_original', '')
                    ans = para.get('answer') if use_paraphrase_answer else base_output
                    if not (ins and ans):
                        continue  # Skip malformed rows
                    examples.append(Example(ins, inp, ans))
                    bucket_counter[int(para["bucket"])] += 1
                    if pc_id in DEBUG_PROMPT_IDS:
                        logging.info("[DEBUG %d-%s] prompt=%r | answer=%r",
                                     pc_id, para.get("instruct_type"),
                                     ins[:140], ans[:140])
    random.shuffle(examples)
    logging.info("Loaded %d filtered examples (buckets=%s).", len(examples), buckets)
    return examples, bucket_counter


# Argument parsing


def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for paraphrase robustness on GEMMA-2-2B-IT")
    p.add_argument('--data_paths', type=str, nargs='+', required=True, help='Paths to JSON dataset files')
    p.add_argument('--model_path', type=str, default='f_finetune/model', help='Directory with base GEMMA model')
    p.add_argument('--output_dir', type=str, required=True, help='Where to save LoRA adapters & checkpoints')
    p.add_argument('--run_name', type=str, default='gemma_paraphrx', help='Name for WandB & logs')
    p.add_argument('--buckets', type=str, default='1-5', help='Bucket spec, e.g. "1", "1-3", "1,2,3"')
    p.add_argument('--use_paraphrase_answer', action='store_true', help='Train on paraphrase answer instead of original output')

    # Training hyper-params
    p.add_argument('--batch_size', type=int, default=4, help='Per-device micro batch size')
    p.add_argument('--gradient_accumulation_steps', type=int, default=4)
    p.add_argument('--num_epochs', type=int, default=3)
    p.add_argument('--learning_rate', type=float, default=2e-4)
    p.add_argument('--warmup_ratio', type=float, default=0.03)
    p.add_argument('--lr_scheduler_type', type=str, default='cosine')

    # LoRA specific
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.05)

    # FOR NOT ALL LAYERS
    #p.add_argument('--target_modules', type=str,
    #               default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
    #               help='Comma-separated list of modules to LoRA-ise')
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
    torch.backends.cuda.matmul.allow_tf32 = True

    # Tokeniser & model
    global tokenizer  # Required inside Example.to_prompt

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # GEMMA has no explicit PAD
    tokenizer.pad_token_id = tokenizer.eos_token_id

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
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # FOR NOT ALL LAYERS
    #lora_cfg = LoraConfig(
    #    r=args.lora_rank,
    #    lora_alpha=args.lora_alpha,
    #    target_modules=[m.strip() for m in args.target_modules.split(',') if m.strip()],
    #    lora_dropout=args.lora_dropout,
    #    bias='none',
    #    task_type='CAUSAL_LM',
    #)

    # FOR ALL LAYERS - test
    target_mods = [m.strip() for m in args.target_modules.split(',') if m.strip()]
    if not target_mods:                       # PEFT needs at least one entry
        target_mods = [
            'q_proj', 'k_proj', 'v_proj',     #   fallback to the usual LLaMA / Gemma
            'o_proj', 'gate_proj',            #   projection names
            'up_proj', 'down_proj',
        ]
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_mods,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )
    # FOR ALL LAYERS - test end

    model = get_peft_model(model, lora_cfg)
    logging.info("LoRA params: %s trainable / %s total", model.num_parameters(only_trainable=True), model.num_parameters())

    # Dataset preparation
    buckets = parse_bucket_spec(args.buckets)
    examples, bucket_counter = load_examples(
        args.data_paths, buckets, args.use_paraphrase_answer
    )
    logging.info("Bucket histogram (loaded set): %s", dict(bucket_counter))
    
    # bucket histogram
    bucket_hist = bucket_counter
    logging.info("Bucket histogram (in loaded set): %s", dict(bucket_hist))

    def to_tokenised_dict(ex: Example):
        """Tokenise one example, add BOS, apply global length cap, build labels mask."""
        # PROMPT (no EOS)
        prefix_ids = tokenizer(
            ex.to_prompt(with_answer=False, add_eos=False),
            add_special_tokens=False, truncation=True, max_length=2048
        )["input_ids"]

        # ANSWER (ensure single EOS)
        answer_ids = tokenizer(
            ex.answer, add_special_tokens=False, truncation=True, max_length=1024
        )["input_ids"]
        if answer_ids and answer_ids[-1] == tokenizer.eos_token_id:
            answer_ids = answer_ids[:-1]
        answer_ids += [tokenizer.eos_token_id]

        bos_id = [tokenizer.bos_token_id]
        input_ids = bos_id + prefix_ids + answer_ids
        labels    = [-100]  + [-100]*len(prefix_ids) + answer_ids

        # global length guard
        max_len = tokenizer.model_max_length or 4096
        if len(input_ids) > max_len:
            # keep the *tail* so EOS is intact
            input_ids = input_ids[-max_len:]
            labels    = labels   [-max_len:]

        return {"input_ids": input_ids, "labels": labels}

    tokenised_ds = Dataset.from_list([dataclasses.asdict(e) for e in examples])

    def batch_tokenise(batch):
        input_ids, labels = [], []
        for inst, inp, ans in zip(batch["instruction"], batch["inp"], batch["answer"]):
            tok = to_tokenised_dict(Example(inst, inp, ans))
            input_ids.append(tok["input_ids"])
            labels.append(tok["labels"])
        return {"input_ids": input_ids, "labels": labels}

    tokenised_ds = tokenised_ds.map(
        batch_tokenise,
        batched=True,
        remove_columns=tokenised_ds.column_names,
    )

    logging.info("Tokenisation finished - %d rows", len(tokenised_ds))

    split = tokenised_ds.train_test_split(test_size=0.05, seed=args.seed)
    train_ds = split["train"]
    val_ds   = split["test"]
    logging.info(
        "Dataset size – train: %d | val: %d",
        len(train_ds), len(val_ds)
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

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
        max_grad_norm=0.3,
        logging_steps      = 100,
        logging_first_step = True,
        eval_strategy = "epoch",
        save_strategy       = "epoch",
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
        eval_dataset=val_ds if "val_ds" in locals() else None,
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
