#!/usr/bin/env python
"""
running in SLURM:

source /scratch_root/ifc24/.hf_token || true
[ -f /scratch_root/ifc24/.wandb_key ] && source /scratch_root/ifc24/.wandb_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_50k}

export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

RUN_SCRIPT="f_finetune/src/finetuning_50k.py"
srun python "$RUN_SCRIPT" \
  --data_paths paraphrx1.json paraphrx2.json \
  --model_path f_finetune/model \
  --output_dir f_finetune/runs/gemma2b_round1 \
  --run_name gemma2b_round1 \
  --instruct_types instruct_sardonic instruct_polite_request \
  --batch_size 4 --gradient_accumulation_steps 8 \
  --num_epochs 3 --bf16 --bnb_8bit_optim
  $WANDB_FLAG

Full-parameter fine-tune (target-modules "none", 10 paraphrase types)
srun python finetuning_50k.py \
  --target_modules none \
  --batch_size 2 --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --run_name gemma2b_fullft_10x \
  --output_dir f_finetune/runs/gemma2b_fullft_10x

LoRA-4-bit adapters (default target-modules list, 10 paraphrase types)
srun python finetuning_50k.py \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --batch_size 8 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lora_rank 16 --lora_alpha 32 \
  --bnb_8bit_optim --bf16 \
  --run_name gemma2b_lora_10x \
  --output_dir f_finetune/runs/gemma2b_lora_10x

Full-parameter fine-tune (target-modules "none", 12 paraphrase types)
srun python finetuning_50k.py \
  --target_modules none \
  --batch_size 2 --gradient_accumulation_steps 16 \
  --num_epochs 2 --learning_rate 3e-5 \
  --run_name gemma2b_fullft_12x \
  --output_dir f_finetune/runs/gemma2b_fullft_12x
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
from typing import List, Tuple

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
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Environment defaults & constants
os.environ.setdefault("TQDM_MININTERVAL", "60")
os.environ.setdefault("TQDM_MINITER", "200")
DEBUG_PROMPT_IDS = {1, 42, 321}  # helpful sanity‑print ids

@dataclasses.dataclass
class Example:
    """Lightweight container used before tokenisation."""

    prompt_count: int
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


# Helper: tokenise a single Example → dict(input_ids, labels)

def tokenise_example(ex: Example):
    prefix = ex.to_prompt(with_answer=False, add_eos=False)
    prefix_ids = tokenizer(
        prefix, add_special_tokens=False, truncation=True, max_length=2048
    )["input_ids"]

    answer_ids = tokenizer(
        ex.answer, add_special_tokens=False, truncation=True, max_length=1024
    )["input_ids"]
    # ensure exactly one EOS at the end
    if answer_ids and answer_ids[-1] == tokenizer.eos_token_id:
        answer_ids = answer_ids[:-1]
    answer_ids.append(tokenizer.eos_token_id)

    bos = [tokenizer.bos_token_id]
    input_ids = bos + prefix_ids + answer_ids
    labels = [-100] + [-100] * len(prefix_ids) + answer_ids

    max_len = tokenizer.model_max_length or 4096
    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        labels = labels[-max_len:]
    return {"input_ids": input_ids, "labels": labels}


# Data‑loading utilities

def three_way_split(
    ds: Dataset, *, val_pct: float = 0.05, test_pct: float = 0.05, seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """Group‑wise split guaranteeing each *prompt_count* appears in one split."""

    import numpy as np

    rng = np.random.default_rng(seed)
    pcs = list({ex["prompt_count"] for ex in ds})
    rng.shuffle(pcs)
    n = len(pcs)
    n_val = int(n * val_pct)
    n_test = int(n * test_pct)
    val_ids = set(pcs[:n_val])
    test_ids = set(pcs[n_val : n_val + n_test])
    train_ids = set(pcs[n_val + n_test :])

    train = ds.filter(lambda ex: ex["prompt_count"] in train_ids)
    val = ds.filter(lambda ex: ex["prompt_count"] in val_ids)
    test = ds.filter(lambda ex: ex["prompt_count"] in test_ids)
    return train, val, test


def load_examples(
    paths: List[str], instruct_types: List[str], use_para_ans: bool
) -> List[Example]:
    examples: List[Example] = []

    for p in paths:
        logging.info("Loading %s", p)
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for item in data:
            pc_id = item["prompt_count"]
            base_ans = item.get("output", "")
            inp = item.get("input", "")

            # always include the original instruction
            examples.append(Example(pc_id, item["instruction_original"], inp, base_ans))

            # Decide which paraphrase keys to keep
            if instruct_types:
                keep_keys = instruct_types
            else:  # default: keep all instruct_* keys found
                keep_keys = [k for k in item.keys() if k.startswith("instruct_")]

            for k in keep_keys:
                if k not in item:
                    continue
                paraphrase = item[k]
                if not paraphrase:
                    continue
                ans = item.get("output_paraphrase", base_ans) if use_para_ans else base_ans
                examples.append(Example(pc_id, paraphrase, inp, ans))

                if pc_id in DEBUG_PROMPT_IDS:
                    logging.debug("[DBG %s-%s] %s", pc_id, k, paraphrase[:80])

    random.shuffle(examples)
    return examples


# CLI arguments

def make_arg_parser():
    p = argparse.ArgumentParser(description="LoRA fine‑tuning on paraphrx data")
    p.add_argument("--data_paths", nargs="+", required=True)
    p.add_argument("--model_path", default="f_finetune/model")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default="gemma_paraphrx")

    p.add_argument(
        "--instruct_types",
        nargs="+",
        default=[],
        help="Space‑separated list of instruct_* keys to include; leave empty to use all paraphrases.",
    )
    p.add_argument("--use_paraphrase_answer", action="store_true")

    p.add_argument("--val_pct", type=float, default=0.05)
    p.add_argument("--test_pct", type=float, default=0.05)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--weight_decay", type=float, default=0.0)  # LoRA params only

    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    p.add_argument("--use_deepspeed", action="store_true")
    p.add_argument("--deepspeed_config", default="ds_zero2.json")
    p.add_argument("--bnb_8bit_optim", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--eval_steps", type=int, default=800)
    p.add_argument("--save_steps", type=int, default=800)
    p.add_argument(
        "--logging_steps", type=int, default=100, help="Log training loss every N steps"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="paraphrx_ft_50k")

    # quick inspection helper
    p.add_argument(
        "--debug_n_samples",
        type=int,
        default=0,
        help="If >0, print N random tokenised samples and exit",
    )
    p.add_argument("--debug_seed", type=int, default=123)
    return p


# Training summary / plots

def summarise_training(trainer: Trainer, out_dir: Path):
    hist = trainer.state.log_history
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_history.json").write_text(json.dumps(hist, indent=2))

    # loss curves
    tr_steps, tr_loss, ev_steps, ev_loss = [], [], [], []
    for h in hist:
        if "loss" in h:
            tr_steps.append(h["step"])
            tr_loss.append(h["loss"])
        if "eval_loss" in h:
            ev_steps.append(h["step"])
            ev_loss.append(h["eval_loss"])
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        if tr_loss:
            plt.plot(tr_steps, tr_loss, label="train")
        if ev_loss:
            plt.plot(ev_steps, ev_loss, label="val")
        plt.legend(); plt.grid(True)
        plt.xlabel("Step"); plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png"); plt.close()
    except Exception as e:
        logging.warning("Plot failed: %s", e)

    # simple text summary
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
        for k, v in summary.items():
            fh.write(f"{k}: {v}\n")

    if wandb.run:  # log back
        wandb.save(str(out_dir / "summary.txt"))
        wandb.save(str(out_dir / "loss_curve.png"))
        wandb.save(str(out_dir / "metrics_history.json"))


# Main

def main(argv=None):
    args = make_arg_parser().parse_args(argv)

    # Early W&B init
    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        job_type="finetune",
        config=vars(args)
    )

    # Logging setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{args.run_name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting run %s", args.run_name)

    # Seed & TF32
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Resolve dataset files / Artifacts
    dataset_paths = []
    for spec in args.data_paths:
        if os.path.exists(spec):
            dataset_paths.append(spec)
        else:
            logging.info("Downloading Artifact %s", spec)
            art = run.use_artifact(f"{args.wandb_project}/{spec}:latest", type="dataset")
            ddir = Path(art.download())
            files = list(ddir.glob("*.json"))
            if not files:
                raise FileNotFoundError(f"No JSON in artifact {spec}")
            dataset_paths.append(str(files[0]))
    logging.info("Datasets: %s", dataset_paths)

    # Tokeniser & model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    full_ft = not args.target_modules or args.target_modules.lower() in {"all", "none"}

    if full_ft:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    else:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map="auto", quantization_config=bnb_cfg
        )
        model = prepare_model_for_kbit_training(model)

    if not full_ft:
        mods = [m.strip() for m in args.target_modules.split(",")]
        lcfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=mods,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
        logging.info("LoRA on modules: %s", mods)
    else:
        logging.info("Full‑parameter fine‑tune (no LoRA)")

    model.config.pad_token_id = tokenizer.pad_token_id
    # Dropout tweaks (kept modest)
    model.config.dropout = 0.05
    model.config.hidden_dropout = 0.05
    model.config.attention_dropout = 0.05
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # needed for checkpointing

    # Load examples
    examples = load_examples(
        dataset_paths, instruct_types=args.instruct_types, use_para_ans=args.use_paraphrase_answer
    )

    if args.debug_n_samples > 0:
        random.seed(args.debug_seed)
        for ex in random.sample(examples, min(len(examples), args.debug_n_samples)):
            toks = tokenise_example(ex)
            logging.info(
                "DEBUG SAMPLE:\nPROMPT:\n%s\nANSWER:\n%s\nTOKENS:%s\nLABELS:%s",
                ex.to_prompt(False),
                ex.answer,
                toks["input_ids"][:40],
                toks["labels"][:40],
            )
        return

    raw_ds = Dataset.from_list([dataclasses.asdict(e) for e in examples])
    train_raw, val_raw, test_raw = three_way_split(
        raw_ds, val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed
    )
    logging.info(
        "raw sizes – train: %d | val: %d | test: %d",
        len(train_raw),
        len(val_raw),
        len(test_raw),
    )

    def batch_tokenise(batch):
        iids, lbls = [], []
        for pc, ins, inp, ans in zip(
            batch["prompt_count"], batch["instruction"], batch["inp"], batch["answer"]
        ):
            tok = tokenise_example(Example(pc, ins, inp, ans))
            iids.append(tok["input_ids"])
            lbls.append(tok["labels"])
        return {"input_ids": iids, "labels": lbls}

    train_ds = train_raw.map(
        batch_tokenise, batched=True, remove_columns=train_raw.column_names
    )
    val_ds = val_raw.map(batch_tokenise, batched=True, remove_columns=val_raw.column_names)
    test_ds = test_raw.map(
        batch_tokenise, batched=True, remove_columns=test_raw.column_names
    )
    logging.info(
        "tokenised – train: %d | val: %d | test: %d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    # DeepSpeed option
    ds_cfg = args.deepspeed_config if args.use_deepspeed else None
    if args.use_deepspeed:
        try:
            importlib.import_module("deepspeed")
        except ImportError:
            logging.warning("DeepSpeed not available, disabling")
            ds_cfg = None

    targs = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        optim=("adamw_bnb_8bit" if args.bnb_8bit_optim else "adamw_torch"),
        bf16=args.bf16,
        fp16=not (args.bf16 or args.bnb_8bit_optim),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=0.3,

        logging_steps=args.logging_steps,
        logging_first_step=True,

        report_to=['wandb'],
        deepspeed=ds_cfg,
        seed=args.seed,
    )

    class StepDigest(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.global_step % args.logging_steps == 0:
                logging.info("step %d | loss %.4f | lr %.3g",
                             state.global_step,
                             logs.get('loss', float('nan')),
                             logs.get('learning_rate', float('nan')))

    if full_ft:
        for p in model.parameters():
            p.requires_grad_(True)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[
            StepDigest(),
            EarlyStoppingCallback(early_stopping_patience=9)
        ],
    )

    # Train
    trainer.train()
    trainer.save_model(Path(args.output_dir)/'final')
    tokenizer.save_pretrained(Path(args.output_dir)/'final')
    logging.info("Training done: adapters & tokenizer in %s/final", args.output_dir)

    summarise_training(trainer, Path(args.output_dir))

    # optional: evaluate on held-out test set
    test_metrics = trainer.evaluate(test_ds)
    logging.info(
        "TEST loss %.4f | ppl %.2f",
        test_metrics["eval_loss"],
        math.exp(test_metrics["eval_loss"]),
    )


    if wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
