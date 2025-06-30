#!/usr/bin/env python3
"""
```
f_finetune/src/ft_inference_alpaca_gsm8k.py \
  --data_paths f_finetune/data/alpaca_gemma-2-2b-it.json \
  --base_model_path f_finetune/model \
  --lora_path        f_finetune/outputs/buckets3/final \
  --buckets 1-3 \
  --split val \
  --batch 8 --max_tokens 128 --output_json out/bkt1_3_val.json
```
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import gc

import torch
from datasets import Dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Optional bitsandbytes / Flash‑Attention 2 – handled gracefully
try:
    from transformers import BitsAndBytesConfig

    _BNB_OK = True
except Exception:  # pragma: no cover – import failure
    BitsAndBytesConfig = None  # type: ignore
    _BNB_OK = False

try:
    import importlib, flash_attn  # noqa: F401

    importlib.import_module("flash_attn.flash_attn_interface")

    _FLASH2_OK = True
except Exception:  # pragma: no cover – flash‑attn missing
    _FLASH2_OK = False

if os.getenv("DISABLE_FLASH_ATTN", "0") == "1":  # user override
    _FLASH2_OK = False

_INFER_CTX = getattr(torch, "inference_mode", torch.no_grad)

# Utility helpers

def parse_bucket_spec(spec: str) -> List[int]:
    """Convert a spec like "1-3,5" -> ``[1, 2, 3, 5]`` (unique, sorted)."""
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            result.update(range(int(start), int(end) + 1))
        else:
            result.add(int(part))
    allowed = [b for b in sorted(result) if 1 <= b <= 5]
    if not allowed:
        raise ValueError("Bucket specification must select at least one bucket between 1 and 5.")
    return allowed


# Prompt construction identical to the fine‑tuning script

def build_prompt(instruction: str, raw_input: str | None = None) -> str:
    if raw_input:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{raw_input.strip()}\n\n"
            "### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# Data loading & held‑out split

def load_examples(paths: List[str], buckets: List[int]) -> List[dict]:
    """Load JSON prompts, keep only paraphrases in *buckets*, return flat list."""

    examples: List[dict] = []
    for p in paths:
        logging.info("Reading %s", p)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            inp = item.get("input", "")
            pc = str(item.get("prompt_count"))
            for para in item.get("paraphrases", []):
                if int(para.get("bucket", 0)) not in buckets:
                    continue
                instruction = para.get("paraphrase") or item.get("instruction_original", "")
                if not instruction:
                    continue
                instruct_type = para.get("instruct_type", "unknown")
                # normalise: original instruction should keep its special key
                if instruct_type == "instruction_original":
                    key_name = "instruction_original"
                else:
                    key_name = instruct_type
                examples.append(
                    {
                        "prompt_count": pc,
                        "key": key_name,
                        "instruction": instruction,
                        "input": inp,
                    }
                )
    logging.info("Loaded %d paraphrases matching buckets %s", len(examples), buckets)
    return examples


def make_holdout_split(examples: List[dict], seed: int, split: str, test_ratio: float = 0.2) -> List[dict]:
    """Replicate the train/val split logic used during fine‑tuning."""

    if split not in {"train", "val"}:
        raise ValueError("--split must be either 'train' or 'val'")

    random.seed(seed)
    random.shuffle(examples)

    ds = Dataset.from_list(examples)
    splitted = ds.train_test_split(test_size=test_ratio, seed=seed, shuffle=True)
    chosen = splitted["test" if split == "val" else "train"]
    logging.info("Using %s split -> %d examples", split, len(chosen))
    return list(chosen)


# Flatten for batched generation

def flatten_examples(examples: List[dict]) -> Tuple[List[Tuple[str, str, str]], Dict[str, Dict[str, str]]]:
    """Return (flat_queue, results_map) – mirrors original script but new format."""

    flat_queue: List[Tuple[str, str, str]] = []
    results_map: Dict[str, Dict[str, str]] = {}

    for ex in examples:
        pc = ex["prompt_count"]
        key = ex["key"]
        prompt = build_prompt(ex["instruction"], ex["input"])

        flat_queue.append((pc, key, prompt))
        if pc not in results_map:
            results_map[pc] = {"prompt_count": pc}

    return flat_queue, results_map

# Main
def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for paraphrase‑robust LoRA‑adapted GEMMA‑2‑2B‑IT")

    # Main I/O
    parser.add_argument("--data_paths", nargs="+", required=True, help="JSON dataset file(s)")
    parser.add_argument("--output_json", required=True, help="Where to dump generations")

    # Model paths
    parser.add_argument("--base_model_path", required=True, help="Directory with the *base* GEMMA model")
    parser.add_argument("--lora_path", required=True, help="Directory with LoRA adapter (the 'final' folder)")
    parser.add_argument("--merge_lora", action="store_true", help="Merge adapter into base weights (memory‑heavy but faster)")

    # Dataset filtering
    parser.add_argument("--buckets", default="1-5", help="Bucket spec, e.g. '1', '1-3', '2,4'")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Evaluate on the held‑out val (default) or train portion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed – must match training to reproduce split")

    # Generation hyper‑params
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Quantisation / device
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")

    args = parser.parse_args()

    # Logging
    Path("logs").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"infer_{Path(args.lora_path).stem}_{ts}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.info("Run args: %s", vars(args))

    # Dataset preparation
    buckets = parse_bucket_spec(args.buckets)
    raw_examples = load_examples(args.data_paths, buckets)
    heldout_examples = make_holdout_split(raw_examples, seed=args.seed, split=args.split)
    flat_queue, results_map = flatten_examples(heldout_examples)

    # Sort shortest -> longest to optimise batching
    flat_queue.sort(key=lambda t: len(t[2]))

    # Model loading (base + LoRA)
    model_kwargs: dict = dict(device_map=args.device)
    if _FLASH2_OK:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        logging.info("Flash‑Attention 2 not available -> using standard attention")

    # bitsandbytes quantisation
    if args.quant != "none" and not _BNB_OK:
        logging.warning("bitsandbytes not available – falling back to bf16")
        args.quant = "none"

    if args.quant == "none":
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        try:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=args.quant == "8bit",
                load_in_4bit=args.quant == "4bit",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as e:  # pragma: no cover
            logging.warning("Quant config failed (%s) – using bf16", e)
            model_kwargs["torch_dtype"] = torch.bfloat16
            args.quant = "none"

    logging.info("Loading base model from %s", args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **model_kwargs)

    logging.info("Loading LoRA adapter from %s", args.lora_path)
    model = PeftModel.from_pretrained(base_model, args.lora_path, is_trainable=False)

    if args.merge_lora:
        logging.info("Merging LoRA weights into base model …")
        model = model.merge_and_unload()
        logging.info("Merge done – adapter modules dropped")

    model.eval()

    try:
        model = torch.compile(model)  # minor speed‑up on PyTorch 2
    except Exception:  # pragma: no cover
        pass

    # Tokeniser (prefer adapter dir – it contains the same copy saved after training)
    tok_path = args.lora_path if (Path(args.lora_path) / "tokenizer_config.json").exists() else args.base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=4096)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generation loop
    def save_partial():
        Path(args.output_json).write_text(
            json.dumps(list(results_map.values()), indent=2, ensure_ascii=False)
        )

    def handler(sig_num, _frame):
        logging.info("Signal %s caught – saving partial results", sig_num)
        save_partial()
        sys.exit(0)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, handler)

    for start in tqdm(range(0, len(flat_queue), args.batch), desc="generating"):
        batch_slice = flat_queue[start : start + args.batch]
        pcounts, keys, prompts = zip(*batch_slice)

        inputs = tokenizer(list(prompts), return_tensors="pt", padding=True).to(model.device)
        input_lens = inputs["attention_mask"].sum(dim=1)

        gen_kwargs = dict(max_new_tokens=args.max_tokens, pad_token_id=tokenizer.eos_token_id)
        if args.temperature > 0:
            gen_kwargs.update(temperature=args.temperature, do_sample=True)
        else:
            gen_kwargs["do_sample"] = False

        with _INFER_CTX():
            outputs = model.generate(**inputs, **gen_kwargs)

        for i in range(len(batch_slice)):
            start_tok = int(input_lens[i])
            completion_ids = outputs[i, start_tok:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            results_map[pcounts[i]][keys[i]] = completion

        # Book‑keeping
        del inputs, outputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        if (start + len(batch_slice)) % 100 == 0:
            logging.info("Processed %d / %d prompts", start + len(batch_slice), len(flat_queue))

    save_partial()
    logging.info("Finished – wrote %d items to %s", len(results_map), args.output_json)
    print(f"Saved {len(results_map)} generations -> {args.output_json}")


if __name__ == "__main__":
    main()
