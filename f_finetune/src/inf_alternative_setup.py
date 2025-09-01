#!/usr/bin/env python3
"""
held-out inference script — CALIPER-aware (FIRST + SECOND), style-whitelist capable,
and generation settings aligned with training

Changes (targeted):
- Load to GPU explicitly via device_map="auto" when CUDA is available; CPU otherwise.
- Use bf16 on GPU and fp32 on CPU when quantization==none.
- Add --compile flag to opt-in to torch.compile (instead of always compiling).
- Log model dtype/device and hf_device_map after load.
- Route tokenized inputs to the proper device (first layer / hf_device_map) instead of model.device.
- Log just before and after the very first generate() call to make progress visible.
"""
from __future__ import annotations
import argparse
import gc
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any, Optional, Set

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Optional import only used when --use_pretokenized is passed
try:
    from datasets import load_from_disk
except Exception:
    load_from_disk = None  # type: ignore


# Helpers

def load_style_list(paths: List[str] | None) -> Optional[Set[str]]:
    """Load one or many JSON lists of style keys (instruct_*). Returns None if not provided/empty"""
    if not paths:
        return None
    styles: Set[str] = set()
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                lst = json.load(fh)
            if isinstance(lst, list):
                styles.update(lst)
        except Exception as e:
            logging.warning("Could not read style list %s: %s", p, e)
    return styles if styles else None


def is_first_format(data: list) -> bool:
    """FIRST dataset items have 'paraphrases' (list) and 'instruction_original' at top level"""
    return bool(
        data
        and isinstance(data, list)
        and isinstance(data[0], dict)
        and "paraphrases" in data[0]
        and isinstance(data[0]["paraphrases"], list)
        and "instruction_original" in data[0]
    )


def build_chat_prompt(tokenizer, instruction: str, inp: str = "") -> str:
    # Match the training meta exactly
    meta = "Answer concisely and directly. Focus on task semantics; ignore stylistic tone cues. End after the answer."
    user_core = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    user_msg = f"{meta}\n\n{user_core}"
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_eot_id(tokenizer) -> int:
    tid = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if tid is None:
        return tokenizer.eos_token_id
    # Some tokenizers return unk for unknown special tokens
    if hasattr(tokenizer, "unk_token_id") and tid == tokenizer.unk_token_id:
        return tokenizer.eos_token_id
    return tid


def three_way_split(ds: Dataset, *, val_pct: float, test_pct: float, seed: int):
    """
    Group-wise split guaranteeing each prompt_count appears in one split.
    Returns (train_ids, val_ids, test_ids) as sets of prompt_count!
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    pcs = list({int(ex["prompt_count"]) for ex in ds})
    rng.shuffle(pcs)
    n = len(pcs)
    n_val = int(n * val_pct)
    n_test = int(n * test_pct)
    val_ids  = set(pcs[:n_val])
    test_ids = set(pcs[n_val:n_val+n_test])
    train_ids = set(pcs[n_val+n_test:])
    return train_ids, val_ids, test_ids


# FIRST-format support

def analyze_first_style_counts_firstfile(
    data: list,
    allowed_styles: Optional[Set[str]],
    require_content_score: int,
) -> Dict[str, int]:
    from collections import Counter
    ctr = Counter()
    for item in data:
        for ph in item.get("paraphrases", []):
            if ph.get("paraphrase_content_score", 0) < require_content_score:
                continue
            sty = ph.get("instruct_type")
            if not sty:
                continue
            if allowed_styles is not None and sty not in allowed_styles:
                continue
            ctr[sty] += 1
    return dict(ctr)


def load_first_rawsplit(
    data_path: str,
    allowed_styles: Optional[Set[str]],
    require_content_score: int,
    first_min_style_count: int,
    val_pct: float,
    test_pct: float,
    seed: int,
    split: str,
    max_samples: int,
    from_prompt_id: int,
    upto_prompt_id: int,
) -> Tuple[List[Tuple[int, str, str, str]], Dict[int, Dict]]:
    """
    Build a flat generation queue from a FIRST-format JSON.
    Returns list of (prompt_count, key_name, prompt_text, raw_input),
    and results_map keyed by prompt_count with input prefilled
    - Always includes 'instruction_original'
    - Keeps paraphrases whose instruct_type passes:
        (a) paraphrase_content_score == require_content_score
        (b) global count >= first_min_style_count
        (c) (optional) allowed_styles whitelist
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ds = Dataset.from_list([{"prompt_count": int(x["prompt_count"])} for x in data])
    train_ids, val_ids, test_ids = three_way_split(ds, val_pct=val_pct, test_pct=test_pct, seed=seed)
    if split == "val":
        keep_ids = val_ids
    elif split == "test":
        keep_ids = test_ids
    else:
        keep_ids = val_ids.union(test_ids)

    # ID filters
    keep_ids = {pid for pid in keep_ids if (from_prompt_id <= pid if from_prompt_id > 0 else True)}
    if upto_prompt_id > 0:
        keep_ids = {pid for pid in keep_ids if pid <= upto_prompt_id}
    if max_samples > 0:
        keep_ids = set(sorted(list(keep_ids))[:max_samples])

    # Which styles qualify globally?
    style_counts = analyze_first_style_counts_firstfile(data, allowed_styles, require_content_score)
    valid_styles = {s for s, c in style_counts.items() if c >= first_min_style_count}
    if allowed_styles is not None:
        valid_styles &= allowed_styles
    logging.info("FIRST: %d styles pass content==%d and count>=%d",
                 len(valid_styles), require_content_score, first_min_style_count)

    flat_queue: List[Tuple[int, str, str, str]] = []
    results_map: Dict[int, Dict] = {}

    for item in data:
        pc = int(item["prompt_count"])
        if pc not in keep_ids:
            continue

        raw_input = (item.get("input", "") or item.get("scenarios", "") or "")
        if pc not in results_map:
            results_map[pc] = {"prompt_count": pc, "input": raw_input}

        # Always include the original instruction
        inst_orig = item.get("instruction_original", "")
        if inst_orig:
            flat_queue.append((pc, "instruction_original", inst_orig, raw_input))

        # Valid paraphrases
        for ph in item.get("paraphrases", []):
            if ph.get("paraphrase_content_score", 0) < require_content_score:
                continue
            sty = ph.get("instruct_type")
            if not sty or (valid_styles and sty not in valid_styles):
                continue
            parap = ph.get("paraphrase", "")
            if not parap:
                continue
            flat_queue.append((pc, sty, parap, raw_input))

    logging.info("Loaded (FIRST) %d groups | flat prompts: %d", len(results_map), len(flat_queue))
    return flat_queue, results_map


# SECOND-format support

def load_second_rawsplit(
    data_path: str,
    allowed_styles: Optional[Set[str]],
    instruct_types: Optional[List[str]],
    val_pct: float,
    test_pct: float,
    seed: int,
    split: str,
    max_samples: int,
    from_prompt_id: int,
    upto_prompt_id: int,
) -> Tuple[List[Tuple[int, str, str, str]], Dict[int, Dict]]:
    """
    Compute split from SECOND-style JSON and return a flat queue:
      list of (prompt_count, key_name, prompt_text, raw_input),
    and a results_map keyed by prompt_count with minimal fields prefilled!
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    ds = Dataset.from_list(raw_data)
    train_ids, val_ids, test_ids = three_way_split(ds, val_pct=val_pct, test_pct=test_pct, seed=seed)
    if split == "val":
        keep_ids = val_ids
    elif split == "test":
        keep_ids = test_ids
    else:
        keep_ids = val_ids.union(test_ids)

    # ID filters
    keep_ids = {pid for pid in keep_ids if (from_prompt_id <= pid if from_prompt_id > 0 else True)}
    if upto_prompt_id > 0:
        keep_ids = {pid for pid in keep_ids if pid <= upto_prompt_id}
    if max_samples > 0:
        keep_ids = set(sorted(list(keep_ids))[:max_samples])

    flat_queue: List[Tuple[int, str, str, str]] = []
    results_map: Dict[int, Dict] = {}
    missing_keys_counter: Dict[str, int] = {}

    # Merge style sources:
    # - If style_whitelist_json was provided -> start from that set
    # - If instruct_types was provided -> intersect with whitelist if present; else use these
    # - Otherwise -> keep all instruct_* keys per item
    explicit_styles: Optional[Set[str]] = None
    if instruct_types:
        explicit_styles = set(instruct_types)
    if allowed_styles is not None and explicit_styles is not None:
        final_style_set = allowed_styles & explicit_styles
    elif allowed_styles is not None:
        final_style_set = allowed_styles
    elif explicit_styles is not None:
        final_style_set = explicit_styles
    else:
        final_style_set = None  # keep all we find

    for item in raw_data:
        pc = int(item["prompt_count"])
        if pc not in keep_ids:
            continue

        raw_input = item.get("input", "")
        if pc not in results_map:
            results_map[pc] = {"prompt_count": pc, "input": raw_input}

        # Always include original
        if item.get("instruction_original"):
            flat_queue.append((pc, "instruction_original", item["instruction_original"], raw_input))

        # Decide style keys for this record
        item_style_keys = [k for k in item.keys() if k.startswith("instruct_")]
        if final_style_set is not None:
            item_style_keys = [k for k in item_style_keys if k in final_style_set]

        seen = set()
        for key in ["instruction_original"] + item_style_keys:
            if key in seen:
                continue
            seen.add(key)
            if key == "instruction_original":
                continue  # already added
            instr = item.get(key)
            if not instr:
                missing_keys_counter[key] = missing_keys_counter.get(key, 0) + 1
                continue
            flat_queue.append((pc, key, instr, raw_input))

    logging.info("Loaded (SECOND) %d groups | flat prompts: %d", len(results_map), len(flat_queue))
    if missing_keys_counter:
        logging.info("Missing paraphrase keys (skipped): %s", missing_keys_counter)
    return flat_queue, results_map


def load_examples_rawsplit(
    data_path: str,
    instruct_types: Optional[List[str]],
    style_whitelist: Optional[Set[str]],
    require_content_score: int,
    first_min_style_count: int,
    val_pct: float,
    test_pct: float,
    seed: int,
    split: str,
    max_samples: int,
    from_prompt_id: int,
    upto_prompt_id: int,
) -> Tuple[List[Tuple[int, str, str, str]], Dict[int, Dict]]:
    """Auto-detect FIRST vs SECOND and dispatch"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if is_first_format(data):
        return load_first_rawsplit(
            data_path=data_path,
            allowed_styles=style_whitelist,
            require_content_score=require_content_score,
            first_min_style_count=first_min_style_count,
            val_pct=val_pct,
            test_pct=test_pct,
            seed=seed,
            split=split,
            max_samples=max_samples,
            from_prompt_id=from_prompt_id,
            upto_prompt_id=upto_prompt_id,
        )
    else:
        return load_second_rawsplit(
            data_path=data_path,
            allowed_styles=style_whitelist,
            instruct_types=instruct_types,
            val_pct=val_pct,
            test_pct=test_pct,
            seed=seed,
            split=split,
            max_samples=max_samples,
            from_prompt_id=from_prompt_id,
            upto_prompt_id=upto_prompt_id,
        )


def load_examples_from_tokenized(
    data_path: str,
    tokenized_data_path: str,
    instruct_types: Optional[List[str]],
    style_whitelist: Optional[Set[str]],
    require_content_score: int,
    first_min_style_count: int,
    max_samples: int,
    from_prompt_id: int,
    upto_prompt_id: int,
) -> Tuple[List[Tuple[int, str, str, str]], Dict[int, Dict]]:
    """
    Use the pre-tokenized dataset's 'test' split to define held-out IDs (strongest parity with training).
    Then read raw JSON and build the queue for those IDs.
    Works for BOTH FIRST and SECOND formats!
    """
    if load_from_disk is None:
        logging.error("datasets.load_from_disk is unavailable but --use_pretokenized was set.")
        sys.exit(1)

    logging.info(f"Loading split information from {tokenized_data_path} …")
    try:
        tokenized = load_from_disk(tokenized_data_path)
    except Exception as e:
        logging.error(f"Failed to load tokenized dataset at {tokenized_data_path}: {e}")
        sys.exit(1)

    if "test" not in tokenized:
        logging.error(f"Tokenized dataset at {tokenized_data_path} must contain a 'test' split.")
        sys.exit(1)

    keep_ids = set(int(x) for x in tokenized["test"]["prompt_count"])

    # Apply range filters
    original_count = len(keep_ids)
    if from_prompt_id > 0:
        keep_ids = {pid for pid in keep_ids if pid >= from_prompt_id}
    if upto_prompt_id > 0:
        keep_ids = {pid for pid in keep_ids if pid <= upto_prompt_id}
    if len(keep_ids) < original_count:
        logging.info("Filtered held-out IDs from %d -> %d by id-range.", original_count, len(keep_ids))

    # Load raw JSON to inspect schema and content
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Subsample group count if requested
    if max_samples > 0:
        keep_ids = set(sorted(list(keep_ids))[:max_samples])
        logging.info(f"Subsampling to at most {len(keep_ids)} prompt groups.")

    # Build flat queue using the IDs
    flat_queue: List[Tuple[int, str, str, str]] = []
    results_map: Dict[int, Dict] = {}

    if is_first_format(raw):
        # Compute valid FIRST styles (counts + whitelist + content score)
        style_counts = analyze_first_style_counts_firstfile(raw, style_whitelist, require_content_score)
        valid_styles = {s for s, c in style_counts.items() if c >= first_min_style_count}
        if style_whitelist is not None:
            valid_styles &= style_whitelist
        logging.info("FIRST (pretok): %d styles pass filters.", len(valid_styles))

        for item in raw:
            pc = int(item["prompt_count"])
            if pc not in keep_ids:
                continue
            raw_input = (item.get("input", "") or item.get("scenarios", "") or "")
            if pc not in results_map:
                results_map[pc] = {"prompt_count": pc, "input": raw_input}

            inst_orig = item.get("instruction_original", "")
            if inst_orig:
                flat_queue.append((pc, "instruction_original", inst_orig, raw_input))

            for ph in item.get("paraphrases", []):
                if ph.get("paraphrase_content_score", 0) != require_content_score:
                    continue
                sty = ph.get("instruct_type")
                if not sty or (valid_styles and sty not in valid_styles):
                    continue
                parap = ph.get("paraphrase", "")
                if parap:
                    flat_queue.append((pc, sty, parap, raw_input))
    else:
        # SECOND
        for item in raw:
            pc = int(item["prompt_count"])
            if pc not in keep_ids:
                continue
            raw_input = item.get("input", "")
            if pc not in results_map:
                results_map[pc] = {"prompt_count": pc, "input": raw_input}

            if item.get("instruction_original"):
                flat_queue.append((pc, "instruction_original", item["instruction_original"], raw_input))

            # Combine style controls
            explicit_styles: Optional[Set[str]] = set(instruct_types) if instruct_types else None
            final_style_set = None
            if style_whitelist is not None and explicit_styles is not None:
                final_style_set = style_whitelist & explicit_styles
            elif style_whitelist is not None:
                final_style_set = style_whitelist
            elif explicit_styles is not None:
                final_style_set = explicit_styles

            item_style_keys = [k for k in item.keys() if k.startswith("instruct_")]
            if final_style_set is not None:
                item_style_keys = [k for k in item_style_keys if k in final_style_set]

            for key in item_style_keys:
                instr = item.get(key)
                if instr:
                    flat_queue.append((pc, key, instr, raw_input))

    logging.info(
        "Loaded %d items | held-out groups: %d | flat prompts: %d",
        len(results_map), len(keep_ids), len(flat_queue)
    )
    return flat_queue, results_map


# Model loading

def load_model_and_tokenizer(args):
    import importlib
    model_kwargs = {}

    # Quantisation (none/4bit/8bit)
    try:
        _BNB_OK = importlib.util.find_spec("bitsandbytes") is not None
    except Exception:
        _BNB_OK = False
    if args.quant != "none" and not _BNB_OK:
        logging.warning("bitsandbytes not available - falling back to bf16/fp32 (quant=none)")
        args.quant = "none"

    # Decide device map & dtype based on availability
    if torch.cuda.is_available():
        device_map = "auto"  # shard across available GPUs if needed
        if args.quant == "none":
            model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        device_map = "cpu"
        if args.quant == "none":
            model_kwargs["torch_dtype"] = torch.float32  # bf16 on CPU is slow

    if args.quant != "none":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=args.quant == "8bit",
            load_in_4bit=args.quant == "4bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logging.info("Loading base model from %s (device_map=%s)", args.base_model_path, device_map)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        device_map=device_map,
        **model_kwargs
    )

    # LoRA
    try:
        _PEFT_OK = importlib.util.find_spec("peft") is not None
    except Exception:
        _PEFT_OK = False

    if args.lora_path:
        if not _PEFT_OK:
            logging.error("peft is not installed but --lora_path provided")
            sys.exit(1)
        from peft import PeftModel
        logging.info("Loading LoRA adapter from %s", args.lora_path)
        model = PeftModel.from_pretrained(base_model, args.lora_path, is_trainable=False)
        if args.merge_lora:
            logging.info("Merging LoRA weights into base model …")
            model = model.merge_and_unload()
            logging.info("Merge completed — adapter dropped")
    else:
        model = base_model

    model.eval()

    # Optional compile (off by default, can be slow on first batch)
    low = args.base_model_path.lower()
    if args.compile and ("falcon" not in low) and ("internlm" not in low):
        try:
            model = torch.compile(model)  # torch 2.x
            logging.info("torch.compile enabled.")
        except Exception as e:
            logging.warning("torch.compile failed: %s", e)

    tok_path = (
        args.lora_path
        if args.lora_path and (Path(args.lora_path) / "tokenizer_config.json").exists()
        else args.base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        model_max_length=4096,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding for some decoder-only families during batching
    low = args.base_model_path.lower()
    if "falcon" in low or "internlm" in low:
        tokenizer.padding_side = "left"

    # Log where the model actually lives
    try:
        first_param = next(p for p in model.parameters() if getattr(p, "device", None) and p.device.type != "meta")
        logging.info("Model dtype=%s on device=%s", first_param.dtype, first_param.device)
    except StopIteration:
        logging.info("Model parameters not materialized (meta tensors).")

    if hasattr(model, "hf_device_map"):
        logging.info("hf_device_map: %s", getattr(model, "hf_device_map"))

    return model, tokenizer


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Held-out inference for CALIPER fine-tuning (FIRST & SECOND aware)")
    # Data / splits
    p.add_argument("--data_path", required=True, help="JSON file used in fine-tuning (FIRST or SECOND format)")
    p.add_argument("--instruct_types", nargs="+", default=[], help="Optional explicit list of instruct_* keys to use (SECOND).")
    p.add_argument("--style_whitelist_json", nargs="*", default=[],
                   help="JSON file(s) each containing a list of allowed instruct_* styles (XS/S/M/L/XL). If omitted, uses all.")
    p.add_argument("--first_min_style_count", type=int, default=200,
                   help="FIRST dataset: require at least this many paraphrases (content_score==require_content_score) per style.")
    p.add_argument("--require_content_score", type=int, default=5,
                   help="FIRST dataset: minimum paraphrase_content_score to keep (inclusive).")
    p.add_argument("--val_pct", type=float, default=0.05, help="Must match training")
    p.add_argument("--test_pct", type=float, default=0.05, help="Must match training")
    p.add_argument("--seed", type=int, default=42, help="Must match training")
    p.add_argument("--split", choices=["val", "test", "heldout"], default="heldout", help="'val', 'test', or both (heldout)'")
    p.add_argument("--max_samples", type=int, default=0, help="Process at most this many prompt_count groups (0 = all)")
    p.add_argument("--from_prompt_id", type=int, default=0, help="Start from this prompt_count ID (inclusive)")
    p.add_argument("--upto_prompt_id", type=int, default=0,
                   help="Process up to this prompt_count ID (inclusive, 0 = no limit)")

    # Pre-tokenised
    p.add_argument("--use_pretokenized", action="store_true", help="Use held-out IDs from a pre-tokenized dataset 'test' split")
    p.add_argument("--tokenized_data_path", type=str, default=None, help="Path passed to datasets.load_from_disk (required if --use_pretokenized)")

    # Model
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--lora_path", help="Path to LoRA adapter; omit for full-FT")
    p.add_argument("--merge_lora", action="store_true", help="Merge adapter into base weights for faster inference")

    # Generation
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    p.add_argument("--compile", action="store_true", help="Use torch.compile (off by default; first run can be slow)")

    # Logging / output
    p.add_argument("--log_name", default=None, help="Optional friendly name for logging/W&B")
    p.add_argument("--wandb_project", default=None, help="If provided, logs to this W&B project")
    p.add_argument("--save_every", type=int, default=0, help="Save every N batches (0 = only at the end)")
    p.add_argument("--output_json", required=True, help="Where to store/merge answers")
    return p.parse_args()


# Main

def main():
    args = parse_args()

    # Logging
    log_name = args.log_name or f"{Path(args.output_json).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.info("CLI Args:\n%s", json.dumps(vars(args), indent=2))

    # Optional W&B
    wb_run = None
    if args.wandb_project:
        try:
            import wandb
            wb_run = wandb.init(
                project=args.wandb_project,
                name=f"infer_{log_name}",
                job_type="inference",
                config=vars(args),
            )
        except Exception as e:
            logging.warning("W&B init failed (%s) - continuing without W&B", e)

    # Style whitelist (if any)
    style_whitelist = load_style_list(args.style_whitelist_json)

    # Data loading
    if args.use_pretokenized:
        if not args.tokenized_data_path:
            logging.error("--use_pretokenized requires --tokenized_data_path")
            sys.exit(1)
        flat_queue, results_map = load_examples_from_tokenized(
            data_path=args.data_path,
            tokenized_data_path=args.tokenized_data_path,
            instruct_types=args.instruct_types or None,
            style_whitelist=style_whitelist,
            require_content_score=args.require_content_score,
            first_min_style_count=args.first_min_style_count,
            max_samples=args.max_samples,
            from_prompt_id=args.from_prompt_id,
            upto_prompt_id=args.upto_prompt_id,
        )
    else:
        flat_queue, results_map = load_examples_rawsplit(
            data_path=args.data_path,
            instruct_types=args.instruct_types or None,
            style_whitelist=style_whitelist,
            require_content_score=args.require_content_score,
            first_min_style_count=args.first_min_style_count,
            val_pct=args.val_pct,
            test_pct=args.test_pct,
            seed=args.seed,
            split=args.split,
            max_samples=args.max_samples,
            from_prompt_id=args.from_prompt_id,
            upto_prompt_id=args.upto_prompt_id,
        )

    # Resume/merge existing output
    output_path = Path(args.output_json)
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            if isinstance(existing, list):
                for rec in existing:
                    pc = int(rec.get("prompt_count", -1))
                    if pc in results_map:
                        for k, v in rec.items():
                            if k in {"prompt_count", "input"}:
                                continue
                            if isinstance(v, str) and v.strip():
                                results_map[pc][k] = v  # keep existing non-empty
                logging.info("Resume: merged %d existing records.", len(existing))
        except Exception as e:
            logging.warning("Failed to read/merge existing output (%s). Starting fresh.", e)

    # Build the pending queue (skip already-answered)
    pending: List[Tuple[int, str, str, str]] = []
    for pc, key, instr, raw_input in flat_queue:
        existing_text = results_map[pc].get(key, "")
        if isinstance(existing_text, str) and existing_text.strip():
            continue
        pending.append((pc, key, instr, raw_input))

    logging.info("Pending prompts to generate: %d / %d", len(pending), len(flat_queue))
    if not pending:
        logging.info("Nothing to do. Exiting.")
        if wb_run:
            wb_run.finish()
        return

    # Model + tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    eot_id = get_eot_id(tokenizer)

    # Determine correct input device
    def _infer_input_device(m):
        # Prefer hf_device_map if present (device of the first module entry)
        if hasattr(m, "hf_device_map") and isinstance(m.hf_device_map, dict) and m.hf_device_map:
            first_dev = next(iter(m.hf_device_map.values()))
            if isinstance(first_dev, int):
                return torch.device(f"cuda:{first_dev}") if torch.cuda.is_available() else torch.device("cpu")
            return torch.device(first_dev)
        # Fallback: device of first real parameter
        try:
            return next(p for p in m.parameters() if p.device.type != "meta").device
        except StopIteration:
            return torch.device("cpu")

    input_device = _infer_input_device(model)
    logging.info("Inputs will be placed on %s", input_device)

    # Safety: catch SIGINT/SIGTERM to save partial results
    def _handler(signum, frame):
        logging.warning("Signal %s received — saving partial results to %s", signum, output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(sorted(list(results_map.values()), key=lambda r: r["prompt_count"]),
                                              ensure_ascii=False, indent=2))
        finally:
            sys.exit(0)
    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _handler)

    _INFER_CTX = getattr(torch, "inference_mode", torch.no_grad)
    SAVE_EVERY_N_BATCHES = args.save_every if args.save_every > 0 else (len(pending) + args.batch - 1) // args.batch

    logging.info("Starting inference: %d prompts | batch=%d | max_tokens=%d | temp=%.2f",
                 len(pending), args.batch, args.max_tokens, args.temperature)

    batches_done = 0
    for start in tqdm(range(0, len(pending), args.batch), desc="generating"):
        batch = pending[start:start + args.batch]
        pcs, keys, instrs, inputs = zip(*batch)
        prompts = [build_chat_prompt(tokenizer, i, inp) for i, inp in zip(instrs, inputs)]

        tokenised = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(input_device)

        input_lens = tokenised["attention_mask"].sum(dim=1)

        # Generation config aligned to training's concise_generate
        gen_cfg = dict(
            max_new_tokens=args.max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=list({tokenizer.eos_token_id, eot_id}),
            do_sample=bool(args.temperature > 0),
            repetition_penalty=1.05,
        )
        if args.temperature > 0:
            gen_cfg["temperature"] = args.temperature
            gen_cfg["top_p"] = 0.6

        low = args.base_model_path.lower()
        if "internlm" in low:
            gen_cfg["cache_implementation"] = "static"  # remote code expectation
        if "falcon" in low:
            gen_cfg["use_cache"] = False  # avoid past_key_values crash

        if start == 0:
            logging.info("Entering first generate()…")
        with _INFER_CTX():
            outputs = model.generate(**tokenised, **gen_cfg)
        if start == 0:
            logging.info("First generate() finished.")

        # Post-process
        for i in range(len(batch)):
            answer_ids = outputs[i, input_lens[i]:]
            text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

            # Strip an echoed prompt if the model returns the user text verbatim
            original_user_message = instrs[i] if not inputs[i] else f"{instrs[i]}\n\nInput:\n{inputs[i]}"
            if text.strip() == original_user_message.strip():
                logging.warning("Empty generation for prompt_count %s-%s (echoed prompt).", pcs[i], keys[i])
                text = ""
            else:
                # Some chat templates include a role marker like "model\n"
                marker = "model\n"
                mp = text.find(marker)
                if mp != -1:
                    text = text[mp + len(marker):].lstrip()

            # Merge into results_map (preserve existing non-empty)
            existing_text = results_map[pcs[i]].get(keys[i], "")
            if isinstance(existing_text, str) and existing_text.strip():
                continue
            results_map[pcs[i]][keys[i]] = text

        # housekeeping
        del tokenised, outputs
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

        batches_done += 1
        if (start + len(batch)) % 100 == 0:
            logging.info("Processed %d / %d prompts", start + len(batch), len(pending))

        if (batches_done % SAVE_EVERY_N_BATCHES) == 0:
            logging.info("Saving partial results to %s", output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(
                sorted(list(results_map.values()), key=lambda r: r["prompt_count"]),
                ensure_ascii=False, indent=2
            ))

    # Final save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        sorted(list(results_map.values()), key=lambda r: r["prompt_count"]),
        ensure_ascii=False, indent=2
    ))
    logging.info("Finished. Wrote %d records to %s", len(results_map), output_path)

    if wb_run:
        try:
            import wandb
            wandb.save(str(output_path))
            wb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
