
#!/usr/bin/env python3
"""
Unified held-out inference script.

Defaults to EXACT behavior of Inference Script 1:
- Reads raw JSON (--data_path)
- Recomputes held-out split via three_way_split (must match training script 1)
- Uses Script-1 prompt formatting and generation config
- Supports LoRA/merge, quantization, resume, W&B logging

Extras integrated from Inference Script 2 (opt-in):
- --use_pretokenized + --tokenized_data_path to identify held-out IDs from a
  preprocessed HF dataset saved via `datasets.load_from_disk` (e.g., when training used pre-tokenized data).
"""

import argparse
import gc
import json
import logging
import math
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Optional import only used when --use_pretokenized is passed
try:
    from datasets import load_from_disk
except Exception:
    load_from_disk = None  # type: ignore


# -----------------------------
# Script-1-compatible utilities
# -----------------------------

def build_chat_prompt(tokenizer, instruction: str, inp: str = "") -> str:
    user_msg = instruction if not inp else f"{instruction}\\n\\nInput:\\n{inp}"
    messages = [{"role": "user", "content": user_msg}]
    # Must match training/inference-1 script exactly.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def three_way_split(ds: Dataset, *, val_pct: float, test_pct: float, seed: int):
    """
    Group-wise split guaranteeing each prompt_count appears in one split.
    Returns (train_ids, val_ids, test_ids) as sets of prompt_count.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    pcs = list({ex["prompt_count"] for ex in ds})
    rng.shuffle(pcs)
    n = len(pcs)
    n_val = int(n * val_pct)
    n_test = int(n * test_pct)
    val_ids  = set(pcs[:n_val])
    test_ids = set(pcs[n_val:n_val+n_test])
    train_ids = set(pcs[n_val+n_test:])
    return train_ids, val_ids, test_ids


# -----------------------------
# Data loading
# -----------------------------

def load_examples_rawsplit(
    data_path: str,
    instruct_types: List[str] | None,
    val_pct: float,
    test_pct: float,
    seed: int,
    split: str,
    max_samples: int,
    from_prompt_id: int,
    upto_prompt_id: int,
) -> Tuple[List[Tuple[int, str, str, str]], Dict[int, Dict]]:
    """
    Script-1 behavior: compute split from raw JSON and return a flat queue:
      list of (prompt_count, key_name, prompt_text, raw_input),
    and a results_map keyed by prompt_count with minimal fields prefilled.
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
        keep_ids = val_ids.union(test_ids)  # heldout

    # prompt_count ID filtering
    keep_ids = set(pid for pid in keep_ids if (from_prompt_id <= pid if from_prompt_id>0 else True))
    if upto_prompt_id > 0:
        keep_ids = {pid for pid in keep_ids if pid <= upto_prompt_id}

    # max_samples = number of groups (prompt_count) to keep
    if max_samples > 0:
        keep_ids = set(sorted(list(keep_ids))[:max_samples])

    flat_queue: List[Tuple[int, str, str, str]] = []
    results_map: Dict[int, Dict] = {}
    missing_keys_counter: Dict[str, int] = {}

    for item in raw_data:
        pc = int(item["prompt_count"])
        if pc not in keep_ids:
            continue

        raw_input = item.get("input", "")

        if pc not in results_map:
            results_map[pc] = {"prompt_count": pc, "input": raw_input}

        if instruct_types:
            keep_keys = ["instruction_original"] + [k for k in instruct_types if k != "instruction_original"]
        else:
            keep_keys = ["instruction_original"] + [k for k in item.keys() if k.startswith("instruct_")]

        processed = set()
        for key in keep_keys:
            if key in processed:
                continue
            processed.add(key)

            instr = item.get(key)
            if not instr:
                missing_keys_counter[key] = missing_keys_counter.get(key, 0) + 1
                continue
            flat_queue.append((pc, key, instr, raw_input))

    logging.info("Loaded %d groups | flat prompts: %d", len(results_map), len(flat_queue))
    if missing_keys_counter:
        logging.info("Missing paraphrase keys (skipped): %s", missing_keys_counter)
    return flat_queue, results_map


def load_examples_from_tokenized(
    data_path: str,
    tokenized_data_path: str,
    instruct_types: List[str] | None,
    max_samples: int,
    from_prompt_id: int,
    upto_prompt_id: int,
) -> Tuple[List[Tuple[int, str, str, str]], Dict[int, Dict]]:
    """
    Script-2 ability: use the *pre-tokenized* dataset's 'test' split to define held-out IDs.
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

    # Apply prompt_count ID filters
    original_count = len(keep_ids)
    if from_prompt_id > 0:
        keep_ids = {pid for pid in keep_ids if pid >= from_prompt_id}
    if upto_prompt_id > 0:
        keep_ids = {pid for pid in keep_ids if pid <= upto_prompt_id}
    if len(keep_ids) < original_count:
        logging.info("Filtered held-out IDs from %d -> %d by id-range.", original_count, len(keep_ids))

    # Load raw JSON to get original text
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if max_samples > 0:
        keep_ids = set(sorted(list(keep_ids))[:max_samples])
        logging.info(f"Subsampling to at most {len(keep_ids)} prompt groups.")

    flat_queue: List[Tuple[int, str, str, str]] = []
    results_map: Dict[int, Dict] = {}
    missing_keys_counter: Dict[str, int] = {}

    for item in raw_data:
        pc = int(item["prompt_count"])
        if pc not in keep_ids:
            continue
        raw_input = item.get("input", "")
        if pc not in results_map:
            results_map[pc] = {"prompt_count": pc, "input": raw_input}

        if instruct_types:
            keep_keys = ["instruction_original"] + [k for k in instruct_types if k != "instruction_original"]
        else:
            keep_keys = ["instruction_original"] + [k for k in item.keys() if k.startswith("instruct_")]

        processed = set()
        for key in keep_keys:
            if key in processed:
                continue
            processed.add(key)
            instr = item.get(key)
            if not instr:
                missing_keys_counter[key] = missing_keys_counter.get(key, 0) + 1
                continue
            flat_queue.append((pc, key, instr, raw_input))

    logging.info(
        "Loaded %d items | held-out groups: %d | flat prompts: %d",
        len(results_map), len(keep_ids), len(flat_queue)
    )
    if missing_keys_counter:
        logging.info("Missing paraphrase keys (skipped): %s", missing_keys_counter)
    return flat_queue, results_map


# -----------------------------
# Model loading (Script-1 style)
# -----------------------------

def load_model_and_tokenizer(args):
    import importlib
    model_kwargs = {}

    # Quantization (none/4bit/8bit), matching Script 1 behavior
    try:
        _BNB_OK = importlib.util.find_spec("bitsandbytes") is not None
    except Exception:
        _BNB_OK = False
    if args.quant != "none" and not _BNB_OK:
        logging.warning("bitsandbytes not available - falling back to bf16")
        args.quant = "none"

    if args.quant == "none":
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=args.quant == "8bit",
            load_in_4bit=args.quant == "4bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logging.info("Loading base model from %s", args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **model_kwargs)

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
    try:
        model = torch.compile(model)  # torch 2.x
    except Exception:
        pass

    tok_path = (
        args.lora_path
        if args.lora_path and (Path(args.lora_path) / "tokenizer_config.json").exists()
        else args.base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=4096)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Held-out inference for paraphrx fine-tuning (with resume)")
    # Data / splits
    p.add_argument("--data_path", required=True, help="JSON file used in fine-tuning")
    p.add_argument("--instruct_types", nargs="+", default=[], help="Optional explicit list of instruct_* keys to use (default = ALL)")
    p.add_argument("--val_pct", type=float, default=0.05, help="Must match training (Script 1 path)")
    p.add_argument("--test_pct", type=float, default=0.05, help="Must match training (Script 1 path)")
    p.add_argument("--seed", type=int, default=42, help="Must match training (Script 1 path)")
    p.add_argument("--split", choices=["val", "test", "heldout"], default="heldout", help="'val', 'test', or both (heldout)")
    p.add_argument("--max_samples", type=int, default=0, help="Process at most this many prompt_count groups (0 = all)")
    p.add_argument("--from_prompt_id", type=int, default=0, help="Start from this prompt_count ID (inclusive)")
    p.add_argument("--upto_prompt_id", type=int, default=0, help="Process up to this prompt_count ID (inclusive, 0 = no limit)")

    # Pre-tokenized (Script 2 capability; opt-in)
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

    # Logging / output
    p.add_argument("--log_name", default=None, help="Optional friendly name for logging/W&B")
    p.add_argument("--wandb_project", default=None, help="If provided, logs to this W&B project")
    p.add_argument("--save_every", type=int, default=0, help="Save every N batches (0 = only at the end)")
    p.add_argument("--output_json", required=True, help="Where to store/merge answers")
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    # Logging
    log_name = args.log_name or f"{Path(args.output_json).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.info("CLI Args:\\n%s", json.dumps(vars(args), indent=2))

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

    # Data loading (default = Script-1 split path)
    if args.use_pretokenized:
        if not args.tokenized_data_path:
            logging.error("--use_pretokenized requires --tokenized_data_path")
            sys.exit(1)
        flat_queue, results_map = load_examples_from_tokenized(
            args.data_path,
            args.tokenized_data_path,
            instruct_types=args.instruct_types,
            max_samples=args.max_samples,
            from_prompt_id=args.from_prompt_id,
            upto_prompt_id=args.upto_prompt_id,
        )
    else:
        flat_queue, results_map = load_examples_rawsplit(
            args.data_path,
            instruct_types=args.instruct_types,
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
                                # preserve existing non-empty text
                                results_map[pc][k] = v
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

    # Safety: catch SIGINT/SIGTERM to save partial results
    def _handler(signum, frame):
        logging.warning("Signal %s received — saving partial results to %s", signum, output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(list(results_map.values()), ensure_ascii=False, indent=2))
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
        ).to(model.device)

        input_lens = tokenised["attention_mask"].sum(dim=1)

        gen_cfg = dict(
            max_new_tokens=args.max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=args.temperature > 0,
        )
        if args.temperature > 0:
            gen_cfg["temperature"] = args.temperature
            gen_cfg["top_p"] = 0.6

        with _INFER_CTX():
            outputs = model.generate(**tokenised, **gen_cfg)

        # Post-process, mirroring Script-1 cleanup
        for i in range(len(batch)):
            answer_ids = outputs[i, input_lens[i]:]
            text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

            # Strip an echoed user block or role marker if present
            original_user_message = instrs[i] if not inputs[i] else f"{instrs[i]}\\n\\nInput:\\n{inputs[i]}"
            if text.strip() == original_user_message.strip():
                logging.warning("Empty generation for prompt_count %s-%s (echoed prompt).", pcs[i], keys[i])
                text = ""
            else:
                model_turn_marker = "model\\n"
                marker_pos = text.find(model_turn_marker)
                if marker_pos != -1:
                    text = text[marker_pos + len(model_turn_marker):].lstrip()

            # Merge into results_map (do NOT overwrite non-empty existing text)
            existing_text = results_map[pcs[i]].get(keys[i], "")
            if isinstance(existing_text, str) and existing_text.strip():
                continue
            results_map[pcs[i]][keys[i]] = text

        # housekeeping
        del tokenised, outputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        batches_done += 1
        if (start + len(batch)) % 100 == 0:
            logging.info("Processed %d / %d prompts", start + len(batch), len(pending))

        if (batches_done % SAVE_EVERY_N_BATCHES) == 0:
            logging.info("Saving partial results to %s", output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(list(results_map.values()), ensure_ascii=False, indent=2))

    # Final save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(results_map.values()), ensure_ascii=False, indent=2))
    logging.info("Finished. Wrote %d records to %s", len(results_map), output_path)

    if wb_run:
        # lazy import here to avoid hard dep
        try:
            import wandb
            wandb.save(str(output_path))
            wb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
