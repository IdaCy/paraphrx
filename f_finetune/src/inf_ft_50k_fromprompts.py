#!/usr/bin/env python3
"""
Inference on the held-out split (val + test) for paraphrase-robust fine-tuning
with RESUME support

If --output_json already exists and contains partial results, this script will:
  - read it,
  - prefill any already-generated answers,
  - queue ONLY the missing (or empty) answers for generation,
  - and keep writing back merged results as it progresses.

# - LoRA run
python $RUN_SCRIPT \
  --data_path a_data/alpaca/50k_phrxed.json \
  --base_model_path f_finetune/model \
  --lora_path f_finetune/outputs/h9x_attn8_sptarg_50k_ft/final \
  --output_json f_finetune/output_inf_ft_50k/h9x_attn8_sptarg_held.json \
  --val_pct 0.05 --test_pct 0.05 \
  --batch 8 --max_tokens 128 --quant 4bit \
  --log_name h9x_attn8_sptarg_held \
  --wandb_project paraphrx_inference

# - Full-parameter run
python $RUN_SCRIPT \
  --data_path a_data/alpaca/50k_phrxed.json \
  --base_model_path f_finetune/outputs/8x_notarg_50k_ft/final \
  --output_json f_finetune/output_inf_ft_50k/8x_notarg_held.json \
  --val_pct 0.05 --test_pct 0.05 \
  --batch 8 --max_tokens 128 \
  --log_name 8x_notarg_held \
  --wandb_project paraphrx_inference

The output file is a list of dicts like:
    {
      "prompt_count": 17,
      "input": "",
      "instruction_original": "<answer text>",
      "instruct_polite_request": "<answer text>",
      ...
    }
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
from typing import Dict, List, Tuple

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Split helper 
def three_way_split(
    ds: Dataset, *, val_pct: float, test_pct: float, seed: int
) -> Tuple[set[int], set[int], set[int]]:
    """Group-wise split guaranteeing each prompt_count stays in one split."""
    import numpy as np

    rng = np.random.default_rng(seed)
    pcs = list({int(ex["prompt_count"]) for ex in ds})
    rng.shuffle(pcs)
    n = len(pcs)
    n_val = int(n * val_pct)
    n_test = int(n * test_pct)
    val_ids = set(pcs[:n_val])
    test_ids = set(pcs[n_val : n_val + n_test])
    train_ids = set(pcs[n_val + n_test :])
    return train_ids, val_ids, test_ids


# Loading prompts + RESUME merge
def load_examples_with_resume(
    data_path: str,
    instruct_types: List[str] | None,
    val_pct: float,
    test_pct: float,
    seed: int,
    split: str,
    max_samples: int,
    from_prompt_id: int,
    upto_prompt_id: int,
    resume_json_path: str | None,
) -> Tuple[List[Tuple[int, str, str, str]], Dict[int, Dict], int]:
    """
    Returns:
        flat_queue - list of (prompt_count, key_name, prompt_text, raw_input)
                     containing ONLY the missing/empty keys to generate.
        results_map - dict keyed by prompt_count with merged existing results:
                      {"prompt_count", "input", <answer fields...>}
        prefilled_answers - count of answers taken from resume_json_path
    """
    # Load prompts JSON
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Build Dataset for split, exactly as in training
    ds = Dataset.from_list(raw_data)
    train_ids, val_ids, test_ids = three_way_split(
        ds, val_pct=val_pct, test_pct=test_pct, seed=seed
    )
    if split == "val":
        keep_ids = val_ids
    elif split == "test":
        keep_ids = test_ids
    else:  # held-out = val ∪ test
        keep_ids = val_ids | test_ids

    # Filter by prompt_count ID ranges
    sorted_keep_ids = sorted(list(keep_ids))

    if from_prompt_id > 0:
        sorted_keep_ids = [pc for pc in sorted_keep_ids if pc >= from_prompt_id]
        logging.info(
            "Filtering from prompt_count >= %d, %d IDs remain.",
            from_prompt_id,
            len(sorted_keep_ids),
        )

    if upto_prompt_id > 0:
        sorted_keep_ids = [pc for pc in sorted_keep_ids if pc <= upto_prompt_id]
        logging.info(
            "Filtering up to prompt_count <= %d, %d IDs remain.",
            upto_prompt_id,
            len(sorted_keep_ids),
        )

    if max_samples > 0:
        sorted_keep_ids = sorted_keep_ids[:max_samples]
        logging.info(
            "Applying max_samples=%d, final group count is %d.",
            max_samples,
            len(sorted_keep_ids),
        )

    keep_ids = set(sorted_keep_ids)

    # Init results_map with prompt_count + input for all kept IDs
    results_map: Dict[int, Dict] = {}
    for item in raw_data:
        pc = int(item["prompt_count"])
        if pc not in keep_ids:
            continue
        if pc not in results_map:
            results_map[pc] = {"prompt_count": pc, "input": item.get("input", "")}

    # Optionally prefill from existing output_json (RESUME)
    prefilled_answers = 0
    resume_seen = set()
    if resume_json_path and Path(resume_json_path).exists():
        try:
            with open(resume_json_path, "r", encoding="utf-8") as rf:
                existing = json.load(rf)
            if isinstance(existing, list):
                for rec in existing:
                    if not isinstance(rec, dict) or "prompt_count" not in rec:
                        continue
                    pc = int(rec["prompt_count"])
                    resume_seen.add(pc)
                    if pc not in keep_ids:
                        # The old file might contain entries from a different split/config.
                        # We ignore those but keep going.
                        continue
                    # Ensure record exists
                    if pc not in results_map:
                        results_map[pc] = {
                            "prompt_count": pc,
                            "input": rec.get("input", ""),
                        }
                    # Merge non-empty answers
                    for k, v in rec.items():
                        if k in {"prompt_count", "input"}:
                            # prefer dataset input if differs
                            if k == "input" and results_map[pc].get("input", "") == "":
                                results_map[pc]["input"] = v
                            continue
                        if isinstance(v, str) and v.strip():
                            # keep existing non-empty answer
                            if k not in results_map[pc] or not str(
                                results_map[pc][k]
                            ).strip():
                                results_map[pc][k] = v
                                prefilled_answers += 1
            else:
                logging.warning(
                    "Existing output file is not a list; ignoring resume data."
                )
        except Exception as e:
            logging.warning(
                "Failed to read/merge existing output_json (%s): %s",
                resume_json_path,
                e,
            )

    # Build the generation queue of MISSING (or empty) keys
    flat_queue: List[Tuple[int, str, str, str]] = []
    missing_keys_counter: Dict[str, int] = {}

    # Normalise instruct_types once
    explicit_types = None
    if instruct_types:
        # Ensure "instruction_original" is included and first
        explicit_types = ["instruction_original"] + [
            k for k in instruct_types if k != "instruction_original"
        ]

    held_groups = 0
    for item in raw_data:
        pc = int(item["prompt_count"])
        if pc not in keep_ids:
            continue
        held_groups += 1
        raw_input = item.get("input", "")

        # Decide which instruction keys to keep for THIS item
        if explicit_types:
            keep_keys = explicit_types
        else:
            # Default: original + all instruct_* keys present in the item
            keep_keys = ["instruction_original"] + [
                k for k in item.keys() if k.startswith("instruct_")
            ]

        # De-duplicate while preserving order
        seen = set()
        keep_keys = [k for k in keep_keys if not (k in seen or seen.add(k))]

        # Ensure results_map has the group scaffold
        if pc not in results_map:
            results_map[pc] = {"prompt_count": pc, "input": raw_input}

        # Queue only missing/empty answers
        for key in keep_keys:
            instr = item.get(key)
            if not instr:
                missing_keys_counter[key] = missing_keys_counter.get(key, 0) + 1
                continue

            existing_text = results_map[pc].get(key, "")
            if isinstance(existing_text, str) and existing_text.strip():
                # Already done; skip
                continue

            flat_queue.append((pc, key, instr, raw_input))

    logging.info(
        "Loaded %d held-out groups (val∪test) | prompts to generate now: %d | prefilled answers: %d",
        held_groups,
        len(flat_queue),
        prefilled_answers,
    )
    if resume_json_path and Path(resume_json_path).exists():
        # Quick sanity: if existing file had prompt_counts outside (val∪test), tell the user
        extra = sorted(pc for pc in resume_seen if pc not in keep_ids)
        if extra:
            logging.warning(
                "Existing output contained %d prompt_count IDs NOT in this run's held-out split (ignored). Example: %s",
                len(extra),
                extra[:10],
            )

    if missing_keys_counter:
        logging.warning(
            "Some items were missing paraphrase keys (count by key): %s",
            missing_keys_counter,
        )

    return flat_queue, results_map, prefilled_answers


# Prompt formatting (MUST match training)
def build_chat_prompt(tokenizer, instruction: str, inp: str = "") -> str:
    user_msg = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    messages = [{"role": "user", "content": user_msg}]
    # Must match training script exactly.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Held-out inference for paraphrx fine-tuning (with resume + reporting)"
    )
    p.add_argument("--data_path", required=True, help="JSON file used in fine-tuning")
    p.add_argument(
        "--instruct_types",
        nargs="+",
        default=[],
        help="Optional explicit list of instruct_* keys to use (default = ALL)",
    )
    p.add_argument("--val_pct", type=float, default=0.05, help="Must match training")
    p.add_argument("--test_pct", type=float, default=0.05, help="Must match training")
    p.add_argument("--seed", type=int, default=42, help="Must match training")
    p.add_argument(
        "--split",
        choices=["val", "test", "heldout"],
        default="heldout",
        help="'val', 'test', or both (heldout)",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Process at most this many prompt_count groups (0 = all)",
    )
    p.add_argument(
        "--from_prompt_id",
        type=int,
        default=0,
        help="Start processing from this prompt_count ID (inclusive)",
    )
    p.add_argument(
        "--upto_prompt_id",
        type=int,
        default=0,
        help="Process up to this prompt_count ID (inclusive, 0 = no limit)",
    )

    p.add_argument("--base_model_path", required=True)
    p.add_argument("--lora_path", help="Path to LoRA adapter; omit for full-FT")
    p.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge adapter into base weights for faster inference",
    )

    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")

    p.add_argument(
        "--queue_order",
        choices=["length", "id"],
        default="length",
        help="Order to process the missing prompts: by 'length' (default) or by 'id'.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Exit after building the queue and print a summary of what would be generated.",
    )

    p.add_argument(
        "--output_json",
        required=True,
        help="Where to write merged results (also used for resume if exists)",
    )
    p.add_argument("--wandb_project", default="paraphrx_50k_inf_ft")
    p.add_argument(
        "--log_name",
        help="Unique name for this inference run (used in log filename and wandb)",
        default=None,
    )
    p.add_argument("--save_every", type=int, default=100, help="Save answers every X batches")
    return p.parse_args()


# Main
def main() -> None:
    args = parse_args()

    # Logging
    Path("logs").mkdir(exist_ok=True)
    log_name = args.log_name or Path(args.base_model_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"infer_{log_name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.info("CLI Args:\n%s", json.dumps(vars(args), indent=2))

    # Optional W&B
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
            logging.warning("W&B init failed (%s) - continuing without logging", e)
            wb_run = None
    else:
        wb_run = None

    # Data (+ RESUME merge)
    flat_queue, results_map, prefilled_answers = load_examples_with_resume(
        args.data_path,
        instruct_types=args.instruct_types,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        seed=args.seed,
        split=args.split,
        max_samples=args.max_samples,
        from_prompt_id=args.from_prompt_id,
        upto_prompt_id=args.upto_prompt_id,
        resume_json_path=args.output_json,
    )

    kept_group_ids = sorted(results_map.keys())
    run_summary = {
        "groups_kept": len(results_map),
        "first_group_ids": kept_group_ids[:10],
        "last_group_ids": kept_group_ids[-10:],
        "prompts_pending_now": len(flat_queue),
        "prefilled_answers_from_resume": prefilled_answers,
        "resume_file_used": args.output_json,
        "split": args.split,
        "seed": args.seed,
        "queue_order": args.queue_order,
    }
    logging.info("RUN SUMMARY:\n%s", json.dumps(run_summary, indent=2))
    print(json.dumps(run_summary, indent=2))

    if args.dry_run:
        # Show a few IDs that would be processed
        example_ids = sorted({pc for (pc, _, _, _) in flat_queue})[:10]
        example_keys = sorted({key for (_, key, _, _) in flat_queue})[:10]
        dry = {
            "example_prompt_count_ids_to_process": example_ids,
            "example_instruction_keys_to_process": example_keys,
            "note": "Dry-run requested, exiting before model load.",
        }
        logging.info("DRY RUN DETAILS:\n%s", json.dumps(dry, indent=2))
        print(json.dumps(dry, indent=2))
        # Still write canonical sorted output to keep resume pipeline deterministic.
        out_items = sorted(results_map.values(), key=lambda d: d["prompt_count"])
        Path(args.output_json).write_text(
            json.dumps(out_items, indent=2, ensure_ascii=False)
        )
        if wb_run:
            wb_run.finish()
        return

    if not flat_queue:
        msg = "Nothing to do: all held-out prompts already completed (or no prompts in split)."
        logging.info(msg)
        print(json.dumps({"status": msg, "groups_kept": len(results_map)}, indent=2))
        # Still write a canonical, sorted output for reproducibility
        out_items = sorted(results_map.values(), key=lambda d: d["prompt_count"])
        Path(args.output_json).write_text(
            json.dumps(out_items, indent=2, ensure_ascii=False)
        )
        if wb_run:
            wb_run.finish()
        return

    if wb_run:
        try:
            wb_run.config.update({"total_prompts_pending": len(flat_queue)}, allow_val_change=True)
        except Exception:
            pass

    # Choose processing order
    if args.queue_order == "length":
        flat_queue.sort(key=lambda t: len(t[2]))  # shortest instruction first
    else:  # "id"
        flat_queue.sort(key=lambda t: (t[0], t[1]))  # (prompt_count, key)

    # Model & Tokeniser
    model_kwargs: dict = dict(device_map=args.device)

    # Flash Attention 2 (if available and not explicitly disabled)
    try:
        import importlib

        _FLASH2_OK = importlib.util.find_spec("flash_attn") is not None
    except Exception:
        _FLASH2_OK = False
    if os.getenv("DISABLE_FLASH_ATTN", "0") == "1":
        _FLASH2_OK = False
    if _FLASH2_OK:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        logging.info("Flash-Attention 2 not available - using standard attention")

    # Quantisation (bitsandbytes)
    try:
        import importlib

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
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, **model_kwargs
    )

    # LoRA?
    try:
        import importlib

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
            logging.info("Merge done - adapter dropped")
    else:
        model = base_model

    model.eval()
    try:
        model = torch.compile(model)  # PT 2.x minor speed-up
    except Exception:
        pass

    tok_path = (
        args.lora_path
        if args.lora_path
        and (Path(args.lora_path) / "tokenizer_config.json").exists()
        else args.base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=4096)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generation helpers
    def _save_partial():
        out_items = sorted(results_map.values(), key=lambda d: d["prompt_count"])
        Path(args.output_json).write_text(
            json.dumps(out_items, indent=2, ensure_ascii=False)
        )
        if wb_run:
            try:
                wb_run.log(
                    {"completed_groups": len([d for d in out_items if len(d) > 2])}
                )
            except Exception:
                pass

    def _handler(sig_num, _frame):
        logging.info("Signal %s caught - saving partial results", sig_num)
        _save_partial()
        sys.exit(0)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _handler)

    _INFER_CTX = getattr(torch, "inference_mode", torch.no_grad)

    SAVE_EVERY_N_BATCHES = (
        args.save_every
        if args.save_every > 0
        else (len(flat_queue) + args.batch - 1) // args.batch
    )

    logging.info(
        "Starting inference loop with %d missing prompts", len(flat_queue)
    )
    logging.info(
        "Batch size: %d, Max tokens: %d, Temperature: %.2f",
        args.batch,
        args.max_tokens,
        args.temperature,
    )

    # New detailed counters
    generated_now = 0
    echo_skips = 0
    per_key_counts: Dict[str, int] = {}
    batches_done = 0

    for start in tqdm(range(0, len(flat_queue), args.batch), desc="generating"):
        batch = flat_queue[start : start + args.batch]
        pcs, keys, instrs, inputs = zip(*batch)
        prompts = [build_chat_prompt(tokenizer, i, inp) for i, inp in zip(instrs, inputs)]

        tokenised = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(model.device)

        # True lengths for right-padding
        input_lens = tokenised["attention_mask"].sum(dim=1)

        # Generation config
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

        # Post-process
        for i in range(len(batch)):
            answer_ids = outputs[i, input_lens[i] :]
            text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

            # Robust cleanup: strip echoed user content / role marker if present
            original_user_message = (
                instrs[i] if not inputs[i] else f"{instrs[i]}\n\nInput:\n{inputs[i]}"
            )
            if text.strip() == original_user_message.strip():
                logging.warning(
                    "Empty generation for prompt_count %s-%s (echoed prompt).",
                    pcs[i],
                    keys[i],
                )
                text = ""
                echo_skips += 1
            else:
                model_turn_marker = "model\n"
                marker_pos = text.find(model_turn_marker)
                if marker_pos != -1:
                    text = text[marker_pos + len(model_turn_marker) :].lstrip()

            # Merge into results_map (do NOT overwrite non-empty existing text)
            existing_text = results_map[pcs[i]].get(keys[i], "")
            if isinstance(existing_text, str) and existing_text.strip():
                # Already had a non-empty answer (from resume); keep it.
                continue

            results_map[pcs[i]][keys[i]] = text
            if text:
                generated_now += 1
                per_key_counts[keys[i]] = per_key_counts.get(keys[i], 0) + 1

        # housekeeping
        del tokenised, outputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        batches_done += 1
        if (start + len(batch)) % 100 == 0:
            logging.info("Processed %d / %d prompts", start + len(batch), len(flat_queue))

        if (batches_done % SAVE_EVERY_N_BATCHES) == 0:
            logging.info("Saving partial results after %d batches", batches_done)
            _save_partial()

    _save_partial()
    logging.info(
        "Finished - wrote %d groups → %s", len(results_map), args.output_json
    )
    print(
        f"Saved {len(results_map)} prompt_count groups → {args.output_json}"
    )

    # W&B artifact
    if wb_run:
        try:
            import wandb

            art = wandb.Artifact(
                name=f"generations_{Path(args.base_model_path).stem}",
                type="inference-results",
                metadata={
                    "num_records": len(results_map),
                    "split": args.split,
                    "model": Path(args.base_model_path).name,
                },
            )
            art.add_file(str(args.output_json))
            art.add_file(str(log_path))
            wb_run.log_artifact(art)
            wb_run.finish()
        except Exception as e:
            logging.warning("W&B artifact upload failed: %s", e)

    # Final check for missing generations
    missing_any = [k for k, v in results_map.items() if len(v) < 3]  # only prompt_count+input present
    if missing_any:
        logging.warning("Some groups have missing generations: %s", missing_any[:10])

    final_report = {
        "groups_saved": len(results_map),
        "answers_generated_now": generated_now,
        "empty_or_echoed": echo_skips,
        "per_key_generated": per_key_counts,
        "output_path": args.output_json,
        "missing_groups_example": missing_any[:10],
    }
    logging.info("FINAL REPORT:\n%s", json.dumps(final_report, indent=2))
    print(json.dumps(final_report, indent=2))
    logging.info("All done.")


if __name__ == "__main__":
    main()
