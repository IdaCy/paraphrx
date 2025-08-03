#!/usr/bin/env python3
"""
Inference on the held-out split (val + test) for paraphrase-robust fine-tuning.

Usage examples (same as before):

# - LoRA run (run #1)
python ft_inference_paraphrx.py \
  --data_path a_data/alpaca/50k_phrxed.json \
  --base_model_path f_finetune/model \
  --lora_path f_finetune/outputs/h9x_attn8_sptarg_50k_ft/final \
  --output_json f_finetune/output_inf_ft_50k/h9x_attn8_sptarg_held.json \
  --val_pct 0.05 --test_pct 0.05 \
  --batch 8 --max_tokens 128 --quant 4bit \
  --log_name h9x_attn8_sptarg_held \
  --wandb_project paraphrx_inference

# - Full-parameter run (run #2 - no LoRA)
python ft_inference_paraphrx.py \
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
One entry per prompt_count covering every prompt phrasing.
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

# ... (all imports and helper functions up to main() are correct and can remain unchanged) ...

# Data helpers - identical split logic to fine-tuning script
def three_way_split(
    ds: Dataset, *, val_pct: float, test_pct: float, seed: int
) -> Tuple[set[int], set[int], set[int]]:
    """Group-wise split guaranteeing each prompt_count stays in one split."""
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
    return train_ids, val_ids, test_ids


def load_examples(
    data_path: str,
    instruct_types: List[str] | None,
    val_pct: float,
    test_pct: float,
    seed: int,
    split: str,
    max_samples: int,
) -> Tuple[List[Tuple[int, str, str, str]], Dict[int, Dict]]:
    """
    Returns:
        flat_queue - list of (prompt_count, key_name, prompt_text, raw_input)
        results_map - dict keyed by prompt_count with {"prompt_count", "input"}
    """

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Convert to HF Dataset for the split helper
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

    if max_samples:
        keep_ids = set(sorted(keep_ids)[: max_samples])

    flat_queue: List[Tuple[int, str, str, str]] = []
    results_map: Dict[int, Dict] = {}
    missing_keys_counter: Dict[str, int] = {}

    # Use a set for faster checking of unique keys
    unique_instruct_types = set(instruct_types) if instruct_types else set()

    for item in raw_data:
        pc = int(item["prompt_count"])
        if pc not in keep_ids:
            continue

        raw_input = item.get("input", "")

        # record group once
        if pc not in results_map:
            results_map[pc] = {"prompt_count": pc, "input": raw_input}

        # Figure out which instruction keys to keep
        if instruct_types:
            # Use the provided list, but still ensure original is first
            keep_keys = ["instruction_original"] + [k for k in instruct_types if k != "instruction_original"]
        else:
            # Default: find all instruct_* keys and add original
            keep_keys = ["instruction_original"] + [k for k in item.keys() if k.startswith("instruct_")]

        # Ensure we don't process duplicate keys
        processed_keys = set()
        for key in keep_keys:
            if key in processed_keys:
                continue
            processed_keys.add(key)
            
            instr = item.get(key)
            if not instr:
                missing_keys_counter[key] = missing_keys_counter.get(key, 0) + 1
                continue
            # Build prompt later once we have the tokenizer
            flat_queue.append((pc, key, instr, raw_input))

    logging.info(
        "Loaded %d items | held-out groups: %d | flat prompts: %d",
        len(results_map),
        len(keep_ids),
        len(flat_queue),
    )
    if missing_keys_counter:
        logging.warning("Some items were missing paraphrase keys: %s", missing_keys_counter)

    return flat_queue, results_map


# Prompt formatting - chat style identical to training
def build_chat_prompt(tokenizer, instruction: str, inp: str = "") -> str:
    user_msg = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    messages = [{"role": "user", "content": user_msg}]
    # IMPORTANT: This must match the training script exactly.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_args() -> argparse.Namespace:
    # This function is correct, no changes needed.
    p = argparse.ArgumentParser(description="Held-out inference for paraphrx fine-tuning")
    p.add_argument("--data_path", required=True, help="JSON file used in fine-tuning")
    p.add_argument("--instruct_types", nargs="+", default=[], help="Optional explicit list of instruct_* keys to use (default = ALL)")
    p.add_argument("--val_pct", type=float, default=0.05, help="Must match training")
    p.add_argument("--test_pct", type=float, default=0.05, help="Must match training")
    p.add_argument("--seed", type=int, default=42, help="Must match training")
    p.add_argument("--split", choices=["val", "test", "heldout"], default="heldout", help="'val', 'test', or both (heldout)")
    p.add_argument("--max_samples", type=int, default=0, help="Process at most this many prompt_count groups (0 = all)")
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--lora_path", help="Path to LoRA adapter; omit for full-FT")
    p.add_argument("--merge_lora", action="store_true", help="Merge adapter into base weights for faster inference")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    p.add_argument("--output_json", required=True)
    p.add_argument("--wandb_project", default="paraphrx_50k_inf_ft")
    p.add_argument("--log_name", help="A unique name for this inference run (used in log filename and wandb)", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Setup, Data Loading, Model Loading ---
    # This part of your script is well-structured and correct. No changes are needed here.
    # ... (logging, wandb, data loading, model loading all as before) ...
    #   LOGGING
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
        import wandb

        wb_run = wandb.init(
            project=args.wandb_project,
            name=f"infer_{log_name}",
            job_type="inference",
            config=vars(args),
        )
    else:
        wb_run = None

    #   DATA
    flat_queue, results_map = load_examples(
        args.data_path,
        instruct_types=args.instruct_types,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        seed=args.seed,
        split=args.split,
        max_samples=args.max_samples,
    )
    if not flat_queue:
        logging.error("No prompts to process - check split/percentages!")
        sys.exit(1)

    if wb_run:
        wb_run.config.update({"total_prompts": len(flat_queue)}, allow_val_change=True)

    # Sort shortest → longest to maximise batch utilisation
    flat_queue.sort(key=lambda t: len(t[2]))

    #   MODEL & TOKENISER
    model_kwargs: dict = dict(device_map=args.device)
    # Check for Flash Attention
    _FLASH2_OK = False
    try:
        import importlib
        if importlib.util.find_spec("flash_attn"):
             _FLASH2_OK = True
    except ImportError:
        pass
    if os.getenv("DISABLE_FLASH_ATTN", "0") == "1":
        _FLASH2_OK = False
    
    if _FLASH2_OK:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        logging.info("Flash-Attention 2 not available - using standard attention")

    # Quant
    _BNB_OK = bool(importlib.util.find_spec("bitsandbytes"))
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

    # LoRA?
    _PEFT_OK = bool(importlib.util.find_spec("peft"))
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
        if args.lora_path and (Path(args.lora_path) / "tokenizer_config.json").exists()
        else args.base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=4096)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # We will use the default right-padding as it's simpler to get the slicing correct.

    #   GENERATION LOOP
    # ... (save_partial and _handler functions are correct) ...
    def save_partial():
        out_items = sorted(results_map.values(), key=lambda d: d["prompt_count"])
        Path(args.output_json).write_text(
            json.dumps(out_items, indent=2, ensure_ascii=False)
        )
        if wb_run:
            wb_run.log({"completed": len([d for d in out_items if len(d) > 2])})
    
    def _handler(sig_num, _frame):
        logging.info("Signal %s caught - saving partial results", sig_num)
        save_partial()
        sys.exit(0)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _handler)

    _INFER_CTX = getattr(torch, "inference_mode", torch.no_grad)

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

        # This logic is CORRECT for right-padding. It gets the true length of each prompt.
        input_lens = tokenised["attention_mask"].sum(dim=1)

        gen_cfg = dict(
            max_new_tokens=args.max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p = 0.6 if args.temperature > 0 else None, # Common practice for sampling
        )

        with _INFER_CTX():
            outputs = model.generate(**tokenised, **gen_cfg)

        # --- THIS IS THE CORRECTED SECTION ---
        for i in range(len(batch)):
            # This slicing is correct. It gets all tokens generated AFTER the prompt.
            answer_ids = outputs[i, input_lens[i] :]
            text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

            # --- ROBUST CLEANUP HEURISTIC ---
            # The model was prompted with a string ending in "...<start_of_turn>model\n".
            # Sometimes it repeats the user turn, or just starts with the model turn.
            # We build the original user message to robustly strip it if it's echoed.

            original_user_message = instrs[i] if not inputs[i] else f"{instrs[i]}\n\nInput:\n{inputs[i]}"
            
            # Check if the generated text is just an echo of the user message.
            # This can happen if the model gets very confused.
            if text.strip() == original_user_message.strip():
                 logging.warning(f"Empty generation for prompt_count {pcs[i]}-{keys[i]}, model echoed prompt.")
                 text = "" # Set to empty to indicate failure
            else:
                # A more common failure is echoing the prompt *then* answering.
                # We can find the model's turn marker and strip everything before it.
                # In Gemma, this is `<start_of_turn>model\n`, which decodes to `model\n`.
                model_turn_marker = "model\n"
                marker_pos = text.find(model_turn_marker)
                if marker_pos != -1:
                    text = text[marker_pos + len(model_turn_marker) :].lstrip()
            
            results_map[pcs[i]][keys[i]] = text

        # housekeeping
        del tokenised, outputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        if (start + len(batch)) % 100 == 0:
            logging.info(
                "Processed %d / %d prompts", start + len(batch), len(flat_queue)
            )

    save_partial()
    logging.info("Finished - wrote %d groups → %s", len(results_map), args.output_json)
    print(f"Saved {len(results_map)} prompt_count groups → {args.output_json}")

    # ... (W&B upload and summary can remain the same) ...
    if wb_run:
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

    missing_any = [k for k, v in results_map.items() if len(v) < 3]
    if missing_any:
        logging.warning("Some groups have missing generations: %s", missing_any[:10])
    logging.info("All done.")


if __name__ == "__main__":
    main()
