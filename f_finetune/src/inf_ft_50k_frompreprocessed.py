#!/usr/bin/env python3
"""
Inference on the held-out split (val + test) for paraphrase-robust fine-tuning.

allows
--from_prompt_id and --upto_prompt_id to limit the range of prompt_count IDs processed

# - LoRA run (run #1)
python $RUN_SCRIPT \
  --data_path a_data/alpaca/50k_phrxed.json \
  --base_model_path f_finetune/model \
  --lora_path f_finetune/outputs/h9x_attn8_sptarg_50k_ft/final \
  --output_json f_finetune/output_inf_ft_50k/h9x_attn8_sptarg_held.json \
  --val_pct 0.05 --test_pct 0.05 \
  --batch 8 --max_tokens 128 --quant 4bit \
  --log_name h9x_attn8_sptarg_held \
  --wandb_project paraphrx_inference

# - Full-parameter run (run #2 - no LoRA)
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
One entry per prompt_count covering every prompt phrasing
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
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_chat_prompt(tokenizer, instruction: str, inp: str = "") -> str:
    user_msg = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# taking path to the tokenized data for held-out prompts
def load_examples(
    data_path: str,
    tokenized_data_path: str,
    instruct_types: List[str] | None,
    max_samples: int,
    from_prompt_id: int,
    upto_prompt_id: int,
) -> Tuple[List[Tuple[int, str, str, str]], Dict[int, Dict]]:
    """
    Loads examples from the raw data file that are part of the held-out
    set defined in the tokenized data directory

    Returns:
        flat_queue - list of (prompt_count, key_name, prompt_text, raw_input)
        results_map - dict keyed by prompt_count with {"prompt_count", "input"}
    """
    # Load the pre-processed dataset to find the held-out prompt IDs
    logging.info(f"Loading split information from {tokenized_data_path}...")
    try:
        tokenized_datasets = load_from_disk(tokenized_data_path)
    except FileNotFoundError:
        logging.error(f"Tokenized data not found at {tokenized_data_path}. This is required to identify the held-out set.")
        sys.exit(1)

    if "test" not in tokenized_datasets:
        logging.error(f"The dataset at {tokenized_data_path} does not contain a 'test' split.")
        sys.exit(1)

    # is definitive set of held-out prompt group IDs
    keep_ids = set(tokenized_datasets["test"]["prompt_count"])
    logging.info(f"Identified {len(keep_ids)} prompt groups in the 'test' split.")

    # Apply prompt_count ID filtering
    original_id_count = len(keep_ids)
    if from_prompt_id > 0:
        keep_ids = {pid for pid in keep_ids if pid >= from_prompt_id}
    if upto_prompt_id > 0:
        keep_ids = {pid for pid in keep_ids if pid <= upto_prompt_id}
    
    if len(keep_ids) < original_id_count:
        logging.info(f"Filtered prompt IDs from {original_id_count} down to {len(keep_ids)} based on --from_prompt_id/--upto_prompt_id.")


    # Load the raw JSON to get the original text
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if max_samples > 0:
        keep_ids = set(sorted(list(keep_ids))[:max_samples])
        logging.info(f"Subsampling to a maximum of {len(keep_ids)} prompt groups.")

    flat_queue: List[Tuple[int, str, str, str]] = []
    results_map: Dict[int, Dict] = {}
    missing_keys_counter: Dict[str, int] = {}

    # Filter the raw data using the held-out IDs
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

        processed_keys = set()
        for key in keep_keys:
            if key in processed_keys:
                continue
            processed_keys.add(key)
            
            instr = item.get(key)
            if not instr:
                missing_keys_counter[key] = missing_keys_counter.get(key, 0) + 1
                continue
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


# argument parser is updated to reflect new data loading strategy
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Held-out inference for paraphrx fine-tuning")
    # Paths
    p.add_argument("--data_path", required=True, help="Path to the original, raw JSON file (e.g., 50k_phrxed.json).")
    p.add_argument("--tokenized_data_path", required=True, help="Path to the pre-tokenized dataset directory created by the preprocessing script. Used to identify the held-out set.")
    p.add_argument("--base_model_path", required=True, help="Path to the fine-tuned model directory.")
    p.add_argument("--lora_path", help="Path to LoRA adapter; omit for full-FT.")
    p.add_argument("--output_json", required=True)

    # Data Selection
    p.add_argument("--instruct_types", nargs="+", default=[], help="Optional explicit list of instruct_* keys to use (default = ALL found in source).")
    p.add_argument("--max_samples", type=int, default=0, help="Process at most this many prompt_count groups from the held-out set (0 = all).")
    p.add_argument("--from_prompt_id", type=int, default=0, help="Process prompt_count IDs starting from this value (inclusive).")
    p.add_argument("--upto_prompt_id", type=int, default=0, help="Process prompt_count IDs up to this value (inclusive).")


    # Generation Config
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)

    # Hardware & Model Config
    p.add_argument("--device", default="auto")
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    p.add_argument("--merge_lora", action="store_true", help="Merge adapter into base weights for faster inference")

    # Logging
    p.add_argument("--wandb_project", default="paraphrx_50k_inf_ft")
    p.add_argument("--log_name", help="A unique name for this inference run (used in log filename and wandb)", default=None)
    p.add_argument("--save_every", type=int, default=100, help="Save answers every X steps")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Setup & Logging
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

    if args.wandb_project:
        import wandb
        wb_run = wandb.init(project=args.wandb_project, name=f"infer_{log_name}", job_type="inference", config=vars(args))
    else:
        wb_run = None

    # Data Loading
    flat_queue, results_map = load_examples(
        args.data_path,
        args.tokenized_data_path,
        instruct_types=args.instruct_types,
        max_samples=args.max_samples,
        from_prompt_id=args.from_prompt_id,
        upto_prompt_id=args.upto_prompt_id,
    )
    
    # LOGIC TO DETECT AND SKIP ALREADY-COMPLETED PROMPTS
    completed_tasks = set()
    if os.path.exists(args.output_json):
        logging.info(f"Output file found at {args.output_json}. Loading existing results to resume.")
        try:
            with open(args.output_json, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            # Re-populate results_map and identify completed tasks
            for item in existing_results:
                pc = item.get("prompt_count")
                if pc is None: continue
                
                # Ensure the entry exists in our current results_map
                if pc not in results_map:
                    results_map[pc] = {"prompt_count": pc, "input": item.get("input", "")}
                
                for key, value in item.items():
                    if key not in ["prompt_count", "input"]:
                        completed_tasks.add((pc, key))
                        results_map[pc][key] = value

            logging.info(f"Loaded {len(existing_results)} existing prompt groups with {len(completed_tasks)} individual completions.")
            
            # Filter the flat_queue to remove completed tasks
            original_queue_size = len(flat_queue)
            flat_queue = [item for item in flat_queue if (item[0], item[1]) not in completed_tasks]
            logging.info(f"Removed {original_queue_size - len(flat_queue)} already completed tasks from the queue.")

        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Could not load or parse existing output file: {e}. Starting from scratch.")
    
    if not flat_queue:
        logging.error("No prompts to process - check data paths, filters, or if all work is already done!")
        sys.exit(0) # Changed to exit 0, as this is not an error if work is just done

    if wb_run:
        wb_run.config.update({"total_prompts": len(flat_queue)}, allow_val_change=True)

    flat_queue.sort(key=lambda t: len(t[2]))

    #   MODEL & TOKENISER
    model_kwargs: dict = dict(device_map=args.device)
    _FLASH2_OK = False
    try:
        import importlib
        if importlib.util.find_spec("flash_attn"): _FLASH2_OK = True
    except ImportError: pass
    if os.getenv("DISABLE_FLASH_ATTN", "0") == "1": _FLASH2_OK = False
    if _FLASH2_OK: model_kwargs["attn_implementation"] = "flash_attention_2"
    else: logging.info("Flash-Attention 2 not available - using standard attention")

    _BNB_OK = bool(importlib.util.find_spec("bitsandbytes"))
    if args.quant != "none" and not _BNB_OK:
        logging.warning("bitsandbytes not available - falling back to bf16")
        args.quant = "none"

    if args.quant == "none":
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=args.quant == "8bit", load_in_4bit=args.quant == "4bit",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logging.info("Loading base model from %s", args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **model_kwargs)

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

    tok_path = (args.lora_path if args.lora_path and (Path(args.lora_path) / "tokenizer_config.json").exists() else args.base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=4096)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    def save_partial():
        out_items = sorted(results_map.values(), key=lambda d: d["prompt_count"])
        Path(args.output_json).write_text(json.dumps(out_items, indent=2, ensure_ascii=False))
        if wb_run: wb_run.log({"completed": len([d for d in out_items if len(d) > 2])})
    def _handler(sig_num, _frame):
        logging.info("Signal %s caught - saving partial results", sig_num)
        save_partial()
        sys.exit(0)
    for _sig in (signal.SIGINT, signal.SIGTERM): signal.signal(_sig, _handler)

    _INFER_CTX = getattr(torch, "inference_mode", torch.no_grad)


    SAVE_EVERY_N_BATCHES = args.save_every
    if SAVE_EVERY_N_BATCHES <= 0:
        SAVE_EVERY_N_BATCHES = len(flat_queue)  # Save only once at the end

    logging.info("Starting inference loop with %d prompts", len(flat_queue))
    logging.info("Batch size: %d, Max tokens: %d, Temperature: %.2f", args.batch, args.max_tokens, args.temperature)

    for start in tqdm(range(0, len(flat_queue), args.batch), desc="generating"):
        batch = flat_queue[start : start + args.batch]
        pcs, keys, instrs, inputs = zip(*batch)
        prompts = [build_chat_prompt(tokenizer, i, inp) for i, inp in zip(instrs, inputs)]
        tokenised = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length).to(model.device)
        input_lens = tokenised["attention_mask"].sum(dim=1)
        gen_cfg = dict(max_new_tokens=args.max_tokens, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=args.temperature > 0, temperature=args.temperature if args.temperature > 0 else None, top_p = 0.6 if args.temperature > 0 else None)
        with _INFER_CTX():
            outputs = model.generate(**tokenised, **gen_cfg)
        for i in range(len(batch)):
            answer_ids = outputs[i, input_lens[i] :]
            text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            model_turn_marker = "model\n"
            marker_pos = text.find(model_turn_marker)
            if marker_pos != -1: text = text[marker_pos + len(model_turn_marker) :].lstrip()
            results_map[pcs[i]][keys[i]] = text
        del tokenised, outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        if (start // args.batch + 1) % SAVE_EVERY_N_BATCHES == 0:
            logging.info("Saving partial results after %d batches", (start // args.batch + 1))
            save_partial()

    save_partial()
    logging.info("Finished - wrote %d groups → %s", len(results_map), args.output_json)

    if wb_run:
        import wandb
        art = wandb.Artifact(name=f"generations_{Path(args.base_model_path).stem}", type="inference-results", metadata={"num_records": len(results_map), "model": Path(args.base_model_path).name})
        art.add_file(str(args.output_json))
        art.add_file(str(log_path))
        wb_run.log_artifact(art)
        wb_run.finish()

    logging.info("All done.")

if __name__ == "__main__":
    main()
