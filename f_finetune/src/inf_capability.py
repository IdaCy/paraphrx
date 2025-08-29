#!/usr/bin/env python3
"""
Inference on the held-out split (val + test) for paraphrase-robust fine-tuning
with automatic RESUME support

Now also supports fetching+running several popular eval datasets online:
  - MMLU (hendrycks_test), with optional subject filtering and a --mmlu-moral shortcut
  - GSM8K (math word problems)
  - DailyDialog (light conversational continuation)

Each selected dataset is saved into its own JSON file, with the SAME schema as the
original output (list of dicts containing "prompt_count", "input", "instruction_*" answers).
Resume/merge works per-file, just like the original.

If a dataset fetch fails, it is skipped with a logged warning.

Examples:
  # Original flow (unchanged)
  python run_infer.py --data_path a_data/alpaca/50k_phrxed.json --base_model_path ... --output_json out/alpaca_held.json ...

  # Benchmarks (each writes out/<stem>.{gsm8k|mmlu|dailydialog}.json)
  python run_infer.py --gsm8k --mmlu --daily_dialog --max_samples 200 --output_json out/bench.json --base_model_path ...

  # MMLU moral subsets only
  python run_infer.py --mmlu-moral --output_json out/mmlu.json --base_model_path ...

Notes:
  - --max_samples limits EACH selected dataset.
  - When *any* benchmark flags are used, --data_path becomes optional. If no
    benchmark flags are used, --data_path is required (backward compatible).
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
from typing import Dict, List, Tuple, Iterable, Optional

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset, DownloadConfig
import datasets
from pathlib import Path
import shutil

# Split helper (MUST match training)
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


# Loading prompts + RESUME merge (original flow)
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
) -> Tuple[List[Tuple[int, str, str, str]], Dict[int, Dict]]:
    """
    Returns:
        flat_queue - list of (prompt_count, key_name, prompt_text, raw_input)
                     containing ONLY the missing/empty keys to generate.
        results_map - dict keyed by prompt_count with merged existing results:
                      {"prompt_count", "input", <answer fields...>}
    """
    # Load prompts JSON
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    return _build_queue_and_resume(
        raw_data=raw_data,
        instruct_types=instruct_types,
        split_mode=True,
        val_pct=val_pct,
        test_pct=test_pct,
        seed=seed,
        split=split,
        max_samples=max_samples,
        from_prompt_id=from_prompt_id,
        upto_prompt_id=upto_prompt_id,
        resume_json_path=resume_json_path,
    )


# NEW: Generic builder used by both original and benchmark flows
def _build_queue_and_resume(
    raw_data: List[Dict],
    instruct_types: Optional[List[str]],
    *,
    split_mode: bool,
    val_pct: float = 0.0,
    test_pct: float = 0.0,
    seed: int = 42,
    split: str = "heldout",
    max_samples: int = 0,
    from_prompt_id: int = 0,
    upto_prompt_id: int = 0,
    resume_json_path: Optional[str] = None,
) -> Tuple[List[Tuple[int, str, str, str]], Dict[int, Dict]]:
    """
    Build the generation queue and results_map with resume-aware merge.

    If split_mode is False (benchmark datasets), we don't do three-way split;
    all items are considered part of the "held-out" set, filtered only by
    from_prompt_id / upto_prompt_id / max_samples.
    """

    ds = Dataset.from_list(raw_data)

    if split_mode:
        train_ids, val_ids, test_ids = three_way_split(
            ds, val_pct=val_pct, test_pct=test_pct, seed=seed
        )
        if split == "val":
            keep_ids = val_ids
        elif split == "test":
            keep_ids = test_ids
        else:  # held-out = val ∪ test
            keep_ids = val_ids | test_ids
        sorted_keep_ids = sorted(list(keep_ids))
    else:
        # Benchmarks: just use all prompt_count IDs present
        sorted_keep_ids = sorted({int(ex["prompt_count"]) for ex in raw_data})

    # Filter by prompt_count ranges
    if from_prompt_id > 0:
        sorted_keep_ids = [pc for pc in sorted_keep_ids if pc >= from_prompt_id]
        logging.info("Filtering from prompt_count >= %d, %d IDs remain.", from_prompt_id, len(sorted_keep_ids))

    if upto_prompt_id > 0:
        sorted_keep_ids = [pc for pc in sorted_keep_ids if pc <= upto_prompt_id]
        logging.info("Filtering up to prompt_count <= %d, %d IDs remain.", upto_prompt_id, len(sorted_keep_ids))

    if max_samples > 0:
        sorted_keep_ids = sorted_keep_ids[:max_samples]
        logging.info("Applying max_samples=%d, final group count is %d.", max_samples, len(sorted_keep_ids))

    keep_ids = set(sorted_keep_ids)

    # Init results_map scaffold
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
                        continue
                    if pc not in results_map:
                        results_map[pc] = {"prompt_count": pc, "input": rec.get("input", "")}
                    for k, v in rec.items():
                        if k in {"prompt_count", "input"}:
                            if k == "input" and results_map[pc].get("input", "") == "":
                                results_map[pc]["input"] = v
                            continue
                        if isinstance(v, str) and v.strip():
                            if k not in results_map[pc] or not str(results_map[pc][k]).strip():
                                results_map[pc][k] = v
                                prefilled_answers += 1
            else:
                logging.warning("Existing output file is not a list; ignoring resume data.")
        except Exception as e:
            logging.warning("Failed to read/merge existing output_json (%s): %s", resume_json_path, e)

    # Build the generation queue of MISSING (or empty) keys
    flat_queue: List[Tuple[int, str, str, str]] = []
    missing_keys_counter: Dict[str, int] = {}

    explicit_types = None
    if instruct_types:
        explicit_types = ["instruction_original"] + [k for k in instruct_types if k != "instruction_original"]

    held_groups = 0
    for item in raw_data:
        pc = int(item["prompt_count"])
        if pc not in keep_ids:
            continue
        held_groups += 1
        raw_input = item.get("input", "")

        if explicit_types:
            keep_keys = explicit_types
        else:
            keep_keys = ["instruction_original"] + [k for k in item.keys() if k.startswith("instruct_")]

        seen = set()
        keep_keys = [k for k in keep_keys if not (k in seen or seen.add(k))]

        if pc not in results_map:
            results_map[pc] = {"prompt_count": pc, "input": raw_input}

        for key in keep_keys:
            instr = item.get(key)
            if not instr:
                missing_keys_counter[key] = missing_keys_counter.get(key, 0) + 1
                continue

            existing_text = results_map[pc].get(key, "")
            if isinstance(existing_text, str) and existing_text.strip():
                continue

            flat_queue.append((pc, key, instr, raw_input))

    logging.info(
        "Prepared %d groups | prompts to generate now: %d | prefilled answers: %d",
        held_groups, len(flat_queue), prefilled_answers
    )
    if resume_json_path and Path(resume_json_path).exists():
        extra = sorted(pc for pc in resume_seen if pc not in keep_ids)
        if extra:
            logging.warning("Existing output contained %d prompt_count IDs NOT in this run (ignored). Example: %s",
                            len(extra), extra[:10])

    if missing_keys_counter:
        logging.warning("Some items were missing paraphrase keys (count by key): %s", missing_keys_counter)

    return flat_queue, results_map


# Prompt formatting (MUST match training)
def build_chat_prompt(tokenizer, instruction: str, inp: str = "") -> str:
    user_msg = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    messages = [{"role": "user", "content": user_msg}]
    # Must match training script exactly.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# NEW: Benchmark dataset loaders
def _as_alpaca_like(items: Iterable[Dict]) -> List[Dict]:
    """
    Normalize a sequence of dicts that at least have:
      - 'instruction_original' (str)
      - optional 'input' (str)
    And assign a sequential 'prompt_count' starting from 1.
    """
    out = []
    for i, ex in enumerate(items, start=1):
        out.append({
            "prompt_count": i,
            "input": ex.get("input", ""),
            "instruction_original": ex["instruction_original"],
            **{k: v for k, v in ex.items() if k.startswith("instruct_")},
        })
    return out


def load_gsm8k(max_samples: int) -> Optional[List[Dict]]:
    """
    Returns Alpaca-like list with instruction_original and empty input.
    """
    try:
        ds = load_dataset("gsm8k", "main", split="test")
    except Exception as e:
        logging.warning("GSM8K fetch failed: %s (skipping)", e)
        return None

    items: List[Dict] = []
    for row in ds:
        problem = str(row.get("question", "")).strip()
        if not problem:
            continue
        instr = (
            "Solve the following math word problem. "
            "Show clear reasoning and finish with a line like: Answer: <number>."
        )
        items.append({"input": "", "instruction_original": f"{instr}\n\nProblem:\n{problem}"})
        if max_samples and len(items) >= max_samples:
            break
    return _as_alpaca_like(items)


def load_mmlu(subjects: Optional[List[str]], max_samples: int) -> Optional[List[Dict]]:
    """
    Robust MMLU loader:
      - Try hendrycks_test with config=<subject>, split='test'
      - Fallback to lukaemon/mmlu with config=<subject>, split='test' then 'dev'
      - Fallback to cais/mmlu with config=<subject>, split='test' then 'validation'
      - Final fallback: 'all' config where available, then filter by subject column
    """
    default_subjects = [
        "abstract_algebra","anatomy","astronomy","business_ethics","clinical_knowledge",
        "college_biology","college_chemistry","college_computer_science","college_mathematics",
        "college_medicine","college_physics","computer_security","conceptual_physics",
        "econometrics","electrical_engineering","elementary_mathematics","formal_logic",
        "global_facts","high_school_biology","high_school_chemistry","high_school_computer_science",
        "high_school_european_history","high_school_geography","high_school_government_and_politics",
        "high_school_macroeconomics","high_school_mathematics","high_school_microeconomics",
        "high_school_physics","high_school_psychology","high_school_statistics","high_school_world_history",
        "human_aging","machine_learning","management","marketing","medical_genetics","miscellaneous",
        "moral_disputes","moral_scenarios","nutrition","philosophy","prehistory","professional_accounting",
        "professional_law","professional_medicine","professional_psychology","public_relations",
        "security_studies","sociology","us_foreign_policy","virology","world_religions"
    ]
    subjects = subjects or default_subjects

    def _dlcfg():
        from datasets import DownloadConfig
        return DownloadConfig(max_retries=3, force_download=False, use_etag=True)

    def _as_items(rows) -> List[Dict]:
        items = []
        letters = "ABCD"
        for row in rows:
            q = str(row.get("question", "")).strip()
            if not q:
                continue
            ch = row.get("choices")
            if isinstance(ch, dict):
                ordered = [ch.get(k, "") for k in letters]
            else:
                ordered = list(ch or [])[:4]
            if not ordered or all(not str(x).strip() for x in ordered):
                continue
            choice_lines = [f"{letters[i]}. {str(ordered[i])}" for i in range(min(4, len(ordered)))]
            instr = "Answer the multiple-choice question by first giving ONLY the letter (A, B, C, or D), then a brief explanation."
            items.append({
                "input": "",
                "instruction_original": f"{instr}\n\nQuestion:\n{q}\n\nChoices:\n" + "\n".join(choice_lines)
            })
        return items

    all_items: List[Dict] = []
    cap = max_samples if max_samples > 0 else None

    def _add(rows):
        nonlocal all_items
        all_items.extend(_as_items(rows))
        if cap and len(all_items) >= cap:
            all_items = all_items[:cap]
            return True
        return False

    for subj in subjects:
        # 1) hendrycks_test
        try:
            ds = load_dataset("hendrycks_test", subj, split="test", download_config=_dlcfg(), trust_remote_code=True)
            if _add(ds): break
            continue
        except Exception as e:
            logging.warning("MMLU '%s' via hendrycks_test failed: %s", subj, e)

        # 2) lukaemon/mmlu (config=subj)
        for split_try in ("test", "dev"):
            try:
                ds = load_dataset("lukaemon/mmlu", subj, split=split_try, download_config=_dlcfg())
                if _add(ds): break
                if len(ds) > 0:  # got something for this subject
                    break
            except Exception as e:
                logging.warning("MMLU '%s' via lukaemon/mmlu split=%s failed: %s", subj, split_try, e)
        if cap and len(all_items) >= cap: break

        # 3) cais/mmlu (config=subj)
        for split_try in ("test", "validation"):
            try:
                ds = load_dataset("cais/mmlu", subj, split=split_try, download_config=_dlcfg())
                if _add(ds): break
                if len(ds) > 0:
                    break
            except Exception as e:
                logging.warning("MMLU '%s' via cais/mmlu split=%s failed: %s", subj, split_try, e)
        if cap and len(all_items) >= cap: break

        # 4) Final: try 'all' config and filter
        for repo, split_try in (("cais/mmlu", "test"), ("cais/mmlu", "validation")):
            try:
                ds_all = load_dataset(repo, "all", split=split_try, download_config=_dlcfg())
                ds_subj = ds_all.filter(lambda r: r.get("subject") == subj)
                if _add(ds_subj): break
                if len(ds_subj) > 0: break
            except Exception as e:
                logging.warning("MMLU '%s' final fallback %s split=%s failed: %s", subj, repo, split_try, e)
        if cap and len(all_items) >= cap: break

    return _as_alpaca_like(all_items) if all_items else None


def load_mmlu_moral(max_samples: int) -> Optional[List[Dict]]:
    return load_mmlu(subjects=["moral_disputes", "moral_scenarios"], max_samples=max_samples)


def load_daily_dialog(max_samples: int, turns: int = 4) -> Optional[List[Dict]]:
    """
    Robust DailyDialog loader with precise cache cleanup and a safe fallback to UltraChat.
    """
    import datasets, shutil
    from datasets import DownloadConfig

    def _as_items(ds_iter) -> List[Dict]:
        items = []
        for row in ds_iter:
            dialog = row.get("dialog") or row.get("dialogue") or row.get("utterances")
            if not dialog:
                continue
            def _to_text(u):
                return u.get("text") if isinstance(u, dict) else u
            dialog_txt = [t for t in map(_to_text, dialog) if isinstance(t, str)]
            if not dialog_txt:
                continue
            context = dialog_txt[-turns:] if len(dialog_txt) >= turns else dialog_txt
            context_text = "\n".join(f"{'A' if i%2==0 else 'B'}: {u}" for i, u in enumerate(context))
            instr = "Given the conversation so far, write the next helpful, natural-sounding reply as one short paragraph."
            items.append({"input": "", "instruction_original": f"{instr}\n\nConversation:\n{context_text}"})
            if max_samples and len(items) >= max_samples:
                break
        return items

    def _try_daily_dialog(force_download=False, streaming=False):
        kwargs = {}
        if force_download:
            kwargs["download_config"] = DownloadConfig(max_retries=3, force_download=True)
        if streaming:
            kwargs["streaming"] = True
        return load_dataset("daily_dialog", split="test", **kwargs)

    # 1) normal attempt
    try:
        ds = _try_daily_dialog(force_download=False, streaming=False)
        return _as_alpaca_like(_as_items(ds))
    except Exception as e:
        logging.warning("DailyDialog load failed: %s", e)

    # 2) clear ONLY DailyDialog cache precisely
    try:
        cache_root = Path(datasets.config.HF_DATASETS_CACHE)
        # Typical structure: <cache_root>/daily_dialog/default/*/*
        for p in cache_root.rglob("daily_dialog"):
            shutil.rmtree(p.parent, ignore_errors=True)  # remove "daily_dialog/default/..."
    except Exception as ee:
        logging.warning("Failed to remove DailyDialog cache: %s", ee)

    # 3) retry clean download
    try:
        ds = _try_daily_dialog(force_download=True, streaming=False)
        return _as_alpaca_like(_as_items(ds))
    except Exception as e:
        logging.warning("DailyDialog re-download failed: %s", e)

    # 4) streaming fallback
    try:
        ds = _try_daily_dialog(force_download=False, streaming=True)
        return _as_alpaca_like(_as_items(ds))
    except Exception as e:
        logging.warning("DailyDialog streaming failed: %s", e)

    # 5) LAST RESORT: small conversational fallback from UltraChat (guarantees you get *something*)
    try:
        uc = load_dataset("HuggingFaceH4/ultrachat_200k", "test_sft", split="test_sft", streaming=True)
        items = []
        for ex in uc:
            # ex["messages"] is list of {"role": "user"/"assistant", "content": "..."}
            msgs = ex.get("messages") or []
            if not msgs:
                continue
            # take last few alternating turns as context; ask for next assistant reply
            hist = [m["content"] for m in msgs[-(2*turns):]]
            context_text = "\n".join(f"{'User' if i%2==0 else 'Assistant'}: {t}" for i, t in enumerate(hist))
            instr = "Continue the chat. Write the next Assistant reply that is helpful and concise."
            items.append({"input": "", "instruction_original": f"{instr}\n\nChat so far:\n{context_text}"})
            if max_samples and len(items) >= max_samples:
                break
        return _as_alpaca_like(items) if items else None
    except Exception as e:
        logging.warning("UltraChat fallback failed: %s", e)
        return None


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Held-out inference for paraphrx fine-tuning (with resume) + benchmarks")

    # Original flow
    p.add_argument("--data_path", help="JSON file used in fine-tuning (required if no benchmark flags are used)")
    p.add_argument("--instruct_types", nargs="+", default=[], help="Optional explicit list of instruct_* keys to use (default = ALL)")
    p.add_argument("--val_pct", type=float, default=0.05, help="Must match training")
    p.add_argument("--test_pct", type=float, default=0.05, help="Must match training")
    p.add_argument("--seed", type=int, default=42, help="Must match training")
    p.add_argument("--split", choices=["val", "test", "heldout"], default="heldout", help="'val', 'test', or both (heldout)")
    p.add_argument("--max_samples", type=int, default=0, help="Process at most this many groups per dataset (0 = all)")
    p.add_argument("--from_prompt_id", type=int, default=0, help="Start processing from this prompt_count ID (inclusive)")
    p.add_argument("--upto_prompt_id", type=int, default=0, help="Process up to this prompt_count ID (inclusive, 0 = no limit)")

    # Model
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--lora_path", help="Path to LoRA adapter; omit for full-FT")
    p.add_argument("--merge_lora", action="store_true", help="Merge adapter into base weights for faster inference")

    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")

    # Output & logging
    p.add_argument("--output_json", required=True, help="If multiple datasets selected, this is treated as a base path; files get suffixed like <stem>.<dataset>.json")
    p.add_argument("--wandb_project", default="paraphrx_50k_inf_ft")
    p.add_argument("--log_name", help="Unique name for this inference run (used in log filename and wandb)", default=None)
    p.add_argument("--save_every", type=int, default=100, help="Save answers every X batches")

    # NEW: benchmark toggles
    p.add_argument("--gsm8k", action="store_true", help="Run GSM8K (math)")
    p.add_argument("--mmlu", action="store_true", help="Run MMLU (hendrycks_test)")
    p.add_argument("--mmlu_moral", action="store_true", help="Shortcut for MMLU moral subjects only")
    p.add_argument("--mmlu_subjects", nargs="*", default=[], help="Specific MMLU subjects (space-separated). If omitted with --mmlu, uses a broad default list.")
    p.add_argument("--daily_dialog", action="store_true", help="Run a light conversational continuation task from DailyDialog")

    return p.parse_args()


# Main
def main() -> None:
    args = parse_args()

    # Determine mode
    benchmarks = []
    if args.gsm8k:
        benchmarks.append("gsm8k")
    if args.mmlu or args.mmlu_moral:
        benchmarks.append("mmlu_moral" if args.mmlu_moral else "mmlu")
    if args.daily_dialog:
        benchmarks.append("dailydialog")

    # Validate required args
    if not benchmarks and not args.data_path:
        print("ERROR: --data_path is required if no benchmark flags are provided (e.g., --gsm8k/--mmlu/--daily_dialog).", file=sys.stderr)
        sys.exit(2)

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

    # -------------------------
    # Model & Tokeniser
    # -------------------------
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
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **model_kwargs)

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
        if args.lora_path and (Path(args.lora_path) / "tokenizer_config.json").exists()
        else args.base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=4096)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # Helpers shared by all flows
    # -------------------------
    def _save_partial(results_map: Dict[int, Dict], out_path: Path, completed_key_name: str = None):
        out_items = sorted(results_map.values(), key=lambda d: d["prompt_count"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_items, indent=2, ensure_ascii=False))
        if wb_run:
            try:
                done_groups = len([d for d in out_items if len(d) > 2])
                wb_run.log({
                    "completed_groups": done_groups,
                    **({completed_key_name: done_groups} if completed_key_name else {})
                })
            except Exception:
                pass

    def _run_generation(flat_queue: List[Tuple[int, str, str, str]], results_map: Dict[int, Dict], out_path: Path, save_every: int):
        if not flat_queue:
            # Still write a canonical, sorted output for reproducibility
            _save_partial(results_map, out_path)
            print(f"Saved {len(results_map)} prompt_count groups → {out_path}")
            return

        # Sort shortest → longest to maximise batch utilisation
        flat_queue.sort(key=lambda t: len(t[2]))

        _INFER_CTX = getattr(torch, "inference_mode", torch.no_grad)
        SAVE_EVERY_N_BATCHES = save_every if save_every > 0 else (len(flat_queue) + args.batch - 1) // args.batch

        logging.info("Starting inference loop with %d missing prompts", len(flat_queue))
        logging.info("Batch size: %d, Max tokens: %d, Temperature: %.2f", args.batch, args.max_tokens, args.temperature)

        batches_done = 0

        def _handler(sig_num, _frame):
            logging.info("Signal %s caught - saving partial results", sig_num)
            _save_partial(results_map, out_path)
            sys.exit(0)

        for _sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(_sig, _handler)

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
                original_user_message = instrs[i] if not inputs[i] else f"{instrs[i]}\n\nInput:\n{inputs[i]}"
                if text.strip() == original_user_message.strip():
                    logging.warning("Empty generation for prompt_count %s-%s (echoed prompt).", pcs[i], keys[i])
                    text = ""
                else:
                    model_turn_marker = "model\n"
                    marker_pos = text.find(model_turn_marker)
                    if marker_pos != -1:
                        text = text[marker_pos + len(model_turn_marker) :].lstrip()

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
                logging.info("Processed %d / %d prompts", start + len(batch), len(flat_queue))

            if (batches_done % SAVE_EVERY_N_BATCHES) == 0:
                logging.info("Saving partial results after %d batches", batches_done)
                _save_partial(results_map, out_path)

        _save_partial(results_map, out_path)
        logging.info("Finished - wrote %d groups → %s", len(results_map), out_path)
        print(f"Saved {len(results_map)} prompt_count groups → {out_path}")

    # -------------------------
    # Decide outputs
    # -------------------------
    out_base = Path(args.output_json)
    if out_base.suffix.lower() == ".json":
        stem = out_base.with_suffix("")  # e.g., out/bench
    else:
        # treat as prefix directory or bare stem
        stem = out_base

    def suffixed(name: str) -> Path:
        # e.g., out/bench.gsm8k.json
        return stem.parent / f"{stem.name}.{name}.json"

    # -------------------------
    # Run either original flow or benchmarks (or both)
    # -------------------------
    any_ran = False

    # 1) Original flow if data_path is supplied (runs exactly as before)
    if args.data_path:
        any_ran = True
        flat_queue, results_map = load_examples_with_resume(
            args.data_path,
            instruct_types=args.instruct_types,
            val_pct=args.val_pct,
            test_pct=args.test_pct,
            seed=args.seed,
            split=args.split,
            max_samples=args.max_samples,
            from_prompt_id=args.from_prompt_id,
            upto_prompt_id=args.upto_prompt_id,
            resume_json_path=str(out_base),
        )
        _run_generation(flat_queue, results_map, out_base, args.save_every)

        # W&B artifact (optional)
        if wb_run:
            try:
                import wandb
                art = wandb.Artifact(
                    name=f"generations_{Path(args.base_model_path).stem}_heldout",
                    type="inference-results",
                    metadata={
                        "num_records": len(results_map),
                        "split": args.split,
                        "model": Path(args.base_model_path).name,
                    },
                )
                art.add_file(str(out_base))
                art.add_file(str(log_path))
                wb_run.log_artifact(art)
            except Exception as e:
                logging.warning("W&B artifact upload failed: %s", e)

    # 2) Benchmarks
    def process_benchmark(dataset_name: str, raw_items: Optional[List[Dict]]):
        if raw_items is None or len(raw_items) == 0:
            logging.warning("No items for %s (skipping)", dataset_name)
            return
        out_path = suffixed(dataset_name)
        logging.info("Preparing benchmark '%s' → %s", dataset_name, out_path)
        flat_queue, results_map = _build_queue_and_resume(
            raw_data=raw_items,
            instruct_types=args.instruct_types,
            split_mode=False,
            max_samples=args.max_samples,  # already applied in loaders too; harmless to re-trim
            from_prompt_id=args.from_prompt_id,
            upto_prompt_id=args.upto_prompt_id,
            resume_json_path=str(out_path),
        )
        _run_generation(flat_queue, results_map, out_path, args.save_every)

        if wb_run:
            try:
                import wandb
                art = wandb.Artifact(
                    name=f"generations_{Path(args.base_model_path).stem}_{dataset_name}",
                    type="inference-results",
                    metadata={"num_records": len(results_map), "dataset": dataset_name},
                )
                art.add_file(str(out_path))
                wb_run.log_artifact(art)
            except Exception as e:
                logging.warning("W&B artifact upload failed for %s: %s", dataset_name, e)

    if "gsm8k" in benchmarks:
        any_ran = True
        process_benchmark("gsm8k", load_gsm8k(args.max_samples))

    if "mmlu_moral" in benchmarks:
        any_ran = True
        process_benchmark("mmlu_moral", load_mmlu_moral(args.max_samples))

    if "mmlu" in benchmarks:
        any_ran = True
        subj_list = args.mmlu_subjects if args.mmlu_subjects else None
        process_benchmark("mmlu", load_mmlu(subj_list, args.max_samples))

    if "dailydialog" in benchmarks:
        any_ran = True
        process_benchmark("dailydialog", load_daily_dialog(args.max_samples, turns=4))

    if wb_run:
        wb_run.finish()

    if not any_ran:
        logging.info("Nothing to do.")
        print("Nothing to do (no data_path and no benchmark flags).")

    # Final friendly check is per-dataset inside _run_generation (written files).
    logging.info("All done.")


if __name__ == "__main__":
    main()
