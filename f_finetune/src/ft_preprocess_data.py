
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Env defaults to make tqdm/logging less spammy unless overridden
os.environ.setdefault("TQDM_MININTERVAL", "30")
os.environ.setdefault("TQDM_MINITER", "500")


@dataclasses.dataclass
class Example:
    prompt_count: int
    instruction: str
    inp: str
    answer: str
    style: str

    def to_prompt(self) -> str:
        # Must match the fine-tuning script's chat template & meta usage
        meta = "Answer concisely and directly. Focus on task semantics; ignore stylistic tone cues. End after the answer."
        user_core = self.instruction if not self.inp else f"{self.instruction}\n\nInput:\n{self.inp}"
        user_msg = f"{meta}\n\n{user_core}"
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def three_way_split(
    ds: Dataset, *, val_pct: float = 0.05, test_pct: float = 0.05, seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Group-wise split guaranteeing each prompt_count appears in exactly one split.
    IMPORTANT: Ordering aligns with fine-tuning script:
      - validation first fraction, then test, then train = rest
    """
    rng = np.random.default_rng(seed)
    pcs = list({ex["prompt_count"] for ex in ds})
    rng.shuffle(pcs)
    n = len(pcs)
    n_val = int(n * val_pct)
    n_test = int(n * test_pct)
    val_ids = set(pcs[:n_val])
    test_ids = set(pcs[n_val: n_val + n_test])
    train_ids = set(pcs[n_val + n_test:])

    train = ds.filter(lambda ex: ex["prompt_count"] in train_ids)
    val = ds.filter(lambda ex: ex["prompt_count"] in val_ids)
    test = ds.filter(lambda ex: ex["prompt_count"] in test_ids)
    return train, val, test


def load_examples(
    paths: List[str],
    instruct_types: List[str],
    use_para_ans: bool,
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

            # original instruction
            examples.append(
                Example(pc_id, item["instruction_original"], inp, base_ans, "instruction_original")
            )

            # paraphrases
            keep_keys = instruct_types if instruct_types else [k for k in item.keys() if k.startswith("instruct_")]
            for k in keep_keys:
                if k not in item: 
                    continue
                paraphrase = item.get(k, "")
                if not paraphrase:
                    continue
                ans = item.get("output_paraphrase", base_ans) if use_para_ans else base_ans
                examples.append(Example(pc_id, paraphrase, inp, ans, k))
    random.shuffle(examples)
    return examples


def load_examples_with_answers(
    prompts_path: str,
    answers_path: str,
    instruct_types: List[str],
    use_para_ans: bool,
) -> List[Example]:
    examples: List[Example] = []
    with open(prompts_path, "r", encoding="utf-8") as fh:
        prompts_data = json.load(fh)
    with open(answers_path, "r", encoding="utf-8") as fh:
        answers_data = json.load(fh)
    answers_map = {a["prompt_count"]: a for a in answers_data}

    for item in prompts_data:
        pc_id = item["prompt_count"]
        if pc_id not in answers_map:
            continue
        ans_rec = answers_map[pc_id]
        inp = item.get("input", "")
        # original instruction
        orig_inst = item["instruction_original"]
        orig_ans = ans_rec.get("instruction_original", "")
        if orig_inst and orig_ans:
            examples.append(Example(pc_id, orig_inst, inp, orig_ans, "instruction_original"))
        # paraphrases
        keep_keys = [k for k in (instruct_types or item.keys()) if str(k).startswith("instruct_") and k in item]
        for k in keep_keys:
            inst = item.get(k, "")
            if not inst:
                continue
            if use_para_ans:
                ans = ans_rec.get(k, ans_rec.get("instruction_original", ""))
            else:
                ans = ans_rec.get("instruction_original", "")
            if ans:
                examples.append(Example(pc_id, inst, inp, ans, k))
    random.shuffle(examples)
    return examples


def batch_tokenise_examples(examples: List[Example], batch_size: int, max_answer_tokens: int):
    """
    High-throughput batch tokenization that EXACTLY mirrors the fine-tuning script:
      - Build prompt with chat template + brevity meta.
      - Tokenize prompt with add_special_tokens=False.
      - Tokenize answer ONLY with truncation to --max_answer_tokens.
      - Append exactly one <end_of_turn> (fallback to eos) to the answer ids.
      - Labels mask the prompt ids as -100, answer ids are labels.
    """
    total = len(examples)
    input_ids_all = []
    labels_all = []
    styles_all = []
    pcs_all = []

    start = time.time()
    last_log = start

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = examples[start_idx:end_idx]

        # Prepare lists
        prompts_txt = [ex.to_prompt() for ex in batch]
        answers_txt = [ex.answer.strip() for ex in batch]

        # Tokenize in batch
        prompt_tok = tokenizer(prompts_txt, add_special_tokens=False)["input_ids"]
        answer_tok = tokenizer(
            answers_txt, add_special_tokens=False, truncation=True, max_length=max_answer_tokens
        )["input_ids"]

        # Compose input/labels
        for ex, p_ids, a_ids in zip(batch, prompt_tok, answer_tok):
            # exactly one EOT
            a_ids = list(a_ids) + [EOT_ID]
            ids = p_ids + a_ids
            labs = [-100] * len(p_ids) + a_ids
            input_ids_all.append(ids)
            labels_all.append(labs)
            styles_all.append(ex.style)
            pcs_all.append(ex.prompt_count)

        # Logging progress every batch
        now = time.time()
        if now - last_log >= 0.5 or end_idx == total:
            done = end_idx
            pct = 100.0 * done / total if total else 100.0
            rate = done / max(1e-6, (now - start))
            eta_s = (total - done) / max(1e-6, rate)
            logging.info(f"[tokenize] {done:,}/{total:,} ({pct:.1f}%) | {rate:.1f} ex/s | ETA {eta_s:.1f}s")
            last_log = now

    # Build tokenized Dataset
    tok_ds = Dataset.from_dict({
        "input_ids": input_ids_all,
        "labels": labels_all,
        "style": styles_all,
        "prompt_count": pcs_all,
    })
    return tok_ds


def main():
    ap = argparse.ArgumentParser("Fast tokenizer+splitter aligned with fine-tuning script")
    ap.add_argument("--prompts_path", nargs="+", required=True, help="One or more JSON prompt files (or shards)")
    ap.add_argument("--answers_path", default=None, help="Optional answers JSON to source targets from")
    ap.add_argument("--instruct_types", nargs="+", default=[], help="Restrict to these instruct_* keys (empty = all)")
    ap.add_argument("--use_paraphrase_answer", action="store_true", help="Use paraphrase-specific answers when available")

    ap.add_argument("--model_path", required=True, help="Tokenizer/model path to ensure identical chat template")
    ap.add_argument("--output_path", required=True, help="Directory for DatasetDict.save_to_disk")

    ap.add_argument("--val_pct", type=float, default=0.05)
    ap.add_argument("--test_pct", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_answer_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1024, help="Examples per tokenization batch")
    ap.add_argument("--log_file", default=None, help="Optional path to log file in addition to stdout")

    args = ap.parse_args()

    # Logging setup
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True
    )
    logging.info("Starting tokenizer-splitter (batched).")
    logging.info("Args:\n%s", json.dumps(vars(args), indent=2))

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Tokenizer init (fast)
    global tokenizer, EOT_ID
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # end-of-turn id (fallback to eos if token not known by tokenizer)
    tid = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    EOT_ID = tid if (tid is not None and tid != tokenizer.unk_token_id) else tokenizer.eos_token_id

    # Load examples from JSON(s)
    if args.answers_path:
        if len(args.prompts_path) != 1:
            raise ValueError("When --answers_path is provided, pass exactly one prompts JSON in --prompts_path.")
        examples = load_examples_with_answers(
            prompts_path=args.prompts_path[0],
            answers_path=args.answers_path,
            instruct_types=args.instruct_types,
            use_para_ans=args.use_paraphrase_answer,
        )
    else:
        examples = load_examples(
            paths=args.prompts_path,
            instruct_types=args.instruct_types,
            use_para_ans=args.use_paraphrase_answer,
        )

    logging.info("Loaded %d examples before split.", len(examples))

    # Build a raw Dataset with the minimal columns required for splitting
    raw = Dataset.from_list([{
        "prompt_count": ex.prompt_count,
        "instruction": ex.instruction,
        "inp": ex.inp,
        "answer": ex.answer,
        "style": ex.style,
    } for ex in examples])

    # Split exactly like fine-tuning (val first, then test, then train)
    train_raw, val_raw, test_raw = three_way_split(
        raw, val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed
    )
    logging.info("Split sizes (raw) — train: %d | val: %d | test: %d", len(train_raw), len(val_raw), len(test_raw))

    # Tokenize each split with batch processing & constant progress logging
    t0 = time.time()
    logging.info("Tokenizing TRAIN...")
    train_examples = [Example(pc, ins, inp, ans, sty) for pc, ins, inp, ans, sty in zip(
        train_raw["prompt_count"], train_raw["instruction"], train_raw["inp"], train_raw["answer"], train_raw["style"]
    )]
    train_tok = batch_tokenise_examples(train_examples, args.batch_size, args.max_answer_tokens)
    logging.info("TRAIN tokenized: %d examples.", len(train_tok))

    logging.info("Tokenizing VAL...")
    val_examples = [Example(pc, ins, inp, ans, sty) for pc, ins, inp, ans, sty in zip(
        val_raw["prompt_count"], val_raw["instruction"], val_raw["inp"], val_raw["answer"], val_raw["style"]
    )]
    val_tok = batch_tokenise_examples(val_examples, args.batch_size, args.max_answer_tokens)
    logging.info("VAL tokenized: %d examples.", len(val_tok))

    logging.info("Tokenizing TEST...")
    test_examples = [Example(pc, ins, inp, ans, sty) for pc, ins, inp, ans, sty in zip(
        test_raw["prompt_count"], test_raw["instruction"], test_raw["inp"], test_raw["answer"], test_raw["style"]
    )]
    test_tok = batch_tokenise_examples(test_examples, args.batch_size, args.max_answer_tokens)
    logging.info("TEST tokenized: %d examples.", len(test_tok))

    total_time = time.time() - t0
    total_ex = len(train_tok) + len(val_tok) + len(test_tok)
    logging.info("All splits tokenized. Total: %d examples in %.1fs (%.1f ex/s)",
                 total_ex, total_time, total_ex / max(1e-6, total_time))

    # Save DatasetDict to disk (load_from_disk compatible)
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    dsd = DatasetDict({
        "train": train_tok,
        "validation": val_tok,
        "test": test_tok,
    })
    dsd.save_to_disk(out_dir.as_posix())
    logging.info("Saved tokenized DatasetDict to %s", out_dir)


if __name__ == "__main__":
    main()
