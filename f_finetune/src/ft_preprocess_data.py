import argparse
import dataclasses
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict, Any

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Data Structures and Utilities
@dataclasses.dataclass
class Example:
    prompt_count: int
    instruction: str
    inp: str
    answer: str
    style: str

def build_chat_prompt(instruction: str, inp: str | None = "") -> str:
    user_msg = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def load_examples(paths: List[str], instruct_types: List[str]) -> List[Example]:
    examples = []
    load_all = not instruct_types
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh: data = json.load(fh)
        for item in data:
            pc_id, inp, base_ans = item["prompt_count"], item.get("input", ""), item.get("output", "")
            if "instruction_original" in item:
                examples.append(Example(pc_id, item["instruction_original"], inp, base_ans, "instruction_original"))
            keys_to_process = [k for k in item if k.startswith("instruct_")] if load_all else instruct_types
            for k in keys_to_process:
                if k in item and item[k] and k != "instruction_original":
                    examples.append(Example(pc_id, item[k], inp, base_ans, k))
    random.shuffle(examples)
    return examples

def tokenise_example(example: Dict[str, Any]) -> Dict[str, List[int]]:
    MAX_TOTAL_LENGTH = 512
    prompt_ids = tokenizer(build_chat_prompt(example['instruction'], example['inp']), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(example['answer'], add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) + 1 > MAX_TOTAL_LENGTH:
        answer_ids = answer_ids[:MAX_TOTAL_LENGTH - len(prompt_ids) - 1]
    answer_ids.append(tokenizer.eos_token_id)
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "labels": labels}

def main():
    global tokenizer
    parser = argparse.ArgumentParser(description="Pre-tokenize the dataset for training.")
    parser.add_argument("--data_paths", nargs="+", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--instruct_types", nargs="+", default=[])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    output_path = Path(args.output_path)
    if os.path.exists(output_path):
        logging.warning(f"Output directory {output_path} already exists. Skipping.")
        return

    logging.info("Starting data preprocessing.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    
    examples = load_examples(args.data_paths, args.instruct_types)
    logging.info(f"Loaded {len(examples)} examples.")
    
    raw_ds = Dataset.from_list([dataclasses.asdict(e) for e in examples])
    
    num_workers = min(8, os.cpu_count() or 1)
    logging.info(f"Tokenizing with {num_workers} processes...")
    
    tokenized_ds = raw_ds.map(tokenise_example, remove_columns=raw_ds.column_names, num_proc=num_workers, desc="Tokenizing dataset")
    
    # Filter out any empty examples that might result from errors
    tokenized_ds = tokenized_ds.filter(lambda example: len(example['input_ids']) > 0)
    
    logging.info(f"Tokenization complete. Final dataset size: {len(tokenized_ds)}")
    
    final_datasets = tokenized_ds.train_test_split(test_size=0.05, seed=42)
    
    logging.info(f"Saving tokenized dataset to {output_path}")
    final_datasets.save_to_disk(str(output_path))
    logging.info("Preprocessing finished successfully.")

if __name__ == "__main__":
    main()
