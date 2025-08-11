#!/usr/bin/env python
"""
preprocessing the data for the finetuning with LAP
- compatible with latest version of ft_lap_optim_continue
(continue was for allowing to go on with stopped checkpoints)
"""
import argparse
import dataclasses
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Union, Set, Tuple

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np

# Global variables
tokenizer = None

# Data Structures and utilities
@dataclasses.dataclass
class Example:
    prompt_count: int
    instruction: str
    inp: str
    answer: str
    style: str

# using Union[str, None] instead of `str | None`
# - syntax is understood by Python 3.9 and older
def build_chat_prompt(instruction: str, inp: Union[str, None] = "") -> str:
    user_msg = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    messages = [{"role": "user", "content": user_msg}]
    if tokenizer:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"

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
    # ! shuffling to happen on the group IDs (prompt_count), not on the individual examples before splitting.
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

def get_group_wise_split_ids(
    examples: List[Example], test_size: float, seed: int
) -> Tuple[Set[int], Set[int]]:
    """
    Splits data by 'prompt_count' to prevent leakage

    Args:
        examples: The list of all loaded Example objects.
        test_size: The fraction of groups to allocate to the test set.
        seed: The random seed for reproducibility.

    Returns:
        A tuple containing two sets: (train_prompt_ids, test_prompt_ids)
    """
    logging.info(f"Performing group-wise split with test_size={test_size} and seed={seed}.")
    
    # Get all unique group identifiers
    all_prompt_ids = sorted(list({ex.prompt_count for ex in examples}))
    
    # Shuffle the group identifiers reproducibly
    rng = np.random.default_rng(seed)
    rng.shuffle(all_prompt_ids)
    
    # Determine the split index
    n_test = int(len(all_prompt_ids) * test_size)
    
    # Create the sets of identifiers
    test_ids = set(all_prompt_ids[:n_test])
    train_ids = set(all_prompt_ids[n_test:])
    
    logging.info(f"Split complete. Train groups: {len(train_ids)}, Test groups: {len(test_ids)}")
    return train_ids, test_ids

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
    
    # Load all examples first
    all_examples = load_examples(args.data_paths, args.instruct_types)
    logging.info(f"Loaded {len(all_examples)} total examples.")

    # 1. Get the IDs for each split based on prompt_count
    TEST_SET_SIZE = 0.05
    RANDOM_SEED = 42
    train_prompt_ids, test_prompt_ids = get_group_wise_split_ids(all_examples, test_size=TEST_SET_SIZE, seed=RANDOM_SEED)

    # 2. Create the two datasets based on the split IDs
    train_examples = [ex for ex in all_examples if ex.prompt_count in train_prompt_ids]
    test_examples = [ex for ex in all_examples if ex.prompt_count in test_prompt_ids]
    
    logging.info(f"Created train set with {len(train_examples)} examples and test set with {len(test_examples)} examples.")

    train_ds = Dataset.from_list([dataclasses.asdict(e) for e in train_examples])
    test_ds = Dataset.from_list([dataclasses.asdict(e) for e in test_examples])

    raw_datasets = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })
    
    # 3. Tokenize both datasets
    num_workers = min(8, os.cpu_count() or 1)
    logging.info(f"Tokenizing with {num_workers} processes...")
    
    tokenized_datasets = raw_datasets.map(
        tokenise_example,
        remove_columns=raw_datasets["train"].column_names,  # Use train columns as reference
        num_proc=num_workers,
        desc="Tokenizing dataset"
    )
    
    # 4. Filter empty examples from both splits
    final_datasets = tokenized_datasets.filter(lambda example: len(example['input_ids']) > 0)
    
    logging.info(f"Tokenization complete. Final dataset sizes: Train={len(final_datasets['train'])}, Test={len(final_datasets['test'])}")
    
    logging.info(f"Saving tokenized dataset to {output_path}")
    final_datasets.save_to_disk(str(output_path))
    logging.info("Preprocessing finished successfully.")

if __name__ == "__main__":
    main()
