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
    return examples

# --- START OF THE ONLY FUNCTIONAL CHANGE ---
# The function signature's return type is updated, and the return statement is modified.
def tokenise_example(example: Dict[str, Any]) -> Dict[str, Any]:
    MAX_TOTAL_LENGTH = 512
    prompt_ids = tokenizer(build_chat_prompt(example['instruction'], example['inp']), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(example['answer'], add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) + 1 > MAX_TOTAL_LENGTH:
        answer_ids = answer_ids[:MAX_TOTAL_LENGTH - len(prompt_ids) - 1]
    answer_ids.append(tokenizer.eos_token_id)
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    
    # THE FIX: This now returns the original columns we need to keep, solving the bug.
    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_count": example["prompt_count"],
        "style": example["style"]
    }
# --- END OF THE ONLY FUNCTIONAL CHANGE ---


def get_group_wise_split_ids(
    examples: List[Example], val_size: float, test_size: float, seed: int
) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Splits data by 'prompt_count' into train, validation, and test sets.

    Args:
        examples: The list of all loaded Example objects.
        val_size: The fraction of groups to allocate to the validation set.
        test_size: The fraction of groups to allocate to the test set.
        seed: The random seed for reproducibility.

    Returns:
        A tuple containing three sets: (train_prompt_ids, val_prompt_ids, test_prompt_ids)
    """
    logging.info(f"Performing group-wise split with val_size={val_size}, test_size={test_size}, and seed={seed}.")
    
    all_prompt_ids = sorted(list({ex.prompt_count for ex in examples}))
    
    rng = np.random.default_rng(seed)
    rng.shuffle(all_prompt_ids)
    
    # Determine the split indices
    n_test = int(len(all_prompt_ids) * test_size)
    n_val = int(len(all_prompt_ids) * val_size)
    
    # Create the sets of identifiers for all three splits
    test_ids = set(all_prompt_ids[:n_test])
    val_ids = set(all_prompt_ids[n_test : n_test + n_val])
    train_ids = set(all_prompt_ids[n_test + n_val:])
    
    logging.info(f"Split complete. Train groups: {len(train_ids)}, Validation groups: {len(val_ids)}, Test groups: {len(test_ids)}")
    return train_ids, val_ids, test_ids

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
        # NOTE: This safety feature means you MUST delete the old output directory
        # manually before re-running this script.
        logging.warning(f"Output directory {output_path} already exists. Skipping.")
        return

    logging.info("Starting data preprocessing.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    
    all_examples = load_examples(args.data_paths, args.instruct_types)
    logging.info(f"Loaded {len(all_examples)} total examples.")

    # 1. Define sizes for both validation and test sets. 90/5/5 is a standard split.
    VALIDATION_SET_SIZE = 0.05
    TEST_SET_SIZE = 0.05
    RANDOM_SEED = 42
    
    # Unpack all three sets of IDs from the updated function
    train_prompt_ids, val_prompt_ids, test_prompt_ids = get_group_wise_split_ids(
        all_examples, val_size=VALIDATION_SET_SIZE, test_size=TEST_SET_SIZE, seed=RANDOM_SEED
    )

    # 2. Create three datasets based on the split IDs
    train_examples = [ex for ex in all_examples if ex.prompt_count in train_prompt_ids]
    val_examples = [ex for ex in all_examples if ex.prompt_count in val_prompt_ids]
    test_examples = [ex for ex in all_examples if ex.prompt_count in test_prompt_ids]
    
    # Update logging to show all three set sizes
    logging.info(f"Created train set with {len(train_examples)}, validation set with {len(val_examples)}, and test set with {len(test_examples)} examples.")

    train_ds = Dataset.from_list([dataclasses.asdict(e) for e in train_examples])
    val_ds = Dataset.from_list([dataclasses.asdict(e) for e in val_examples])
    test_ds = Dataset.from_list([dataclasses.asdict(e) for e in test_examples])

    # Create the DatasetDict with the 'validation' key included
    raw_datasets = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })
    
    # 3. Tokenize all datasets
    #num_workers = min(8, os.cpu_count() or 1)
    num_workers = max(1, (os.cpu_count() or 1) - 2)
    # num_workers = max(1, (os.cpu_count() or 1) - 2)  # use all, reserve 2
    # num_workers = max(1, (os.cpu_count() or 1) // 2) # reserve 50%
    logging.info(f"Tokenizing with {num_workers} processes...")
    
    tokenized_datasets = raw_datasets.map(
            tokenise_example,
            # This line is now correct, as it runs *after* tokenise_example has preserved the other columns.
            remove_columns=["instruction", "inp", "answer"],
            num_proc=num_workers,
            desc="Tokenizing dataset"
        )
    
    # 4. Filter empty examples
    final_datasets = tokenized_datasets.filter(lambda example: len(example['input_ids']) > 0)
    
    # Update final logging to include the validation set size
    logging.info(f"Tokenization complete. Final dataset sizes: Train={len(final_datasets['train'])}, Validation={len(final_datasets['validation'])}, Test={len(final_datasets['test'])}")
    
    logging.info(f"Saving tokenized dataset to {output_path}")
    final_datasets.save_to_disk(str(output_path))

    # --- START OF NEW VERIFICATION BLOCK ---
    logging.info("\n" + "="*80)
    logging.info("STARTING POST-PROCESSING VERIFICATION")
    logging.info("="*80)

    # Check 1: Final dataset structure and sizes
    logging.info(f"Final columns in dataset: {final_datasets['train'].column_names}")
    if "prompt_count" not in final_datasets['train'].column_names:
        logging.error("CRITICAL ERROR: 'prompt_count' column is MISSING from the final dataset.")
    else:
        logging.info("SUCCESS: 'prompt_count' column is present.")
    
    logging.info(f"Final example counts -> Train: {len(final_datasets['train'])}, Validation: {len(final_datasets['validation'])}, Test: {len(final_datasets['test'])}")

    # Check 2: Verify that there is no data leakage between splits
    logging.info("\nChecking for ID overlap between splits (data leakage)...")
    train_ids = set(final_datasets['train']['prompt_count'])
    val_ids = set(final_datasets['validation']['prompt_count'])
    test_ids = set(final_datasets['test']['prompt_count'])

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    if not train_val_overlap and not train_test_overlap and not val_test_overlap:
        logging.info("SUCCESS: No overlap found between train, validation, and test sets.")
    else:
        logging.error("CRITICAL ERROR: Overlap detected between splits! This is a data leak.")
        if train_val_overlap:
            logging.error(f"  - Train/Validation Overlap IDs: {train_val_overlap}")
        if train_test_overlap:
            logging.error(f"  - Train/Test Overlap IDs: {train_test_overlap}")
        if val_test_overlap:
            logging.error(f"  - Validation/Test Overlap IDs: {val_test_overlap}")
            
    logging.info("="*80)
    logging.info("VERIFICATION COMPLETE")
    logging.info("="*80 + "\n")
    # --- END OF NEW VERIFICATION BLOCK ---

    logging.info("Preprocessing finished successfully.")

if __name__ == "__main__":
    main()
