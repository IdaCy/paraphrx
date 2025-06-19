#!/usr/bin/env python3
"""
python a_data/preproc/downloads/download_mmlu.py --output-dir a_data/mmlu

Requires:  pip install datasets tqdm
"""

import argparse
import json
import os
from typing import List, Optional

from datasets import get_dataset_config_names, load_dataset


def dump_subject(subject: str, output_dir: str, splits: List[str],
                 indent: Optional[int] = 2) -> None:
    """Download one subject, attach prompt_count, and save to JSON."""
    examples = []
    prompt_count = 1

    for split in splits:
        try:
            ds = load_dataset("cais/mmlu", name=subject, split=split,
                              trust_remote_code=True)
        except Exception:
            # Some subjects don’t provide every split – skip missing ones
            continue

        for ex in ds:
            record = {**ex, "prompt_count": prompt_count, "split": split}
            examples.append(record)
            prompt_count += 1

    outfile = os.path.join(output_dir, f"{subject}.json")
    with open(outfile, "w", encoding="utf-8") as fp:
        json.dump(examples, fp, ensure_ascii=False, indent=indent)

    print(f"Saved {len(examples):>5} records → {outfile}")


def main(output_dir: str, splits: Optional[List[str]] = None,
         indent: Optional[int] = 2) -> None:
    if splits is None:
        splits = ["auxiliary_train", "dev", "val", "test"]

    os.makedirs(output_dir, exist_ok=True)

    subjects = get_dataset_config_names("cais/mmlu")
    print(f"Discovered {len(subjects)} subjects.")

    for subject in subjects:
        print(f"Processing: {subject:>30}")
        dump_subject(subject, output_dir, splits, indent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MMLU and store per‑subject JSON files.")
    parser.add_argument("--output-dir", "-o", default="mmlu_json",
                        help="Directory to hold JSON files.")
    parser.add_argument(
        "--splits",
        "-s",
        nargs="+",
        help="Which splits to include (default: auxiliary_train dev val test)."
    )
    parser.add_argument("--no-indent",
                        action="store_true",
                        help="Write compact JSON without newlines.")

    args = parser.parse_args()
    indent = None if args.no_indent else 2

    main(args.output_dir, splits=args.splits, indent=indent)
