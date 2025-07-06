#!/usr/bin/env python3
"""
python a_data/preproc/downloads/download_gsm8k.py \
    --output-dir a_data/gsm8k
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from datasets import get_dataset_config_names, load_dataset

DATASET_NAME = "gsm8k"  # HuggingFace hub ID
DEFAULT_SPLITS = ["train", "validation"]


def dump_config(cfg: str, output_dir: str, splits: List[str], indent: Optional[int] = 2) -> None:  # noqa: D401
    """Download *one* configuration, attach prompt_count, and save to JSON"""

    examples: list[dict] = []
    prompt_count = 1

    for split in splits:
        try:
            ds = load_dataset(DATASET_NAME, name=cfg, split=split)
        except Exception:
            # Some configurations may miss a split – skip it gracefully
            continue

        for ex in ds:
            record = {**ex, "prompt_count": prompt_count, "split": split}
            examples.append(record)
            prompt_count += 1

    outfile = os.path.join(output_dir, f"{cfg}.json")
    with open(outfile, "w", encoding="utf-8") as fp:
        json.dump(examples, fp, ensure_ascii=False, indent=indent)

    print(f"Saved {len(examples):>5} records -> {outfile}")


def main(output_dir: str, configs: Optional[List[str]] = None, splits: Optional[List[str]] = None, *, indent: Optional[int] = 2) -> None:
    # Resolve defaults.
    if configs is None:
        configs = get_dataset_config_names(DATASET_NAME)
    if splits is None:
        splits = DEFAULT_SPLITS

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\nConfigs       : {configs}\nSplits        : {splits}\n")

    for cfg in configs:
        print(f"Processing: {cfg:>10}")
        dump_config(cfg, output_dir, splits, indent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GSM8K and store per‑configuration JSON files.")
    parser.add_argument("--output-dir", "-o", default="gsm8k_json", help="Directory to hold JSON files.")
    parser.add_argument("--configs", "-c", nargs="+", help="Specific configurations to fetch (default: all).")
    parser.add_argument("--splits", "-s", nargs="+", help="Which splits to include (default: train validation).")
    parser.add_argument("--no-indent", action="store_true", help="Write compact JSON (no newlines/indentation).")

    args = parser.parse_args()
    indent: Optional[int] = None if args.no_indent else 2

    main(args.output_dir, configs=args.configs, splits=args.splits, indent=indent)
