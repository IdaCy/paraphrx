#!/usr/bin/env python3

"""
python e_eval/src/replace_equi_scores.py \
    --data e_eval/data/xx_mmlu_gemma-2-2b-it.json \
    --scores a_data/mmlu/equi_scores_paraphrases_500.json \
    --out e_eval/data/mmlu_gemma-2-2b-it.json
"""

import json
import argparse
import pathlib
from typing import Dict


def load_score_map(score_path: pathlib.Path) -> Dict[int, Dict[str, int]]:
    """Return {prompt_count: {instruct_type: score_int}}g"""
    with score_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    return {
        item["prompt_count"]: item["scores"]
        for item in raw
        if "scores" in item
    }


def rewrite_file(data_path: pathlib.Path,
                 score_map: Dict[int, Dict[str, int]],
                 out_path: pathlib.Path) -> None:
    """Write a copy of data_path to out_path, swapping in new scores"""
    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)

    changed = 0
    for prompt in data:
        mapping = score_map.get(prompt["prompt_count"])
        if not mapping:
            continue
        for para in prompt["paraphrases"]:
            new_score = mapping.get(para["instruct_type"])
            if new_score is not None:
                para["paraphrase_content_score"] = new_score
                changed += 1

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Finished. Updated {changed:,} paraphrases.")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True, type=pathlib.Path)
    p.add_argument("--scores", required=True, type=pathlib.Path)
    p.add_argument("--out",    required=True, type=pathlib.Path)
    args = p.parse_args(argv)

    score_map = load_score_map(args.scores)
    rewrite_file(args.data, score_map, args.out)


if __name__ == "__main__":
    main()
