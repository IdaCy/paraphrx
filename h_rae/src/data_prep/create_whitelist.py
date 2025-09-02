#!/usr/bin/env python3
"""
python3 h_rae/src/data_prep/create_whitelist.py \
    f_finetune/outputs_alternat/alta3/hinf_alta3_gsm8k100_inferences.json \
    h_rae/data/whitelists/hinf_alta3_gsm8k100.json

python3 h_rae/src/data_prep/create_whitelist.py \
    f_finetune/model/bases_hinf_alta4_alpaca_inferences_onlyheldout5.json \
    h_rae/data/whitelists/bases_hinf_alta4_alpaca_inferences_onlyheldout5.json

python3 h_rae/src/data_prep/create_whitelist.py \
    f_finetune/outputs_alternat/alta4/hinf_alta4_alpaca_inferences.json\
    h_rae/data/whitelists/hinf_alta4_alpaca_inferences.json
"""

import argparse
import json
import sys
from typing import Any, List

def load_json(path: str) -> Any:
    data = sys.stdin.read() if path == "-" else open(path, "r", encoding="utf-8").read()
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Error: failed to parse JSON from '{path}': {e}\n")
        sys.exit(1)

def write_json(path: str, obj: Any) -> None:
    out = json.dumps(obj, ensure_ascii=False, indent=2)
    if path == "-":
        sys.stdout.write(out + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(out + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract prompt_count IDs from a JSON list of objects.")
    parser.add_argument("input", help="Path to input JSON file (use '-' for stdin)")
    parser.add_argument("output", help="Path to output JSON file (use '-' for stdout)")
    args = parser.parse_args()

    data = load_json(args.input)

    if not isinstance(data, list):
        sys.stderr.write("Error: input JSON must be a list of objects.\n")
        sys.exit(1)

    ids: List[int] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            sys.stderr.write(f"Warning: skipping non-object at index {i}.\n")
            continue
        if "prompt_count" not in item:
            sys.stderr.write(f"Warning: 'prompt_count' missing at index {i}; skipping.\n")
            continue
        val = item["prompt_count"]
        if not isinstance(val, int):
            sys.stderr.write(f"Warning: 'prompt_count' at index {i} is not an int; skipping.\n")
            continue
        ids.append(val)

    write_json(args.output, ids)

if __name__ == "__main__":
    main()
