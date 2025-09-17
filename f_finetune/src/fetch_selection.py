#!/usr/bin/env python3
"""
fetch_selection.py

Create a subset ("selection file") of paraphrased prompts from a larger
Alpaca-style JSON, constrained by:
  - a whitelist of allowed paraphrase keys (from a selection JSON),
  - a maximum number of prompt_count IDs,
  - a maximum number of paraphrases kept per ID,
  - optional RNG seed for reproducibility.

INPUTS
--full_json:      Path to the full prompts JSON. Expects a list of dicts, each
                  with "prompt_count", "instruction_original", optional "input",
                  "output", and many "instruct_*" paraphrase keys.
--selected_keys:  Path to a JSON array of allowed paraphrase keys (strings),
                  e.g. ["instruct_all_caps", "instruct_aave", ...]
--out_json:       Path to write the selection JSON.

OPTIONS
--max_ids:          Max number of distinct prompt_count items to include
                    (default: include all).
--max_paraphrases:  Max number of paraphrase keys to keep per item, sampled
                    from the allowed keys that actually exist in that item
                    (default: include all available allowed keys).
--seed:             RNG seed for reproducibility (default: 1337).
--keep_output:      If set, keep the original "output" field when present
                    (default: not set -> drop "output").
--verbose:          Print simple progress info.

OUTPUT
A JSON list of objects. For each selected prompt, the script retains:
  - "prompt_count" (required),
  - "instruction_original" (if present),
  - "input" (if present),
  - "output" (if present and --keep_output is set),
  - up to --max_paraphrases paraphrase fields whose keys are in the
    --selected_keys whitelist and present in the source item.

python3 f_finetune/src/fetch_selection.py \
    --full_json a_data/alpaca/paraphrases_500.json \
    --selected_keys a_data/alpaca/selections/selection_useable_s.json \
    --out_json a_data/alpaca/selections/selected_200_with_s.json \
    --max_ids 200 \
    --max_paraphrases 50 \
    --keep_output
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set

BASE_FIELDS = ["prompt_count", "instruction_original", "input"]  # minimal keep set


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def pick_records(data: List[Dict[str, Any]], max_ids: int | None, rng: random.Random) -> List[Dict[str, Any]]:
    if max_ids is None or max_ids >= len(data):
        return data
    # Sample without replacement deterministically using seed
    indices = list(range(len(data)))
    rng.shuffle(indices)
    chosen_idx = set(indices[:max_ids])
    return [data[i] for i in range(len(data)) if i in chosen_idx]


def select_paraphrase_keys(
    item: Dict[str, Any],
    allowed: Set[str],
    max_paraphrases: int | None,
    rng: random.Random,
) -> List[str]:
    available = [k for k in item.keys() if k in allowed]
    if not available:
        return []
    if max_paraphrases is None or max_paraphrases >= len(available):
        return sorted(available)  # stable order if taking all
    return rng.sample(available, k=max_paraphrases)


def build_selected_item(
    item: Dict[str, Any],
    chosen_keys: List[str],
    keep_output: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Always include prompt_count if present; skip item if absent
    if "prompt_count" not in item:
        return out
    out["prompt_count"] = item["prompt_count"]

    # Include standard base fields if present
    for f in BASE_FIELDS:
        if f in item:
            out[f] = item[f]

    # Optionally keep the original model output (from the original instruction)
    if keep_output and "output" in item:
        out["output"] = item["output"]

    # Add selected paraphrase fields
    for k in chosen_keys:
        out[k] = item[k]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a constrained selection of paraphrases.")
    parser.add_argument("--full_json", required=True, type=Path, help="Path to the full paraphrases JSON.")
    parser.add_argument("--selected_keys", required=True, type=Path, help="Path to JSON array of allowed paraphrase keys.")
    parser.add_argument("--out_json", required=True, type=Path, help="Path to write the selected subset JSON.")
    parser.add_argument("--max_ids", type=int, default=None, help="Max number of prompt_count IDs to include.")
    parser.add_argument("--max_paraphrases", type=int, default=None, help="Max paraphrase fields per ID.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility.")
    parser.add_argument("--keep_output", action="store_true", help="Keep the original 'output' field if present.")
    parser.add_argument("--verbose", action="store_true", help="Print basic progress info.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load inputs
    data = load_json(args.full_json)
    if not isinstance(data, list):
        raise ValueError(f"{args.full_json} must be a JSON list of objects")
    allowed_keys_list = load_json(args.selected_keys)
    if not isinstance(allowed_keys_list, list) or not all(isinstance(k, str) for k in allowed_keys_list):
        raise ValueError(f"{args.selected_keys} must be a JSON array of strings")
    allowed_keys: Set[str] = set(allowed_keys_list)

    # Filter by max_ids
    picked_records = pick_records(data, args.max_ids, rng)

    selected: List[Dict[str, Any]] = []
    skipped_no_id = 0
    empty_after_filter = 0

    for item in picked_records:
        if "prompt_count" not in item:
            skipped_no_id += 1
            continue
        chosen_keys = select_paraphrase_keys(item, allowed_keys, args.max_paraphrases, rng)
        # Even if no paraphrase keys match, we still keep the base fields; but you can
        # choose to skip empty paraphrase cases by toggling the condition below
        if not chosen_keys:
            empty_after_filter += 1
        selected_item = build_selected_item(item, chosen_keys, args.keep_output)
        if selected_item:
            selected.append(selected_item)

    # Write output
    dump_json(selected, args.out_json)

    if args.verbose:
        total = len(picked_records)
        print(f"Loaded: {len(data)} items")
        print(f"Picked for ID cap: {total} items (max_ids={args.max_ids})")
        print(f"Allowed paraphrase keys: {len(allowed_keys)}")
        print(f"Selected: {len(selected)} items -> wrote to {args.out_json}")
        if skipped_no_id:
            print(f"Skipped (no prompt_count): {skipped_no_id}")
        if empty_after_filter:
            print(f"Items with 0 matching paraphrase keys: {empty_after_filter}")


if __name__ == "__main__":
    main()
