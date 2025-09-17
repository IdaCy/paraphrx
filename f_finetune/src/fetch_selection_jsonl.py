#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a JSONL selection file for paraphrase robustness experiments.

- Input 1 (required): a JSON list of prompt objects (Alpaca-like), each having:
    - "prompt_count" (int)
    - "instruction_original" (str)
    - optional "input" (str)
    - many "instruct_*" keys (str)

- Input 2 (required): a JSON list of allowed paraphrase keys (e.g. ["instruct_aave", "instruct_all_caps", ...])

- Output: JSONL where each line is:
    {
      "prompt_count": int,
      "instruction_original": str,
      "input": str,
      "paraphrases": [{"key": str, "text": str}, ...]
    }

Notes:
- We merge duplicate entries with the same prompt_count by unifying their instruct_* fields.
- We only keep paraphrases whose keys are in the provided allowlist and have non-empty strings.
- Randomly sample up to --max_ids families, and up to --max_paraphrases paraphrases per family.
- Output is JSONL; the filename extension is arbitrary.

Example:
    python3 f_finetune/src/fetch_selection_jsonl.py \
      --full_json a_data/alpaca/paraphrases_500.json \
      --selected_keys a_data/alpaca/selections/selection_largest_performance_impr_alta42.json \
      --out_json a_data/alpaca/selections/selected_largest_performance_impr_alta42_limited.jsonl \
      --ids_json a_data/alpaca/selections/selection_verylargest_ids.json \
      --max_ids 200 \
      --max_paraphrases 50 \
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def load_allowed_keys(path: Path) -> Set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array of strings.")
    out = {str(k).strip() for k in data if isinstance(k, str)}
    return {k for k in out if k.startswith("instruct_")}


def load_ids_json(path: Optional[Path]) -> Optional[Set[int]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array of integers.")
    ids: Set[int] = set()
    for x in data:
        try:
            ids.add(int(x))
        except Exception:
            continue
    return ids


def load_and_merge_full_json(path: Path) -> Dict[int, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array of prompt objects.")

    merged: Dict[int, Dict[str, Any]] = {}
    for obj in data:
        if not isinstance(obj, dict) or "prompt_count" not in obj:
            continue
        try:
            pid = int(obj["prompt_count"])
        except Exception:
            continue

        dst = merged.get(pid)
        if dst is None:
            dst = {
                "prompt_count": pid,
                "instruction_original": obj.get("instruction_original", "") or "",
                "input": obj.get("input", "") or "",
            }
            merged[pid] = dst
        else:
            if not dst.get("instruction_original"):
                dst["instruction_original"] = obj.get("instruction_original", "") or ""
            if not dst.get("input"):
                dst["input"] = obj.get("input", "") or ""

        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("instruct_"):
                if isinstance(v, str) and v.strip():
                    dst[k] = v

    return {pid: o for pid, o in merged.items() if (o.get("instruction_original") or "").strip()}


def build_selection_rows(
    merged: Dict[int, Dict[str, Any]],
    allowed: Set[str],
    max_ids: Optional[int],
    max_paraphrases: Optional[int],
    rng: random.Random,
    whitelist_ids: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
    # Families that have ≥1 allowed paraphrase
    eligible: List[Dict[str, Any]] = []
    for o in merged.values():
        avail = [k for k in o.keys() if k in allowed and isinstance(o[k], str) and o[k].strip()]
        if avail:
            o["_avail_keys"] = avail
            eligible.append(o)

    if whitelist_ids is not None:
        # Restrict to requested IDs
        eligible = [o for o in eligible if int(o["prompt_count"]) in whitelist_ids]

    if not eligible:
        return []

    # Sort ascending by prompt_count
    eligible.sort(key=lambda x: int(x["prompt_count"]))

    # Take first N families (if limited)
    if isinstance(max_ids, int) and max_ids > 0:
        eligible = eligible[:max_ids]

    rows: List[Dict[str, Any]] = []
    for o in eligible:
        keys = list(o["_avail_keys"])
        rng.shuffle(keys)  # random paraphrase sampling
        if isinstance(max_paraphrases, int) and max_paraphrases > 0:
            keys = keys[:max_paraphrases]
        paraphrases = [{"key": k, "text": o[k]} for k in keys]
        rows.append({
            "prompt_count": int(o["prompt_count"]),
            "instruction_original": o.get("instruction_original", "") or "",
            "input": o.get("input", "") or "",
            "paraphrases": paraphrases,
        })

    # Final safety: keep rows sorted ascending
    rows.sort(key=lambda r: int(r["prompt_count"]))
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Create a JSONL selection file of prompt families and sampled paraphrases (first-N by prompt_count, with optional whitelist).")
    p.add_argument("--full_json", required=True, type=str, help="Path to the full paraphrases JSON (array).")
    p.add_argument("--selected_keys", required=True, type=str, help="Path to JSON array of allowed instruct_* keys.")
    p.add_argument("--out_json", required=True, type=str, help="Output JSONL path (extension can be .json).")
    p.add_argument("--max_ids", type=int, default=None, help="Max number of prompt_count IDs to include (first-N by ascending prompt_count).")
    p.add_argument("--max_paraphrases", type=int, default=None, help="Max paraphrases per ID (randomly sampled).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for paraphrase sampling.")
    p.add_argument("--ids_json", type=str, default=None, help="Optional JSON array of prompt_count IDs to include.")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)

    allowed = load_allowed_keys(Path(args.selected_keys))
    if not allowed:
        print(f"[warn] No allowed keys loaded from {args.selected_keys}. Output may be empty.", file=sys.stderr)

    whitelist_ids = load_ids_json(Path(args.ids_json)) if args.ids_json else None
    if args.ids_json and not whitelist_ids:
        print(f"[warn] Provided --ids_json={args.ids_json} produced zero valid IDs.", file=sys.stderr)

    merged = load_and_merge_full_json(Path(args.full_json))

    rows = build_selection_rows(
        merged=merged,
        allowed=allowed,
        max_ids=args.max_ids,
        max_paraphrases=args.max_paraphrases,
        rng=rng,
        whitelist_ids=whitelist_ids,
    )

    if not rows:
        if whitelist_ids is not None:
            print("[warn] No rows selected (IDs not found and/or no allowed paraphrases for those IDs).", file=sys.stderr)
        else:
            print("[warn] No rows selected (check allowlist and inputs).", file=sys.stderr)

    out_path = Path(args.out_json)
    write_jsonl(rows, out_path)

    n_ids = len(rows)
    avg_p = (sum(len(r.get("paraphrases", [])) for r in rows) / n_ids) if n_ids else 0.0
    origin = f" (restricted to {len(whitelist_ids)} IDs)" if whitelist_ids is not None else ""
    print(f"[ok] Wrote {n_ids} prompt families{origin} to {out_path} (avg paraphrases per family: {avg_p:.2f}).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
