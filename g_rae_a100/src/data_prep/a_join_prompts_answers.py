#!/usr/bin/env python3
# save as prompts_answers_to_alpacaeval.py
"""
Create AlpacaEval-ready outputs by joining your prompts (instructions) and answers.

python g_rae/src/data_prep/a_join_prompts_answers.py \
  --prompts a_data/alpaca/paraphrases_500.json \
  --answers f_finetune/model/alta_alpaca_base100.json \
  --out g_rae/data/rae_ready_alta_alpaca_base100.json \
  --strict
"""

import json, argparse
from pathlib import Path
from collections import defaultdict

def key_to_variant(key: str):
    if key == "instruction_original":
        return "original"
    if key.startswith("instruct_"):
        return key[len("instruct_"):]
    return None

def index_prompts(prompts_list):
    idx = {}
    for item in prompts_list:
        pc = item.get("prompt_count")
        if pc is None: 
            continue
        pc = int(pc)
        for k, v in item.items():
            var = key_to_variant(k)
            if var is None: 
                continue
            # v is prompt text for that variant
            idx[(pc, var)] = v
    return idx

def extract_answers(answers_list):
    rows = []
    for item in answers_list:
        pc = item.get("prompt_count")
        if pc is None: 
            continue
        pc = int(pc)
        for k, v in item.items():
            var = key_to_variant(k)
            if var is None: 
                continue
            # v is the model's answer for that variant
            rows.append((pc, var, v))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="Your prompts JSON")
    ap.add_argument("--answers", required=True, help="Your answers JSON")
    ap.add_argument("--out", required=True, help="Path to write AlpacaEval-ready JSON")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any (prompt_count,variant) is missing in prompts or answers")
    args = ap.parse_args()

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts_list = json.load(f)
    with open(args.answers, "r", encoding="utf-8") as f:
        answers_list = json.load(f)

    prompt_idx = index_prompts(prompts_list)
    answer_rows = extract_answers(answers_list)

    out = []
    missing_prompts = []
    for pc, var, answer_text in answer_rows:
        key = (pc, var)
        prompt_text = prompt_idx.get(key)
        if prompt_text is None:
            missing_prompts.append(key)
            if args.strict:
                raise SystemExit(f"Missing prompt text for {key}")
            # fallback placeholder (not recommended, but avoids hard fail)
            prompt_text = f"[MISSING PROMPT TEXT] variant={var} prompt_count={pc}"
        out.append({
            "instruction": prompt_text,
            "output": answer_text,
            "meta": {"prompt_count": pc, "variant": var}
        })

    # Optional: warn if prompts exist with no corresponding answers
    orphan_prompts = []
    seen = {(pc, var) for pc, var, _ in answer_rows}
    for key in prompt_idx.keys():
        if key not in seen:
            orphan_prompts.append(key)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out)} rows to {args.out}.")
    if missing_prompts:
        print(f"WARNING: {len(missing_prompts)} answers had no matching prompt text. Example: {missing_prompts[:3]}")
    if orphan_prompts:
        print(f"NOTE: {len(orphan_prompts)} prompts had no matching answer. Example: {orphan_prompts[:3]}")

if __name__ == "__main__":
    main()
