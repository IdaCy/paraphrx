#!/usr/bin/env python3
"""
Convert a JSONL dataset like:
  {"instruction": "...", "paraphrases": ["p1", "p2", ...], "index": 0}
into a JSON array like:
  [
    {
      "input": "",
      "instruction_original": "...",
      "instruct_1": "p1",
      "instruct_2": "p2",
      ...
      "prompt_count": 1
    },
    ...
  ]

python3 h_rae/src/data_prep/rae_convert.py \
    h_rae/data/rae_official/RobustAlpacaEval.jsonl \
    h_rae/data/rae_official/RobustAlpacaEval_converted_all.json

Notes:
- If a line has no `paraphrases`, will still emit the base fields
+ emit an alias key `intruct_1` for the first paraphrase
"""

import json
import sys
from pathlib import Path

def main(inp: str, outp: str) -> None:
    input_path = Path(inp)
    output_path = Path(outp)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    results = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue  # skip empty lines
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[Line {line_no}] Skipping invalid JSON: {e}", file=sys.stderr)
                continue

            instruction = obj.get("instruction", "")
            paraphrases = obj.get("paraphrases", []) or []

            rec = {
                "input": "",
                "instruction_original": instruction,
                "prompt_count": line_no
            }

            # Map paraphrases → instruct_1, instruct_2, ...
            for i, p in enumerate(paraphrases, start=1):
                rec[f"instruct_{i}"] = p
                if i == 1:
                    # Add the exact alias the prompt mentioned ("intruct_1") for the first paraphrase
                    rec["intruct_1"] = p

            results.append(rec)

    # Write pretty, stable JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(results)} records to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python jsonl_to_instruct_json.py input.jsonl output.json", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
