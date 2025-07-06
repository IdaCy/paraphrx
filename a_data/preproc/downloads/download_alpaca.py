"""
python a_data/preproc/download_alpaca.py --out a_data/alpaca/alpaca_52k_clean.jsonl
"""

import argparse
import json
import re
import uuid
from pathlib import Path

from datasets import load_dataset


def clean(text: str | None) -> str:
    """Normalise newlines and strip surrounding whitespace"""
    if text is None:
        return ""
    # unify CRLF / CR to LF
    text = re.sub(r"\r\n?", "\n", text)
    return text.strip()


def main(out_file: Path) -> None:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    seen: set[tuple[str, str]] = set()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for ex in ds:
            key = (ex["instruction"], ex["input"])
            if key in seen:
                continue
            seen.add(key)

            record = {
                "prompt_id": str(uuid.uuid4()),
                "instruction": clean(ex["instruction"]),
                "input": clean(ex["input"]),
                "output": clean(ex["output"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(seen):,} unique prompts -> {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default="alpaca_52k_clean.jsonl")
    args = p.parse_args()
    main(args.out)
