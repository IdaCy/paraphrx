#!/usr/bin/env python3
"""
Generate RobustAlpacaEval reference outputs with the OpenAI API.

writes a JSON list with: {"instruction","output","group_id","variant","model"}.

python3 h_rae/src/data_prep/generate_official_rae_baseline.py \
  --dataset_jsonl h_rae/data/rae_official/RobustAlpacaEval.jsonl \
  --out_json h_rae/data/baseline/rae_gpt4_turbo_reference.json \
  --model gpt-4-turbo-2024-04-09 \
  --max_tokens 512 \
  --rate_limit_rps 1.5

trying run official reference for RobustAlpacaEval

Install deps
    pip install --upgrade openai

Set API key
    export OPENAI_API_KEY="sk-..."

Run with the historical GPT-4 Turbo reference model used by the benchmark:
    (If your org has access to this exact model ID, use it for apples-to-apples.)

python3 h_rae/src/data_prep/generate_official_rae_baseline.py \
    --dataset_jsonl h_rae/data/rae_official/RobustAlpacaEval.jsonl \
    --out_json h_rae/data/baseline/rae_gpt4_turbo_reference.json \
    --model gpt-4-turbo-2024-04-09 \
    --max_tokens 512 \
    --rate_limit_rps 1.5 \
    --resume \
    --flush_every 20
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# OpenAI Python SDK (v1+)
try:
    from openai import OpenAI
    from openai._exceptions import RateLimitError, APIError, APITimeoutError
except Exception as e:
    print("Please install the OpenAI SDK:  pip install --upgrade openai", file=sys.stderr)
    raise


def load_robust_jsonl(p: str) -> List[Dict[str, Any]]:
    """
    Load RobustAlpacaEval.jsonl and flatten to a list of {group_id, variant, prompt}.
    Each line: {"index": int, "instruction": str, "paraphrases": [str, ...]}
    """
    flat = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            gid = int(o["index"])
            flat.append({"group_id": gid, "variant": "original", "prompt": o["instruction"]})
            for j, ph in enumerate(o.get("paraphrases", [])):
                flat.append({"group_id": gid, "variant": f"paraphrase_{j}", "prompt": ph})
    return flat


def backoff_sleep(i: int, base: float = 1.0, cap: float = 30.0):
    t = min(cap, base * (2 ** (i - 1)))
    time.sleep(t)


def flush_now(results: List[Dict[str, Any]], out_path: Path):
    """
    Atomically write JSON and JSONL so progress isn't lost on crash.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)

    jsonl_path = out_path.with_suffix(".jsonl")
    tmp_jsonl = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, jsonl_path)


def main():
    ap = argparse.ArgumentParser(description="Generate RobustAlpacaEval reference outputs with OpenAI.")
    ap.add_argument("--dataset_jsonl", required=True, help="Path to RobustAlpacaEval.jsonl")
    ap.add_argument("--out_json", required=True, help="Where to write outputs JSON")
    ap.add_argument("--model", default="gpt-4-turbo-2024-04-09",
                    help="Use the historical GPT-4 Turbo model for official comparability.")
    ap.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--rate_limit_rps", type=float, default=2.0, help="Simple client-side pacing")
    ap.add_argument("--resume", action="store_true", help="Skip prompts already present in out_json")
    ap.add_argument("--flush_every", type=int, default=0, help="Write partial results every N prompts (0 = only at end)")
    args = ap.parse_args()

    if not args.api_key:
        print("Set OPENAI_API_KEY or pass --api_key.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=args.api_key)

    flat = load_robust_jsonl(args.dataset_jsonl)
    print(f"Loaded {len(flat)} prompts (original + paraphrases).", file=sys.stderr)

    # Resume: load prior results keyed by (group_id, variant), keep full record for exact reuse
    done_rec: Dict[Tuple[int, str], Dict[str, Any]] = {}
    out_path = Path(args.out_json)
    if args.resume and out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            kept = 0
            for r in prev:
                k = (int(r["group_id"]), str(r["variant"]))
                done_rec[k] = r
                kept += 1
            print(f"Resuming with {kept} existing generations.", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Could not parse existing {out_path}: {e}", file=sys.stderr)

    # Start results with existing ones (preserve order by dataset later)
    results: List[Dict[str, Any]] = []

    last_ts = 0.0
    min_interval = 1.0 / max(0.001, args.rate_limit_rps)

    processed = 0
    for ex in flat:
        k = (ex["group_id"], ex["variant"])

        # Reuse from resume file
        if k in done_rec:
            results.append(done_rec[k])
            processed += 1
            if args.flush_every and (processed % args.flush_every == 0):
                flush_now(results, out_path)
            continue

        # crude client-side pacing
        now = time.time()
        if now - last_ts < min_interval:
            time.sleep(max(0.0, min_interval - (now - last_ts)))
        last_ts = time.time()

        prompt = ex["prompt"]
        attempts = 0
        while True:
            attempts += 1
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,              # deterministic
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                text = (resp.choices[0].message.content or "").strip()
                break
            except (RateLimitError,) as e:
                if attempts <= 5:
                    backoff_sleep(attempts, base=1.5, cap=45)
                    continue
                else:
                    raise
            except (APITimeoutError, APIError) as e:
                if attempts <= 4:
                    backoff_sleep(attempts, base=1.5, cap=30)
                    continue
                else:
                    raise

        rec = {
            "instruction": prompt,
            "output": text,
            "group_id": ex["group_id"],
            "variant": ex["variant"],
            "model": args.model,
        }
        results.append(rec)
        done_rec[k] = rec  # so we don't duplicate if we crash between flushes

        processed += 1
        if args.flush_every and (processed % args.flush_every == 0):
            flush_now(results, out_path)

    # Final write
    flush_now(results, out_path)
    print(f"Wrote {len(results)} generations to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
