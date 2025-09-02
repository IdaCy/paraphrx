"""
OPENAI WAY OF GENERATING ANSWERS

generate LLM answers for every paraphrase in a prompts file using the OpenAI API
set key either with the flag or as an environment variable: export OPENAI_API_KEY="sk-..

cargo h_rae/src/data_prep/robust_alpaca_genbasline.rs \
    --prompts a_data/alpaca/paraphrases_500.json \
    --whitelist h_rae/data/whitelists/alta_alpaca_base100.json \
    --output h_rae/data/baseline/gpt4_answers_100.json \
    --model gpt-4o \
    --api-key "xxxxxxxx" \
    --log-name "GPT4o_basline_gen_100whitelist" \
    --api-call-max 100 \
    >> logs/GPT4o_basline_gen_100whitelist_$(date +%F_%T).log 2>&1 &


Answer generator for paraphrased instructions using the Gemini API.

Parses a prompts JSON file (array of records with `prompt_count`, `input` (opt),
and instruction keys like `instruction_original` / `instruct_*`), calls a Gemini
model for each missing instruction, and writes/updates an answers JSON file
(resume-friendly), with live logging, retries, token/cost tracking, whitelist,
API call budget, and optional fallback.


  python3 h_rae/src/data_prep/robust_alpaca_genbasline.py \
    --prompts a_data/alpaca/paraphrases_500.json \
    --whitelist h_rae/data/whitelists/alta_alpaca_base100.json \
    --output h_rae/data/baseline/gemini25f_answers.json \
    --model gemini-2.5-flash \
    --api-key xxxxxxx \
    --log-name AnswerGenGem25f \
    --delay-ms 7000 \
    --max-attempts 1 \
    --api-call-max 970 \
    --max-input-tokens 120 \
    --max-output-tokens 256 \
    >> logs/AnswerGenGem25f_$(date +%F_%T).log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import contextlib

def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

@contextlib.contextmanager
def file_lock(lock_path: Path):
    """
    Cross-platform advisory file lock. Blocks until the lock is acquired.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import msvcrt
        f = open(lock_path, "w")
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            f.close()
    else:
        import fcntl
        with open(lock_path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

try:
    from tqdm import tqdm  # real progress bar if installed
except Exception:  # ModuleNotFoundError on most Pythons
    class _NoOpTqdm:
        def __init__(self, *args, **kwargs): pass
        def update(self, n=1): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): pass
    def tqdm(*args, **kwargs):
        return _NoOpTqdm()

# Gemini SDK
try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
except Exception as e:
    print("This script requires the Google Gen AI SDK. Install with: pip install google-genai", file=sys.stderr)
    raise

# ----------------------------
# Logger
# ----------------------------
class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        try:
            self._fh.write(line + "\n")
            self._fh.flush()
        except Exception:
            pass
    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

# ----------------------------
# Data model
# ----------------------------
def _coerce_prompt_count(v: Any) -> int:
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        return int(v.strip())
    raise ValueError(f"Invalid prompt_count: {v!r}")

@dataclass
class PromptRecord:
    prompt_count: int
    instructions: Dict[str, Any] = field(default_factory=dict)
    input: str = ""

    @staticmethod
    def from_obj(o: Dict[str, Any]) -> "PromptRecord":
        # pull prompt_count (string or number)
        pc = _coerce_prompt_count(o.get("prompt_count"))
        # flatten everything else into instructions except 'input'
        input_val = o.get("input", "")
        # keep original keys; we only process instruction_original and instruct_*
        instructions = {k: v for k, v in o.items() if k not in ("prompt_count", "input")}
        return PromptRecord(prompt_count=pc, instructions=instructions, input=input_val or "")

# ----------------------------
# Files: load/repair/save
# ----------------------------
def remove_trailing_commas(s: str) -> str:
    """Remove trailing commas before ] or } to repair slightly-invalid JSON."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == ",":
            # look ahead to next non-space
            j = i + 1
            while j < n and s[j].isspace():
                j += 1
            if j < n and s[j] in ("]", "}"):
                i += 1
                continue  # drop the comma
        out.append(c)
        i += 1
    return "".join(out)

def read_json_array(path: Path, logger: Logger) -> List[Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.log(f"[FATAL] Could not read {path}: {e}")
        raise
    if not raw.strip():
        return []
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return val
        raise ValueError("Root must be a JSON array")
    except Exception as e1:
        logger.log(f"[WARN] Strict JSON parse failed ({e1}). Attempting to repair trailing commas…")
        cleaned = remove_trailing_commas(raw)
        try:
            val = json.loads(cleaned)
            if isinstance(val, list):
                # write back cleaned
                #path.write_text(json.dumps(val, indent=2, ensure_ascii=False), encoding="utf-8")
                _atomic_write_text(path, json.dumps(val, indent=2, ensure_ascii=False))

                logger.log("[WARN] Recovered existing file by removing trailing commas. Wrote cleaned JSON.")
                return val
            raise ValueError("Root must be a JSON array")
        except Exception as e2:
            # backup corrupt file
            logger.log(f"[WARN] Repair failed ({e2}). Leaving file untouched; proceeding without loading it.")
            return []

def load_prompt_records(path: Path, logger: Logger) -> List[PromptRecord]:
    arr = read_json_array(path, logger)
    out: List[PromptRecord] = []
    for o in arr:
        if not isinstance(o, dict):
            logger.log(f"[WARN] Skipping non-object item in prompts: {o!r}")
            continue
        try:
            out.append(PromptRecord.from_obj(o))
        except Exception as e:
            logger.log(f"[WARN] Skipping invalid record: {e}")
    return out

def load_existing_answers(path: Path, logger: Logger) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        logger.log("No existing output file found. Starting fresh.")
        return {}
    logger.log(f"Loading existing output for resume mode: {path}")
    arr = read_json_array(path, logger)
    answers: Dict[str, Dict[str, Any]] = {}
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        # accept number or string prompt_count
        pc = obj.get("prompt_count")
        try:
            id_str = str(_coerce_prompt_count(pc))
        except Exception:
            logger.log("[WARN] Missing or invalid prompt_count in existing output; skipping an item.")
            continue
        answers[id_str] = obj
    return answers

def save_answers(answers: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """
    Merge current in-memory answers with what's on disk and atomically replace
    the file. If the on-disk JSON is corrupt, back it up before replacing.
    The whole read-merge-write is protected by a file lock.
    """
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    with file_lock(lock_path):
        on_disk: Dict[str, Dict[str, Any]] = {}
        parsed_ok = True

        if output_path.exists():
            try:
                raw = output_path.read_text(encoding="utf-8")
                if raw.strip():
                    arr = json.loads(raw)
                    if isinstance(arr, list):
                        for obj in arr:
                            if isinstance(obj, dict) and "prompt_count" in obj:
                                try:
                                    pid = str(int(obj["prompt_count"]))
                                    on_disk[pid] = obj
                                except Exception:
                                    pass
                    else:
                        parsed_ok = False
            except Exception:
                parsed_ok = False

        # Merge: on-disk first, in-memory wins per-key
        merged: Dict[str, Dict[str, Any]] = dict(on_disk)
        for pid, rec in answers.items():
            base = merged.get(pid, {"prompt_count": rec.get("prompt_count")})
            base.update(rec)
            merged[pid] = base

        # Sort and write atomically
        vec = sorted(merged.values(), key=lambda m: int(m.get("prompt_count") or 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If previous file looked corrupt, back it up BEFORE replacing
        if output_path.exists() and not parsed_ok:
            backup = output_path.with_suffix(output_path.suffix + f".corrupt-{datetime.now():%Y%m%d-%H%M%S}")
            try:
                os.replace(output_path, backup)
            except Exception:
                pass  # best-effort

        _atomic_write_text(output_path, json.dumps(vec, indent=2, ensure_ascii=False))


def mark_empty_failure(
    answers: Dict[str, Dict[str, Any]],
    rec: "PromptRecord",
    id_str: str,
    key: str,
    output_path: Path,
    logger: "Logger",
    reason: Optional[str] = None,
    use_null: bool = False,
) -> None:
    """
    Mark a key as completed-with-empty so resume won't retry it.
    Set use_null=True if you prefer JSON null instead of "".
    """
    entry = answers.get(id_str)
    if entry is None:
        entry = {"prompt_count": rec.prompt_count}
        if rec.input:
            entry["input"] = rec.input
        answers[id_str] = entry
    entry[key] = (None if use_null else "")
    if reason:
        # keep lightweight failure context without changing resume logic
        entry.setdefault("_failures", {})[key] = str(reason)
    try:
        save_answers(answers, output_path)
    except Exception as e:
        logger.log(f"[ERROR] Failed to save failure marker for {id_str}/{key}: {e}")


# ----------------------------
# Token counting (fast local approx)
# ----------------------------
def rough_count_tokens(text: str) -> int:
    # Gemini docs: ~4 characters ≈ 1 token (very rough). Good enough for a guardrail.
    # If you prefer exact counting, you can swap in client.models.count_tokens.
    if not text:
        return 0
    return math.ceil(len(text) / 4.0)

# ----------------------------
# Gemini client & call
# ----------------------------
def build_client(explicit_key: Optional[str]) -> genai.Client:
    # Accept CLI key or env vars; prefer explicit, then GOOGLE_API_KEY, then GEMINI_API_KEY.
    api_key = explicit_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Provide --api-key or set GOOGLE_API_KEY (or GEMINI_API_KEY).")
    return genai.Client(api_key=api_key)

def extract_text(response) -> str:
    # Prefer response.text (SDK concatenates text parts). Fallback: collect parts.
    try:
        if getattr(response, "text", None):
            return response.text
        # Fallback: concatenate any text parts from first candidate
        cands = getattr(response, "candidates", None) or []
        if cands:
            parts = getattr(cands[0], "content", None)
            if parts and getattr(parts, "parts", None):
                buf = []
                for p in parts.parts:
                    t = getattr(p, "text", None)
                    if t:
                        buf.append(t)
                return "".join(buf)
    except Exception:
        pass
    return ""

def query_gemini_once(
    client: genai.Client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float],
    max_output_tokens: int,
):
    cfg_kwargs = {}
    if system_prompt:
        cfg_kwargs["system_instruction"] = system_prompt
    if temperature is not None:
        cfg_kwargs["temperature"] = float(temperature)
    if max_output_tokens and max_output_tokens > 0:
        cfg_kwargs["max_output_tokens"] = int(max_output_tokens)

    config = types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None

    resp = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config
    )
    text = extract_text(resp).strip()

    # usage metadata (snake_case in SDK; camelCase in REST)
    p_tok = 0
    c_tok = 0
    finish_reason = None
    try:
        um = getattr(resp, "usage_metadata", None)
        if um:
            p_tok = getattr(um, "prompt_token_count", 0) or getattr(um, "promptTokenCount", 0) or 0
            c_tok = getattr(um, "candidates_token_count", 0) or getattr(um, "candidatesTokenCount", 0) or 0
        cands = getattr(resp, "candidates", None) or []
        if cands:
            finish_reason = getattr(cands[0], "finish_reason", None)
    except Exception:
        pass

    return text, int(p_tok), int(c_tok), finish_reason, resp

def query_gemini_with_fallbacks(
    client: genai.Client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float],
    max_output_tokens: int,
    fallback_on_empty: bool,
    fallback_model: str,
    logger: Logger,
    id_str: str,
    key: str,
) -> Tuple[str, int, int]:
    # Primary attempt
    text, p_tok, c_tok, finish_reason, _ = query_gemini_once(
        client, model, system_prompt, user_prompt, temperature, max_output_tokens
    )
    if text:
        return text, p_tok, c_tok

    # “Starved by length”: empty text, MAX_TOKENS
    starved = (not text) and (str(finish_reason).upper() == "MAX_TOKENS")
    if starved:
        logger.log(f"ID {id_str} key {key}: Starved. Retrying with +64 token headroom.")
        text2, p2, c2, _, _ = query_gemini_once(
            client, model, system_prompt, user_prompt, temperature, max_output_tokens + 64
        )
        if text2:
            return text2, p2, c2

    if fallback_on_empty:
        logger.log(f"ID {id_str} key {key}: Empty content. Falling back to {fallback_model}.")
        text3, p3, c3, _, _ = query_gemini_once(
            client, fallback_model, system_prompt, user_prompt, temperature, max_output_tokens
        )
        if text3:
            return text3, p3, c3
        raise RuntimeError(f"Empty content after fallback to {fallback_model}")

    raise RuntimeError("Empty content in successful response")

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generates LLM answers for all paraphrases in a prompt file using the Gemini API."
    )
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--whitelist", type=Path)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--api-key")
    ap.add_argument("--log-name", default="AnswerGen")
    ap.add_argument("--system-prompt", default="You are a helpful assistant. Provide a direct and concise answer to the user's request.")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--delay-ms", type=int, default=200)
    ap.add_argument("--api-call-max", type=int, default=10_000)
    ap.add_argument("--max-input-tokens", type=int, default=120)
    ap.add_argument("--price-per-million-input", type=float, default=0.05)
    ap.add_argument("--price-per-million-output", type=float, default=0.40)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--fallback-on-empty", type=lambda s: s.lower() in ("1","true","yes","y"), default=True)
    ap.add_argument("--fallback-model", default="gemini-1.5-flash")

    args = ap.parse_args()

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{args.log_name}_{ts}.log"
    logger = Logger(log_path)

    try:
        logger.log(f"Run started – model={args.model}")
        client = build_client(args.api_key)

        # Optional whitelist
        whitelist_set: Optional[set[str]] = None
        if args.whitelist:
            logger.log(f"Reading whitelist from: {args.whitelist}")
            ids_raw = read_json_array(args.whitelist, logger)
            ids: set[str] = set()
            for v in ids_raw:
                if isinstance(v, (int, float)):
                    ids.add(str(int(v)))
                elif isinstance(v, str):
                    ids.add(v)
                else:
                    logger.log(f"[WARN] Ignoring non-string/non-number value in whitelist: {v!r}")
            if not ids:
                logger.log("[WARN] Whitelist provided but resulted in an empty set of IDs.")
            whitelist_set = ids

        logger.log(f"Reading prompts from: {args.prompts}")
        all_records = load_prompt_records(args.prompts, logger)

        if whitelist_set is not None:
            before = len(all_records)
            records = [r for r in all_records if str(r.prompt_count) in whitelist_set]
            logger.log(f"Whitelist applied: {len(records)} of {before} records selected for processing.")
        else:
            records = all_records

        answers = load_existing_answers(args.output, logger)

        # Count planned total calls (respect resume & api_call_max)
        total_pending = 0
        for rec in records:
            keys = [k for k in rec.instructions.keys()
                    if k == "instruction_original" or k.startswith("instruct_")]
            done_keys = set(answers.get(str(rec.prompt_count), {}).keys())
            total_pending += sum(1 for k in keys if k not in done_keys)
        planned_total_calls = min(total_pending, args.api_call_max)
        logger.log(f"Planned total calls (bounded by api_call_max): {planned_total_calls}")

        total_in_tokens = 0
        total_out_tokens = 0
        api_calls_used = 0

        bar = tqdm(total=len(records), ncols=80, desc="Records")

        for rec in records:
            bar.update(1)
            id_str = str(rec.prompt_count)

            keys_to_process = [k for k in rec.instructions.keys()
                               if k == "instruction_original" or k.startswith("instruct_")]
            already_done = set(answers.get(id_str, {}).keys())

            for key in keys_to_process:
                if key in already_done:
                    continue
                if api_calls_used >= args.api_call_max:
                    logger.log("API call limit reached -> aborting early.")
                    bar.close()
                    save_answers(answers, args.output)
                    logger.close()
                    return

                val = rec.instructions.get(key)
                if not isinstance(val, str):
                    logger.log(f"ID {id_str}: Key '{key}' has non-string value, marking empty and skipping.")
                    mark_empty_failure(answers, rec, id_str, key, args.output, logger, reason="non-string value")
                    continue

                user_prompt = val if not rec.input else f"{val}\n\n[Input Data]\n{rec.input}"

                # Input token cap (approximate)
                instr_tokens = rough_count_tokens(user_prompt)
                if instr_tokens > args.max_input_tokens:
                    logger.log(f"ID {id_str} key {key}: {instr_tokens} tokens exceeds cap {args.max_input_tokens} -> marking empty.")
                    mark_empty_failure(
                        answers, rec, id_str, key, args.output, logger,
                        reason=f"input tokens {instr_tokens} > cap {args.max_input_tokens}"
                    )
                    continue

                # Call Gemini with retries/backoff
                answer: Optional[str] = None
                p_tok = 0
                c_tok = 0
                last_err = None

                for attempt in range(1, args.max_attempts + 1):
                    try:
                        txt, pin, pout = query_gemini_with_fallbacks(
                            client=client,
                            model=args.model,
                            system_prompt=args.system_prompt,
                            user_prompt=user_prompt,
                            temperature=args.temperature,
                            max_output_tokens=args.max_output_tokens,
                            fallback_on_empty=args.fallback_on_empty,
                            fallback_model=args.fallback_model,
                            logger=logger,
                            id_str=id_str,
                            key=key,
                        )
                        answer = txt
                        p_tok = pin
                        c_tok = pout
                        logger.log(f"ID {id_str} key {key}: saved {len(answer)} chars ({p_tok} in / {c_tok} out tokens)")
                        break
                    except (genai_errors.APIError, RuntimeError) as e:
                        last_err = e
                        # Special handling for 429 quota/rate-limit errors
                        if "429" in str(e):
                            #m = re.search(r'"retryDelay":\s*"(\d+)s"', str(e))
                            m = re.search(r'"retryDelay":\s*"(\d+)s"', str(e))
                            delay_s = int(m.group(1)) if m else 30
                            logger.log(f"ID {id_str} key {key}: 429 quota/rate limit. Sleeping {delay_s}s per RetryInfo…")
                            time.sleep(delay_s)
                            # one grace retry even if max_attempts==1
                            if attempt >= args.max_attempts:
                                # allow one extra attempt
                                continue
                            continue

                        # Fallback to normal retry logic
                        if attempt < args.max_attempts:
                            logger.log(f"ID {id_str} key {key}: API attempt {attempt} failed: {e}. Retrying…")
                            backoff_ms = 500 * (2 ** attempt)
                            time.sleep(backoff_ms / 1000.0)
                        else:
                            logger.log(f"ID {id_str} key {key}: All API attempts failed: {e}")
                            # leave answer as None to signal a skip
                            break

                api_calls_used += 1

                # Skip write when no usable answer
                if answer is None or not str(answer).strip():
                    transient = last_err and ("429" in str(last_err) or "503" in str(last_err))
                    if transient:
                        logger.log(f"ID {id_str} key {key}: Transient error ({last_err}). Leaving unfilled to retry later.")
                        # (do NOT mark empty)
                    else:
                        logger.log(f"ID {id_str} key {key}: Permanent failure -> marking empty. Error: {last_err}")
                        mark_empty_failure(answers, rec, id_str, key, args.output, logger, reason=str(last_err))
                    if args.delay_ms > 0:
                        time.sleep(args.delay_ms / 1000.0)
                    continue

                total_in_tokens += p_tok
                total_out_tokens += c_tok

                cost_in = (total_in_tokens / 1_000_000.0) * args.price_per_million_input
                cost_out = (total_out_tokens / 1_000_000.0) * args.price_per_million_output
                cost_total = cost_in + cost_out
                pct = (api_calls_used / max(1, planned_total_calls)) * 100.0
                logger.log(f"[PROGRESS] {api_calls_used}/{planned_total_calls} ({pct:.1f}%) • "
                           f"usage_in={total_in_tokens} usage_out={total_out_tokens} • "
                           f"est_cost=${cost_total:.4f} (in=${cost_in:.4f}, out=${cost_out:.4f})")

                # Write/update output record
                entry = answers.get(id_str)
                if entry is None:
                    entry = {"prompt_count": rec.prompt_count}
                    if rec.input:
                        entry["input"] = rec.input
                    answers[id_str] = entry
                entry[key] = answer

                # Save incremental progress
                try:
                    save_answers(answers, args.output)
                except Exception as e:
                    logger.log(f"[ERROR] Failed to save intermediate results: {e}")

                # polite pacing
                if args.delay_ms > 0:
                    time.sleep(args.delay_ms / 1000.0)

        bar.close()
        logger.log(f"Finished. Writing final results to {args.output}")
        save_answers(answers, args.output)

    finally:
        logger.close()

if __name__ == "__main__":
    main()

