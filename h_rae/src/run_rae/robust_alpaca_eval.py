import argparse
import collections
import dataclasses
import datetime as dt
import functools
import gzip
import io
import json
import math
import os
import random
import re
import statistics
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("This script requires numpy. Please `pip install numpy`.", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("This script requires requests. Please `pip install requests`.", file=sys.stderr)
    sys.exit(1)


# Config and Arg Parsing

PARAPHRASE_PREFIX = "instruct_"
ORIGINAL_KEY = "instruction_original"
REFERENCE_OUTPUT_KEY = "output"
DEFAULT_METRICS = [
    # Keep TF first (most important)
    "Task Fulfilment/Relevance",
    "Structure",
    "Creativity",
    "Factuality/Correctness",
    "Clarity",
    "Safety/Non-Harm",
    "Conciseness",
    "Helpfulness/Actionability",
    "Coverage/Completeness",
    "Style/Tone",
]
DEFAULT_MODEL = "gemini-2.0-flash"


@dataclasses.dataclass
class RunConfig:
    prompts_path: str
    baseline_answers_path: str
    compare_answers_path: str
    outdir: str
    run_name: str
    api_keys: List[str]
    max_per_key: int
    max_total: Optional[int]
    judging_model: str
    delay_ms: int
    seed: int
    phase: str  # "judging" | "assessment" | "both"
    original_only_for_judge: bool
    include_optimal_output: bool
    length_unit: str  # "char" | "word"
    temperature: float
    top_p: float
    top_k: int
    max_output_tokens: int
    json_only: bool
    paraphrases: Optional[List[str]]


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Robust Alpaca Paraphrase Evaluation with Gemini Judge")
    p.add_argument("--prompts", required=True, help="Path to prompts JSON (list of objects).")
    p.add_argument("--answers-baseline", required=True, help="Path to baseline answers JSON (list of objects).")
    p.add_argument("--answers-compare", required=True, help="Path to compare/target answers JSON (list of objects).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--run-name", required=True, help="Run name used as prefix for files.")
    p.add_argument("--api-key", action="append", default=[], help="Google API key (repeatable).")
    p.add_argument("--max-per-key", type=int, default=1000, help="Max successful requests to send per API key.")
    p.add_argument("--max-total", type=int, default=None, help="Optional overall cap on total cases to judge.")
    p.add_argument("--judging-model", default=DEFAULT_MODEL, help="Gemini model (default: gemini-2.0-flash).")
    p.add_argument("--delay-ms", type=int, default=100, help="Delay between requests in milliseconds.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    p.add_argument("--phase", choices=["judging", "assessment", "both"], default="both",
                   help="Which phase(s) to run.")
    p.add_argument("--original-only-for-judge", action="store_true",
                   help="Judge sees ONLY the original instruction (no paraphrase).")
    p.add_argument("--include-optimal-output", action="store_true",
                   help="Provide the 'output' field from prompts JSON to the judge as a reference anchor.")
    p.add_argument("--length-unit", choices=["char", "word"], default="char",
                   help="Unit for length difference (default: char).")
    p.add_argument("--temperature", type=float, default=0.0, help="Judge temperature.")
    p.add_argument("--top-p", type=float, default=1.0, help="Judge nucleus sampling p.")
    p.add_argument("--top-k", type=int, default=1, help="Judge top-k.")
    p.add_argument("--max-output-tokens", type=int, default=1024, help="Judge max output tokens.")
    p.add_argument("--json-only", action="store_true",
                   help="Force judge to emit JSON strictly (the prompt already asks this).")
    p.add_argument("--paraphrases", nargs='*', help="An optional list of paraphrase keys to consider (e.g., instruct_polite_request instruct_one_typo_punctuation). If not provided, all paraphrases will be used.")

    args = p.parse_args()

    if not args.api_key:
        print("ERROR: provide at least one --api-key", file=sys.stderr)
        sys.exit(2)

    return RunConfig(
        prompts_path=args.prompts,
        baseline_answers_path=args.answers_baseline,
        compare_answers_path=args.answers_compare,
        outdir=args.outdir,
        run_name=args.run_name,
        api_keys=args.api_key,
        max_per_key=args.max_per_key,
        max_total=args.max_total,
        judging_model=args.judging_model,
        delay_ms=args.delay_ms,
        seed=args.seed,
        phase=args.phase,
        original_only_for_judge=args.original_only_for_judge,
        include_optimal_output=args.include_optimal_output,
        length_unit=args.length_unit,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_output_tokens=args.max_output_tokens,
        json_only=args.json_only,
        paraphrases=args.paraphrases,
    )


# Utilities

def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

def mkdir_p(path: str):
    os.makedirs(path, exist_ok=True)


def _load_json_strict(path: str) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:  # -sig handles BOM
        return json.load(f)

def load_json_flexible(path: str) -> Any:
    """
    Robust loader:
      1) Try strict JSON.
      2) Try JSONL (one object per line).
      3) Try 'salvage' mode: stream the top-level array, extract well-formed objects
         using a state machine that respects strings/escapes, and parse them individually.
         Trailing commas are trimmed safely (outside strings).
      4) If still failing, raise with a helpful snippet
    """
    # Strict
    try:
        return _load_json_strict(path)
    except Exception:
        pass

    # JSONL fallback
    objs = []
    had_lines = False
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            had_lines = True
            try:
                objs.append(json.loads(line))
            except Exception:
                objs = None
                break
    if objs is not None and had_lines:
        return objs

    # Salvage top-level array
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    salvaged = _salvage_top_level_array_of_objects(raw)
    if salvaged and isinstance(salvaged, list):
        return salvaged

    # Final helpful error
    try:
        json.loads(raw)  # will raise JSONDecodeError with position
    except json.JSONDecodeError as je:
        pos = je.pos
        start = max(0, pos - 200)
        end = min(len(raw), pos + 200)
        snippet = raw[start:end]
        raise json.JSONDecodeError(
            f"{je.msg} at char {je.pos}. Surrounding text:\n<<<{snippet}>>>",
            raw, je.pos
        ) from None
    # If not a JSONDecodeError, just raise a generic
    raise ValueError("Unrecognized file format; not JSON, not JSONL, and salvage failed.")


def _salvage_top_level_array_of_objects(raw: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a file that SHOULD be a JSON array of objects, but may contain:
      - Trailing commas
      - A truncated final object
    """
    # Locate array bounds
    i_lbrack = raw.find('[')
    i_rbrack = raw.rfind(']')
    if i_lbrack == -1 or i_rbrack == -1 or i_rbrack <= i_lbrack:
        return None
    s = raw[i_lbrack+1:i_rbrack]  # inside the array

    in_string = False
    escape = False
    brace_depth = 0
    current_start = -1
    items: List[Dict[str, Any]] = []

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == '{':
                if brace_depth == 0:
                    current_start = i
                brace_depth += 1
                continue
            if ch == '}':
                if brace_depth > 0:
                    brace_depth -= 1
                    if brace_depth == 0 and current_start >= 0:
                        # We got a complete object substring
                        obj_text = s[current_start:i+1]
                        obj_text = _trim_trailing_commas(obj_text)
                        try:
                            obj = json.loads(obj_text)
                            if isinstance(obj, dict):
                                items.append(obj)
                        except Exception:
                            # skip malformed object
                            pass
                        current_start = -1
                continue
            # other chars (commas/whitespace) are fine

    # If we found at least one object, return them (even if some were skipped)
    if items:
        return items
    return None


def _trim_trailing_commas(obj_text: str) -> str:
    # Quick path: if there's clearly no ',}' or ',]' we're done
    if ',}' not in obj_text and ',]' not in obj_text:
        return obj_text

    out_chars = []
    in_string = False
    escape = False
    # will look ahead with a small buffer
    for i, ch in enumerate(obj_text):
        if in_string:
            out_chars.append(ch)
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                out_chars.append(ch)
                continue
            if ch == ',':
                # peek next non-space
                j = i + 1
                while j < len(obj_text) and obj_text[j] in ' \t\r\n':
                    j += 1
                if j < len(obj_text) and obj_text[j] in '}]':
                    # skip this comma (i.e., do not append)
                    continue
                else:
                    out_chars.append(ch)
                    continue
            out_chars.append(ch)
    return ''.join(out_chars)


def write_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_text(text: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def gzip_jsonl_append(path: str, item: Dict[str, Any]):
    # If .jsonl.gz path exists -> append, else create
    mode = "ab" if os.path.exists(path) else "wb"
    with gzip.open(path, mode) as g:
        line = (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")
        g.write(line)

def safe_len(s: Optional[str], unit: str) -> int:
    if not s:
        return 0
    if unit == "word":
        return len(s.split())
    return len(s)

def short(s: str, n: int = 120) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "…"

def guess_paraphrase_keys(prompts: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for rec in prompts:
        for k in rec.keys():
            if k.startswith(PARAPHRASE_PREFIX) or k == ORIGINAL_KEY:
                keys.add(k)
    # keep original at end for readability in reports
    keys = sorted([k for k in keys if k != ORIGINAL_KEY]) + ([ORIGINAL_KEY] if ORIGINAL_KEY in keys else [])
    return keys

def to_bool(x) -> bool:
    return str(x).lower() in ("1", "true", "yes", "y", "t")


# Key Manager (multi-key, 429 handling)

class KeyManager:
    def __init__(self, keys: List[str], max_per_key: int):
        self.keys = list(keys)
        self.max_per_key = max_per_key
        self.usage = {k: 0 for k in keys}
        self.dead = {k: False for k in keys}  # dead after too many 429s
        self.strikes = {k: 0 for k in keys}
        self._i = 0

    def next_key(self) -> Optional[str]:
        for _ in range(len(self.keys)):
            k = self.keys[self._i]
            self._i = (self._i + 1) % len(self.keys)
            if not self.dead[k] and self.usage[k] < self.max_per_key:
                return k
        return None

    def mark_success(self, key: str):
        self.usage[key] += 1

    def mark_429(self, key: str):
        self.strikes[key] += 1
        if self.strikes[key] >= 4:  # retire the key
            self.dead[key] = True

    def summary(self) -> Dict[str, Any]:
        return {
            "usage": self.usage,
            "dead": self.dead,
            "strikes": self.strikes,
            "max_per_key": self.max_per_key,
        }


# Gemini HTTP client (REST)

def gemini_generate_content(
        api_key: str,
        model: str,
        user_text: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        max_output_tokens: int = 1024,
) -> Tuple[int, Dict[str, Any], str]:
    """
    Returns: (status_code, raw_json, text_output)
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json; charset=utf-8"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "topK": top_k,
            "maxOutputTokens": max_output_tokens,
        }
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload).encode("utf-8"), timeout=60)
    except requests.RequestException as e:
        return 599, {"error": str(e)}, ""

    status = resp.status_code
    try:
        data = resp.json()
    except Exception:
        data = {"raw_text": resp.text}

    # Extract text
    out_text = ""
    try:
        cands = data.get("candidates") or []
        if cands:
            parts = cands[0].get("content", {}).get("parts") or []
            if parts and "text" in parts[0]:
                out_text = parts[0]["text"]
    except Exception:
        pass

    return status, data, out_text


# Judge Prompt

def build_judge_prompt(
    instruction_for_judge: str,
    input_text: str,
    answer_a: str,
    answer_b: str,
    include_reference_output: bool,
    reference_output: Optional[str],
    metrics: List[str],
    json_only: bool,
) -> str:
    metrics_list = "\\n".join([f"- {m}" for m in metrics])
    schema = {
        "decision": "A | B | Tie",
        "confidence": "float in [0,1]",
        "scores": {
            "A": { "Task Fulfilment/Relevance": "0-10", "Structure": "0-10", "Creativity": "0-10",
                   "Factuality/Correctness": "0-10", "Clarity": "0-10", "Safety/Non-Harm": "0-10",
                   "Conciseness": "0-10", "Helpfulness/Actionability": "0-10",
                   "Coverage/Completeness": "0-10", "Style/Tone": "0-10" },
            "B": { "Task Fulfilment/Relevance": "0-10", "Structure": "0-10", "Creativity": "0-10",
                   "Factuality/Correctness": "0-10", "Clarity": "0-10", "Safety/Non-Harm": "0-10",
                   "Conciseness": "0-10", "Helpfulness/Actionability": "0-10",
                   "Coverage/Completeness": "0-10", "Style/Tone": "0-10" }
        },
        "reasoning_short": "1-3 sentences justifying the decision"
    }
    # will embed schema as JSON string in instructions
    schema_s = json.dumps(schema, ensure_ascii=False, indent=2)

    ref_block = ""
    if include_reference_output and reference_output:
        ref_block = f"\\n\\nReference 'optimal' output (from dataset; use only as a quality anchor, not ground truth):\\n---\\n{reference_output}\\n---"

    json_only_note = (
        "\\nYou MUST respond with **JSON only** and nothing else (no prose before or after)."
        if json_only else
        "\\nRespond with JSON as the final message."
    )

    prompt = f"""You are a strict, careful evaluation judge.

Task:
Given an instruction and two candidate answers (A = baseline/reference; B = target/system-under-test),
CHOOSE WHICH ANSWER BETTER FULFILLS THE INSTRUCTION. Your primary criterion is **Task Fulfilment/Relevance (TF)**.
If both are about equally good overall, return "Tie".

Also score each answer along these 10 metrics (0–10, decimals allowed):
{metrics_list}

Important judging guidelines:
- Prioritize **Task Fulfilment/Relevance** (TF) above all. If TF differs notably, choose the higher-TF answer even if other metrics are strong.
- Penalize factual errors, unsafe content, or failure to follow instructions.
- Be fair to style paraphrases: do not reward mere verbosity or fancy tone if it doesn't improve TF.
- Short but complete answers can be better than long but off-target ones.
- Use the input (if any) as required by the instruction.
- If the instruction asks for a specific format (e.g., Markdown list), consider adherence in TF/Structure.

Instruction:
---
{instruction_for_judge}
---

Additional input (may be empty):
---
{input_text}
---

Candidate Answer A (baseline/reference):
---
{answer_a}
---

Candidate Answer B (target/system-under-test):
---
{answer_b}
---{ref_block}

Output format (JSON schema you MUST follow):
{schema_s}
{json_only_note}
"""
    return prompt


# JSON Output Repair

def try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        pass
    # Attempt bracket capture
    if "{" in s and "}" in s:
        first = s.find("{")
        last = s.rfind("}")
        if first >= 0 and last > first:
            chunk = s[first:last+1]
            try:
                return json.loads(chunk)
            except Exception:
                pass
    # Quote fix
    try:
        s2 = re.sub(r"(\w+):", r'"\1":', s)  # very rough
        return json.loads(s2)
    except Exception:
        return None


# Judging Phase

def load_indexed(records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for r in records:
        if "prompt_count" not in r:
            continue
        out[int(r["prompt_count"])] = r
    return out

def build_cases(
    prompts: List[Dict[str, Any]],
    answers_base_idx: Dict[int, Dict[str, Any]],
    answers_comp_idx: Dict[int, Dict[str, Any]],
    cfg: RunConfig,
) -> List[Dict[str, Any]]:
    all_keys = guess_paraphrase_keys(prompts)
    if cfg.paraphrases:
        all_keys = [key for key in all_keys if key in cfg.paraphrases]
    cases = []
    for rec in prompts:
        pc = int(rec.get("prompt_count"))
        base = answers_base_idx.get(pc)
        comp = answers_comp_idx.get(pc)
        if base is None or comp is None:
            continue
        for key in all_keys:
            instr_text = rec.get(key)
            if not isinstance(instr_text, str):
                continue
            # Need both answers present for this key
            aA = base.get(key)
            aB = comp.get(key)
            if not isinstance(aA, str) or not isinstance(aB, str):
                continue
            cases.append({
                "prompt_count": pc,
                "paraphrase_key": key,
                "instruction_text": instr_text,
                "instruction_original": rec.get(ORIGINAL_KEY, ""),
                "input": rec.get("input", ""),
                "answer_a": aA,
                "answer_b": aB,
                "reference_output": rec.get(REFERENCE_OUTPUT_KEY, None),
            })
    return cases

def load_jsonl_gz(path: str) -> List[Dict[str, Any]]:
    out = []
    with gzip.open(path, "rb") as g:
        for line in g:
            try:
                out.append(json.loads(line.decode("utf-8")))
            except Exception:
                pass
    return out

def _existing_case_ids(judgments_path: str) -> Tuple[set, int, int, int]:
    """
    Read existing judgments file and return:
      - set of case_ids already present (from any status row)
      - successes count
      - failures count
      - attempted count
    This lets us skip duplicates and accumulate counts on resume.
    """
    if not os.path.exists(judgments_path):
        return set(), 0, 0, 0
    rows = load_jsonl_gz(judgments_path)
    ids = set()
    succ = 0
    fail = 0
    att = 0
    for r in rows:
        cid = r.get("case_id")
        if cid:
            ids.add(cid)
        status = r.get("status")
        if status in ("ok", "bad_json", "error"):
            att += 1
        if status == "ok":
            succ += 1
        elif status in ("bad_json", "error"):
            fail += 1
    return ids, succ, fail, att

def judging_phase(cfg: RunConfig, logf, rnd: random.Random) -> str:
    prompts = load_json_flexible(cfg.prompts_path)
    ans_base = load_json_flexible(cfg.baseline_answers_path)
    ans_comp = load_json_flexible(cfg.compare_answers_path)

    if not isinstance(prompts, list) or not isinstance(ans_base, list) or not isinstance(ans_comp, list):
        print("ERROR: JSON files must be lists of objects.", file=sys.stderr)
        sys.exit(3)

    base_idx = load_indexed(ans_base)
    comp_idx = load_indexed(ans_comp)

    # Build full case list (all available in current data)
    all_cases = build_cases(prompts, base_idx, comp_idx, cfg)

    mkdir_p(cfg.outdir)
    run_prefix = os.path.join(cfg.outdir, cfg.run_name)
    meta_path = f"{run_prefix}__judging_meta.json"
    gz_path = f"{run_prefix}__judgments.jsonl.gz"
    log_path = f"{run_prefix}__judging_log.txt"

    # Resume detection and existing stats
    resume_mode = os.path.exists(gz_path)
    existing_ids, prev_successes, prev_failures, prev_attempted = _existing_case_ids(gz_path)

    # Filter out already-processed cases (skip duplicates entirely)
    remaining_cases = [
        c for c in all_cases
        if f"{c['prompt_count']}::{c['paraphrase_key']}" not in existing_ids
    ]

    # Enforce overall cap across runs (if provided)
    if cfg.max_total is not None:
        # how many we may still process this run
        already_counted = len(existing_ids)
        remaining_cap = max(0, cfg.max_total - already_counted)
        if remaining_cap < len(remaining_cases):
            remaining_cases = remaining_cases[:remaining_cap]

    # Shuffle remaining only (stable randomness via provided seed)
    rnd.shuffle(remaining_cases)

    # Prepare/Load metadata (preserve original created_utc on resume)
    if resume_mode and os.path.exists(meta_path):
        try:
            metadata = _load_json_strict(meta_path)
            # Ensure structure exists
            if "counts" not in metadata or not isinstance(metadata["counts"], dict):
                metadata.setdefault("counts", {})
        except Exception:
            # If meta is unreadable, fall back to minimal structure without clobbering created_utc
            metadata = {"created_utc": now_utc_iso(), "counts": {}}
    else:
        metadata = {
            "created_utc": now_utc_iso(),
            "config": dataclasses.asdict(cfg),
            "counts": {
                "total_cases_built": len(all_cases),
            },
        }

    # Ensure total_cases_built reflects *current* dataset size (but keep original created_utc)
    metadata.setdefault("counts", {})
    metadata["counts"]["total_cases_built"] = len(all_cases)

    # Open log in append mode if resuming; otherwise create new
    log_mode = "a" if resume_mode and os.path.exists(log_path) else "w"
    with open(log_path, log_mode, encoding="utf-8") as lf:
        lf.write(f"[{now_utc_iso()}] Start judging ({'resume' if resume_mode else 'fresh'}): "
                 f"{len(remaining_cases)} remaining (had {len(existing_ids)} done)\n")

    # Save (or refresh) meta header now (non-destructive to created_utc)
    write_json(metadata, meta_path)

    metrics = DEFAULT_METRICS[:]

    km = KeyManager(cfg.api_keys, cfg.max_per_key)
    total = len(remaining_cases)
    successes = 0
    failures = 0
    skipped = 0  # left for compatibility with existing code
    last_progress_print = -1

    for i, case in enumerate(remaining_cases, 1):
        # Progress heartbeat (every 5%)
        pct = int(i * 100 / max(1, total))
        if pct >= last_progress_print + 5:
            msg = f"[{now_utc_iso()}] Progress {i}/{total} ({pct}%)"
            print(msg)
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
            last_progress_print = pct

        key_to_use = km.next_key()
        if key_to_use is None:
            print("All API keys exhausted (usage caps or retired). Stopping judging.")
            break

        instruction_for_judge = (
            case["instruction_original"] if cfg.original_only_for_judge else case["instruction_text"]
        )
        prompt_text = build_judge_prompt(
            instruction_for_judge=instruction_for_judge,
            input_text=case.get("input", "") or "",
            answer_a=case.get("answer_a", "") or "",
            answer_b=case.get("answer_b", "") or "",
            include_reference_output=cfg.include_optimal_output,
            reference_output=case.get("reference_output"),
            metrics=metrics,
            json_only=cfg.json_only or True,  # enforce
        )

        # send
        attempt = 0
        backoff = 1.0
        judged = False
        last_status = None
        raw_json = None
        out_text = ""

        while attempt < 5 and not judged:
            if attempt > 0:
                time.sleep(backoff)
                backoff *= 2.0
            time.sleep(cfg.delay_ms / 1000.0)
            status, raw_json, out_text = gemini_generate_content(
                api_key=key_to_use,
                model=cfg.judging_model,
                user_text=prompt_text,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                max_output_tokens=cfg.max_output_tokens,
            )
            last_status = status
            if status == 200:
                judged = True
                km.mark_success(key_to_use)
            elif status == 429:
                km.mark_429(key_to_use)
                attempt += 1
            else:
                attempt += 1
                # Non-429 error; we retry up to 5 as well
                pass

        case_id = f"{case['prompt_count']}::{case['paraphrase_key']}"

        if not judged:
            failures += 1
            rec = {
                "ts_utc": now_utc_iso(),
                "phase": "judging",
                "status": "error",
                "http_status": last_status,
                "error": raw_json,
                "case_id": case_id,
            }
            gzip_jsonl_append(gz_path, rec)
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"[{now_utc_iso()}] ERROR case {case_id} status={last_status}\n")
            continue

        # Parse judge JSON
        parsed = try_parse_json(out_text)
        if parsed is None:
            failures += 1
            rec = {
                "ts_utc": now_utc_iso(),
                "phase": "judging",
                "status": "bad_json",
                "http_status": last_status,
                "raw_text": out_text,
                "raw_json": raw_json,
                "case_id": case_id,
            }
            gzip_jsonl_append(gz_path, rec)
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"[{now_utc_iso()}] BAD_JSON case {case_id}\n")
            continue

        decision = str(parsed.get("decision", "")).strip()
        confidence = parsed.get("confidence", None)
        if confidence is None:
            confidence = 0.5
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5
        scores = parsed.get("scores", {})
        reasoning_short = parsed.get("reasoning_short", "")

        # assemble judgment row
        len_a = safe_len(case["answer_a"], cfg.length_unit)
        len_b = safe_len(case["answer_b"], cfg.length_unit)
        delta_len = len_b - len_a

        row = {
            "ts_utc": now_utc_iso(),
            "status": "ok",
            "http_status": last_status,
            "prompt_count": case["prompt_count"],
            "paraphrase_key": case["paraphrase_key"],
            "instruction_used": instruction_for_judge,
            "instruction_original": case["instruction_original"],
            "input": case.get("input", ""),
            "include_reference_output": bool(cfg.include_optimal_output),
            "reference_output_present": bool(case.get("reference_output")),
            "decision": decision,
            "confidence": confidence,
            "scores": scores,
            "reasoning_short": reasoning_short,
            "length_unit": cfg.length_unit,
            "len_a": len_a,
            "len_b": len_b,
            "delta_len": delta_len,
            # store short previews to keep JSONL compact; full texts are large
            "answer_a_preview": short(case["answer_a"]),
            "answer_b_preview": short(case["answer_b"]),
            "case_id": case_id,
        }

        gzip_jsonl_append(gz_path, row)
        successes += 1

        # occasional progress logging
        if i % 25 == 0 or i == total:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"[{now_utc_iso()}] judged {i}/{total}, successes={successes}, failures={failures}\n")

    # finalise metadata (preserve created_utc if present)
    # Accumulate counts with previous run (if any)
    attempted_this_run = successes + failures + skipped
    counts = metadata.setdefault("counts", {})
    counts["attempted_cases"] = (prev_attempted or 0) + attempted_this_run
    counts["successes"] = (prev_successes or 0) + successes
    counts["failures"] = (prev_failures or 0) + failures
    metadata["key_manager"] = KeyManager(cfg.api_keys, cfg.max_per_key).summary()  # summary shape preserved
    write_json(metadata, meta_path)

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"[{now_utc_iso()}] Done judging. Successes(+{successes}) total={counts['successes']}, "
                 f"Failures(+{failures}) total={counts['failures']}\n")

    return gz_path


# Assessment Phase

def score_to_winfrac(decision: str) -> float:
    d = (decision or "").strip().lower()
    # B is the target/system-under-test; A is baseline
    if d == "b":
        return 1.0
    if d == "a":
        return 0.0
    return 0.5  # tie or unknown

def aggregate_assessment(cfg: RunConfig, judgments_path: str) -> Dict[str, Any]:
    rows = load_jsonl_gz(judgments_path)
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {"error": "No successful judgments found.", "judgments_path": judgments_path}

    total = len(ok)
    wins = sum(1 for r in ok if str(r.get("decision", "")).strip().lower() == "b")
    ties = sum(1 for r in ok if str(r.get("decision", "")).strip().lower() == "tie")
    losses = sum(1 for r in ok if str(r.get("decision", "")).strip().lower() == "a")

    win_rate_tie_half = (wins + 0.5 * ties) / total if total else 0.0

    # Confidence-weighted win rate
    cw_num = 0.0
    cw_den = 0.0
    for r in ok:
        c = float(r.get("confidence", 0.5) or 0.5)
        s = score_to_winfrac(r.get("decision", ""))
        cw_num += c * s
        cw_den += c
    confidence_weighted_wr = (cw_num / cw_den) if cw_den > 0 else None

    # Per paraphrase type
    by_type = collections.defaultdict(list)
    for r in ok:
        by_type[r["paraphrase_key"]].append(r)
    win_rate_by_type = {}
    for k, arr in by_type.items():
        n = len(arr)
        w = sum(1 for r in arr if str(r.get("decision","")).lower() == "b")
        t = sum(1 for r in arr if str(r.get("decision","")).lower() == "tie")
        l = sum(1 for r in arr if str(r.get("decision","")).lower() == "a")
        wr = (w + 0.5 * t) / (n if n else 1)
        win_rate_by_type[k] = {
            "n": n, "wins": w, "ties": t, "losses": l,
            "win_rate": wr
        }

    # RobustAlpaca-style per-task aggregation (macro-avg)
    # Compute per-task win-fractions across paraphrase variants, where
    # tie=0.5, win=1, loss=0. Then for each task, take worst/best/avg/stdev across its paraphrases
    by_task = collections.defaultdict(list)
    for r in ok:
        by_task[r["prompt_count"]].append(score_to_winfrac(r.get("decision","")))
    per_task_stats = []
    for pc, vals in by_task.items():
        if not vals:
            continue
        worst = min(vals)
        best = max(vals)
        avg = sum(vals)/len(vals)
        stdev = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        per_task_stats.append({"prompt_count": pc, "worst": worst, "best": best, "avg": avg, "stdev": stdev})
    macro_avg = {
        "worst": sum(x["worst"] for x in per_task_stats)/len(per_task_stats) if per_task_stats else None,
        "best":  sum(x["best"]  for x in per_task_stats)/len(per_task_stats)  if per_task_stats else None,
        "avg":   sum(x["avg"]   for x in per_task_stats)/len(per_task_stats)   if per_task_stats else None,
        "stdev": sum(x["stdev"] for x in per_task_stats)/len(per_task_stats) if per_task_stats else None,
    }

    # Length-controlled regressions
    # OLS on y in {0,0.5,1}
    y_ols = np.array([score_to_winfrac(r.get("decision","")) for r in ok], dtype=float)
    x = np.array([float(r.get("delta_len", 0) or 0.0) for r in ok], dtype=float)
    X = np.column_stack([np.ones_like(x), x])  # [1, Δlen]
    # β = (X^T X)^-1 X^T y
    try:
        beta = np.linalg.inv(X.T @ X) @ (X.T @ y_ols)
        alpha_hat, beta_hat = float(beta[0]), float(beta[1])
        # Predicted p at Δlen=0 is alpha_hat
        ols_p_at_0 = alpha_hat
        ols_summary = {
            "alpha": alpha_hat,
            "beta": beta_hat,
            "p_at_delta_len_0": ols_p_at_0,
        }
    except np.linalg.LinAlgError:
        ols_summary = {"error": "Singular matrix in OLS"}

    # GLM-Logit on binary y; drop ties
    xy = [(float(r.get("delta_len",0)), 1.0 if str(r.get("decision","")).lower()=="b" else 0.0)
          for r in ok if str(r.get("decision","")).lower() in ("a","b")]
    if len(xy) >= 10 and len(set(y for _,y in xy)) == 2:
        xb = np.array([[1.0, xi] for xi,_ in xy], dtype=float)
        yb = np.array([yi for _,yi in xy], dtype=float)
        # IRLS Newton method
        b = np.zeros(2, dtype=float)
        for _ in range(50):
            z = xb @ b
            p = 1.0 / (1.0 + np.exp(-z))
            W = p * (1 - p) + 1e-9
            # X^T W X
            XTW = xb.T * W
            H = XTW @ xb
            # X^T (y - p)
            g = xb.T @ (yb - p)
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            b_new = b + step
            if np.linalg.norm(b_new - b) < 1e-8:
                b = b_new
                break
            b = b_new
        logit_alpha, logit_beta = float(b[0]), float(b[1])
        p_at_0 = 1.0 / (1.0 + math.exp(-logit_alpha))
        glm_summary = {
            "alpha": logit_alpha,
            "beta": logit_beta,
            "p_at_delta_len_0": p_at_0,
        }
    else:
        glm_summary = {"note": "Insufficient non-tie data for GLM-Logit"}

    assessment = {
        "created_utc": now_utc_iso(),
        "config": dataclasses.asdict(cfg),
        "judgments_path": judgments_path,
        "counts": {
            "total": total,
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
        "overall": {
            "win_rate_tie_half": win_rate_tie_half,
            "confidence_weighted_win_rate": confidence_weighted_wr,
        },
        "by_paraphrase_type": win_rate_by_type,
        "per_task_macro": macro_avg,
        "ols_length_control": ols_summary,
        "glm_logit_length_control": glm_summary,
    }
    return assessment

def format_assessment_human(a: Dict[str, Any]) -> str:
    if "error" in a:
        return f"Assessment error: {a['error']}\n"

    total = a["counts"]["total"]
    wins = a["counts"]["wins"]
    ties = a["counts"]["ties"]
    losses = a["counts"]["losses"]
    wr = a["overall"]["win_rate_tie_half"] * 100.0
    cw = a["overall"]["confidence_weighted_win_rate"]
    cw_s = f"{cw*100.0:.2f}%" if cw is not None else "N/A"

    lines = []
    lines.append("--- Overall Results ---")
    lines.append(f"Total Comparisons: {total}")
    lines.append(f"Win Rate (target vs reference): {wr:.2f}%")
    # show detailed rates as percentages of total
    lines.append(f"  - Wins: {wins} ({wins*100/total:.2f}%)")
    lines.append(f"  - Ties: {ties} ({ties*100/total:.2f}%)")
    lines.append(f"  - Losses: {losses} ({losses*100/total:.2f}%)")
    lines.append("")
    lines.append("--- Win Rate by Paraphrase Type ---")
    # stable order: alphabetical, but ensure ORIGINAL last for readability
    items = list(a["by_paraphrase_type"].items())
    items.sort(key=lambda kv: (kv[0] == ORIGINAL_KEY, kv[0]))
    for k, v in items:
        wr_k = v["win_rate"] * 100.0
        n = v["n"]
        lines.append(f"- {k: <28}: {wr_k:.2f}% (n={n})")
    lines.append(f"Weighted Win Rate (confidence-weighted): {cw_s}")
    lines.append("")
    lines.append("--- RobustAlpacaEval Summary (per-task worst/best/avg/stdev; then macro-avg) ---")
    macro = a["per_task_macro"]
    if macro["avg"] is not None:
        lines.append("Macro-Average across tasks (vs reference):")
        lines.append(f"  worst: {macro['worst']*100:.2f}%   best: {macro['best']*100:.2f}%   "
                     f"avg: {macro['avg']*100:.2f}%   stdev: {macro['stdev']*100:.2f}%")
    else:
        lines.append("  (Not enough data.)")
    lines.append("")
    lines.append("--- Length-Controlled (OLS) Win Rate (y ~ α + β·Δlen) ---")
    ols = a["ols_length_control"]
    if "alpha" in ols:
        lines.append(f"OLS α (pred. p@Δlen=0): {ols['p_at_delta_len_0']*100:.2f}%   β (len effect): {ols['beta']:.6f}")
        lines.append(f"OLS Length-Controlled Win Rate: {ols['p_at_delta_len_0']*100:.2f}%")
    else:
        lines.append(str(ols))
    lines.append("")
    lines.append("--- Length-Controlled (GLM-Logit) Win Rate (logit(p)=α+β·Δlen) ---")
    glm = a["glm_logit_length_control"]
    if "alpha" in glm:
        lines.append(f"Logit α (p@Δlen=0): {glm['p_at_delta_len_0']*100:.2f}%   β (len effect): {glm['beta']:.6f}")
    else:
        lines.append(str(glm))
    return "\n".join(lines)


def assessment_phase(cfg: RunConfig) -> Tuple[str, str]:
    run_prefix = os.path.join(cfg.outdir, cfg.run_name)
    judgments_path = f"{run_prefix}__judgments.jsonl.gz"
    if not os.path.exists(judgments_path):
        return "", "No judgments file found to assess."

    assessment = aggregate_assessment(cfg, judgments_path)
    assessment_path = f"{run_prefix}__assessment.json"
    write_json(assessment, assessment_path)

    human = format_assessment_human(assessment)
    summary_path = f"{run_prefix}__summary.txt"
    write_text(human, summary_path)

    return assessment_path, summary_path


# Main

def main():
    cfg = parse_args()
    rnd = random.Random(cfg.seed)
    mkdir_p(cfg.outdir)

    run_prefix = os.path.join(cfg.outdir, cfg.run_name)
    master_log = f"{run_prefix}__master.log"
    with open(master_log, "w", encoding="utf-8") as lf:
        lf.write(f"[{now_utc_iso()}] Run start\n")
        lf.write(json.dumps(dataclasses.asdict(cfg), indent=2) + "\n")

    try:
        if cfg.phase in ("judging", "both"):
            with open(master_log, "a", encoding="utf-8") as lf:
                lf.write(f"[{now_utc_iso()}] Starting judging phase\n")
            judgments_path = judging_phase(cfg, master_log, rnd)
            with open(master_log, "a", encoding="utf-8") as lf:
                lf.write(f"[{now_utc_iso()}] Judging phase finished: {judgments_path}\n")

        if cfg.phase in ("assessment", "both"):
            with open(master_log, "a", encoding="utf-8") as lf:
                lf.write(f"[{now_utc_iso()}] Starting assessment phase\n")
            assessment_path, summary_path = assessment_phase(cfg)
            with open(master_log, "a", encoding="utf-8") as lf:
                lf.write(f"[{now_utc_iso()}] Assessment outputs:\n  {assessment_path}\n  {summary_path}\n")

        with open(master_log, "a", encoding="utf-8") as lf:
            lf.write(f"[{now_utc_iso()}] Run complete\n")

        print("\nDone.")
        print(f"- Logs:     {master_log}")
        print(f"- Outputs (prefixed with run name) in: {cfg.outdir}")

    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
    except Exception as e:
        traceback.print_exc()
        with open(master_log, "a", encoding="utf-8") as lf:
            lf.write(f"[{now_utc_iso()}] FATAL ERROR: {e}\n")


if __name__ == "__main__":
    main()
