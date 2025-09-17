#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
import pathlib
from datetime import datetime
from typing import Any, Dict, List, Tuple

import httpx
from tqdm import tqdm

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
ANCHOR_KEY = "instruction_original"

# Limits
EST_TOKENS_LIMIT = 250_000
MAX_SECTION_BYTES = 1_500_000
MAX_PROMPT_BYTES  = 1_600_000
MAX_KEYS_PER_SCHEMA = 8
DEFAULT_MODEL = "gemini-2.5-flash"

# Free tier vers
FREE_RPM = 10
FREE_TPM = 250_000
PACER_HEADROOM = 0.6

# Output sizing
OUT_TOKENS_PER_SCORE = 6
MAX_OUTPUT_TOKENS_CAP = 16384
MIN_OUTPUT_TOKENS     = 256

def _safe_meta(meta: Dict[str, Any] | None) -> Dict[str, Any]:
    try:
        return {str(k): v for k, v in (meta or {}).items() if k != "prompt"}
    except Exception as e:
        return {"_meta_error": f"{type(e).__name__}: {e}"}

import base64
from binascii import Error as B64Error

def _hunt_any_json_string(obj) -> str | None:
    """
    Walk the response recursively; return the first string that parses as JSON.
    Tries to strip code fences and whitespace. Returns the raw JSON string.
    """
    import re
    def try_parse(s: str) -> str | None:
        t = s.strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
            t = re.sub(r"\s*```$", "", t)
        if not (t.startswith("{") or t.startswith("[")):
            m = re.search(r"(\{.*\}|\[.*\])", t, flags=re.DOTALL)
            if m:
                t = m.group(1).strip()
            else:
                return None
        try:
            json.loads(t)
            return t
        except Exception:
            return None

    if isinstance(obj, str):
        return try_parse(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            got = _hunt_any_json_string(v)
            if got: return got
    elif isinstance(obj, list):
        for v in obj:
            got = _hunt_any_json_string(v)
            if got: return got
    return None

def _extract_json_text_from_response(data: Dict[str, Any]) -> str:
    """
    Find JSON/text in Gemini responses across all candidates and part types.
    Supports: text, inlineData (json), functionCall.args, functionResponse,
    executableCode.code, and finally hunts any JSON-looking string
    """
    pf = data.get("promptFeedback") or {}
    if pf.get("blockReason"):
        raise RuntimeError(f"BLOCKED_SAFETY: {pf.get('blockReason')}")

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("NO_CANDIDATES")

    for cand in candidates:
        parts = ((cand.get("content") or {}).get("parts") or [])
        # Plain text
        for p in parts:
            t = p.get("text")
            if isinstance(t, str) and t.strip():
                return t
        # inlineData (json)
        for p in parts:
            inline = p.get("inlineData") or p.get("inline_data")
            if isinstance(inline, dict):
                mt = (inline.get("mimeType") or inline.get("mime_type") or "").lower()
                if mt.startswith("application/json"):
                    b = inline.get("data")
                    if isinstance(b, str) and b:
                        try:
                            return base64.b64decode(b).decode("utf-8", "replace")
                        except B64Error:
                            return b
        # functionCall.args
        for p in parts:
            fc = p.get("functionCall") or p.get("function_call")
            if isinstance(fc, dict) and isinstance(fc.get("args"), (dict, list)):
                try:
                    return json.dumps(fc["args"], ensure_ascii=False)
                except Exception:
                    pass
        # functionResponse nested text
        for p in parts:
            fr = p.get("functionResponse") or p.get("function_response")
            if isinstance(fr, dict):
                args = fr.get("args")
                if isinstance(args, (dict, list)):
                    try:
                        return json.dumps(args, ensure_ascii=False)
                    except Exception:
                        pass
                resp = fr.get("response")
                if isinstance(resp, dict):
                    content = resp.get("content")
                    if isinstance(content, list):
                        for it in content:
                            txt = (it or {}).get("text")
                            if isinstance(txt, str) and txt.strip():
                                return txt
        # executableCode.code starting with JSON
        for p in parts:
            exe = p.get("executableCode") or p.get("executable_code")
            if isinstance(exe, dict):
                code = exe.get("code")
                if isinstance(code, str) and code.strip().startswith(("{", "[")):
                    return code

    hunted = _hunt_any_json_string(data)
    if hunted:
        return hunted

    raise RuntimeError("NO_TEXT_IN_RESPONSE")


class Logger:
    def __init__(self, path: pathlib.Path, run_prefix: str, stem: str, ts: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh_text = open(path, "w", encoding="utf-8", buffering=1)
        self.text_path = path
        self.events_path = path.with_name(f"{run_prefix}{stem}_{ts}.events.ndjson")
        self._fh_json = open(self.events_path, "w", encoding="utf-8", buffering=1)

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            self._fh_text.write(f"[{ts}] {msg}\n")
            self._fh_text.flush()
        except Exception:
            pass

    def event(self, level: str, **fields: Any) -> None:
        payload = {"ts": datetime.now().isoformat(timespec="milliseconds"),
                   "level": level, **fields}
        try:
            self._fh_json.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._fh_json.flush()
        except Exception:
            pass

    def close(self) -> None:
        try: self._fh_text.close()
        except Exception: pass
        try: self._fh_json.close()
        except Exception: pass


class RunMetrics:
    def __init__(self):
        self.total_ids = 0
        self.skipped_missing = 0
        self.already_processed = 0
        self.processed = 0
        self.calls_total = 0
        self.calls_success = 0
        self.calls_recovery = 0
        self.calls_failed = 0
        self.shrink_events = 0
        self.decode_errors = 0
        self.rate_limit_429 = 0
        self.internal_500 = 0

class FailureTally:
    def __init__(self):
        self.counts: Dict[str, int] = {}
        self.samples: Dict[str, str] = {}

    def note(self, key: str, sample: str = ""):
        self.counts[key] = self.counts.get(key, 0) + 1
        if key not in self.samples and sample:
            self.samples[key] = sample[:300]

    def summary_lines(self) -> List[str]:
        keys = sorted(self.counts.keys(), key=lambda k: -self.counts[k])
        out = []
        for k in keys:
            s = f"{k}: {self.counts[k]}"
            if k in self.samples:
                s += f" | e.g. {self.samples[k]}"
            out.append(s)
        return out

# data handling
def read_records(path: pathlib.Path, logger: Logger) -> Dict[str, Dict[str, Any]]:
    try:
        s = path.read_text(encoding="utf-8")
        vec = json.loads(s)
        out: Dict[str, Dict[str, Any]] = {}
        for r in vec:
            if "instruction_original" not in r:
                r["instruction_original"] = r.get("instruction", "") or ""
            pc = str(r.get("prompt_count"))
            out[pc] = r
        return out
    except Exception as e:
        logger.log(f"[fatal-but-skipped] could not parse {path}: {e}")
        logger.event("error", event="load_records_failed", path=str(path), error=str(e))
        return {}


def load_existing_results(path: pathlib.Path, logger: Logger) -> Tuple[List[Dict[str, Any]], set]:
    if path.exists():
        try:
            s = path.read_text(encoding="utf-8")
            arr = json.loads(s)
            processed = {int(v["prompt_count"]) for v in arr if isinstance(v, dict) and "prompt_count" in v}
            return arr, processed
        except Exception as e:
            logger.log(f"[warn] could not parse existing results from {path}: {e}. Starting fresh.")
            logger.event("warn", event="load_existing_results_failed", path=str(path), error=str(e))
            return [], set()
    return [], set()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def schema_for_keys(keys: List[str]) -> Dict[str, Any]:
    props = {k: {"type": "array", "items": {"type": "integer"}} for k in keys}
    return {"type": "object", "properties": props, "required": keys}

def is_ten_ints(v: Any) -> bool:
    return isinstance(v, list) and len(v) == 10 and all(isinstance(x, int) for x in v)

def coerce_ten_ints(v: Any) -> List[int] | None:
    if is_ten_ints(v):
        return v
    if isinstance(v, list) and len(v) == 10:
        out: List[int] = []
        for x in v:
            try:
                if isinstance(x, (int, float)):
                    n = int(round(float(x)))
                elif isinstance(x, str):
                    n = int(round(float(x.strip())))
                else:
                    return None
                out.append(max(0, min(10, n)))
            except Exception:
                return None
        return out if len(out) == 10 else None
    return None

def build_eval_prompt(section: str, keys: List[str]) -> str:
    key_list = '", "'.join(keys)
    return f'''You are an expert evaluator.

For every answer below, assess it against **ten metrics**. Each metric must be scored on a 0–10 integer scale (higher is better).

Metrics (use **exact** order):
1. Task Fulfilment / Relevance
2. Usefulness & Actionability
3. Factual Accuracy & Verifiability
4. Efficiency / Depth & Completeness
5. Reasoning Quality / Transparency
6. Tone & Likeability
7. Adaptation to Context
8. Safety & Bias Avoidance
9. Structure & Formatting & UX Extras
10. Creativity

Return **only** JSON whose **top-level object has exactly these keys**:
["{key_list}"]
Each key maps to a list of **10 integers** (0–10) in the metric order above. No explanations, no extra keys.

Begin data to evaluate:

{section}
'''.strip()

def build_section_for_keys(inst: Dict[str, Any],
                           ans: Dict[str, Any],
                           keys: List[str]) -> str:
    input_opt = inst.get("input")
    if not isinstance(input_opt, str) or not input_opt.strip():
        input_opt = (inst.get("extra", {}) or {}).get("input")
    input_opt = input_opt.strip() if isinstance(input_opt, str) else None

    def get_field(d: Dict[str, Any], key: str, fallback: str) -> str:
        if key in d and isinstance(d[key], str):
            return d[key]
        ex = d.get("extra") or {}
        if key in ex and isinstance(ex[key], str):
            return ex[key]
        return fallback

    lines: List[str] = []
    for key in keys:
        instr = get_field(inst, key, inst.get("instruction_original", ""))
        ans_txt = get_field(ans, key, "")
        lines.append(f"### {key}\n[Instruction]\n{instr}\n")
        if input_opt:
            lines.append(f"\n[Input]\n{input_opt}\n")
        lines.append(f"\n[Answer]\n{ans_txt}\n\n")
    return "".join(lines)

def collect_all_keys(inst: Dict[str, Any], ans: Dict[str, Any], only_answered: bool = True) -> List[str]:
    keys = []
    if isinstance(ans.get(ANCHOR_KEY), str) and ans.get(ANCHOR_KEY).strip():
        keys.append(ANCHOR_KEY)
    for k, v in ans.items():
        if isinstance(k, str) and k.startswith("instruct_"):
            if (not only_answered) or (isinstance(v, str) and v.strip()):
                keys.append(k)
    rest = sorted([k for k in keys if k != ANCHOR_KEY])
    return ([ANCHOR_KEY] + rest) if ANCHOR_KEY in keys else rest

def greedy_batches_by_size(all_keys: List[str],
                           inst: Dict[str, Any],
                           ans: Dict[str, Any],
                           target_batch_size: int) -> List[List[str]]:
    if not all_keys:
        return []
    rest = [k for k in all_keys if k != ANCHOR_KEY]
    batches: List[List[str]] = []
    i = 0
    hard_limit = max(1, min(target_batch_size, MAX_KEYS_PER_SCHEMA))
    while i < len(rest):
        chunk = ([ANCHOR_KEY] if ANCHOR_KEY in all_keys else []) + rest[i:i + max(1, hard_limit - (1 if ANCHOR_KEY in all_keys else 0))]
        while True:
            section = build_section_for_keys(inst, ans, chunk)
            tok_est = estimate_tokens(section)
            if len(section.encode("utf-8")) <= MAX_SECTION_BYTES and tok_est <= EST_TOKENS_LIMIT:
                break
            if len(chunk) <= (1 if ANCHOR_KEY not in chunk else 2):
                break
            chunk = chunk[:-1]
        batches.append(chunk)
        i += max(1, len(chunk) - (1 if ANCHOR_KEY in chunk else 0))
    if not batches and all_keys:
        batches = [all_keys[:1]]
    return batches


def compute_sleep_for_free_tier(tokens_for_request: int) -> float:
    calls_by_tpm = max(1, FREE_TPM // max(1, tokens_for_request))
    target_rpm = max(1, int(min(FREE_RPM, calls_by_tpm) * PACER_HEADROOM))
    return 60.0 / target_rpm


def build_client() -> httpx.Client:
    headers = {"Content-Type": "application/json"}
    return httpx.Client(
        headers=headers,
        timeout=httpx.Timeout(60.0, read=300.0),
        limits=httpx.Limits(max_keepalive_connections=2, max_connections=2),
        follow_redirects=True,
    )

def query_gemini(client: httpx.Client,
                 api_key: str,
                 model: str,
                 schema: Dict[str, Any],
                 prompt: str,
                 logger: Logger,
                 meta: Dict[str, Any],
                 max_output_tokens: int,
                 thinking_budget: int,
                 failure_tally,  # FailureTally
                 verbose_errors: bool,
                 dump_quota: List[int] | None = None) -> Dict[str, Any]:
    url = f"{ENDPOINT}/models/{model}:generateContent?key={api_key}"

    def _post(body: Dict[str, Any]) -> Dict[str, Any]:
        log_meta = _safe_meta(meta)
        logger.event("info", event="api_request", url=url, model=model,
                     bytes_request=len(json.dumps(body, ensure_ascii=False).encode("utf-8")),
                     meta=log_meta)
        resp = client.post(url, json=body)
        if not resp.is_success:
            txt = (resp.text or "").strip()
            logger.log(f"[error] http {resp.status_code} body={txt[:500]}")
            logger.event("warn", event="api_non_2xx", status=resp.status_code, body=txt[:5000], meta=log_meta)
            if resp.status_code == 429:
                failure_tally.note("rate_limit_429", txt)
                raise RuntimeError(f"RATE_LIMIT_429: {txt}")
            failure_tally.note(f"http_{resp.status_code}", txt)
            raise RuntimeError(f"{resp.status_code} — {txt}")
        return resp.json()

    def _dump_raw(tag: str, data_obj: Any):
        if not dump_quota or dump_quota[0] <= 0:
            return
        dump_quota[0] -= 1
        fn = logger.text_path.with_name(f"{logger.text_path.stem}.{tag}.{meta.get('req_id','xxxx')}.raw.json")
        try:
            fn.write_text(json.dumps(data_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.log(f"[dump] wrote raw response to {fn}")
        except Exception as e:
            logger.log(f"[dump-error] failed to write raw response: {e}")

    gen_cfg_base = {
        "temperature": 0.0,
        "topK": 1,
        "topP": 1.0,
        "maxOutputTokens": int(max_output_tokens),
        # HARD FIX: disable thinking for Flash so it doesn't consume all tokens
        "thinkingConfig": {"thinkingBudget": int(thinking_budget)},
    }

    # First attempt: schema + JSON mime
    body1 = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            **gen_cfg_base,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY",   "threshold": "BLOCK_ONLY_HIGH"},
        ],
    }

    try:
        data1 = _post(body1)
        json_text = _extract_json_text_from_response(data1)
    except RuntimeError as e:
        msg = str(e)
        if "NO_TEXT_IN_RESPONSE" in msg or "NO_CANDIDATES" in msg or "BLOCKED_CANDIDATE" in msg or "MAX_TOKENS" in msg:
            failure_tally.note("schema_fallback", msg)
            if verbose_errors:
                logger.log("[warn] schema fallback due to odd/empty response")
            _dump_raw("noschema_probe", locals().get("data1", {}))
            body2 = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    **gen_cfg_base,
                    # keep JSON mime even without schema to bias structured output
                    "responseMimeType": "application/json",
                },
                "safetySettings": body1["safetySettings"],
            }
            data2 = _post(body2)
            try:
                json_text = _extract_json_text_from_response(data2)
            except RuntimeError as e2:
                _dump_raw("plain_probe", data2)
                raise
        else:
            if verbose_errors:
                logger.log(f"[error] response-shape cause={msg} data_preview={str(locals().get('data1', {}))[:400]}")
            raise

    if not isinstance(json_text, str):
        failure_tally.note("not_string_text", type(json_text).__name__)
        logger.log(f"[error] response text not str (got {type(json_text).__name__})")
        raise RuntimeError("unexpected response text type")

    # Robust parse: try full string, then salvage the first valid JSON substring
    text_try = json_text.strip()
    try:
        parsed = json.loads(text_try)
        logger.event("info", event="api_success", meta=_safe_meta(meta))
        return parsed
    except json.JSONDecodeError:
        salvage = _hunt_any_json_string(text_try)
        if salvage:
            try:
                parsed = json.loads(salvage)
                logger.event("info", event="api_success_salvaged", meta=_safe_meta(meta))
                return parsed
            except Exception as e:
                pass
        failure_tally.note("decode_json_error", text_try[:200])
        if verbose_errors:
            logger.log(f"[error] decode-json text_preview={text_try[:400]}")
        raise RuntimeError(f"DECODE_JSON_ERROR: unable to parse JSON")

def process_single_id(
    id_str: str,
    inst: Dict[str, Any],
    ans_map: Dict[str, Dict[str, Any]],
    client: httpx.Client,
    api_key: str,
    model: str,
    max_attempts: int,
    target_batch_size: int,
    delay_between_batches_ms: int,
    logger: Logger,
    results: List[Dict[str, Any]],
    issues: List[str],
    global_rate_limit_hits: List[int],
    metrics: RunMetrics,
    max_global_429: int,
    failure_tally: FailureTally,
    verbose_errors: bool,
    dump_bad_responses: int,
    thinking_budget: int,
) -> Tuple[int, bool, int]:
    ans = ans_map.get(id_str)
    if not ans:
        issues.append(f"answers missing id {id_str}")
        logger.event("warn", event="missing_answers_for_id", id=id_str)
        return (max_attempts, False, 0)

    keys = collect_all_keys(inst, ans, only_answered=True)
    if not keys:
        issues.append(f"id {id_str}: no answered keys; skipping")
        logger.event("warn", event="no_answered_keys", id=id_str)
        return (1, False, 0)

    batches = greedy_batches_by_size(keys, inst, ans, max(1, target_batch_size))

    logger.log(f"[batch-plan] id {id_str}: {len(batches)} batch(es) (first batch size={len(batches[0]) if batches else 0})")
    logger.event("info", event="batch_plan", id=id_str, batches=len(batches),
                 first_batch_keys=len(batches[0]) if batches else 0, total_keys=len(keys))

    eval_json_all: Dict[str, Any] = {}
    attempts_used_overall = 1
    api_calls_used = 0

    for bix, batch_keys_initial in enumerate(batches, start=1):
        batch_keys = list(batch_keys_initial)
        success = False
        one_batch_result: Dict[str, Any] = {}
        internal_hits_local = 0

        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            section = build_section_for_keys(inst, ans, batch_keys)
            sec_bytes = len(section.encode("utf-8"))
            tok_est_in = estimate_tokens(section)

            expected_out_tokens = max(MIN_OUTPUT_TOKENS, len(batch_keys) * 10 * OUT_TOKENS_PER_SCORE)
            max_output_tokens = min(MAX_OUTPUT_TOKENS_CAP, expected_out_tokens + 128)

            logger.log(f"[call] id {id_str} batch {bix}/{len(batches)} attempt {attempt}/{max_attempts} "
                       f"keys={len(batch_keys)} bytes={sec_bytes} est_tokens_in={tok_est_in} base_out={MIN_OUTPUT_TOKENS} max_out={max_output_tokens}")
            logger.event("info", event="batch_attempt",
                         id=id_str, batch=bix, batches=len(batches), attempt=attempt,
                         keys=len(batch_keys), section_bytes=sec_bytes,
                         est_tokens_in=tok_est_in, est_tokens_out=expected_out_tokens,
                         max_output_tokens=max_output_tokens)

            if sec_bytes > MAX_SECTION_BYTES:
                if len(batch_keys) > 2:
                    metrics.shrink_events += 1
                    logger.log(f"[shrink] id {id_str} batch {bix}: section too large ({sec_bytes}). shrink keys {len(batch_keys)}→{len(batch_keys)-1}")
                    logger.event("info", event="shrink_preflight", id=id_str, batch=bix,
                                 from_keys=len(batch_keys), to_keys=len(batch_keys)-1, reason="section_bytes_exceeded")
                    batch_keys = batch_keys[:-1]
                    continue
                else:
                    issues.append(f"id {id_str}: prompt too large for batch {bix} ({sec_bytes} bytes)")
                    logger.event("error", event="batch_too_large_minimal", id=id_str, batch=bix, bytes=sec_bytes)
                    break

            schema = schema_for_keys(batch_keys)
            prompt = build_eval_prompt(section, batch_keys)

            total_tokens_est = tok_est_in + expected_out_tokens
            pre_sleep = compute_sleep_for_free_tier(total_tokens_est)
            logger.event("info", event="pacer_sleep_pre", id=id_str, batch=bix, seconds=round(pre_sleep, 3))
            time.sleep(pre_sleep)

            req_id = str(uuid.uuid4())[:8]
            req_meta = dict(req_id=req_id, id=id_str, batch=bix, attempt=attempt, keys=len(batch_keys),
                            section_bytes=sec_bytes, est_tokens=tok_est_in,
                            est_tokens_out=expected_out_tokens, pacer_sleep_s=round(pre_sleep, 3))

            try:
                got_raw = query_gemini(client, api_key, model, schema, prompt, logger, req_meta,
                                       max_output_tokens, thinking_budget, failure_tally, verbose_errors,
                                       dump_quota=[dump_bad_responses])
                metrics.calls_total += 1
                metrics.calls_success += 1

                got: Dict[str, Any] = {}
                missing: List[str] = []
                for k in batch_keys:
                    coerced = coerce_ten_ints(got_raw.get(k))
                    if coerced is None:
                        missing.append(k)
                    else:
                        got[k] = coerced

                if missing:
                    logger.log(f"[recover] id {id_str} batch {bix}: recovering {len(missing)} key(s)")
                    logger.event("info", event="recovery_start", id=id_str, batch=bix, missing=len(missing))
                    for rr in range(1, 2 + 1):
                        section_retry = build_section_for_keys(inst, ans, missing)
                        sec_bytes_r = len(section_retry.encode("utf-8"))
                        tok_est_r_in = estimate_tokens(section_retry)
                        expected_out_tokens_r = max(MIN_OUTPUT_TOKENS, len(missing) * 10 * OUT_TOKENS_PER_SCORE)
                        max_output_tokens_r = min(MAX_OUTPUT_TOKENS_CAP, expected_out_tokens_r + 64)
                        schema_retry = schema_for_keys(missing)
                        prompt_retry = build_eval_prompt(section_retry, missing)
                        req_id_r = str(uuid.uuid4())[:8]

                        logger.event("info", event="recovery_attempt", id=id_str, batch=bix, round=rr,
                                     keys=len(missing), section_bytes=sec_bytes_r,
                                     est_tokens_in=tok_est_r_in, est_tokens_out=expected_out_tokens_r,
                                     max_output_tokens=max_output_tokens_r, req_id=req_id_r)
                        time.sleep(compute_sleep_for_free_tier(tok_est_r_in + expected_out_tokens_r))

                        try:
                            got2_raw = query_gemini(client, api_key, model, schema_retry, prompt_retry, logger,
                                                    {"req_id": req_id_r, "id": id_str, "batch": bix,
                                                     "round": rr, "keys": len(missing),
                                                     "section_bytes": sec_bytes_r,
                                                     "est_tokens_in": tok_est_r_in,
                                                     "est_tokens_out": expected_out_tokens_r},
                                                    max_output_tokens_r, thinking_budget, failure_tally,
                                                    verbose_errors, dump_quota=[dump_bad_responses])
                            metrics.calls_total += 1
                            metrics.calls_success += 1
                            metrics.calls_recovery += 1
                            newly_ok = []
                            for k in list(missing):
                                coerced2 = coerce_ten_ints(got2_raw.get(k))
                                if coerced2 is not None:
                                    got[k] = coerced2
                                    newly_ok.append(k)
                            missing = [k for k in missing if k not in newly_ok]
                            if not missing:
                                logger.event("info", event="recovery_done", id=id_str, batch=bix, round=rr)
                                break
                        except RuntimeError as e2:
                            metrics.calls_total += 1
                            metrics.calls_failed += 1
                            failure_tally.note("recovery_failed", str(e2))
                            logger.log(f"[error] recovery failed id={id_str} batch={bix} round={rr} cause={e2}")
                            logger.event("warn", event="recovery_failed", id=id_str, batch=bix, round=rr, error=str(e2))
                            break

                one_batch_result = got
                success = True
                attempts_used_overall = max(attempts_used_overall, attempt)
                api_calls_used += 1
                time.sleep(max(0.2, pre_sleep * 0.5))
                break

            except RuntimeError as e:
                msg = str(e)
                metrics.calls_total += 1
                metrics.calls_failed += 1
                logger.log(f"[error] call failed id={id_str} batch={bix} attempt={attempt} cause={msg}")
                logger.event("warn", event="batch_call_error", id=id_str, batch=bix, attempt=attempt, error=msg)

                if ("NO_TEXT_IN_RESPONSE" in msg or "NO_CANDIDATES" in msg or "BLOCKED_SAFETY" in msg or
                    "MAX_TOKENS" in msg):
                    failure_tally.note("odd_or_blocked_response", msg)
                    if len(batch_keys) > 2:
                        back_half = ([batch_keys[0]] if batch_keys and batch_keys[0] == ANCHOR_KEY else []) + batch_keys[(len(batch_keys) + 1) // 2:]
                        if back_half and back_half != batch_keys:
                            logger.event("info", event="try_other_half", id=id_str, batch=bix,
                                         from_len=len(batch_keys), to_len=len(back_half))
                            batch_keys = back_half
                            time.sleep(1.0)
                            continue
                        new_len = max(2, (len(batch_keys) + 1) // 2)
                        metrics.shrink_events += 1
                        logger.event("info", event="shrink_odd_response", id=id_str, batch=bix, attempt=attempt,
                                     from_keys=len(batch_keys), to_keys=new_len, reason=msg)
                        batch_keys = batch_keys[:new_len]
                        time.sleep(1.0)
                        continue
                    else:
                        if len(batch_keys) == 2:
                            anchor, lone = batch_keys[0], batch_keys[1]
                            try:
                                section1 = build_section_for_keys(inst, ans, [lone])
                                schema1  = schema_for_keys([lone])
                                prompt1  = build_eval_prompt(section1, [lone])
                                tot_est  = estimate_tokens(section1) + (10 * OUT_TOKENS_PER_SCORE)
                                time.sleep(compute_sleep_for_free_tier(tot_est))
                                got1 = query_gemini(client, api_key, model, schema1, prompt1, logger,
                                                    {"req_id": str(uuid.uuid4())[:8], "id": id_str, "batch": bix, "attempt": attempt, "keys": 1},
                                                    max(MIN_OUTPUT_TOKENS, 10 * OUT_TOKENS_PER_SCORE + 64),
                                                    thinking_budget, failure_tally, verbose_errors,
                                                    dump_quota=[dump_bad_responses])
                                metrics.calls_total += 1
                                metrics.calls_success += 1
                                coerced = coerce_ten_ints(got1.get(lone))
                                if coerced is not None:
                                    one_batch_result = {lone: coerced}
                                    success = True
                                    api_calls_used += 1
                                    break
                            except RuntimeError as _:
                                pass

                if ("INVALID_ARGUMENT" in msg and "schema" in msg.lower()):
                    failure_tally.note("schema_complexity", msg)
                    if len(batch_keys) > 2:
                        new_len = max(2, (len(batch_keys) + 1) // 2)
                        metrics.shrink_events += 1
                        logger.log(f"[shrink] id {id_str} batch {bix} attempt {attempt}: schema too complex — keys {len(batch_keys)}→{new_len}")
                        logger.event("info", event="shrink_schema_complexity", id=id_str, batch=bix,
                                     attempt=attempt, from_keys=len(batch_keys), to_keys=new_len)
                        batch_keys = batch_keys[:new_len]
                        time.sleep(1.0)
                        continue
                    else:
                        time.sleep(2.0)
                        continue

                if "RATE_LIMIT_429" in msg:
                    global_rate_limit_hits[0] += 1
                    metrics.rate_limit_429 += 1
                    backoff = min(120, 5 * (2 ** (attempt - 1)))
                    logger.log(f"[429] id {id_str} batch {bix} attempt {attempt} (global hits {global_rate_limit_hits[0]}). backoff {backoff}s")
                    logger.event("warn", event="rate_limit_429", id=id_str, batch=bix, attempt=attempt,
                                 global_hits=global_rate_limit_hits[0], backoff_s=backoff)
                    if global_rate_limit_hits[0] >= max_global_429:
                        raise SystemExit("QUIT_AFTER_MAX_429")
                    time.sleep(backoff)
                    continue

                if "500" in msg or '"status":"INTERNAL"' in msg or "Internal Server Error" in msg:
                    failure_tally.note("internal_500", msg)
                    internal_hits_local += 1
                    if internal_hits_local >= 3:
                        logger.log(f"[warn] id {id_str} batch {bix}: 3×500 INTERNAL — skipping this batch.")
                        logger.event("error", event="internal_500x3_skip_batch", id=id_str, batch=bix)
                        break
                    wait = (1000 * (2 ** attempt)) + (int(time.time() * 1000) % 500)
                    logger.log(f"[500] id {id_str} batch {bix} attempt {attempt}: backoff {wait}ms")
                    logger.event("warn", event="internal_500_backoff", id=id_str, batch=bix,
                                 attempt=attempt, wait_ms=wait)
                    time.sleep(wait / 1000.0)
                    continue

                if "DECODE_JSON_ERROR" in msg or "EOF while parsing" in msg or "unterminated" in msg:
                    failure_tally.note("decode_json_error", msg)
                    metrics.decode_errors += 1
                    if len(batch_keys) > 2:
                        new_len = 1 + max(1, math.ceil((len(batch_keys) - 1) / 2))
                        metrics.shrink_events += 1
                        logger.log(f"[shrink] id {id_str} batch {bix} attempt {attempt}: decode error — keys {len(batch_keys)}→{new_len}")
                        logger.event("info", event="shrink_decode_error", id=id_str, batch=bix,
                                     from_keys=len(batch_keys), to_keys=new_len)
                        batch_keys = batch_keys[:new_len]
                        time.sleep(1.0)
                        continue
                    else:
                        time.sleep(2.0)
                        continue

                if "overloaded" in msg.lower() or "503" in msg:
                    failure_tally.note("overloaded_503", msg)
                    wait = (1500 * (2 ** attempt))
                    logger.log(f"[503] id {id_str} batch {bix} attempt {attempt}: overloaded — backoff {wait}ms")
                    logger.event("warn", event="overloaded_backoff", id=id_str, batch=bix, attempt=attempt, wait_ms=wait)
                    time.sleep(wait / 1000.0)
                    continue

                if attempt < max_attempts:
                    wait = (500 * (2 ** attempt)) + (int(time.time() * 1000) % 300)
                    failure_tally.note("generic_retry", msg)
                    logger.log(f"[retry] id {id_str} batch {bix} attempt {attempt}: {msg} — backoff {wait}ms")
                    logger.event("info", event="generic_backoff", id=id_str, batch=bix, attempt=attempt, wait_ms=wait, error=msg)
                    time.sleep(wait / 1000.0)
                else:
                    issues.append(f"id {id_str} batch {bix}: {msg}")
                    failure_tally.note("exhausted_attempts", msg)
                    logger.event("error", event="batch_attempts_exhausted", id=id_str, batch=bix, error=msg)
                    break

        if not success:
            issues.append(f"id {id_str}: skipped batch {bix}/{len(batches)} after retries")
            logger.event("error", event="batch_skipped", id=id_str, batch=bix)
            continue

        for k in batch_keys:
            v = coerce_ten_ints(one_batch_result.get(k))
            if v is not None:
                eval_json_all[k] = v
            else:
                issues.append(f"id {id_str}: bad or missing shape for key {k} in batch {bix}/{len(batches)}")
                logger.event("warn", event="bad_shape_key", id=id_str, key=k, batch=bix)

    if not eval_json_all:
        issues.append(f"id {id_str}: no keys scored after all batches")
        logger.event("warn", event="id_no_scores", id=id_str)
        return (attempts_used_overall, False, api_calls_used)

    res_obj: Dict[str, Any] = {"prompt_count": inst.get("prompt_count")}
    for k in keys:
        if k in eval_json_all:
            res_obj[k] = eval_json_all[k]
        else:
            issues.append(f"id {id_str}: missing eval key {k} after merge")
            logger.event("warn", event="missing_key_after_merge", id=id_str, key=k)

    results.append(res_obj)
    logger.log(f"[done] id {id_str} processed (partial ok) in {api_calls_used} batch call(s)")
    logger.event("info", event="id_done", id=id_str, api_calls_used=api_calls_used)
    return (attempts_used_overall, True, api_calls_used)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assess paraphrase answers with Gemini (Python port).")
    p.add_argument("instructions", type=pathlib.Path)
    p.add_argument("answers", type=pathlib.Path)
    p.add_argument("output", type=pathlib.Path)
    p.add_argument("--run-name", default=None, help="A name for the run, prepended to log & issues files")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--max-calls", type=int, default=250, dest="max_calls",
                   help="Hard cap on number of API calls this run (counts batches)")
    p.add_argument("--delay-ms", type=int, default=0,
                   help="Milliseconds to wait after every successful batch (legacy; pacer handles free tier)")
    p.add_argument("--api-key", default=None, dest="api_key",
                   help="Google API key (overrides $GOOGLE_API_KEY)")
    p.add_argument("--batch-size", type=int, default=1000,
                   help="Max paraphrases per request (always includes instruction_original)")
    p.add_argument("--max-429", type=int, default=5,
                   help="Abort the whole run after this many total 429s (default 5)")
    p.add_argument("--verbose-errors", action="store_true",
                   help="Log extra error context to the human-readable log")
    p.add_argument("--dump-bad-responses", type=int, default=3,
                   help="Dump up to N raw bad responses to logs/ for inspection (default 3)")
    p.add_argument("--thinking-budget", type=int, default=0,
                   help="Thinking tokens budget for Gemini 2.5 models (set 0 to disable on Flash).")
    return p.parse_args()


def main():
    args = parse_args()

    log_dir = pathlib.Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = args.output.stem
    run_prefix = f"{args.run_name}_" if args.run_name else ""
    log_path = log_dir / f"{run_prefix}{stem}_{ts}.logs"
    logger = Logger(log_path, run_prefix, stem, ts)
    logger.log(f"run started -> model={args.model} log={log_path}")
    logger.event("info", event="run_start", model=args.model, log=str(log_path))

    instr_map = read_records(args.instructions, logger)
    ans_map = read_records(args.answers, logger)
    results, processed_ids = load_existing_results(args.output, logger)

    metrics = RunMetrics()
    metrics.total_ids = len(instr_map)
    metrics.already_processed = len(processed_ids)

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.log("ERROR: provide --api-key or set GOOGLE_API_KEY")
        logger.event("error", event="missing_api_key")
        print("Error: provide --api-key or set GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(2)

    client = build_client()
    failure_tally = FailureTally()

    instr_sorted = sorted(instr_map.items(), key=lambda kv: kv[1].get("prompt_count", 0))
    todo_pairs = []
    missing_ids = []
    for id_str, rec in instr_sorted:
        pc = int(rec.get("prompt_count", 0))
        if pc in processed_ids:
            continue
        if id_str in ans_map:
            todo_pairs.append((id_str, rec))
        else:
            missing_ids.append(id_str)

    issues: List[str] = []
    for mid in missing_ids:
        metrics.skipped_missing += 1
        issues.append(f"answers missing id {mid}")
        logger.event("warn", event="missing_answers_enqueued", id=mid)

    remaining_unprocessed = len(instr_sorted) - len(processed_ids)
    logger.log(f"Total instructions: {len(instr_map)}, Already processed: {len(processed_ids)}, "
               f"Remaining (unprocessed): {remaining_unprocessed}, With answers: {len(todo_pairs)}. "
               f"max_calls={args.max_calls} (counts batches).")
    logger.event("info", event="inventory", total=len(instr_map),
                 already_processed=len(processed_ids), remaining_unprocessed=remaining_unprocessed,
                 with_answers=len(todo_pairs), max_calls=args.max_calls)

    if not todo_pairs:
        print(f"Nothing to do (either all processed or missing answers). Log: {log_path}")
        logger.event("info", event="nothing_to_do")
        logger.close()
        return

    calls_made = 0
    bar = tqdm(total=len(todo_pairs), unit="id")

    global_rate_limit_hits = [0]

    try:
        for idx, (id_str, inst) in enumerate(todo_pairs, start=1):
            if calls_made >= args.max_calls:
                logger.log(f"Reached max-calls cap ({args.max_calls}) — stopping cleanly.")
                logger.event("info", event="max_calls_reached", calls=calls_made, cap=args.max_calls)
                break

            progress_pct = round(100.0 * (idx-1) / len(todo_pairs), 2)
            logger.log(f"▶ start id {id_str} ({idx}/{len(todo_pairs)} | {progress_pct}% done)")
            logger.event("info", event="id_start", id=id_str, index=idx, total=len(todo_pairs), progress_pct=progress_pct)

            try:
                attempts_used_overall, processed_this_run, calls_used_for_id = process_single_id(
                    id_str=id_str,
                    inst=inst,
                    ans_map=ans_map,
                    client=client,
                    api_key=api_key,
                    model=args.model,
                    max_attempts=args.max_attempts,
                    target_batch_size=args.batch_size,
                    delay_between_batches_ms=args.delay_ms,
                    logger=logger,
                    results=results,
                    issues=issues,
                    global_rate_limit_hits=global_rate_limit_hits,
                    metrics=metrics,
                    max_global_429=args.max_429,
                    failure_tally=failure_tally,
                    verbose_errors=args.verbose_errors,
                    dump_bad_responses=args.dump_bad_responses,
                    thinking_budget=args.thinking_budget,
                )
            except SystemExit as se:
                with open(args.output, "w", encoding="utf-8") as fh:
                    json.dump(results, fh, ensure_ascii=False, indent=2)
                if issues:
                    issues_path = args.output.with_name(f"{run_prefix}{stem}.issues.json")
                    with open(issues_path, "w", encoding="utf-8") as fh:
                        json.dump(issues, fh, ensure_ascii=False, indent=2)
                reason = str(se)
                logger.event("error", event="run_abort", reason=reason, results_path=str(args.output))
                if "QUIT_AFTER_MAX_429" in reason:
                    logger.log("[fatal] hit max 429s; progress saved. Quitting.")
                    print(f"Stopped after {global_rate_limit_hits[0]}×429 per --max-429. Saved progress to {args.output}. Log: {log_path}")
                else:
                    logger.log(f"[fatal] {reason}")
                    print(f"fatal: {reason}. Saved progress to {args.output}. Log: {log_path}")
                print("\nFailure summary:")
                for line in failure_tally.summary_lines():
                    print(" -", line)
                return

            calls_made += calls_used_for_id
            bar.update(1)

            if processed_this_run:
                metrics.processed += 1
                try:
                    with open(args.output, "w", encoding="utf-8") as fh:
                        json.dump(results, fh, ensure_ascii=False, indent=2)
                    logger.event("info", event="checkpoint_saved", path=str(args.output),
                                 ids_done=metrics.processed, calls_so_far=calls_made)
                except Exception as e:
                    logger.log(f"[error] id {id_str}: Failed to save intermediate results: {e}")
                    logger.event("error", event="checkpoint_save_failed", id=id_str, error=str(e))

            if calls_made >= args.max_calls:
                logger.log(f"Reached max-calls cap ({args.max_calls}) — stopping cleanly.")
                logger.event("info", event="max_calls_reached", calls=calls_made, cap=args.max_calls)
                break

        bar.close()
        logger.log("run finished, results are up-to-date")

        summary = {
            "total_ids": metrics.total_ids,
            "already_processed": metrics.already_processed,
            "skipped_missing": metrics.skipped_missing,
            "processed_this_run": metrics.processed,
            "calls_total": metrics.calls_total,
            "calls_success": metrics.calls_success,
            "calls_failed": metrics.calls_failed,
            "calls_recovery": metrics.calls_recovery,
            "shrink_events": metrics.shrink_events,
            "decode_errors": metrics.decode_errors,
            "rate_limit_429_hits": metrics.rate_limit_429,
            "internal_500_hits": metrics.internal_500,
            "results_path": str(args.output),
            "log_path": str(log_path),
            "events_path": str(logger.events_path),
        }
        logger.log(f"SUMMARY: {json.dumps(summary, ensure_ascii=False)}")
        logger.event("info", event="run_summary", **summary)

        if issues:
            issues_path = args.output.with_name(f"{run_prefix}{stem}.issues.json")
            with open(issues_path, "w", encoding="utf-8") as fh:
                json.dump(issues, fh, ensure_ascii=False, indent=2)
            logger.log(f"wrote {len(issues)} issues to {issues_path}")
            logger.event("info", event="issues_written", count=len(issues), path=str(issues_path))

        print("\nFailure summary:")
        for line in failure_tally.summary_lines():
            print(" -", line)

        print(f"\nDone. Log: {log_path}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
