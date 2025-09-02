#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate paraphrase answers for a single `prompt_count` against the gold answer
from the prompts JSON. Produces a plain-text report with both percentages and
absolute numbers, plus a log section with progress and unexpected events

  python3 h_rae/src/maths_eval/run_eval.py\
    --prompts_json a_data/gsm8k/main.json \
    --answers_json f_finetune/model/alta_gsm8k_base100.json \
    --out h_rae/output/maths_eval/base_report

  python3 h_rae/src/maths_eval/run_eval.py\
    --prompts_json a_data/gsm8k/main.json \
    --answers_json f_finetune/outputs_alternat/alta3/hinf_alta3_gsm8k100_inferences.json \
    --out h_rae/output/maths_eval/hinf_alta3_gsm8k100

  python3 h_rae/src/maths_eval/run_eval.py\
    --prompts_json a_data/gsm8k/main.json \
    --answers_json f_finetune/outputs_alternat/alta4/linf_alta4_gsm8k20_inferences.json \
    --out h_rae/output/maths_eval/hinf_alta4_gsm8k20
"""

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation, getcontext
from typing import Any, Dict, List, Optional, Tuple

getcontext().prec = 28  # high precision for Decimal math

# Regex helpers
NUM_RE = re.compile(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?')
HASH_ANSWER_RE = re.compile(r'####\s*([-+]?\d+(?:\.\d+)?)\b')
SCI_BLOCK_RE = re.compile(
    r'(?P<a>[-+]?\d+(?:\.\d+)?)\s*(?:x|×|\*)\s*10\^(?P<b>[-+]?\d+)',
    flags=re.IGNORECASE
)

# Utilities
def d(x: str) -> Optional[Decimal]:
    """Safe Decimal converter (strip thousands separators)"""
    if x is None:
        return None
    try:
        x2 = re.sub(r'(?<=\d),(?=\d{3}\b)', '', x)
        return Decimal(x2)
    except (InvalidOperation, ValueError):
        return None

def float_equal(a: Decimal, b: Decimal, tol: Decimal = Decimal("1e-9")) -> bool:
    return (a - b).copy_abs() <= tol

def extract_last_scientific_value(text: str, logs: List[str]) -> List[Tuple[int, int, Decimal]]:
    """Find all 'a x 10^b' matches; return list of (start, end, value)"""
    out = []
    for m in SCI_BLOCK_RE.finditer(text):
        a = d(m.group('a'))
        b = d(m.group('b'))
        if a is None or b is None:
            continue
        try:
            val = a * (Decimal(10) ** int(b))
            out.append((m.start(), m.end(), val))
        except Exception as e:
            logs.append(f"[WARN] Sci-notation parse error: {e}")
    return out

def extract_last_number(text: str, logs: List[str], context: str = "") -> Optional[Decimal]:
    """
    Heuristic to extract a single final numeric answer from arbitrary text.
      1) '#### <num>'
      2) last 'a x 10^b' block
      3) last plain numeric token (excluding exponent digits of '10^')
    """
    if not isinstance(text, str) or not text.strip():
        return None

    m = HASH_ANSWER_RE.search(text)
    if m:
        val = d(m.group(1))
        if val is not None:
            return val

    sci = extract_last_scientific_value(text, logs)
    last_sci = max(sci, key=lambda t: t[0]) if sci else None

    nums: List[Tuple[int, int, str]] = [(m.start(), m.end(), m.group(0)) for m in NUM_RE.finditer(text)]
    cleaned: List[Tuple[int, int, str]] = []
    for s, e, tok in nums:
        pre = text[max(0, s-2):s]
        if pre.endswith('^') and '10' in text[max(0, s-4):s-1]:
            continue
        cleaned.append((s, e, tok))

    last_plain = max(cleaned, key=lambda t: t[0]) if cleaned else None

    if last_sci and (not last_plain or last_sci[0] > last_plain[0]):
        logs.append(f"[INFO] ({context}) Using last scientific block as answer.")
        return last_sci[2]
    if last_plain:
        val = d(last_plain[2])
        if val is not None:
            if context:
                logs.append(f"[INFO] ({context}) Using last numeric token: '{last_plain[2]}'")
            return val
    return None

def parse_gold_answer(answer_field: str, logs: List[str]) -> Optional[Decimal]:
    """Extract gold answer from the prompts JSON 'answer' text"""
    if not isinstance(answer_field, str):
        logs.append("[WARN] Gold answer field is not a string.")
        return None
    m = HASH_ANSWER_RE.search(answer_field)
    if m:
        val = d(m.group(1))
        if val is not None:
            return val
        logs.append(f"[WARN] Could not Decimal() gold value: {m.group(1)}")
        return None
    cand = extract_last_number(answer_field, logs, context="gold-fallback")
    if cand is None:
        logs.append("[ERROR] Could not find gold answer (#### <num> missing).")
    else:
        logs.append("[INFO] Gold answer extracted via fallback last-number.")
    return cand

def percent(n: int, dnom: int) -> str:
    if dnom == 0:
        return "0.00%"
    return f"{(n / dnom) * 100:.2f}%"

def is_instruct_key(k: str) -> bool:
    return k.startswith("instruct_") or k == "instruction_original"

# Table rendering (TXT + MD)
def render_table_text(headers: List[str], rows: List[List[str]]) -> str:
    """Render a neat, aligned plain-text table"""
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)

def render_table_md(headers: List[str], rows: List[List[str]]) -> str:
    """Render a Markdown pipe table"""
    head = "| " + " | ".join(headers) + " |"
    sep  = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])

# Builders return (headers, rows)
def build_overall_table(total: int, identified: int, correct: int) -> Tuple[List[str], List[List[str]]]:
    not_identified = total - identified
    wrong = identified - correct
    headers = ["Metric", "Count", "Percent"]
    rows = [
        ["Identified", str(identified), percent(identified, total)],
        ["Not identified", str(not_identified), percent(not_identified, total)],
        ["Correct", str(correct), percent(correct, total)],
        ["Wrong", str(wrong), percent(wrong, total)],
    ]
    if identified > 0:
        rows.append(["Accuracy (of identified)", f"{correct}/{identified}", percent(correct, identified)])
    return headers, rows

def build_per_paraphrase_type_table(stats_by_key: Dict[str, Dict[str, int]]) -> Tuple[List[str], List[List[str]]]:
    headers = [
        "Paraphrase Key", "Total",
        "Identified (n / %)", "Correct (n / %)",
        "Wrong (n / %)", "Not Identified (n / %)"
    ]
    rows: List[List[str]] = []
    for key in sorted(stats_by_key.keys()):
        st = stats_by_key[key]
        total = st.get('total', 0)
        identified = st.get('identified', 0)
        correct = st.get('correct', 0)
        wrong = identified - correct
        not_id = total - identified
        rows.append([
            key, str(total),
            f"{identified} / {percent(identified, total)}",
            f"{correct} / {percent(correct, total)}",
            f"{wrong} / {percent(wrong, total)}",
            f"{not_id} / {percent(not_id, total)}",
        ])
    return headers, rows

def build_per_prompt_id_table(stats_by_id: Dict[int, Dict[str, int]]) -> Tuple[List[str], List[List[str]]]:
    headers = [
        "prompt_count", "Total",
        "Identified (n / %)", "Correct (n / %)",
        "Wrong (n / %)", "Not Identified (n / %)",
        "Accuracy (of identified)"
    ]
    rows: List[List[str]] = []
    for pid in sorted(stats_by_id.keys()):
        st = stats_by_id[pid]
        total = st.get('total', 0)
        identified = st.get('identified', 0)
        correct = st.get('correct', 0)
        wrong = identified - correct
        not_id = total - identified
        acc = percent(correct, identified) if identified > 0 else "0.00%"
        rows.append([
            str(pid), str(total),
            f"{identified} / {percent(identified, total)}",
            f"{correct} / {percent(correct, total)}",
            f"{wrong} / {percent(wrong, total)}",
            f"{not_id} / {percent(not_id, total)}",
            acc
        ])
    return headers, rows

# Main
def main():
    ap = argparse.ArgumentParser(description="Evaluate paraphrases for ALL prompt_count IDs and write .txt and .md reports.")
    ap.add_argument("--prompts_json", required=True, help="Path to prompts JSON (array of dicts).")
    ap.add_argument("--answers_json", required=True, help="Path to answers JSON (array of dicts).")
    ap.add_argument("--out_prefix", required=True, help="Prefix for output files (writes <prefix>.txt and <prefix>.md).")
    args = ap.parse_args()

    logs: List[str] = []

    # Load inputs
    try:
        with open(args.prompts_json, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if not isinstance(prompts, list):
            raise ValueError("prompts_json is not a list.")
    except Exception as e:
        print(f"[FATAL] Failed to read prompts_json: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        with open(args.answers_json, "r", encoding="utf-8") as f:
            answers = json.load(f)
        if not isinstance(answers, list):
            raise ValueError("answers_json is not a list.")
    except Exception as e:
        print(f"[FATAL] Failed to read answers_json: {e}", file=sys.stderr)
        sys.exit(3)

    # Map prompt_count -> prompt record (for gold)
    prompt_map: Dict[int, Dict[str, Any]] = {}
    for rec in prompts:
        if not isinstance(rec, dict):
            logs.append("[WARN] Non-dict record in prompts_json skipped.")
            continue
        pc = rec.get("prompt_count")
        if isinstance(pc, int):
            prompt_map[pc] = rec
        else:
            logs.append("[WARN] prompts_json record missing/invalid prompt_count.")

    # Prepare tracking
    overall_total = 0
    overall_identified = 0
    overall_correct = 0

    stats_by_key: Dict[str, Dict[str, int]] = {}  # paraphrase key aggregates
    stats_by_id: Dict[int, Dict[str, int]] = {}   # per prompt_count aggregates

    total_answer_rows = sum(1 for r in answers if isinstance(r, dict) and isinstance(r.get("prompt_count"), int))
    processed_rows = 0
    processed_paraphrases = 0

    # Process all answer rows
    for ans_row in answers:
        if not isinstance(ans_row, dict):
            logs.append("[WARN] Non-dict record in answers_json skipped.")
            continue
        pid = ans_row.get("prompt_count")
        if not isinstance(pid, int):
            logs.append("[WARN] answers_json row missing/invalid prompt_count; skipping.")
            continue

        p_rec = prompt_map.get(pid)
        if p_rec is None:
            logs.append(f"[WARN] No prompts_json record for prompt_count {pid}; skipping this ID.")
            processed_rows += 1
            logs.append(f"[PROGRESS] IDs processed: {processed_rows}/{total_answer_rows} ({percent(processed_rows, total_answer_rows)})")
            continue

        gold_text = p_rec.get("answer", "")
        gold = parse_gold_answer(gold_text, logs)
        if gold is None:
            logs.append(f"[ERROR] Could not extract gold answer for prompt_count {pid}; skipping this ID.")
            processed_rows += 1
            logs.append(f"[PROGRESS] IDs processed: {processed_rows}/{total_answer_rows} ({percent(processed_rows, total_answer_rows)})")
            continue

        keys = [k for k in ans_row.keys() if is_instruct_key(k)]
        if not keys:
            logs.append(f"[WARN] No paraphrase keys for prompt_count {pid}.")
            stats_by_id.setdefault(pid, {"total": 0, "identified": 0, "correct": 0})
            processed_rows += 1
            logs.append(f"[PROGRESS] IDs processed: {processed_rows}/{total_answer_rows} ({percent(processed_rows, total_answer_rows)})")
            continue

        id_total = 0
        id_identified = 0
        id_correct = 0

        for k in keys:
            id_total += 1
            overall_total += 1
            stats_by_key.setdefault(k, {"total": 0, "identified": 0, "correct": 0})
            stats_by_key[k]["total"] += 1

            txt = ans_row.get(k, "")
            pred_val: Optional[Decimal] = None
            if isinstance(txt, str) and txt.strip():
                pred_val = extract_last_number(txt, logs, context=f"{pid}:{k}")
                if pred_val is None:
                    if re.search(r'\d', txt):
                        logs.append(f"[WARN] ({pid}:{k}) Has digits but no parseable final number.")
            else:
                logs.append(f"[WARN] ({pid}:{k}) Missing or non-string paraphrase text.")

            if pred_val is not None:
                id_identified += 1
                overall_identified += 1
                stats_by_key[k]["identified"] += 1
                if float_equal(pred_val, gold):
                    id_correct += 1
                    overall_correct += 1
                    stats_by_key[k]["correct"] += 1

            processed_paraphrases += 1
            logs.append(f"[PROGRESS] Paraphrases processed: {processed_paraphrases} (current ID {pid})")

        stats_by_id[pid] = {
            "total": id_total,
            "identified": id_identified,
            "correct": id_correct
        }

        processed_rows += 1
        logs.append(f"[PROGRESS] IDs processed: {processed_rows}/{total_answer_rows} ({percent(processed_rows, total_answer_rows)})")

    # Build sections
    txt_sections: List[str] = []
    md_sections: List[str]  = []

    # Header
    title = "Evaluation Report (ALL IDs)"
    txt_sections.append(title)
    txt_sections.append("=" * len(title))
    txt_sections.append("")  # spacing

    md_sections.append(f"# {title}")
    md_sections.append("")

    # Overall
    headers, rows = build_overall_table(overall_total, overall_identified, overall_correct)
    txt_sections.append("OVERALL SUMMARY")
    txt_sections.append("----------------")
    txt_sections.append(render_table_text(headers, rows))
    txt_sections.append("")

    md_sections.append("## Overall Summary")
    md_sections.append(render_table_md(headers, rows))
    md_sections.append("")

    # Per-paraphrase-type
    headers, rows = build_per_paraphrase_type_table(stats_by_key)
    txt_sections.append("PER-PARAPHRASE-TYPE SUMMARY")
    txt_sections.append("---------------------------")
    txt_sections.append(render_table_text(headers, rows))
    txt_sections.append("")

    md_sections.append("## Per-Paraphrase-Type Summary")
    md_sections.append(render_table_md(headers, rows))
    md_sections.append("")

    # Per-prompt-id
    headers, rows = build_per_prompt_id_table(stats_by_id)
    txt_sections.append("PER-PROMPT-ID SUMMARY")
    txt_sections.append("---------------------")
    txt_sections.append(render_table_text(headers, rows))
    txt_sections.append("")

    md_sections.append("## Per-Prompt-ID Summary")
    md_sections.append(render_table_md(headers, rows))
    md_sections.append("")

    # Logs
    txt_sections.append("LOGS")
    txt_sections.append("----")
    txt_sections.extend(logs)
    txt_sections.append("")

    md_sections.append("## Logs")
    md_sections.append("")
    md_sections.append("```text")
    md_sections.extend(logs)
    md_sections.append("```")
    md_sections.append("")

    # Write files
    out_txt = f"{args.out_prefix}.txt"
    out_md  = f"{args.out_prefix}.md"

    try:
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_sections))
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_sections))
    except Exception as e:
        print(f"[FATAL] Failed to write outputs: {e}", file=sys.stderr)
        sys.exit(4)

    print(f"[OK] Wrote: {out_txt}")
    print(f"[OK] Wrote: {out_md}")

if __name__ == "__main__":
    main()
