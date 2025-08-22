
#!/usr/bin/env python3
"""
SCRIPT A

prep_first_dataset_sampler.py

Purpose:
- From the FIRST dataset (500 prompts, many paraphrases per prompt),
  build the selection required for the Jacobian experiments under the expert's constraints:

  * Only use paraphrase styles (instruct_* types) with >= MIN_TYPE_COVERAGE items
    whose paraphrase_content_score == MIN_PCS (default 5).
  * For each prompt_count, sample PER_PROMPT_N paraphrases among those that satisfy
    the constraints (score==5 AND type ∈ allowed_types). If a prompt has fewer than
    PER_PROMPT_N eligible paraphrases, it is skipped.
  * Target up to MAX_PROMPTS prompts.

Outputs (to --outdir):
- selection_summary.json: global counts per instruct_type and run metadata.
- type_counts.csv: table of counts per instruct_type (score==MIN_PCS).
- selected_prompts.csv: table of selected prompt_count IDs and how many paraphrases were taken.
- jacobian_prompts.jsonl: one JSONL per selected prompt with fields:
    {
      "prompt_count": int,
      "instruction_original": str,
      "input": str,
      "paraphrases": [{"key": "instruct_...", "text": "..."} * PER_PROMPT_N]
    }
- Graphics:
  * type_counts_bar.png: bar chart of the top-N (configurable) eligible types by count.
  * paraphrases_per_prompt_hist.png: histogram of eligible paraphrase counts per prompt.
  * selection_progress.png: cumulative number of selected prompts vs processed prompts.

Notes:
- Reproducible sampling with --seed (default: 42).
- Very verbose logging; logs every step and warnings when unexpected structures are found.

prep_first_dataset_sampler.py  (STRICT uniqueness logging + default skip list)

Changes requested:
1) If a prompt has multiple paraphrases of the **same style**, just **log** it
   (write duplicates_found.csv) and continue. We also de-duplicate within the
   selection pool so at most **one** paraphrase per style enters a prompt's pool.
2) By default, **skip** the following styles:
   - Exact matches:
     instruct_esperanto, instruct_base64, instruct_chinese_simplified, instruct_contradictory_ask,
     instruct_deadpan, instruct_dogmatic, instruct_double_negative, instruct_email, instruct_emojy_only,
     instruct_error_message, instruct_exact_numbers, instruct_evidence_cited_md, instruct_expert_consensus,
     instruct_fact_check_inline, instruct_fashion_jargon, instruct_fewest_words, instruct_finance_jargon,
     instruct_footnotes, instruct_french, instruct_fuzzy_numbers, instruct_garden_path, instruct_german,
     instruct_haiku, instruct_hinglish, instruct_json_format, instruct_klington, instruct_leet_speak,
     instruct_legal, instruct_lighthearted, instruct_lyrical, instruct_malpropism, instruct_marketing,
     instruct_medical_jargon, instruct_meta_question, instruct_morse_code, instruct_musical_notation,
     instruct_news_headline, instruct_no_spaces, instruct_noninalization, instruct_output, instruct_paradox,
     instruct_plan_execute_reflect, instruct_poetic, instruct_rap_verse, instruct_react_tool_calls,
     instruct_redundant_waffle, instruct_regex_pattern, instruct_reversed_text, instruct_roman_numeral,
     instruct_rot13, instruct_rubric_scored, instruct_salsey, instruct_scientific_notation,
     instruct_see_attached_diagram, instruct_silly, instruct_singlish, instruct_small_hex_blob,
     instruct_software_jargon, instruct_spanglish, instruct_spanish, instruct_sports_jargon,
     instruct_sql_snippet, instruct_summary_then_detail, instruct_surreal, instruct_table_layout,
     instruct_tech, instruct_validator_pass, instruct_yaml_block, instruct_yes_no
   - Prefix matches:
     instruct_condensed_then_expand*, instruct_csv*, instruct_with*

Other behavior unchanged:
- Only styles with paraphrase_content_score==MIN_PCS are counted/eligible.
- Allowed types require global coverage >= --min_type_coverage.
- Per-prompt, sample exactly --per_prompt_n eligible styles (or skip prompt),
  unless --allow_partial_if_less is used.
"""

from __future__ import annotations
import argparse, json, logging, random, sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def _is_paraphrase_obj_valid(pobj: dict) -> bool:
    return isinstance(pobj, dict) and \
           ("instruct_type" in pobj or "instruct_type_name" in pobj) and \
           ("paraphrase_content_score" in pobj) and \
           ("paraphrase" in pobj)

# --- Skip list (exact + prefixes) ---
SKIP_EXACT = {
    "instruct_esperanto","instruct_base64","instruct_chinese_simplified","instruct_contradictory_ask",
    "instruct_deadpan","instruct_dogmatic","instruct_double_negative","instruct_email","instruct_emojy_only",
    "instruct_error_message","instruct_exact_numbers","instruct_evidence_cited_md","instruct_expert_consensus",
    "instruct_fact_check_inline","instruct_fashion_jargon","instruct_fewest_words","instruct_finance_jargon",
    "instruct_footnotes","instruct_french","instruct_fuzzy_numbers","instruct_garden_path","instruct_german",
    "instruct_haiku","instruct_hinglish","instruct_json_format","instruct_klington","instruct_leet_speak",
    "instruct_legal","instruct_lighthearted","instruct_lyrical","instruct_malpropism","instruct_marketing",
    "instruct_medical_jargon","instruct_meta_question","instruct_morse_code","instruct_musical_notation",
    "instruct_news_headline","instruct_no_spaces","instruct_noninalization","instruct_output","instruct_paradox",
    "instruct_plan_execute_reflect","instruct_poetic","instruct_rap_verse","instruct_react_tool_calls",
    "instruct_redundant_waffle","instruct_regex_pattern","instruct_reversed_text","instruct_roman_numeral",
    "instruct_rot13","instruct_rubric_scored","instruct_salsey","instruct_scientific_notation",
    "instruct_see_attached_diagram","instruct_silly","instruct_singlish","instruct_small_hex_blob",
    "instruct_software_jargon","instruct_spanglish","instruct_spanish","instruct_sports_jargon",
    "instruct_sql_snippet","instruct_summary_then_detail","instruct_surreal","instruct_table_layout",
    "instruct_tech","instruct_validator_pass","instruct_yaml_block","instruct_yes_no",
    "instruct_condensed_then_expand", "instruct_csv", "instruct_with"
}
SKIP_PREFIXES = ["instruct_condensed_then_expand", "instruct_csv", "instruct_with", "instruct_output"]

def style_is_skipped(t: str) -> bool:
    if t in SKIP_EXACT:
        return True
    for p in SKIP_PREFIXES:
        if t.startswith(p):
            return True
    return False

def load_first_dataset(first_json_path: str | Path) -> List[dict]:
    path = Path(first_json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"FIRST dataset must be a JSON array; got {type(data)}")
    return data

def find_duplicates(data: List[dict]) -> List[Tuple[int,str,int]]:
    """
    Returns list of (prompt_count, style, occurrences) where occurrences > 1.
    (Checked across all styles, including ones we may later skip.)
    """
    dup_rows: List[Tuple[int,str,int]] = []
    for obj in data:
        pc = int(obj.get("prompt_count", -1))
        seen: Dict[str,int] = {}
        paras = obj.get("paraphrases", [])
        if not isinstance(paras, list): continue
        for pobj in paras:
            if not _is_paraphrase_obj_valid(pobj): continue
            t = pobj.get("instruct_type") or pobj.get("instruct_type_name")
            if not t: continue
            seen[t] = seen.get(t, 0) + 1
        for t, c in seen.items():
            if c > 1:
                dup_rows.append((pc, t, c))
    return dup_rows

def main():
    ap = argparse.ArgumentParser(description="Prepare FIRST dataset selection for Jacobian experiments (log duplicates, skip style list).")
    ap.add_argument("--first_json", type=str, required=True, help="Path to FIRST dataset JSON.")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--min_paraphrase_content_score", type=int, default=5)
    ap.add_argument("--min_type_coverage", type=int, default=200,
                    help="Only styles with >= this many paraphrases at the specified score are allowed.")
    ap.add_argument("--per_prompt_n", type=int, default=50,
                    help="Paraphrases sampled per selected prompt.")
    ap.add_argument("--max_prompts", type=int, default=100,
                    help="Max number of prompts to include in selection.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bar_top_n", type=int, default=30, help="Top-N types to show in bar chart.")
    ap.add_argument("--allow_partial_if_less", action="store_true",
                    help="If set, include prompts with <per_prompt_n eligible paraphrases, as long as >= min_per_prompt_n.")
    ap.add_argument("--min_per_prompt_n", type=int, default=25,
                    help="Minimum paraphrases per prompt when --allow_partial_if_less is set.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("prep_sampler")

    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    log.info("Loading FIRST dataset from %s", args.first_json)
    data = load_first_dataset(args.first_json)
    log.info("Loaded %d prompt objects.", len(data))

    # Log duplicates (do not exit)
    dups = find_duplicates(data)
    if dups:
        df = pd.DataFrame(dups, columns=["prompt_count","instruct_type","occurrences"]).sort_values(["prompt_count","instruct_type"])
        df.to_csv(outdir / "duplicates_found.csv", index=False)
        log.warning("Detected %d prompts with duplicate styles. Logged to duplicates_found.csv. Continuing.", df.shape[0])
    else:
        log.info("No duplicate styles per prompt detected.")

    # Pass 1: global type counts @ score==MIN_PCS, applying skip list
    MIN_PCS = int(args.min_paraphrase_content_score)
    type_counts: Dict[str, int] = {}
    count_all_score5_per_prompt: List[Tuple[int,int]] = []  # after skip list
    prompt_meta: Dict[int, dict] = {}

    for obj in data:
        pc = int(obj.get("prompt_count", -1))
        prompt_meta[pc] = {
            "instruction_original": obj.get("instruction_original", ""),
            "input": obj.get("input", "") or obj.get("scenarios", "") or ""
        }
        paras = obj.get("paraphrases", [])
        if not isinstance(paras, list):
            count_all_score5_per_prompt.append((pc, 0))
            continue
        cnt_all = 0
        # local de-duplication per style to honor the invariant in the pool later
        seen_styles: Dict[str,bool] = {}
        for pobj in paras:
            if not _is_paraphrase_obj_valid(pobj): continue
            t = pobj.get("instruct_type") or pobj.get("instruct_type_name")
            if not t or style_is_skipped(t): continue
            if seen_styles.get(t, False):  # duplicate within this prompt
                continue
            pcs = int(pobj.get("paraphrase_content_score", -1))
            if pcs == MIN_PCS:
                type_counts[t] = type_counts.get(t, 0) + 1
                cnt_all += 1
                seen_styles[t] = True
        count_all_score5_per_prompt.append((pc, cnt_all))

    # Allowed types by global coverage
    allowed_types = sorted([t for t,c in type_counts.items() if c >= args.min_type_coverage])
    log.info("Allowed types (>= %d at score==%d): %d", args.min_type_coverage, MIN_PCS, len(allowed_types))

    # Pass 2: per-prompt counts AFTER allowed-type filtering
    count_allowed_score5_per_prompt: List[Tuple[int,int]] = []
    for obj in data:
        pc = int(obj.get("prompt_count", -1))
        paras = obj.get("paraphrases", [])
        cnt_allowed = 0
        if isinstance(paras, list):
            seen_styles = set()
            for pobj in paras:
                if not _is_paraphrase_obj_valid(pobj): continue
                t = pobj.get("instruct_type") or pobj.get("instruct_type_name")
                if not t or style_is_skipped(t): continue
                if t in seen_styles:  # duplicate style in prompt
                    continue
                pcs = int(pobj.get("paraphrase_content_score", -1))
                if t in allowed_types and pcs == MIN_PCS:
                    cnt_allowed += 1
                    seen_styles.add(t)
        count_allowed_score5_per_prompt.append((pc, cnt_allowed))

    # Save type counts + charts
    type_counts_df = pd.DataFrame(sorted(type_counts.items(), key=lambda x: (-x[1], x[0])),
                                  columns=["instruct_type", "count_at_score"])
    type_counts_df.to_csv(outdir / "type_counts.csv", index=False)

    top_df = type_counts_df.head(args.bar_top_n).copy()
    if not top_df.empty:
        x = top_df["instruct_type"].tolist()[::-1]
        y = top_df["count_at_score"].astype(int).to_numpy()[::-1]
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top_df))))
        ax.barh(x, y); ax.set_xlabel(f"count where paraphrase_content_score == {MIN_PCS}")
        ax.set_title("Paraphrase styles with score=={MIN_PCS} (top-N by count)".format(MIN_PCS=MIN_PCS))
        fig.tight_layout(); fig.savefig(outdir / "type_counts_bar.png", dpi=160); plt.close(fig)

    allowed_df = type_counts_df[type_counts_df["instruct_type"].isin(allowed_types)] \
                    .sort_values("count_at_score", ascending=False) \
                    .head(args.bar_top_n)
    if not allowed_df.empty:
        x2 = allowed_df["instruct_type"].tolist()[::-1]
        y2 = allowed_df["count_at_score"].astype(int).to_numpy()[::-1]
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(allowed_df))))
        ax.barh(x2, y2); ax.set_xlabel(f"count where paraphrase_content_score == {MIN_PCS}")
        ax.set_title(f"ALLOWED styles (>= {args.min_type_coverage} at score=={MIN_PCS}) — top-N")
        fig.tight_layout(); fig.savefig(outdir / "type_counts_bar_ALLOWED.png", dpi=160); plt.close(fig)

    # Histograms (ALL vs ALLOWED) — note: "ALL" here already respects the skip list
    all_df = pd.DataFrame(count_all_score5_per_prompt, columns=["prompt_count","count_score5_all"]) \
               .sort_values("prompt_count")
    all_df.to_csv(outdir / "eligible_paraphrases_per_prompt_ALL.csv", index=False)
    if len(all_df) > 0:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(all_df["count_score5_all"], bins=30)
        ax.set_xlabel(f"# paraphrases with score=={MIN_PCS} (ALL non-skipped types)")
        ax.set_ylabel("# prompts"); ax.set_title("Distribution of score-5 paraphrases per prompt (non-skipped types)")
        fig.tight_layout(); fig.savefig(outdir / "paraphrases_per_prompt_hist_ALL.png", dpi=160); plt.close(fig)

    allowed_per_prompt_df = pd.DataFrame(count_allowed_score5_per_prompt, columns=["prompt_count","count_score5_allowed"]) \
                               .sort_values("prompt_count")
    allowed_per_prompt_df.to_csv(outdir / "eligible_paraphrases_per_prompt_ALLOWED.csv", index=False)
    if len(allowed_per_prompt_df) > 0:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(allowed_per_prompt_df["count_score5_allowed"], bins=30)
        ax.set_xlabel(f"# paraphrases with score=={MIN_PCS} AND type∈allowed_types (non-skipped)")
        ax.set_ylabel("# prompts"); ax.set_title("Eligible paraphrases per prompt (after allowed-types filter)")
        fig.tight_layout(); fig.savefig(outdir / "paraphrases_per_prompt_hist_ALLOWED.png", dpi=160); plt.close(fig)

    # Selection
    selected: List[dict] = []
    selected_rows: List[Tuple[int,int]] = []
    skipped_rows: List[Tuple[int,int,int]] = []  # (prompt_count, n_allowed, required_n)
    processed = 0
    cumulative_selected = []

    for obj in data:
        if len(selected) >= args.max_prompts:
            break
        pc = int(obj.get("prompt_count", -1))
        paras = obj.get("paraphrases", [])
        pool: List[Tuple[str,str]] = []
        if isinstance(paras, list):
            seen_styles = set()
            for pobj in paras:
                if not _is_paraphrase_obj_valid(pobj): continue
                t = pobj.get("instruct_type") or pobj.get("instruct_type_name")
                if not t or style_is_skipped(t): continue
                if t in seen_styles:  # de-duplicate within this prompt
                    continue
                pcs = int(pobj.get("paraphrase_content_score", -1))
                txt = pobj.get("paraphrase", "")
                if t in allowed_types and pcs == MIN_PCS and isinstance(txt, str) and txt.strip():
                    pool.append((t, txt))
                    seen_styles.add(t)
        processed += 1

        need = int(args.per_prompt_n); have = len(pool)
        if have < need:
            if args.allow_partial_if_less and have >= int(args.min_per_prompt_n):
                keep = pool[:]  # include all
                selected.append({
                    "prompt_count": pc,
                    "instruction_original": prompt_meta.get(pc, {}).get("instruction_original",""),
                    "input": prompt_meta.get(pc, {}).get("input",""),
                    "paraphrases": [{"key": k, "text": v} for (k,v) in keep]
                })
                selected_rows.append((pc, len(keep)))
                cumulative_selected.append((processed, len(selected)))
            else:
                skipped_rows.append((pc, have, need))
            continue

        # Sample reproducibly from pool
        idxs = list(range(have)); random.shuffle(idxs)
        keep = [pool[i] for i in idxs[:need]]
        selected.append({
            "prompt_count": pc,
            "instruction_original": prompt_meta.get(pc, {}).get("instruction_original",""),
            "input": prompt_meta.get(pc, {}).get("input",""),
            "paraphrases": [{"key": k, "text": v} for (k,v) in keep]
        })
        selected_rows.append((pc, len(keep)))
        cumulative_selected.append((processed, len(selected)))

    # Outputs
    sel_df = pd.DataFrame(selected_rows, columns=["prompt_count", "selected_n"]).sort_values("prompt_count")
    sel_df.to_csv(outdir / "selected_prompts.csv", index=False)

    if skipped_rows:
        sk_df = pd.DataFrame(skipped_rows, columns=["prompt_count","eligible_allowed_n","required_n"]) \
                  .sort_values("prompt_count")
        sk_df.to_csv(outdir / "skipped_prompts.csv", index=False)

    if cumulative_selected:
        xs, ys = zip(*cumulative_selected)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(xs, ys, marker="."); ax.set_xlabel("prompts processed"); ax.set_ylabel("prompts selected")
        ax.set_title("Selection progress"); fig.tight_layout()
        fig.savefig(outdir / "selection_progress.png", dpi=160); plt.close(fig)

    # Save jsonl
    jpath = outdir / "jacobian_prompts.jsonl"
    with jpath.open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    summary = {
        "first_json": str(Path(args.first_json).resolve()),
        "min_paraphrase_content_score": int(args.min_paraphrase_content_score),
        "min_type_coverage": int(args.min_type_coverage),
        "per_prompt_n": int(args.per_prompt_n),
        "max_prompts": int(args.max_prompts),
        "seed": int(args.seed),
        "n_prompts_total": int(len(data)),
        "n_prompts_selected": int(len(selected)),
        "n_allowed_types": int(len(allowed_types)),
        "allowed_types": allowed_types,
        "allow_partial_if_less": bool(args.allow_partial_if_less),
        "min_per_prompt_n": int(args.min_per_prompt_n),
        "skip_exact_count": len(SKIP_EXACT),
        "skip_prefixes": SKIP_PREFIXES,
    }
    (outdir / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # README
    lines = []
    lines.append("# FIRST dataset selection — summary")
    lines.append("")
    lines.append(f"- Source: `{args.first_json}`")
    lines.append(f"- min paraphrase_content_score: **{int(args.min_paraphrase_content_score)}**")
    lines.append(f"- min type coverage: **{int(args.min_type_coverage)}** (global across dataset)")
    lines.append(f"- per_prompt_n: **{int(args.per_prompt_n)}**  |  max_prompts: **{int(args.max_prompts)}**  |  seed: **{int(args.seed)}**")
    lines.append(f"- allow_partial_if_less: **{bool(args.allow_partial_if_less)}** (min_per_prompt_n={int(args.min_per_prompt_n)})")
    lines.append(f"- skip_exact_count: **{len(SKIP_EXACT)}**, skip_prefixes: {SKIP_PREFIXES}")
    lines.append("")
    lines.append("## What to use next")
    lines.append("- `jacobian_prompts.jsonl` → feed into `paraphrase_subspace_and_jacobian.py`.")
    lines.append("")
    lines.append("## Files")
    lines.append("- `selection_summary.json`")
    lines.append("- `duplicates_found.csv` (if any duplicates detected)")
    lines.append("- `type_counts.csv`")
    lines.append("- `type_counts_bar.png` (ALL non-skipped types)")
    lines.append("- `type_counts_bar_ALLOWED.png` (allowed types only)")
    lines.append("- `eligible_paraphrases_per_prompt_ALL.csv` + `paraphrases_per_prompt_hist_ALL.png` (non-skipped)")
    lines.append("- `eligible_paraphrases_per_prompt_ALLOWED.csv` + `paraphrases_per_prompt_hist_ALLOWED.png`")
    lines.append("- `selected_prompts.csv`")
    if (outdir / "skipped_prompts.csv").exists():
        lines.append("- `skipped_prompts.csv` (eligible counts per skipped prompt)")
    lines.append("- `selection_progress.png`")
    lines.append("- `jacobian_prompts.jsonl`")
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    logging.getLogger("prep_sampler").info("Done. Selected %d prompts. Artifacts written to %s", len(selected), outdir)

if __name__ == "__main__":
    main()

