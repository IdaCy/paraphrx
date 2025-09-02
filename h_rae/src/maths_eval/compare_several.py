#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare multiple paraphrase-answer runs against the same prompts file

python compare_answer_runs.py \
  --prompts_json path/to/prompts.json \
  --answers_jsons path/to/run1.json path/to/run2.json path/to/run3.json \
  --run_names run1 run2 run3 \
  --out_prefix out/comparison \
  --keys_filter instruct_polite,instruct_double_negative \
  --log_file out/compare.log
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict, Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# Utilities

Number = float

def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d > 0 else 0.0

def pct_s(n: int, d: int) -> str:
    return f"{pct(n, d):6.2f}%"

def log(log_fh, msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    log_fh.write(f"[{ts} UTC] {msg}\n")
    log_fh.flush()

def load_json_array(path: str, log_fh) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            log(log_fh, f"ERROR: Failed to parse JSON: {path}: {e}")
            raise
    if not isinstance(data, list):
        raise ValueError(f"Top-level JSON at {path} is not a list/array.")
    return data

def try_parse_number(s: str) -> Optional[Number]:
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None

def parse_truth_from_prompt_answer(ans_text: str) -> Optional[Number]:
    if not isinstance(ans_text, str):
        return None
    matches = re.findall(r"####\s*([+-]?\d+(?:\.\d+)?)", ans_text)
    if matches:
        return try_parse_number(matches[-1])
    return None

def parse_answer_from_text(text: str) -> Optional[Number]:
    """
    Heuristic extraction of a final numeric answer from a paraphrase string.
    Order:
      1) Last '#### <number>'
      2) Last '= <number>' (right-hand result)
      3) Rightmost standalone number
    """
    if not isinstance(text, str) or not text.strip():
        return None
    m1 = re.findall(r"####\s*([+-]?\d+(?:\.\d+)?)", text)
    if m1:
        return try_parse_number(m1[-1])
    m2 = re.findall(r"=\s*([+-]?\d+(?:\.\d+)?)\b", text)
    if m2:
        return try_parse_number(m2[-1])
    m3 = re.findall(r"([+-]?\d+(?:\.\d+)?)", text)
    if m3:
        return try_parse_number(m3[-1])
    return None

def is_correct(pred: Number, truth: Number, tol: float = 1e-9) -> bool:
    return (
        pred is not None and truth is not None
        and math.isfinite(pred) and math.isfinite(truth)
        and abs(pred - truth) <= tol
    )

def format_table(rows: List[List[str]], col_sep: str = "  ") -> str:
    if not rows:
        return ""
    cols = max(len(r) for r in rows)
    widths = [0] * cols
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    out_lines = []
    for r in rows:
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(r)]
        out_lines.append(col_sep.join(padded))
    return "\n".join(out_lines)

# Evaluation / Aggregation

def collect_truth(prompts: List[Dict[str, Any]], log_fh) -> Dict[int, Number]:
    truth_map: Dict[int, Number] = {}
    seen = set()
    for i, item in enumerate(prompts):
        pc = item.get("prompt_count")
        if pc is None:
            log(log_fh, f"WARNING: prompts[{i}] missing prompt_count; skipping.")
            continue
        t = parse_truth_from_prompt_answer(item.get("answer", ""))
        if t is None:
            log(log_fh, f"WARNING: prompt_count={pc} has no parseable ground-truth.")
        if pc in seen:
            log(log_fh, f"WARNING: duplicate prompt_count in prompts: {pc} (overwriting).")
        truth_map[pc] = t
        seen.add(pc)
    return truth_map

def union_paraphrase_keys(answer_runs: List[List[Dict[str, Any]]]) -> List[str]:
    keys = set()
    for answers in answer_runs:
        for entry in answers:
            for k in entry.keys():
                if k not in ("prompt_count", "input"):
                    keys.add(k)
    return sorted(keys)

class RunStats:
    def __init__(self, name: str):
        self.name = name
        self.overall = Counter()
        self.by_key: Dict[str, Counter] = defaultdict(Counter)
        # For boxplot: per-key vector of 0/1 correctness for identified & truth-present items
        self.correct_vectors: Dict[str, List[int]] = defaultdict(list)

def evaluate_runs(
    truth: Dict[int, Number],
    answer_runs: List[List[Dict[str, Any]]],
    run_names: List[str],
    all_keys: List[str],
    log_fh
) -> Dict[str, RunStats]:
    total_entries = sum(len(ans) for ans in answer_runs)
    done_entries = 0
    stats: Dict[str, RunStats] = {name: RunStats(name) for name in run_names}

    for run_idx, (run_name, answers) in enumerate(zip(run_names, answer_runs)):
        rs = stats[run_name]
        for entry in answers:
            pc = entry.get("prompt_count")
            if pc is None:
                log(log_fh, f"WARNING: run={run_name} entry missing prompt_count; skipping.")
                continue
            truth_val = truth.get(pc, None)
            for key, text in entry.items():
                if key in ("prompt_count", "input"):
                    continue
                rs.overall["total"] += 1
                rs.by_key[key]["total"] += 1

                pred = None
                try:
                    pred = parse_answer_from_text(text)
                except Exception as e:
                    log(log_fh, f"ERROR: parsing run={run_name}, pc={pc}, key={key}: {e}")

                if pred is None:
                    rs.overall["not_identified"] += 1
                    rs.by_key[key]["not_identified"] += 1
                else:
                    rs.overall["identified"] += 1
                    rs.by_key[key]["identified"] += 1
                    if truth_val is None or not math.isfinite(truth_val):
                        rs.overall["judgement_skipped_truth_missing"] += 1
                        rs.by_key[key]["judgement_skipped_truth_missing"] += 1
                    else:
                        if is_correct(pred, truth_val):
                            rs.overall["correct"] += 1
                            rs.by_key[key]["correct"] += 1
                            rs.correct_vectors[key].append(1)
                        else:
                            rs.overall["wrong"] += 1
                            rs.by_key[key]["wrong"] += 1
                            rs.correct_vectors[key].append(0)

            done_entries += 1
            if done_entries % 50 == 0 or done_entries == total_entries:
                log(log_fh, f"Progress: {done_entries}/{total_entries} ({pct(done_entries, total_entries):.2f}%)")

    # Ensure every known key exists in by_key for consistency
    for rs in stats.values():
        for k in all_keys:
            _ = rs.by_key[k]  # touch to create default counters if missing
            _ = rs.correct_vectors[k]
    return stats

def accuracy_tuple(c: Counter) -> Tuple[float, float, int, int]:
    """Return tuple for ranking: (acc_ident, parse_rate, correct, -wrong)"""
    tot = c.get("total", 0)
    idn = c.get("identified", 0)
    cor = c.get("correct", 0)
    wrg = c.get("wrong", 0)
    acc_ident = (cor / idn) if idn > 0 else -1.0
    parse_rate = (idn / tot) if tot > 0 else -1.0
    return (acc_ident, parse_rate, cor, -wrg)

def rank_runs_overall(stats: Dict[str, RunStats]) -> List[Tuple[str, Counter]]:
    runs = [(name, rs.overall) for name, rs in stats.items()]
    runs.sort(key=lambda x: accuracy_tuple(x[1]), reverse=True)
    return runs

def rank_runs_for_key(stats: Dict[str, RunStats], key: str) -> List[Tuple[str, Counter]]:
    runs = [(name, rs.by_key[key]) for name, rs in stats.items()]
    runs.sort(key=lambda x: accuracy_tuple(x[1]), reverse=True)
    return runs

# Reporting

def add_overall_table(lines: List[str], title: str, ranked: List[Tuple[str, Counter]]) -> None:
    lines.append(title)
    rows = [["Run", "Total", "Identified", "%Parsed", "Correct", "Wrong", "%Acc(Ident)", "%Acc(Total)"]]
    for name, c in ranked:
        tot = c.get("total", 0)
        idn = c.get("identified", 0)
        cor = c.get("correct", 0)
        wrg = c.get("wrong", 0)
        rows.append([
            name,
            str(tot),
            str(idn), pct_s(idn, tot),
            str(cor),
            str(wrg),
            pct_s(cor, idn),
            pct_s(cor, tot),
        ])
    lines.append("```\n" + format_table(rows) + "\n```\n")

def add_key_winners(lines: List[str], stats: Dict[str, RunStats], keys: List[str], title: str) -> None:
    lines.append(title)
    hdr = [["Key", "Best Run", "Acc(Ident)", "Parsed", "Second Best", "Δ Acc (pp)"]]
    rows = []
    for key in keys:
        ranked = rank_runs_for_key(stats, key)
        if not ranked:
            continue
        # Filter only runs that actually have any identified examples; otherwise acc_ident = -1
        ranked_valid = []
        for name, c in ranked:
            idn = c.get("identified", 0)
            cor = c.get("correct", 0)
            if idn > 0:
                ranked_valid.append((name, c, cor / idn))
        if not ranked_valid:
            rows.append([key, "(no data)", "", "", "", ""])
            continue
        ranked_valid.sort(key=lambda x: x[2], reverse=True)
        best_name, best_c, best_acc = ranked_valid[0]
        best_idn = best_c.get("identified", 0)
        best_tot = best_c.get("total", 0)
        if len(ranked_valid) >= 2:
            second_name, second_c, second_acc = ranked_valid[1]
            delta_pp = (best_acc - second_acc) * 100.0
            second_label = f"{second_name} ({second_acc*100:.2f}%)"
            delta_label = f"{delta_pp:.2f}"
        else:
            second_label = "(n/a)"
            delta_label = ""
        rows.append([
            key,
            best_name,
            f"{best_acc*100:.2f}%",
            f"{best_idn}/{best_tot} ({pct_s(best_idn, best_tot)})",
            second_label,
            delta_label
        ])
    lines.append("```\n" + format_table(hdr + rows) + "\n```\n")

def build_markdown_report(
    stats: Dict[str, RunStats],
    all_keys: List[str],
    filtered_keys: Optional[List[str]],
    out_prefix: str,
) -> str:
    lines: List[str] = []
    lines.append(f"# Multi-Run Comparison Report\n")
    lines.append(f"Generated by `compare_answer_runs.py`.\n")

    # OVERALL ranking (all keys)
    overall_ranked = rank_runs_overall(stats)
    add_overall_table(lines, "## Overall Ranking (all keys)\n", overall_ranked)

    # If filtered keys were provided: ranking restricted to those keys
    if filtered_keys:
        # Build synthetic "overall filtered" counters by summing per-key counters
        filtered_overall: List[Tuple[str, Counter]] = []
        for name, rs in stats.items():
            agg = Counter()
            for k in filtered_keys:
                agg.update(rs.by_key[k])
            filtered_overall.append((name, agg))
        filtered_overall.sort(key=lambda x: accuracy_tuple(x[1]), reverse=True)
        add_overall_table(lines, f"## Overall Ranking (filtered keys: {', '.join(filtered_keys)})\n", filtered_overall)

    # Per-key winners (all keys)
    add_key_winners(lines, stats, all_keys, "## Per-Key Winners (all keys)\n")

    # Per-key winners (filtered subset)
    if filtered_keys:
        add_key_winners(lines, stats, filtered_keys, f"## Per-Key Winners (filtered keys: {', '.join(filtered_keys)})\n")

    # Box plot section
    fig_path = f"{out_prefix}_boxplot.png"
    lines.append("## Box Plot\n")
    if filtered_keys:
        lines.append(f"Keys shown: `{', '.join(filtered_keys)}` (always including `instruction_original`).")
    else:
        lines.append("Keys shown: all available paraphrase keys.")
    lines.append("Each box shows the distribution of correctness (1=correct, 0=wrong) **among identified items**.\n")
    lines.append(f"![Correctness by Key and Run]({os.path.basename(fig_path)})\n")

    return "\n".join(lines)

def build_text_report(
    stats: Dict[str, RunStats],
    all_keys: List[str],
    filtered_keys: Optional[List[str]],
    out_prefix: str,
) -> str:
    lines: List[str] = []
    lines.append("Multi-Run Comparison Report\n")
    lines.append("=" * 30 + "\n")

    # OVERALL (all keys)
    lines.append("Overall Ranking (all keys)\n")
    overall_ranked = rank_runs_overall(stats)
    add_overall_table(lines, "", overall_ranked)

    # FILTERED
    if filtered_keys:
        filtered_overall: List[Tuple[str, Counter]] = []
        for name, rs in stats.items():
            agg = Counter()
            for k in filtered_keys:
                agg.update(rs.by_key[k])
            filtered_overall.append((name, agg))
        filtered_overall.sort(key=lambda x: accuracy_tuple(x[1]), reverse=True)
        lines.append(f"\nOverall Ranking (filtered keys: {', '.join(filtered_keys)})\n")
        add_overall_table(lines, "", filtered_overall)

    # Per-key winners
    lines.append("\nPer-Key Winners (all keys)\n")
    add_key_winners(lines, stats, all_keys, "")

    if filtered_keys:
        lines.append(f"\nPer-Key Winners (filtered keys: {', '.join(filtered_keys)})\n")
        add_key_winners(lines, stats, filtered_keys, "")

    fig_path = f"{out_prefix}_boxplot.png"
    lines.append("\nBox plot saved to: " + os.path.basename(fig_path) + "\n")

    return "\n".join(lines)

# Plotting

def draw_boxplot(
    stats: Dict[str, RunStats],
    keys_for_plot: List[str],
    run_names: List[str],
    out_png: str,
    title: str = "Correctness by Key (among identified items)",
    figsize: Tuple[float, float] = (14, 6),
    dpi: int = 160,
) -> None:
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    # Colors per run from Matplotlib's default cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key().get('color', [])
    if len(colors) < len(run_names):
        # Extend colors if many runs
        colors = (colors * ((len(run_names) + len(colors) - 1) // len(colors)))[:len(run_names)]

    group_width = 1.0
    n_runs = len(run_names)
    if n_runs == 0:
        n_runs = 1
    # offsets so boxes sit side-by-side within each key group
    spacing = min(0.8 / max(1, n_runs), 0.25)
    offsets = np.linspace(-spacing*(n_runs-1)/2, spacing*(n_runs-1)/2, n_runs)
    box_width = spacing * 0.8

    plotted_any = False
    # plot per run to control colors/legend
    for r_idx, run in enumerate(run_names):
        rs = stats[run]
        data = []
        positions = []
        for i, key in enumerate(keys_for_plot):
            vec = rs.correct_vectors.get(key, [])
            # vec contains 0/1 for items that were identified and had truth
            if len(vec) == 0:
                # We still create an empty placeholder? Skip to avoid mpl errors
                continue
            data.append(vec)
            positions.append(i * group_width + offsets[r_idx])

        if not data:
            continue
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showmeans=True,
            meanline=False,
            manage_ticks=False,
        )
        # color the boxes for this run
        for patch in bp['boxes']:
            patch.set_facecolor(colors[r_idx])
            patch.set_alpha(0.5)
        for whisk in bp['whiskers']:
            whisk.set_color(colors[r_idx])
        for cap in bp['caps']:
            cap.set_color(colors[r_idx])
        for med in bp['medians']:
            med.set_color("black")
        for mean in bp['means']:
            mean.set_color(colors[r_idx])
        plotted_any = True

    ax.set_title(title)
    ax.set_ylabel("Correctness (0=wrong, 1=correct)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([i * group_width for i in range(len(keys_for_plot))])
    ax.set_xticklabels(keys_for_plot, rotation=35, ha="right")

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors[i], edgecolor='none', alpha=0.5, label=run_names[i]) for i in range(len(run_names))]
    ax.legend(handles=legend_patches, title="Runs", loc="best")

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if not plotted_any:
        # Create a dummy note on the canvas if no data
        ax.text(0.5, 0.5, "No identified+truth data for selected keys.", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png)
    plt.close()

# CLI

def main():
    ap = argparse.ArgumentParser(description="Compare multiple answer runs against prompts ground-truth.")
    ap.add_argument("--prompts_json", required=True, help="Path to prompts JSON.")
    ap.add_argument("--answers_jsons", nargs="+", required=True, help="Paths to answers JSON files (one per run).")
    ap.add_argument("--run_names", nargs="+", required=True, help="Names for each run (same arity/order as --answers_jsons).")
    ap.add_argument("--out_prefix", default="comparison", help="Output prefix for .md, .txt and plot image.")
    ap.add_argument("--keys_filter", default="", help="Optional comma-separated keys to include (instruction_original always included).")
    ap.add_argument("--log_file", default="compare.log", help="Log file path.")
    args = ap.parse_args()

    if len(args.answers_jsons) != len(args.run_names):
        print("ERROR: --answers_jsons and --run_names must have the same length.", file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)

    with open(args.log_file, "w", encoding="utf-8") as log_fh:
        log(log_fh, f"START compare: prompts={args.prompts_json}")
        log(log_fh, f"Runs: {list(zip(args.run_names, args.answers_jsons))}")

        prompts = load_json_array(args.prompts_json, log_fh)
        truth = collect_truth(prompts, log_fh)

        answer_runs = []
        for p in args.answers_jsons:
            answer_runs.append(load_json_array(p, log_fh))

        # Collect union of all paraphrase keys across all runs
        all_keys = union_paraphrase_keys(answer_runs)

        # Evaluate
        stats = evaluate_runs(truth, answer_runs, args.run_names, all_keys, log_fh)

        # Determine filtered keys, if any
        filtered_keys = None
        if args.keys_filter.strip():
            filt = [k.strip() for k in args.keys_filter.split(",") if k.strip()]
            if "instruction_original" not in filt:
                filt.insert(0, "instruction_original")
            # Keep only those that actually exist in the data
            filtered_keys = [k for k in filt if k in all_keys]
            missing = set(filt) - set(filtered_keys)
            if missing:
                log(log_fh, f"NOTICE: some requested --keys_filter keys not present and will be ignored: {sorted(missing)}")

        # Plot
        keys_for_plot = filtered_keys if filtered_keys else all_keys
        fig_path = f"{args.out_prefix}_boxplot.png"
        draw_boxplot(stats, keys_for_plot, args.run_names, fig_path)

        # Reports
        md_report = build_markdown_report(stats, all_keys, filtered_keys, args.out_prefix)
        txt_report = build_text_report(stats, all_keys, filtered_keys, args.out_prefix)

        md_path = f"{args.out_prefix}.md"
        txt_path = f"{args.out_prefix}.txt"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_report)

        log(log_fh, f"DONE. Wrote: {md_path}, {txt_path}, {fig_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"FATAL ERROR: {e}\n")
        sys.exit(1)
