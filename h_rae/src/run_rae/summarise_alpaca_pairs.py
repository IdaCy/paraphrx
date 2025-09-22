#!/usr/bin/env python3
"""
Summarize AlpacaEval-style pairwise annotation files produced by the judge

- Input: one or more files, each a JSON array or JSONL with fields like:
    instruction, output_1, output_2, generator_1, generator_2, preference, ...

- Output (stdout): compact table per file + overall summary when multiple files

interp:
- numeric: >0 -> output_1 wins; <0 -> output_2 wins; ==0 tie
- string:  "generator_1"/"model_1"/"1" -> output_1 wins
           "generator_2"/"model_2"/"2" -> output_2 wins
           "tie"/"equal"                -> tie

Approx length-controlled win rate (LC WR):
- bucket by the *model* side's length (output_2) into quintiles and
  average the within-bucket win rate for that side. This is a pragmatic
  approximation to remove verbosity effects. For the reference side,
  we symmetrically bucket on output_1
"""

import argparse, json, math, sys, pathlib, glob
from typing import List, Dict, Any, Tuple, Optional

def _read_any(path: pathlib.Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    txt = txt.strip()
    if not txt:
        return []
    # Try JSON array first
    if txt[0] == "[":
        try:
            arr = json.loads(txt)
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
        except Exception:
            pass
    # Fallback: JSONL
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            # skip bad lines
            continue
    return out

def _winner_from_preference(pref: Any) -> Optional[int]:
    """
    Returns 1 if output_1 wins, 2 if output_2 wins, 0 for tie, None if unknown.
    """
    if isinstance(pref, (int, float)):
        if pref > 0: return 1
        if pref < 0: return 2
        return 0
    if isinstance(pref, str):
        s = pref.strip().lower()
        if s in {"1", "model_1", "generator_1", "a", "first"}: return 1
        if s in {"2", "model_2", "generator_2", "b", "second"}: return 2
        if s in {"tie", "equal", "draw"}: return 0
    return None

def _length(s: Optional[str]) -> int:
    # character count; cheap and consistent with the printed ‘avg_length’ style
    return len(s or "")

def _quantile_bins(values: List[int], q: int = 5) -> List[int]:
    """
    Returns length thresholds (q-1 cut points) to split values into q buckets.
    If too few points, returns empty -> caller handles as 'no LC WR'.
    """
    if not values:
        return []
    vals = sorted(values)
    cuts = []
    for i in range(1, q):
        # quantile position
        pos = i * (len(vals) - 1) / q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            cuts.append(vals[lo])
        else:
            w = pos - lo
            cuts.append(int(vals[lo] * (1 - w) + vals[hi] * w))
    # de-duplicate in degenerate cases
    uniq = []
    for c in cuts:
        if not uniq or c != uniq[-1]:
            uniq.append(c)
    return uniq

def _bucket_index(v: int, cuts: List[int]) -> int:
    # returns 0..len(cuts); right-open buckets
    for i, c in enumerate(cuts):
        if v <= c:
            return i
    return len(cuts)

def _compute_stats(items: List[Dict[str, Any]], consider_model_side: int = 2,
                   approx_lc_quintiles: int = 5) -> Dict[str, Any]:
    """
    consider_model_side: which side's win rate to report as "model" (1 or 2).
    """
    n = 0
    w1 = 0
    w2 = 0
    ties = 0
    lens1: List[int] = []
    lens2: List[int] = []

    # collect winners and lengths
    winners: List[Optional[int]] = []
    for row in items:
        pref = row.get("preference")
        win = _winner_from_preference(pref)
        out1 = row.get("output_1") or ""
        out2 = row.get("output_2") or ""
        l1 = _length(out1)
        l2 = _length(out2)

        if win is None:
            # Skip rows with no interpretable preference
            continue
        n += 1
        winners.append(win)
        if win == 1: w1 += 1
        elif win == 2: w2 += 1
        else: ties += 1

        lens1.append(l1)
        lens2.append(l2)

    if n == 0:
        return {
            "n_total": 0,
            "win_rate_model": None,
            "win_rate_ref": None,
            "ties": 0,
            "avg_length_model": None,
            "avg_length_ref": None,
            "std_error_model": None,
            "approx_lc_winrate_model": None,
        }

    # raw win rates (ties = 0.5 for each)
    # model side is 1 or 2
    wm = w1 if consider_model_side == 1 else w2
    wr_model = (wm + 0.5 * ties) / n
    wr_ref   = 1.0 - wr_model

    # standard error for model’s win rate
    se_model = math.sqrt(max(wr_model * (1 - wr_model), 0.0) / n) if n > 0 else None

    # avg lengths
    avg_len1 = sum(lens1) / len(lens1) if lens1 else 0.0
    avg_len2 = sum(lens2) / len(lens2) if lens2 else 0.0
    avg_len_model = avg_len1 if consider_model_side == 1 else avg_len2
    avg_len_ref   = avg_len2 if consider_model_side == 1 else avg_len1

    # approximate length-controlled win rate for model side:
    # bucket by the *model* side’s length into quintiles and average per-bin WR
    lc_wr_model = None
    if approx_lc_quintiles >= 2:
        model_lengths = lens1 if consider_model_side == 1 else lens2
        cuts = _quantile_bins(model_lengths, q=approx_lc_quintiles)
        if cuts:
            bins_total = [0] * (len(cuts) + 1)
            bins_win   = [0.0] * (len(cuts) + 1)  # include 0.5 for ties

            for row, win, l1, l2 in zip(items, winners, lens1, lens2):
                if win is None:
                    continue
                mlen = l1 if consider_model_side == 1 else l2
                b = _bucket_index(mlen, cuts)
                bins_total[b] += 1
                if (consider_model_side == 1 and win == 1) or (consider_model_side == 2 and win == 2):
                    bins_win[b] += 1.0
                elif win == 0:
                    bins_win[b] += 0.5

            per_bin = []
            for bt, bw in zip(bins_total, bins_win):
                if bt > 0:
                    per_bin.append(bw / bt)
            if per_bin:
                lc_wr_model = sum(per_bin) / len(per_bin)

    return {
        "n_total": n,
        "win_rate_model": wr_model,
        "win_rate_ref": wr_ref,
        "ties": ties,
        "avg_length_model": avg_len_model,
        "avg_length_ref": avg_len_ref,
        "std_error_model": se_model,
        "approx_lc_winrate_model": lc_wr_model,
    }

def _pick_unique(values: List[str]) -> str:
    uniq = sorted(set([v for v in values if v]))
    if not uniq:
        return "(unknown)"
    if len(uniq) == 1:
        return uniq[0]
    return f"{uniq[0]} (+{len(uniq)-1} more)"

def main():
    ap = argparse.ArgumentParser(description="Summarize AlpacaEval pairwise annotation files.")
    ap.add_argument("paths", nargs="+", help="Files (JSON/JSONL) or globs")
    ap.add_argument("--max-samples", type=int, default=0, help="Cap number of rows per file (0 = all)")
    ap.add_argument("--model-side", type=int, default=2, choices=[1,2],
                    help="Which side to report as the 'model' (1 or 2). Default 2.")
    ap.add_argument("--no-length-control", action="store_true",
                    help="Disable approximate length-controlled win rate calculation.")
    args = ap.parse_args()

    # Expand globs
    files: List[pathlib.Path] = []
    for p in args.paths:
        matched = [pathlib.Path(x) for x in glob.glob(p)]
        if not matched:
            matched = [pathlib.Path(p)]
        files.extend(matched)

    if not files:
        print("No files matched.", file=sys.stderr)
        sys.exit(2)

    grand_items: List[Dict[str, Any]] = []
    print("\nRESULTS (per file):\n")
    for f in files:
        if not f.exists():
            print(f"[skip] {f} (not found)")
            continue
        rows = _read_any(f)
        if args.max_samples and args.max_samples > 0:
            rows = rows[:args.max_samples]

        # Extract (likely constant) generator labels for context
        gens1 = [str(r.get("generator_1") or r.get("model_1") or "") for r in rows]
        gens2 = [str(r.get("generator_2") or r.get("model_2") or "") for r in rows]
        g1 = _pick_unique(gens1)
        g2 = _pick_unique(gens2)

        stats = _compute_stats(rows, consider_model_side=args.model_side,
                               approx_lc_quintiles=(1 if args.no_length_control else 5))

        def pct(x):
            return "—" if x is None else f"{100.0 * x:6.2f}"

        def num(x):
            return "—" if x is None else f"{x:.2f}"

        # 95% CI
        ci = "—"
        if stats["std_error_model"] is not None and stats["win_rate_model"] is not None:
            se = stats["std_error_model"]
            p  = stats["win_rate_model"]
            lo = max(0.0, p - 1.96 * se)
            hi = min(1.0, p + 1.96 * se)
            ci = f"[{100*lo:.2f}, {100*hi:.2f}]"

        print(f"File: {f}")
        print(f"  generator_1: {g1}")
        print(f"  generator_2: {g2}   <-- reported as 'model' side" if args.model_side == 2 else
              f"  generator_2: {g2}")
        print(f"  n_total: {stats['n_total']}  | ties: {stats['ties']}")
        print(f"  avg_length_model: {num(stats['avg_length_model'])}  | avg_length_ref: {num(stats['avg_length_ref'])}")
        print(f"  win_rate_model (ties=0.5): {pct(stats['win_rate_model'])}%   SE: {pct(stats['std_error_model'])}%   95% CI: {ci}")
        if not args.no_length_control:
            print(f"  approx_length_controlled_winrate_model: {pct(stats['approx_lc_winrate_model'])}%")
        print()

        grand_items.extend(rows)

    if len(files) > 1:
        print("\nAGGREGATED across all files:\n")
        agg = _compute_stats(grand_items, consider_model_side=args.model_side,
                             approx_lc_quintiles=(1 if args.no_length_control else 5))
        def pct(x): return "—" if x is None else f"{100.0 * x:6.2f}"
        def num(x): return "—" if x is None else f"{x:.2f}"
        ci = "—"
        if agg["std_error_model"] is not None and agg["win_rate_model"] is not None:
            se = agg["std_error_model"]; p = agg["win_rate_model"]
            lo = max(0.0, p - 1.96 * se)
            hi = min(1.0, p + 1.96 * se)
            ci = f"[{100*lo:.2f}, {100*hi:.2f}]"

        print(f"  n_total: {agg['n_total']}  | ties: {agg['ties']}")
        print(f"  avg_length_model: {num(agg['avg_length_model'])}  | avg_length_ref: {num(agg['avg_length_ref'])}")
        print(f"  win_rate_model (ties=0.5): {pct(agg['win_rate_model'])}%   SE: {pct(agg['std_error_model'])}%   95% CI: {ci}")
        if not args.no_length_control:
            print(f"  approx_length_controlled_winrate_model: {pct(agg['approx_lc_winrate_model'])}%")

if __name__ == "__main__":
    main()
