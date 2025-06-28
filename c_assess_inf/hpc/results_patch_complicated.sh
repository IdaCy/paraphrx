#!/usr/bin/env bash

set -uo pipefail
trap 'echo "› CTRL-C – stopping"; kill -TERM -- -$$' INT TERM
set -m

MODEL="gemini-2.5-flash-preview-05-20"
INSTR_DIR="a_data/alpaca/prxed_main_slice_100"
ANSW_DIR="c_assess_inf/output/alpaca/gemma-2-9b-it/answers"
SCORE_DIR="c_assess_inf/output/alpaca/gemma-2-9b-it/scores"
ISSUES_DIR="c_assess_inf/output/alpaca/gemma-2-9b-it/scores_issues"

# default comes from env, but can be overridden with -k|--key
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--key)
      [[ -n "$2" ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      GOOGLE_API_KEY="$2"
      shift 2
      ;;
    -m|--model)
      [[ -n "$2" ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      MODEL="$2"
      shift 2
      ;;
    -ind|--instr-dir)
      [[ -n "$2" ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      INSTR_DIR="$2"
      shift 2
      ;;
    -and|--answers-dir)
      [[ -n "$2" ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      ANSW_DIR="$2"
      shift 2
      ;;
    -sc|--score-dir)
      [[ -n "$2" ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      SCORE_DIR="$2"
      shift 2
      ;;
    -is|--issues-dir)
      [[ -n "$2" ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      ISSUES_DIR="$2"
      shift 2
      ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

[[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"

if [[ -z "${DETACHED_RESULTS_ASSESS:-}" ]]; then
  export DETACHED_RESULTS_ASSESS=1            # marks the second run
  nohup caffeinate -dimsu "$0" "$@" \
        </dev/null >/dev/null 2>&1 &          # background + keep-awake
  disown                                      # release from this shell
  echo "results_q_assess batch detached → check logs/ directory for progress"
  exit 0                                      # give control back to the terminal
fi

# make sure we actually have a key now
if [[ -z "$GOOGLE_API_KEY" ]]; then
  echo "Error: Google API key not set.  Supply it with -k KEY or env GOOGLE_API_KEY." >&2
  exit 1
fi

# everything that remains are explicit types
TYPES=("$@")

# build dirs
MODEL="gemini-2.5-flash-preview-05-20"
INSTR_DIR="a_data/alpaca/prxed_main_slice_100"
ANSW_DIR="c_assess_inf/output/alpaca/gemma-2-9b-it/answers"
SCORE_DIR="c_assess_inf/output/alpaca/gemma-2-9b-it/scores"
ISSUES_DIR="c_assess_inf/output/alpaca/gemma-2-9b-it/scores_issues"

# create a single master log and redirect everything
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/patch_results-$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

TS_START=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS_START – patching run started" >>"$HOME/times.log"

# each type needs its own "--type" argument- build a string
TYPE_STRING=""
for TYPE in "${TYPES[@]}"; do
    TYPE_STRING+=" --type $TYPE"
done

cargo results_patch \
  --instructions-dir "$INSTR_DIR" \
  --answers-dir      "$ANSW_DIR" \
  --scores-dir       "$SCORE_DIR" \
  --issues-dir       "$ISSUES_DIR" \
  $TYPE_STRING \
  --model "$MODEL" \
  --api-key "$GOOGLE_API_KEY"

TS_END=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS_END – patching run finished" >>"$HOME/times.log"
exit 0
