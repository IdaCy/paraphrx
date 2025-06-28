#!/usr/bin/env bash
set -uo pipefail
trap 'echo "› CTRL-C – stopping"; kill -TERM -- -$$' INT TERM
set -m

SCORING_MODEL="gemini-2.5-flash-preview-05-20"
DATASET="alpaca"
INF_MODEL="gemma-2-2b-it"

GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"

ORIG_ARGS=("$@") 

while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--key)           GOOGLE_API_KEY="$2"; shift 2;;
    -sm|--scoring-model)         SCORING_MODEL="$2"; shift 2;;
    -d|--dataset)       DATASET="$2";      shift 2;;
    -im|--inf-model)    INF_MODEL="$2";    shift 2;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

[[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"

if [[ -z "${DETACHED_RESULTS_ASSESS:-}" ]]; then
  export DETACHED_RESULTS_ASSESS=1
  nohup caffeinate -dimsu "$0" "${ORIG_ARGS[@]}" </dev/null >/dev/null 2>&1 &
  disown
  echo "results_q_assess batch detached → check logs/ directory for progress"
  exit 0
fi

[[ -n "$GOOGLE_API_KEY" ]] || { echo "Error: Google API key not set."; exit 1; }

TYPES=("$@")

# paths (instructions path points to folder that contains the json files)
INSTR_DIR="a_data/${DATASET}/paraphrases_500"
ANSW_DIR="c_assess_inf/output/${DATASET}/${INF_MODEL}/answers"
SCORE_DIR="c_assess_inf/output/${DATASET}/${INF_MODEL}/scores_500"
ISSUES_DIR="c_assess_inf/output/${DATASET}/${INF_MODEL}/scores_issues_500"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/patch_results-$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%Y-%m-%d %H:%M:%S') – patching run started" >>"$HOME/times.log"

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
  --model "$SCORING_MODEL" \
  --api-key "$GOOGLE_API_KEY"

echo "$(date '+%Y-%m-%d %H:%M:%S') – patching run finished" >>"$HOME/times.log"
exit 0
