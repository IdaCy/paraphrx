#!/usr/bin/env bash
set -uo pipefail
trap 'echo "› CTRL-C - stopping"; kill -TERM -- -$$' INT TERM
set -m

SCORING_MODEL="gemini-2.5-flash-preview-05-20"
DATASET="alpaca"
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
ORIG_ARGS=("$@") 

while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--key)           GOOGLE_API_KEY="$2"; shift 2;;
    -sm|--scoring-model)         SCORING_MODEL="$2"; shift 2;;
    -d|--dataset)       DATASET="$2";      shift 2;;
    -n|--name)       NAME="$2";      shift 2;;
    -ln|--logname)       LOG_NAME="$2";      shift 2;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

[[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"
[[ -n "$GOOGLE_API_KEY" ]] || { echo "Error: Google API key not set."; exit 1; }


# paths (instructions path points to folder that contains the json files)
INSTR_DIR="a_data/${DATASET}/${NAME}.json"
SCORE_DIR="a_data/${DATASET}/equi_scores/${NAME}_phrx_scores.json"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_phrx_scoring_$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring started"
echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring started" >>"$HOME/times.log"

cargo phrx_equivalence_score \
  --model "$SCORING_MODEL" \
  --log-name "${LOG_NAME}" \
  --api-key "$GOOGLE_API_KEY" \
  --delay-ms 200 \
  --api-call-maximum 250 \
  "$INSTR_DIR" "$SCORE_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring finished"
echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring finished" >>"$HOME/times.log"
exit 0
