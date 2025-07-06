#!/usr/bin/env bash
set -uo pipefail
trap 'echo "› CTRL-C - stopping"; kill -TERM -- -$$' INT TERM
set -m

SCORING_MODEL="gemini-2.5-flash-preview-05-20"
DATASET="alpaca"
GOOGLE_API_KEY1="${GOOGLE_API_KEY1:-}"
ORIG_ARGS=("$@") 

while [[ $# -gt 0 ]]; do
  case "$1" in
    -k1|--key1)           GOOGLE_API_KEY1="$2"; shift 2;;
    -k2|--key2)           GOOGLE_API_KEY2="$2"; shift 2;;
    -sm|--scoring-model)         SCORING_MODEL="$2"; shift 2;;
    -d|--dataset)       DATASET="$2";      shift 2;;
    -n1|--name1)       NAME1="$2";      shift 2;;
    -n2|--name2)       NAME2="$2";      shift 2;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

[[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"
[[ -n "$GOOGLE_API_KEY1" ]] || { echo "Error: Google API key 1 not set."; exit 1; }
[[ -n "$GOOGLE_API_KEY2" ]] || { echo "Error: Google API key 2 not set."; exit 1; }

echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring received ${NAME1} and ${NAME2} with model ${SCORING_MODEL} and dataset ${DATASET}"

# FIRST ONE

# paths (instructions path points to folder that contains the json files)
INSTR_DIR="a_data/${DATASET}/${NAME1}.json"
SCORE_DIR="a_data/${DATASET}/${NAME1}_phrx_scores.json"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_phrx_scoring_$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring started ${NAME1}"
echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring started ${NAME1}" >>"$HOME/times.log"

cargo phrx_equivalence_score \
  --model "$SCORING_MODEL" \
  --api-key "$GOOGLE_API_KEY1" \
  --delay-ms 200 \
  --api-call-maximum 250 \
  "$INSTR_DIR" "$SCORE_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring finished ${NAME1}"
echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring finished ${NAME1}" >>"$HOME/times.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring started ${NAME2}"
echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring started ${NAME2}" >>"$HOME/times.log"

# SECOND ONE

# paths (instructions path points to folder that contains the json files)
INSTR_DIR="a_data/${DATASET}/${NAME2}.json"
SCORE_DIR="a_data/${DATASET}/${NAME2}_phrx_scores.json"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_phrx_scoring_$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring started ${NAME2}"
echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring started ${NAME2}" >>"$HOME/times.log"

cargo phrx_equivalence_score \
  --model "$SCORING_MODEL" \
  --api-key "$GOOGLE_API_KEY2" \
  --delay-ms 200 \
  --api-call-maximum 250 \
  "$INSTR_DIR" "$SCORE_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring finished ${NAME2}"
echo "$(date '+%Y-%m-%d %H:%M:%S') - phrx scoring finished ${NAME2}" >>"$HOME/times.log"
exit 0
