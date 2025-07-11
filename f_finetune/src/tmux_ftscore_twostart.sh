#!/usr/bin/env bash

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - tmux_ftscore_all_layers script started" >>"$HOME/times.log"
echo "$TS - tmux_ftscore_all_layers script started"

set -euo pipefail
trap 'echo "CTRL-C - stopping"; kill -- -$$' INT TERM

MODEL="gemini-2.5-flash-preview-05-20"
# could "gemini-2.5-flash-lite-preview-06-17"
BASE_DIR="f_finetune/outputs/alpaca"
INSTR_DIR="f_finetune/data/alpaca_gemma-2-2b-it.json"
LAYERS1="all_layers"
LAYERS2="all_layers"
IN_NAME1="buckets_1-1"
IN_NAME2="buckets_1-2"

# hard-coded per-call keys
KEY_A=""
KEY_B=""

# light option parser (-k / --key , -m / --model)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k1|--key1)   GOOGLE_API_KEY1="$2"; shift 2 ;;
    -k2|--key2)   GOOGLE_API_KEY2="$2"; shift 2 ;;
    -m|--model) MODEL="$2";         shift 2 ;;
    -l1|--layers1) LAYERS1="$2";         shift 2 ;;
    -l2|--layers2) LAYERS2="$2";         shift 2 ;;
    -n1|--in_name1) IN_NAME1="$2";         shift 2 ;;
    -n2|--in_name2) IN_NAME2="$2";         shift 2 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

KEY_A="$GOOGLE_API_KEY1"
KEY_B="$GOOGLE_API_KEY2"

if [[ -f "$HOME/.cargo/env" ]]; then
  . "$HOME/.cargo/env"
fi

# logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/SCORING_master_one_score_starter_batch-$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%F %T') - score_starter batch started"


# first IN_NAME (A)

ANSW_DIR="$BASE_DIR/${LAYERS1}/inference_results"
OUT_DIR="$BASE_DIR/${LAYERS1}/ft_inf_scores"
ANSWERS="$ANSW_DIR/${IN_NAME1}.json"
OUTPUT="$OUT_DIR/${IN_NAME1}_results_${MODEL//[^[:alnum:]]/_}_$(date '+%Y%m%d_%H%M%S').json"

if [[ ! -f $INSTR_DIR || ! -f $ANSWERS ]]; then
  echo "⚠  Skipping $IN_NAME1 - file(s) missing"
else
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/SCORING_mainlog_score_starter__${IN_NAME1}-${TS}.txt"
  echo "-> $IN_NAME1 - starting $(date)  (log -> $LOG_FILE)"

  if cargo score_results \
       --model "$MODEL" \
       --api-key "$KEY_A" \
       --api-call-max 250 \
       --log-name "SCORING_$LAYERS1" \
       "$INSTR_DIR" "$ANSWERS" "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $IN_NAME1 - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $IN_NAME1 - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

echo "$TS - tmux_ftscore_all_layers finished $IN_NAME1"


# second IN_NAME (B)

ANSW_DIR="$BASE_DIR/${LAYERS2}/inference_results"
OUT_DIR="$BASE_DIR/${LAYERS2}/ft_inf_scores"
ANSWERS="$ANSW_DIR/${IN_NAME2}.json"
OUTPUT="$OUT_DIR/${IN_NAME2}_results_${MODEL//[^[:alnum:]]/_}_$(date '+%Y%m%d_%H%M%S').json"

if [[ ! -f $INSTR_DIR || ! -f $ANSWERS ]]; then
  echo "⚠  Skipping $IN_NAME2 - file(s) missing"
else
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/mainlog_score_starter__${IN_NAME2}-${TS}.txt"
  echo "-> $IN_NAME2 - starting $(date)  (log -> $LOG_FILE)"

  if cargo score_results \
       --model "$MODEL" \
       --api-key "$KEY_B" \
       --api-call-max 250 \
       --log-name "SCORING_$LAYERS2" \
       "$INSTR_DIR" "$ANSWERS" "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $IN_NAME2 - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $IN_NAME2 - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

echo "$TS - tmux_ftscore_all_layers finished $IN_NAME2"



TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - tmux_ftscore_all_layers finished" >>"$HOME/times.log"
echo "$TS - tmux_ftscore_all_layers finished"

echo "$(date '+%F %T') - score_starter batch finished"
exit 0
