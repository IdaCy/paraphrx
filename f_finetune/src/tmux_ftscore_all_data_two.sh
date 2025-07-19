#!/usr/bin/env bash

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - tmux_ftscore_all_data_two script started" >>"$HOME/times.log"
echo "$TS - tmux_ftscore_all_data_two script started"

set -euo pipefail
trap 'echo "CTRL-C - stopping"; kill -- -$$' INT TERM

MODEL="gemini-2.5-flash-preview-05-20"
# could "gemini-2.5-flash-lite-preview-06-17"
BASE_DIR="f_finetune/outputs"
INSTR_DIR="f_finetune/data"
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

LAYERS1="all_data_${LAYERS1}"
LAYERS2="all_data_${LAYERS2}"

if [[ -f "$HOME/.cargo/env" ]]; then
  . "$HOME/.cargo/env"
fi

# logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/all_data_SCORE_master_one_score_starter_batch-$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%F %T') - score_starter batch started"


# first IN_NAME (A)

ANSW_DIR="$BASE_DIR/${LAYERS1}/inference_results"
OUT_DIR="$BASE_DIR/${LAYERS1}/ft_inf_scores"
ANSWERS="$ANSW_DIR/${IN_NAME1}.json"
OUTPUT="$OUT_DIR/${IN_NAME1}_results_${MODEL//[^[:alnum:]]/_}_$(date '+%Y%m%d_%H%M%S').json"

A1="$INSTR_DIR/output_splits_alpaca/${IN_NAME1}_test.json"
A2="$INSTR_DIR/output_splits_gsm8k/${IN_NAME1}_test.json"
A3="$INSTR_DIR/output_splits_mmlu/${IN_NAME1}_test.json"

if [[ ! -f "$A1" || ! -f "$A2" || ! -f "$A3" || ! -f "$ANSWERS" ]]; then
  echo "⚠  Skipping $IN_NAME1 - missing one of $A1, $A2, $A3, or $ANSWERS"
else
  mkdir -p "$(dirname "$OUTPUT")"
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/all_data_SCORE_mainlog_score_starter__${IN_NAME1}-${TS}.txt"
  echo "-> $IN_NAME1 - starting $(date)  (log -> $LOG_FILE)"

  if cargo score_results_all_data \
      --instructions "$A1" \
      --instructions "$A2" \
      --instructions "$A3" \
      --datasets alpaca \
      --datasets gsm8k \
      --datasets mmlu \
      --model "$MODEL" \
      --api-key "$KEY_A" \
      --api-call-max 250 \
      --log-name "all_data_SCORE_$LAYERS1" \
      "$ANSWERS" \
      "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $IN_NAME1 - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $IN_NAME1 - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

echo "$TS - tmux_ftscore_all_data_two finished $IN_NAME1"


# second IN_NAME (B)

ANSW_DIR="$BASE_DIR/${LAYERS2}/inference_results"
OUT_DIR="$BASE_DIR/${LAYERS2}/ft_inf_scores"
ANSWERS="$ANSW_DIR/${IN_NAME2}.json"
OUTPUT="$OUT_DIR/${IN_NAME2}_results_${MODEL//[^[:alnum:]]/_}_$(date '+%Y%m%d_%H%M%S').json"

if [[ ! -d $INSTR_DIR || ! -f $ANSWERS ]]; then
  echo "⚠  Skipping $IN_NAME2 - file(s) missing"
else
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/all_data_SCORE_mainlog_score_starter__${IN_NAME2}-${TS}.txt"
  echo "-> $IN_NAME2 - starting $(date)  (log -> $LOG_FILE)"

  if cargo score_results_all_data \
      --instructions "$INSTR_DIR/output_splits_alpaca/${IN_NAME2}.json" \
      --instructions "$INSTR_DIR/output_splits_gsm8k/${IN_NAME2}.json" \
      --instructions "$INSTR_DIR/output_splits_mmlu/${IN_NAME2}.json" \
      --datasets alpaca \
      --datasets gsm8k \
      --datasets mmlu \
      --model "$MODEL" \
      --api-key "$KEY_B" \
      --api-call-max 250 \
      --log-name "all_data_SCORE_$LAYERS2" \
      "$ANSWERS" \
      "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $IN_NAME2 - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $IN_NAME2 - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

echo "$TS - tmux_ftscore_all_data_two finished $IN_NAME2"


TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - tmux_ftscore_all_data_two finished" >>"$HOME/times.log"
echo "$TS - tmux_ftscore_all_data_two finished"

echo "$(date '+%F %T') - score_starter batch finished"
exit 0
