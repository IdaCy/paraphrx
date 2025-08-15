#!/usr/bin/env bash

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - tmux_ftscore_threestart script started" >>"$HOME/times.log"
echo "$TS - tmux_ftscore_threestart script started"

set -euo pipefail
trap 'echo "CTRL-C - stopping"; kill -- -$$' INT TERM

MODEL="gemini-2.5-flash-preview-05-20"
INF_NAME="inf_results.json"
INSTR_ALPACA="f_finetune/data/output_splits_alpaca/buckets_1-4_test.json"
INSTR_GSM8K="f_finetune/data/output_splits_gsm8k/buckets_1-4_test.json"
INSTR_MMLU="f_finetune/data/output_splits_mmlu/buckets_1-4_test.json"

CARGO_HOME=/scratch_dgxl/ifc24/proj/paraphrx/.cargo
RUSTUP_HOME=/scratch_dgxl/ifc24/proj/paraphrx/.rustup
export PATH="$CARGO_HOME/bin:$PATH"

command -v cargo >/dev/null || {
  echo "ERROR: cargo not found in $CARGO_HOME/bin"; exit 1;
}

echo "PATH = $PATH"
command -v cargo || echo "!! cargo not found in PATH"


# logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/SCORING_master_one_score_starter_batch-$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%F %T') - score_starter batch started"


# 1A


RUN_NAME="outputs_buckets_1-3_D-qlora-midLR-1p2e-4"
RUN_DIR="f_finetune/outputs_6/all_data_specific_layers/${RUN_NAME}"
ANSWERS="$RUN_DIR/${INF_NAME}"
OUTPUT="$RUN_DIR/results_${MODEL//[^[:alnum:]]/_}_$(date '+%Y%m%d_%H%M%S').json"

if [[ ! -d $RUN_DIR || ! -f $ANSWERS ]]; then
  echo "⚠  Skipping $RUN_NAME - file(s) missing"
else
  mkdir -p "$(dirname "$OUTPUT")"
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/SCORING_mainlog_${RUN_NAME}_${TS}.txt"
  echo "-> $RUN_NAME - starting $(date)  (log -> $LOG_FILE)"

  if cargo score_results_all_data \
      --instructions "$INSTR_ALPACA" \
      --instructions "$INSTR_GSM8K" \
      --instructions "$INSTR_MMLU" \
      --datasets alpaca \
      --datasets gsm8k \
      --datasets mmlu \
      --model "$MODEL" \
      --api-key "AIzaSyD8j2JZaJ-b3HY9lYZYp-l3sYLoNFZi_lA" \
      --api-call-max 250 \
      --log-name "SCORE_${RUN_NAME}" \
      "$ANSWERS" \
      "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $RUN_NAME - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $RUN_NAME - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

echo "$TS - tmux_ftscore_threestart finished $RUN_NAME"


# 1B

RUN_NAME="outputs_buckets_1-3_B-qlora-lowLR-8e-5"
RUN_DIR="f_finetune/outputs_6/alpaca_specific_layers/${RUN_NAME}"
ANSWERS="$RUN_DIR/${INF_NAME}"
OUTPUT="$RUN_DIR/results_${MODEL//[^[:alnum:]]/_}_$(date '+%Y%m%d_%H%M%S').json"

if [[ ! -d $RUN_DIR || ! -f $ANSWERS ]]; then
  echo "⚠  Skipping $RUN_NAME - file(s) missing"
else
  mkdir -p "$(dirname "$OUTPUT")"
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/SCORING_mainlog_${RUN_NAME}_${TS}.txt"
  echo "-> $RUN_NAME - starting $(date)  (log -> $LOG_FILE)"

  if cargo score_results_all_data \
      --instructions "$INSTR_ALPACA" \
      --datasets alpaca \
      --model "$MODEL" \
      --api-key "AIzaSyCz_C0NPi02zpFCXFOCZg5ALRPxjiOTi3U" \
      --api-call-max 250 \
      --log-name "SCORE_${RUN_NAME}" \
      "$ANSWERS" \
      "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $RUN_NAME - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $RUN_NAME - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

echo "$TS - tmux_ftscore_threestart finished $RUN_NAME"





TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - tmux_ftscore_threestart finished" >>"$HOME/times.log"
echo "$TS - tmux_ftscore_threestart finished"

echo "$(date '+%F %T') - score_starter batch finished"
exit 0
