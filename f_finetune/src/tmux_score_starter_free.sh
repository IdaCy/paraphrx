#!/usr/bin/env bash

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - tmux_score_starter_BUCKETS4 script started" >>"$HOME/times.log"
echo "$TS - tmux_score_starter_BUCKETS4 script started"

set -euo pipefail
trap 'echo "CTRL-C - stopping"; kill -- -$$' INT TERM

MODEL="gemini-2.5-flash-preview-05-20"
# could "gemini-2.5-flash-lite-preview-06-17"
INSTR_DIR="f_finetune/data/alpaca_gemma-2-2b-it.json"
ANSW_DIR="f_finetune/outputs/ft_inf_results"
OUT_DIR="f_finetune/outputs/ft_inf_scores"

GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"   # can be overridden with -k

# hard-coded per-call keys (optional; leave empty to fall back to GOOGLE_API_KEY or no key)
KEY_A=""   # first call
KEY_B=""   # second call

# light option parser (-k / --key , -m / --model)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--key)   GOOGLE_API_KEY="$2"; shift 2 ;;
    -m|--model) MODEL="$2";         shift 2 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

[[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"   # make cargo available

# logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_score_starter_batch-$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%F %T') - score_starter batch started"


# first IN_NAME (A)

IN_NAME="buckets4_part1"
ANSWERS="$ANSW_DIR/${IN_NAME}.json"
OUTPUT="$OUT_DIR/${IN_NAME}_results_${MODEL//[^[:alnum:]]/_}_$(date '+%Y%m%d_%H%M%S')_${IN_NAME}.json"

if [[ ! -f $INSTR_DIR || ! -f $ANSWERS ]]; then
  echo "⚠  Skipping $IN_NAME - file(s) missing"
else
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/mainlog_score_starter__${IN_NAME}-${TS}.txt"
  echo "-> $IN_NAME - starting $(date)  (log -> $LOG_FILE)"

  if cargo score_results \
       --model "$MODEL" \
       --api-key "$KEY_A" \
       --api-call-max 250 \
       "$INSTR_DIR" "$ANSWERS" "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $IN_NAME - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $IN_NAME - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

echo "$TS - tmux_score_starter_BUCKETS4 finished $IN_NAME"


# second IN_NAME (B)

IN_NAME="buckets4_part2"
ANSWERS="$ANSW_DIR/${IN_NAME}.json"
OUTPUT="$OUT_DIR/${IN_NAME}_results_${MODEL//[^[:alnum:]]/_}_$(date '+%Y%m%d_%H%M%S')_${IN_NAME}.json"

if [[ ! -f $INSTR_DIR || ! -f $ANSWERS ]]; then
  echo "⚠  Skipping $IN_NAME - file(s) missing"
else
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/mainlog_score_starter__${IN_NAME}-${TS}.txt"
  echo "-> $IN_NAME - starting $(date)  (log -> $LOG_FILE)"

  if cargo score_results \
       --model "$MODEL" \
       --api-key "$KEY_B" \
       --api-call-max 250 \
       "$INSTR_DIR" "$ANSWERS" "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $IN_NAME - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $IN_NAME - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

echo "$TS - tmux_score_starter_BUCKETS4 finished $IN_NAME"

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - tmux_score_starter_BUCKETS4 finished" >>"$HOME/times.log"
echo "$TS - tmux_score_starter_BUCKETS4 finished"

echo "$(date '+%F %T') - score_starter batch finished"
exit 0
