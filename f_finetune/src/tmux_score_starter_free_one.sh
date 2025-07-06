#!/usr/bin/env bash
xxx
TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - tmux_score_starter_free_one script started" >>"$HOME/times.log"
echo "$TS - tmux_score_starter_free_one script started"

set -euo pipefail
trap 'echo "CTRL-C - stopping"; kill -- -$$' INT TERM

MODEL="gemini-2.5-flash-preview-05-20"
# could "gemini-2.5-flash-lite-preview-06-17"
INSTR_DIR="f_finetune/data/alpaca_gemma-2-2b-it.json"
ANSW_DIR="f_finetune/outputs/ft_inf_results"
OUT_DIR="f_finetune/outputs/ft_inf_scores"
IN_NAME="buckets2_part1"

# hard-coded per-call keys
KEY_A=""

# light option parser (-k / --key , -m / --model)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--key)   GOOGLE_API_KEY="$2"; shift 2 ;;
    -m|--model) MODEL="$2";         shift 2 ;;
    -n|--in_name) IN_NAME="$2";         shift 2 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

KEY_A="$GOOGLE_API_KEY"

if [[ -f "$HOME/.cargo/env" ]]; then
  . "$HOME/.cargo/env"
fi

# logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_score_starter_batch-$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%F %T') - score_starter batch started"


# first IN_NAME (A)

ANSWERS="$ANSW_DIR/${IN_NAME}.json"
OUTPUT="$OUT_DIR/${IN_NAME}_results_${MODEL//[^[:alnum:]]/_}_$(date '+%Y%m%d_%H%M%S').json"

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

echo "$TS - tmux_score_starter_free_one finished $IN_NAME"

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - tmux_score_starter_free_one finished" >>"$HOME/times.log"
echo "$TS - tmux_score_starter_free_one finished"

echo "$(date '+%F %T') - score_starter batch finished"
exit 0
