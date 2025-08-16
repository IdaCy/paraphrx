#!/usr/bin/env bash

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - phrx_50k script started" >>"$HOME/times.log"
echo "$TS - phrx_50k script started"

set -euo pipefail
trap 'echo "CTRL-C - stopping"; kill -- -$$' INT TERM

MODEL="gemini-2.5-flash"
GOOGLE_API_KEY1=""
GOOGLE_API_KEY2=""
PART="part1"

# parser
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k1|--key1)   GOOGLE_API_KEY1="$2"; shift 2 ;;
    -k2|--key2)   GOOGLE_API_KEY2="$2"; shift 2 ;;
    -p| --part)   PART="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

API_KEY1="$GOOGLE_API_KEY1"
API_KEY2="$GOOGLE_API_KEY2"
INSTR="a_data/alpaca/50k_phrxed.json"
ANSWERS="f_finetune/output_inf_ft_50k/${PART}.json"
OUTPUT="f_finetune/output_inf_ft_50k_scores/${PART}.json"

if [[ -f "$HOME/.cargo/env" ]]; then
  . "$HOME/.cargo/env"
fi

# logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/scoring_${PART}_$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%F %T') - scoretwo batch started"


# run 1

if [[ ! -f $INSTR ]]; then
  echo "⚠  Skipping $INSTR $PART - file(s) missing"
else
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/scoring_${PART}_${TS}.txt"
  echo "-> $INSTR $PART - starting $(date)  (log -> $LOG_FILE)"

  if cargo score_inf_ft_50k \
        --model "$MODEL" \
        --api-key "$API_KEY1" \
        --api-call-max 250 \
        --log-name "scoring50k_${PART}" \
        "$INSTR" \
        "$ANSWERS" \
        "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $INSTR $PART - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $INSTR $PART - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

TS="$(date '+%Y%m%d_%H%M%S')"
echo "$TS - phrx_50k finished $INSTR $PART"

# run 2
if [[ ! -f $INSTR ]]; then
  echo "⚠  Skipping $INSTR $PART - file(s) missing"
else
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/scoring_${PART}_2_${TS}.txt"
  echo "-> $INSTR $PART - starting 2nd run $(date)  (log -> $LOG_FILE)"

  if cargo score_inf_ft_50k \
        --model "$MODEL" \
        --api-key "$API_KEY2" \
        --api-call-max 250 \
        --log-name "scoring50k_${PART}" \
        "$INSTR" \
        "$ANSWERS" \
        "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $INSTR $PART - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $INSTR $PART - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

TS="$(date '+%Y%m%d_%H%M%S')"
echo "$TS - phrx_50k finished $INSTR $PART"


TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - phrx_50k finished" >>"$HOME/times.log"
echo "$TS - phrx_50k finished"

echo "$(date '+%F %T') - scoretwo batch finished"
exit 0
