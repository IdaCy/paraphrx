#!/usr/bin/env bash

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - phrx_50k script started" >>"$HOME/times.log"
echo "$TS - phrx_50k script started"

set -euo pipefail
trap 'echo "CTRL-C - stopping"; kill -- -$$' INT TERM

PART="part2"

KEY_A=""
KEY_B=""
KEY_C=""
KEY_D=""
KEY_E=""
KEY_F=""

# parser
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k1|--key1)   GOOGLE_API_KEY1="$2"; shift 2 ;;
    -k2|--key2)   GOOGLE_API_KEY2="$2"; shift 2 ;;
    -k3|--key3)   GOOGLE_API_KEY3="$2"; shift 2 ;;
    -k4|--key4)   GOOGLE_API_KEY4="$2"; shift 2 ;;
    -k5|--key5)   GOOGLE_API_KEY5="$2"; shift 2 ;;
    -k6|--key6)   GOOGLE_API_KEY6="$2"; shift 2 ;;
    -p| --part)   PART="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

KEY_A="$GOOGLE_API_KEY1"
KEY_B="$GOOGLE_API_KEY2"
KEY_C="$GOOGLE_API_KEY3"
KEY_D="$GOOGLE_API_KEY4"
KEY_E="$GOOGLE_API_KEY5"
KEY_F="$GOOGLE_API_KEY6"

INSTR="a_data/alpaca/alpaca_10k_${PART}.json"

if [[ -f "$HOME/.cargo/env" ]]; then
  . "$HOME/.cargo/env"
fi

# logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/phrxing_${PART}_$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%F %T') - score24hl batch started"

# API key array and loop control
API_KEYS=("$KEY_A" "$KEY_B" "$KEY_C" "$KEY_D" "$KEY_E" "$KEY_F")
NUM_KEYS=${#API_KEYS[@]}
LOOP_COUNT=0
MAX_LOOPS=18

while true; do
  LOOP_COUNT=$((LOOP_COUNT+1))
  if [[ $LOOP_COUNT -gt $MAX_LOOPS ]]; then
    echo "Reached max loops ($MAX_LOOPS) at $(date), exiting"
    break
  fi

  idx=$(( (LOOP_COUNT - 1) % NUM_KEYS ))
  CURRENT_KEY="${API_KEYS[$idx]}"
  RUN_NAME="phrxed${LOOP_COUNT}"

  if [[ ! -f $INSTR ]]; then
    echo "⚠  Skipping $INSTR $RUN_NAME - file(s) missing"
  else
    TS="$(date '+%Y%m%d_%H%M%S')"
    LOG_FILE="$LOG_DIR/phrxing_${PART}_${RUN_NAME}_${TS}.txt"
    echo "-> $INSTR $RUN_NAME - starting $(date)  (log -> $LOG_FILE)"

    if cargo generate_11_paraphrases \
          --input "$INSTR" \
          --output "a_data/alpaca/alpaca_10k_${PART}_phrxed.json" \
          --log-name "phrxing50k_${PART}" \
          --api-call-maximum 250 \
          --api-key "$CURRENT_KEY" \
         &> "$LOG_FILE"
    then
      echo "✔ $INSTR $RUN_NAME - finished OK $(date)"
    else
      STATUS=$?
      echo "⚠ $INSTR $RUN_NAME - cargo exited $STATUS  (see $LOG_FILE)"
    fi
  fi

  echo "$TS - phrx_50k finished $INSTR $RUN_NAME"
done

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - phrx_50k finished" >>"$HOME/times.log"
echo "$TS - phrx_50k finished"

echo "$(date '+%F %T') - score24hl batch finished"
exit 0
