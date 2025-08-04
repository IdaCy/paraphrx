#!/usr/bin/env bash

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - phrx_50k script started" >>"$HOME/times.log"
echo "$TS - phrx_50k script started"

set -euo pipefail
trap 'echo "CTRL-C - stopping"; kill -- -$$' INT TERM

MODEL="gemini-2.5-flash"
GOOGLE_API_KEY1=""
GOOGLE_API_KEY2=""
GOOGLE_API_KEY3=""
GOOGLE_API_KEY4=""
PART1="part1"
PART2="part2"

# parser
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k1|--key1)   GOOGLE_API_KEY1="$2"; shift 2 ;;
    -k2|--key2)   GOOGLE_API_KEY2="$2"; shift 2 ;;
    -k3|--key3)   GOOGLE_API_KEY3="$2"; shift 2 ;;
    -k4|--key4)   GOOGLE_API_KEY4="$2"; shift 2 ;;
    -p1| --part)   PART1="$2"; shift 2 ;;
    -p2| --part2) PART2="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

API_KEY1="$GOOGLE_API_KEY1"
API_KEY2="$GOOGLE_API_KEY2"
INSTR="a_data/alpaca/50k_phrxed.json"
ANSWERS="f_finetune/output_inf_ft_50k/${PART1}.json"
OUTPUT="f_finetune/output_inf_ft_50k_scores/${PART1}.json"

if [[ -f "$HOME/.cargo/env" ]]; then
  . "$HOME/.cargo/env"
fi

# logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/scoring_${PART1}_$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%F %T') - scoretwotwice batch started"

### PART1

# run 1

if [[ ! -f $INSTR ]]; then
  echo "⚠  Skipping $INSTR $PART1 - file(s) missing"
else
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/scoring_${PART1}_${TS}.txt"
  echo "-> $INSTR $PART1 - starting $(date)  (log -> $LOG_FILE)"

  if cargo score_inf_ft_50k \
        --model "$MODEL" \
        --api-key "$API_KEY1" \
        --api-call-max 250 \
        --log-name "scoring50k_${PART1}" \
        "$INSTR" \
        "$ANSWERS" \
        "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $INSTR $PART1 - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $INSTR $PART1 - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

TS="$(date '+%Y%m%d_%H%M%S')"
echo "$TS - phrx_50k finished $INSTR $PART1"

# run 2
if [[ ! -f $INSTR ]]; then
  echo "⚠  Skipping $INSTR $PART1 - file(s) missing"
else
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/scoring_${PART1}_2_${TS}.txt"
  echo "-> $INSTR $PART1 - starting 2nd run $(date)  (log -> $LOG_FILE)"

  if cargo score_inf_ft_50k \
        --model "$MODEL" \
        --api-key "$API_KEY2" \
        --api-call-max 250 \
        --log-name "scoring50k_${PART1}" \
        "$INSTR" \
        "$ANSWERS" \
        "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $INSTR $PART1 - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $INSTR $PART1 - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

TS="$(date '+%Y%m%d_%H%M%S')"
echo "$TS - phrx_50k finished $INSTR $PART1"


### PART2

ANSWERS="f_finetune/output_inf_ft_50k/${PART2}.json"
OUTPUT="f_finetune/output_inf_ft_50k_scores/${PART2}.json"

echo "$(date '+%F %T') - starting 3rd and 4th runs for $PART2"

# run 3
if [[ ! -f $INSTR ]]; then
  echo "⚠  Skipping $INSTR $PART2 - file(s) missing"
else
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/scoring_${PART2}_3_${TS}.txt"
  echo "-> $INSTR $PART2 - starting 3rd run $(date)  (log -> $LOG_FILE)"

  if cargo score_inf_ft_50k \
        --model "$MODEL" \
        --api-key "$API_KEY3" \
        --api-call-max 250 \
        --log-name "scoring50k_${PART2}" \
        "$INSTR" \
        "$ANSWERS" \
        "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $INSTR $PART2 - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $INSTR $PART2 - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

TS="$(date '+%Y%m%d_%H%M%S')"
echo "$TS - phrx_50k finished $INSTR $PART2"

# run 4
if [[ ! -f $INSTR ]]; then
  echo "⚠  Skipping $INSTR $PART2 - file(s) missing"
else
  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/scoring_${PART2}_4_${TS}.txt"
  echo "-> $INSTR $PART2 - starting 4th run $(date)  (log -> $LOG_FILE)"

  if cargo score_inf_ft_50k \
        --model "$MODEL" \
        --api-key "$API_KEY4" \
        --api-call-max 250 \
        --log-name "scoring50k_${PART2}" \
        "$INSTR" \
        "$ANSWERS" \
        "$OUTPUT" \
       &> "$LOG_FILE"
  then
    echo "✔ $INSTR $PART2 - finished OK $(date)"
  else
    STATUS=$?
    echo "⚠ $INSTR $PART2 - cargo exited $STATUS  (see $LOG_FILE)"
  fi
fi

TS="$(date '+%Y%m%d_%H%M%S')"
echo "$TS - phrx_50k finished $INSTR $PART2"

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - phrx_50k finished" >>"$HOME/times.log"
echo "$TS - phrx_50k finished"

echo "$(date '+%F %T') - scoretwotwice batch finished"
exit 0
