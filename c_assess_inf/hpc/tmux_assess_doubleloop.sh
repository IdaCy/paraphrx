#!/usr/bin/env bash

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - tmux_assess_doubleloop_alpaca script started" >>"$HOME/times.log"

set -euo pipefail
trap 'echo "CTRL-C - stopping"; kill -- -$$' INT TERM

MODEL="gemini-2.5-flash-preview-05-20"
INSTR_DIR="a_data/alpaca/paraphrases_slice_100"
ANSW_DIR="c_assess_inf/output/alpaca/Qwen2.5-3B-Instruct/answers_slice_100"
OUT_DIR="c_assess_inf/output/alpaca/Qwen2.5-3B-Instruct/scores_slice_100"

GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"   # can be overridden with -k

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

[[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"   # make `cargo` available

# sanity checks
if [[ -z "$GOOGLE_API_KEY" ]]; then
  echo "Error: Google API key not set.  Use -k KEY or env GOOGLE_API_KEY." >&2
  exit 1
fi
if (( $# < 1 )); then
  echo "Usage: $(basename "$0") TYPE [TYPE …]" >&2
  echo "   e.g.  $(basename "$0") style syntax tone" >&2
  exit 1
fi

TYPES=("$@")              # <── all remaining args are file names (types)
SLICES=(1 2 3 4 5)        # <── always process five fixed slices

# logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/a_Q2_batch-$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%F %T') - Q2 batch started"

# main loop
for TYPE in "${TYPES[@]}"; do
  for SLICE in "${SLICES[@]}"; do

    # ─── possible hard-coded skip ───
    #if [[ "$TYPE" == "style" && "$SLICE" -eq 2 ]]; then
    #  echo "• Skipping style slice2 (hard-coded rule)"
    #  continue          # jump to next slice
    #fi
    # ───────────────────────

    SLICE_TAG="${TYPE}_slice${SLICE}"
    INSTRUCTIONS="$INSTR_DIR/${SLICE_TAG}.json"
    ANSWERS="$ANSW_DIR/${SLICE_TAG}.json"
    OUTPUT="$OUT_DIR/${SLICE_TAG}_results_${MODEL//[^[:alnum:]]/_}.json"

    if [[ ! -f $INSTRUCTIONS || ! -f $ANSWERS ]]; then
      echo "⚠  Skipping $SLICE_TAG - file(s) missing"
      continue
    fi

    TS="$(date '+%Y%m%d_%H%M%S')"
    LOG_FILE="$LOG_DIR/alpaca_Q2__${SLICE_TAG}-${TS}.txt"
    echo "▶ $SLICE_TAG - starting $(date)  (log → $LOG_FILE)"

    if cargo results_assess_noID \
         --model "$MODEL" \
         --api-key "$GOOGLE_API_KEY" \
         "$INSTRUCTIONS" "$ANSWERS" "$OUTPUT" \
         &> "$LOG_FILE"
    then
      echo "✔ $SLICE_TAG - finished OK $(date)"
    else
      STATUS=$?
      echo "⚠ $SLICE_TAG - cargo exited $STATUS  (see $LOG_FILE)"
    fi
  done
done

TS=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS - tmux_assess_doubleloop_alpaca finished" >>"$HOME/times.log"

echo "$(date '+%F %T') - Q2 batch finished"
exit 0
