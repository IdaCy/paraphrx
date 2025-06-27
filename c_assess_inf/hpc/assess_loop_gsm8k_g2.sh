#!/usr/bin/env bash

set -uo pipefail
trap 'echo "› CTRL-C – stopping"; kill -TERM -- -$$' INT TERM
set -m

MODEL="gemini-2.5-flash-preview-05-20"
INSTR_DIR="a_data/gsm8k/prxed_main_slice_100"
ANSW_DIR="c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100"
OUT_DIR="c_assess_inf/output/gsm8k/gemma-2-2b-it/scores_slice_100"

#export GOOGLE_API_KEY=""
# Default comes from env, but can be overridden with -k|--key
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"

# simple option parser (only -k / --key)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--key)
      [[ -n "$2" ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      GOOGLE_API_KEY="$2"
      shift 2
      ;;
    -m|--model)
      [[ -n "$2" ]] || { echo "Error: $1 needs a value" >&2; exit 1; }
      MODEL="$2"
      shift 2
      ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *)  break ;;
  esac
done

[[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"

if [[ -z "${DETACHED_RESULTS_ASSESS:-}" ]]; then
  export DETACHED_RESULTS_ASSESS=1            # marks the second run
  nohup caffeinate -dimsu "$0" "$@" \
        </dev/null >/dev/null 2>&1 &          # background + keep-awake
  disown                                      # release from this shell
  echo "results_assess batch detached → check logs/ directory for progress"
  exit 0                                      # give control back to the terminal
fi

# make sure we actually have a key now
if [[ -z "$GOOGLE_API_KEY" ]]; then
  echo "Error: Google API key not set.  Supply it with -k KEY or env GOOGLE_API_KEY." >&2
  exit 1
fi

# slice selection
if (( $# < 2 )); then
  echo "Usage: $(basename "$0") TYPE SLICE [SLICE …]" >&2
  echo "   e.g.  $(basename "$0") syntax 1 3" >&2
  exit 1
fi

TYPE="$1"
shift               # everything that remains are explicit slice numbers
SLICES=("$@")

# create a single master log and redirect everything
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/gsm8k_g2_batch-$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

TS_START=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS_START – batch run started" >>"$HOME/times.log"

for SLICE in "${SLICES[@]}"; do
  SLICE_TAG="${TYPE}_slice${SLICE}"

  INSTRUCTIONS="$INSTR_DIR/${SLICE_TAG}.json"
  ANSWERS="$ANSW_DIR/${SLICE_TAG}.json"
  OUTPUT="$OUT_DIR/${SLICE_TAG}_results_${MODEL//[^[:alnum:]]/_}.json"

  if [[ ! -f $INSTRUCTIONS || ! -f $ANSWERS ]]; then
    echo "⚠  Skipping $SLICE_TAG – matching file missing"
    continue
  fi

  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/gsm8k_g2__${SLICE_TAG}-${TS}.txt"
  echo "▶ $SLICE_TAG – starting $(date)  (log → $LOG_FILE)"

  {
    cargo results_assess_noID \
      --model "$MODEL" \
      --api-key "$GOOGLE_API_KEY" \
      "$INSTRUCTIONS" "$ANSWERS" "$OUTPUT"
  } &> "$LOG_FILE"

  STATUS=$?
  if [[ $STATUS -eq 0 ]]; then
    echo "✔ $SLICE_TAG – finished OK $(date)"
  else
    echo "⚠ $SLICE_TAG – cargo exited $STATUS  (see $LOG_FILE)"
  fi
done

TS_END=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS_END – batch run finished" >>"$HOME/times.log"
exit 0
