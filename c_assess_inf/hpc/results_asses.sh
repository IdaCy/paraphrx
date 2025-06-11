#!/usr/bin/env bash
set -euo pipefail

# user-configurable variables
MODEL="gemini-2.0-flash"

INSTRUCTIONS="a_data/alpaca/merge_instructs/all.json"
ANSWERS="c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/instruct_merged/all.json"
OUTPUT="c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/instruct_merged/all_results.json"
MANIFEST="c_assess_inf/Cargo.toml"
BIN="cargo assess_robust"

# Optional: ensure Google API key is set
: "${GOOGLE_API_KEY:?GOOGLE_API_KEY environment variable not set}"

# Logs go to logs/run-YYYYmmdd_HHMMSS.txt
TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run-${TS}.txt"

echo "▶ Starting run at $(date). Logging to $LOG_FILE"
{
  set -x
  $BIN \
    --manifest-path "$MANIFEST" \
    --release \
    -- \
    --model "$MODEL" \
    "$INSTRUCTIONS" \
    "$ANSWERS" \
    "$OUTPUT"
} &> "$LOG_FILE"

echo "✔ Finished at $(date). Exit code $?"
