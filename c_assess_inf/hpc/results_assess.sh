#!/usr/bin/env bash
# caffeinate -dimsu c_assess_inf/hpc/results_assess.sh
set -uo pipefail
trap 'echo "› CTRL-C – stopping"; kill -TERM -- -$$' INT TERM
set -m

# user-configurable variables
MODEL="gemini-2.5-flash-preview-05-20"

INSTRUCTIONS="a_data/alpaca/merge_instructs/bolastto.json"
ANSWERS="c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/instruct_merged/bolastto.json"
OUTPUT="c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/instruct_merged/bolastto_results_gem25flash52.json"

# required environment
export GOOGLE_API_KEY="AIzaSyBx1W8ovOHCOhfuuUcYJ-wquS_oDLxgMBc"

# logging
TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/results_assess-${TS}.txt"

# load Cargo environment & change into PBS working dir
[[ -n "${PBS_O_WORKDIR:-}" ]] && cd "$PBS_O_WORKDIR"
[[ -f "$HOME/.cargo/env"    ]] && . "$HOME/.cargo/env"

echo "$(date '+%F %T') – results_assess started" >> "$HOME/times.log"

# main run wrapped in caffeinate
echo "▶ Starting at $(date). Output → $LOG_FILE"
{
  set -x
  cargo results_assess \
         --model "$MODEL" \
         "$INSTRUCTIONS" "$ANSWERS" "$OUTPUT"
} &> "$LOG_FILE"

STATUS=$?

# summary + exit
if [[ $STATUS -eq 0 ]]; then
  echo "✔ Finished OK at $(date)" | tee -a "$LOG_FILE"
else
  echo "⚠ Binary exited with $STATUS – check $LOG_FILE or issues file" \
       | tee -a "$LOG_FILE"
fi

echo "$(date '+%F %T') – results_assess finished" >> "$HOME/times.log"
exit 0
