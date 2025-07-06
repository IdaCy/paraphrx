#!/usr/bin/env bash

#!/usr/bin/env bash
set -uo pipefail

echo "$(date) - starting tmux_polling_ft_scor.sh"

# tiny option parser – only -k|--key
KEY_A=""   # first call
KEY_B=""   # second call

# detach on first invocation
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/tmux_polling_ft_scor-$(date '+%Y%m%d_%H%M%S').log"

# locate the running first batch
pid=$(pgrep -f "./c_assess_inf/hpc/tmux_assess_noslice_mmlu_g9_two.sh" | head -n1)
if [[ -z "$pid" ]]; then
  echo "$(date) – original job not found" >&2
  exit 1
fi
echo "$(date) – waiting for PID $pid to finish"

# wait (poll every 10 min)
while kill -0 "$pid" 2>/dev/null; do
  sleep 600
done
echo "$(date) – first batch finished"

# launch the second batch
if [[ -z "$KEY_A" ]]; then
  echo "Error: Google API key not supplied (-k) and not in env." >&2
  exit 1
fi

# paths (instructions path points to folder that contains the json files)
SCORING_MODEL="gemini-2.5-flash-preview-05-20"
DATASET="alpaca"
INF_MODEL="gemma-2-9b-it"
INSTR_DIR="a_data/${DATASET}/paraphrases_500"
ANSW_DIR="c_assess_inf/output/${DATASET}/${INF_MODEL}/answers"
SCORE_DIR="c_assess_inf/output/${DATASET}/${INF_MODEL}/scores_500"
ISSUES_DIR="c_assess_inf/output/${DATASET}/${INF_MODEL}/scores_issues_500"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/patch_results-$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

echo "$(date '+%Y-%m-%d %H:%M:%S') – patching run started" >>"$HOME/times.log"

TYPE_STRING=""
for TYPE in "${TYPES[@]}"; do
  TYPE_STRING+=" --type $TYPE"
done

cargo results_patch \
  --instructions-dir "$INSTR_DIR" \
  --answers-dir      "$ANSW_DIR" \
  --scores-dir       "$SCORE_DIR" \
  --issues-dir       "$ISSUES_DIR" \
  --type extra_b --type style \
  --model "$SCORING_MODEL" \
  --api-key "$KEY_A"


if [[ -z "$GOOGLE_API_KEY2" ]]; then
  echo "Error: Google API key not supplied (-k) and not in env." >&2
  exit 1
fi

echo "$(date) - tmux_polling_ft_scor.sh finished"
