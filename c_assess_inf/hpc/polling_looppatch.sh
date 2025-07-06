#!/usr/bin/env bash

#!/usr/bin/env bash
set -uo pipefail

echo "$(date) - starting polling_loop_patchmmlu.sh"

# tiny option parser - only -k|--key
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"        # default from env
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--key)  GOOGLE_API_KEY="$2"; shift 2 ;;
    --)        shift; break ;;              # end of options
    -*)        echo "Unknown option $1" >&2; exit 1 ;;
    *)         break ;;
  esac
done

# detach on first invocation
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/polling_loop_patchmmlu-$(date '+%Y%m%d_%H%M%S').log"

if [[ -z "${DETACHED_polling_loop_patchmmlu:-}" ]]; then
  export DETACHED_polling_loop_patchmmlu=1      # mark the second copy
  exec nohup caffeinate -dimsu "$0" -k "$GOOGLE_API_KEY" \
       </dev/null >>"$LOG_FILE" 2>&1 &
  disown
  echo "polling_loop_patchmmlu detached → log → $LOG_FILE"
  exit 0
fi

# from here on we are the background copy; all output already
# goes to $LOG_FILE because of the 'exec' above

# locate the running first batch
pid=$(pgrep -f "results_patch.sh.*-d gsm8k.*-im gemma-2-9b-it" | head -n1)
if [[ -z "$pid" ]]; then
  echo "$(date) - original job not found" >&2
  exit 1
fi
echo "$(date) - waiting for PID $pid to finish"

# wait (poll every 10 min)
while kill -0 "$pid" 2>/dev/null; do
  sleep 600
done
echo "$(date) - first batch finished"

# launch the second batch
if [[ -z "$GOOGLE_API_KEY" ]]; then
  echo "Error: Google API key not supplied (-k) and not in env." >&2
  exit 1
fi

# paths (instructions path points to folder that contains the json files)
SCORING_MODEL="gemini-2.5-flash-preview-05-20"
DATASET="gsm8k"
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

echo "$(date '+%Y-%m-%d %H:%M:%S') - patching run started" >>"$HOME/times.log"

TYPE_STRING=""
for TYPE in "${TYPES[@]}"; do
  TYPE_STRING+=" --type $TYPE"
done

cargo results_patch \
  --instructions-dir "$INSTR_DIR" \
  --answers-dir      "$ANSW_DIR" \
  --scores-dir       "$SCORE_DIR" \
  --issues-dir       "$ISSUES_DIR" \
  --type boundary --type context --type extra_a \
  --type extra_b --type language --type length \
  --type obstruction --type special_char --type style \
  --type syntax --type tone \
  --model "$SCORING_MODEL" \
  --api-key "$GOOGLE_API_KEY"

echo "$(date) - polling_loop_patchmmlu.sh finished"
