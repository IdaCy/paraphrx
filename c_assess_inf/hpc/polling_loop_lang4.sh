#!/usr/bin/env bash

# done: 
# use for:
# mmlu
# from: length obstruction speci_char style
# to: voice
# to: mmlu_q - all

#!/usr/bin/env bash
set -uo pipefail

echo "$(date) - starting polling_loop_lang4.sh"

# tiny option parser – only -k|--key
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"        # default from env
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--key)  GOOGLE_API_KEY="$2"; shift 2 ;;
    --)        shift; break ;;              # end of options
    -*)        echo "Unknown option $1" >&2; exit 1 ;;
    *)         break ;;
  esac
done

# expect two positional args
if (( $# < 2 )); then
  echo "Usage: $0 [-k KEY] FIRST_TYPE SECOND_TYPE" >&2
  exit 1
fi
FIRST_TYPE="$1"; SECOND_TYPE="$2"
shift 2

# detach on first invocation
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/polling_loop_lang4-$(date '+%Y%m%d_%H%M%S').log"

if [[ -z "${DETACHED_polling_loop_lang4:-}" ]]; then
  export DETACHED_polling_loop_lang4=1      # mark the second copy
  exec nohup caffeinate -dimsu "$0" -k "$GOOGLE_API_KEY" \
       "$FIRST_TYPE" "$SECOND_TYPE" \
       </dev/null >>"$LOG_FILE" 2>&1 &
  disown
  echo "polling_loop_lang4 detached → log → $LOG_FILE"
  exit 0
fi

# from here on we are the background copy; all output already
# goes to $LOG_FILE because of the 'exec' above

# locate the running first batch
pid=$(pgrep -f "assess_loop_gsm8k_q.sh $FIRST_TYPE 5" | head -n1)
if [[ -z "$pid" ]]; then
  echo "$(date) – original job not found" >&2
  exit 1
fi
echo "$(date) – waiting for PID $pid to finish"

# wait (poll every 2 min)
while kill -0 "$pid" 2>/dev/null; do
  sleep 120
done
echo "$(date) – first batch finished"

# launch the follow-up batch
if [[ -z "$GOOGLE_API_KEY" ]]; then
  echo "Error: Google API key not supplied (-k) and not in env." >&2
  exit 1
fi

caffeinate -dimsu ./c_assess_inf/hpc/assess_loop_gsm8k_q.sh \
           -k "$GOOGLE_API_KEY"  "$SECOND_TYPE" 4

echo "$(date) – started follow-up batch for $SECOND_TYPE 4"
