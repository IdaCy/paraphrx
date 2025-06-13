#!/usr/bin/env bash
# ps -o pid,pgid,comm,args -u "$USER" | egrep 'results_assess_[0-9].sh'
# chmod +x c_assess_inf/hpc/results_assess_4.sh
#
#   # run all 11 types
#   caffeinate -dimsu ./c_assess_inf/hpc/results_assess_4.sh
#
#   # run only specific types
#   caffeinate -dimsu ./c_assess_inf/hpc/results_assess_4.sh tone voice

set -uo pipefail
trap 'echo "› CTRL-C – stopping"; kill -TERM -- -$$' INT TERM
set -m

MODEL="gemini-2.5-flash-preview-05-20"
INSTR_DIR="a_data/alpaca/slice_500"
ANSW_DIR="c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/merged"
OUT_DIR="$ANSW_DIR"

export GOOGLE_API_KEY="AIzaSyBx1W8ovOHCOhfuuUcYJ-wquS_oDLxgMBc"
[[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"

if [[ -z "${DETACHED_RESULTS_ASSESS:-}" ]]; then
  export DETACHED_RESULTS_ASSESS=1            # marks the second run
  nohup caffeinate -dimsu "$0" "$@" \
        </dev/null >/dev/null 2>&1 &          # background + keep-awake
  disown                                      # release from this shell
  echo "results_assess batch detached → check logs/ directory for progress"
  exit 0                                      # give control back to the terminal
fi

if (( $# == 0 )); then
  TYPES=(boundary context extra language length obstruction \
         speci_char style syntax tone voice)
else
  TYPES=("$@")
fi

# create a single master log and redirect everything
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/extrab_batch-$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x                              # keep command echoing, goes to log

TS_START=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS_START – batch run started" >>"$HOME/times.log"

for TYPE in "${TYPES[@]}"; do
  INSTRUCTIONS="$INSTR_DIR/${TYPE}.json"
  ANSWERS="$ANSW_DIR/${TYPE}.json"
  OUTPUT="$OUT_DIR/${TYPE}_results_${MODEL//[^[:alnum:]]/_}.json"

  if [[ ! -f $INSTRUCTIONS || ! -f $ANSWERS ]]; then
    echo "⚠  Skipping $TYPE – matching file missing"
    continue
  fi

  TS="$(date '+%Y%m%d_%H%M%S')"
  LOG_FILE="$LOG_DIR/extrab_${TYPE}-${TS}.txt"

  echo "▶ $TYPE – starting $(date)  (log → $LOG_FILE)"

  {
    cargo results_assess \
      --model "$MODEL" \
      "$INSTRUCTIONS" "$ANSWERS" "$OUTPUT"
  } &> "$LOG_FILE"

  STATUS=$?
  if [[ $STATUS -eq 0 ]]; then
    echo "✔ $TYPE – finished OK $(date)"
  else
    echo "⚠ $TYPE – cargo exited $STATUS  (see $LOG_FILE)"
  fi
done

TS_END=$(date '+%Y-%m-%d %H:%M:%S')
echo "$TS_END – batch run finished" >>"$HOME/times.log"
exit 0
