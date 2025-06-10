#!/usr/bin/env bash
# Wrapper that launches inference_run_betterbatch.py
# and writes stdout/stderr to a unique log file.

# find the value that follows --type (if any)
type_tag=""
for (( i=1; i<=$#; i++ )); do
  if [[ "${!i}" == "--type" ]]; then
    j=$((i+1))
    type_tag="${!j}"
    break
  fi
done

# build log filename
timestamp=$(date +%Y%m%d_%H%M%S)
if [[ -n "$type_tag" ]]; then
  run_log="logs/${type_tag}_${timestamp}.out"
else
  run_log="logs/run_${timestamp}.out"
fi

# launch job
nohup python c_assess_inf/src/inference_run_betterbatch.py "$@" \
      >"$run_log" 2>&1 &

echo "Job started — log file: $run_log"
