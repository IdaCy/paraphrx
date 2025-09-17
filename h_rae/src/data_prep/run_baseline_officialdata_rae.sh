#!/usr/bin/env bash
# Runs one-after-another + Kills all descendants on exit

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: ./run_with_keys_sequential.sh -k KEY [-k KEY2 ...]
Starts one run per key, sequentially. On exit/interrupt, kills all descendants.
EOF
}

keys=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k|--key) [[ $# -ge 2 ]] || { echo "Error: missing value for $1" >&2; exit 2; }
              keys+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
(( ${#keys[@]} > 0 )) || { echo "Error: provide at least one -k/--key" >&2; exit 2; }

mkdir -p logs

mask_key(){ local k="$1"; local l=${#k}; (( l<=8 )) && printf '****' || printf '%s...%s' "${k:0:4}" "${k: -4}"; }

CURRENT_PGID=""

cleanup() {
  echo
  echo "[CLEANUP] Terminating current run and its descendants..."
  if [[ -n "${CURRENT_PGID:-}" ]]; then
    # Kill the whole process group for the current run (child + grandchildren)
    kill -TERM -- -"${CURRENT_PGID}" 2>/dev/null || true
    sleep 0.5
    kill -KILL -- -"${CURRENT_PGID}" 2>/dev/null || true
  fi
  # Reap anything left
  wait || true
}
trap cleanup EXIT INT TERM HUP

status=0
for key in "${keys[@]}"; do
  log="logs/AnswerGenGem25f_$(date +%F_%H-%M-%S-%N).log"
  echo "[START] $(date -Is) with key $(mask_key "$key") → $log"

  # Start this run in its own process group (PGID = child PID) so we can kill all descendants
  # Run in background, then `wait` to enforce sequential execution
  set +e
  setsid python3 h_rae/src/data_prep/gen_officialdata_rae.py \
    --prompts h_rae/data/rae_official/RobustAlpacaEval_converted.json \
    --output h_rae/data/baseline/gemini15f_answers_rae.json \
    --model gemini-1.5-flash \
    --api-key "$key" \
    --log-name AnswerGenGem15f \
    --delay-ms 4000 \
    --max-attempts 1 \
    --api-call-max 200 \
    --max-input-tokens 120 \
    --max-output-tokens 256 \
    >> "$log" 2>&1 &
  child_pid=$!
  set -e

  # The PGID of a setsid-launched process is its own PID
  CURRENT_PGID="$child_pid"

  # Wait for this run to complete before starting the next
  if ! wait "$child_pid"; then
    status=1
  fi

  # Belt-and-suspenders: ensure no stragglers remain in that group
  kill -TERM -- -"${CURRENT_PGID}" 2>/dev/null || true
  sleep 0.2
  kill -KILL -- -"${CURRENT_PGID}" 2>/dev/null || true
  CURRENT_PGID=""
done

exit "$status"
