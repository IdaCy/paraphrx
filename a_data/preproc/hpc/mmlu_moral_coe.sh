# Purpose: Re-phrase Alpaca JSON slices locally
# Usage:
# chmod +x a_data/preproc/hpc/mmlu_moral_coe.sh
# nohup a_data/preproc/hpc/mmlu_moral_coe.sh > logs/mmlu_moral_coe_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
# caffeinate -dims nohup a_data/preproc/hpc/mmlu_moral_coe.sh > logs/mmlu_moral_coe_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
#
# ps -f -u "$USER" | grep mmlu_moral_coe.sh | grep -v grep

set -euo pipefail

# user-configurable paths
#WORKDIR="$HOME/path/to/repo"
WORKDIR="/Users/ifc24/Develop/paraphrx"
DATA_DIR="$WORKDIR/a_data"
LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR"

cd "$WORKDIR"
# Ensure Rust env is loaded
if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1090
  . "$HOME/.cargo/env"
fi

# choice:
#const MODEL: &str = "gemini-2.5-pro-preview-05-06";
#const MODEL: &str = "gemini-2.5-flash-preview-05-20";
#const MODEL: &str = "gemini-2.5-pro-preview-05-06";
#const MODEL: &str = "gemini-2.5-flash-preview-04-17";

echo "$(date '+%Y-%m-%d %H:%M:%S') - mmlu_moral_coe started" >> "$WORKDIR/times.log"

export GOOGLE_API_KEY="AIzaSyAHiQDQ0Zs-2wY8gNWWEu2bNW_Pt6Mdb68"

### mmlu main dataset
IN_JSON="$DATA_DIR/mmlu/selection_original/moral_scenarios_500.json"

for TYPE in context extra; do
  OUT_JSON="$DATA_DIR/mmlu/moral_500_prxed_${TYPE}.json"

  echo "..."
  echo "..."
  echo "..."
  echo "Processing mmlu ($TYPE)..."

  if ! cargo gen_phrx_skipfail \
      --version-set "$TYPE" \
      --model "gemini-2.5-flash-preview-05-20" \
      --max-attempts 6 \
      "$IN_JSON" \
      "$OUT_JSON"; then
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - type $TYPE failed" >> "$WORKDIR/times.log"
  fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - mmlu_moral_coe finished " >> "$WORKDIR/times.log"
echo "All slices complete."
