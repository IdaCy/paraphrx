# Purpose: Re-phrase Alpaca JSON slices locally (CPU-only)
# Usage:
# chmod +x ./a_data/preproc/hpc/run_prx_alpaca_local_modelchoice_mmlu_moral.sh
# nohup ./a_data/preproc/hpc/run_prx_alpaca_local_modelchoice_mmlu_moral.sh >  logs/prx_modelchoice_mmlu_moral_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
#
# ps -f -u "$USER" | grep run_prx_alpaca_local_modelchoice_mmlu_moral.sh | grep -v grep

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

echo "$(date '+%Y-%m-%d %H:%M:%S') - mmlu_moral_prx_alpaca_gen_phrx started alpaca" >> "$WORKDIR/times.log"

export GOOGLE_API_KEY="AIzaSyCkeYN7o2S8yZYaGt7-KifEiGejGvP1KmY"


### MMLU moral dataset

IN_JSON="$DATA_DIR/mmlu/moral_scenarios_500.json"

for TYPE in voice tone syntax style special_chars obstruction length extra context; do
#for TYPE in boundary; do
  OUT_JSON="$DATA_DIR/mmlu/moral_scenarios_500_prxed_${TYPE}.json"

  echo "▶︎ Processing slice ($TYPE)..."

  if ! cargo gen_phrx_modchoice \
      --version-set "$TYPE" \
      --model "gemini-2.5-flash-preview-05-20" \
      "$IN_JSON" \
      "$OUT_JSON"; then
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - type $TYPE failed" >> "$WORKDIR/times.log"
  fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - mmlu_moral_prx_alpaca_gen_phrx finished mmlu language" >> "$WORKDIR/times.log"
echo "All slices complete."
