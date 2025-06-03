# Purpose: Re-phrase Alpaca JSON slices locally (CPU-only)
# Usage:
# chmod +x ./a_data/preproc/hpc/run_prx_alpaca_local_modelchoice_further.sh
# nohup ./a_data/preproc/hpc/run_prx_alpaca_local_modelchoice_further.sh >  logs/prx_modelchoice_further_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
#
# ps -f -u "$USER" | grep run_prx_alpaca_local_modelchoice_further.sh | grep -v grep

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

echo "$(date '+%Y-%m-%d %H:%M:%S') - further_1-2_prx_alpaca_gen_phrx started alpaca" >> "$WORKDIR/times.log"

export GOOGLE_API_KEY=""

for SLICE in 5 4; do
  IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"

  for TYPE in language context; do
    OUT_JSON="$DATA_DIR/alpaca/slice_100/prxed_${TYPE}_slice${SLICE}.json"

    echo "▶︎ Processing slice $SLICE ($TYPE)..."

    if ! cargo gen_phrx_modchoice \
        --version-set "$TYPE" \
        --model "gemini-2.5-flash-preview-05-20" \
        "$IN_JSON" \
        "$OUT_JSON"; then
      echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - slice $SLICE type $TYPE failed" >> "$WORKDIR/times.log"
    fi
  done
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - further_1-2_prx_alpaca_gen_phrx finished alpaca" >> "$WORKDIR/times.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') - further_1-2_prx_alpaca_gen_phrx started gsm8k language" >> "$WORKDIR/times.log"

### GSM8K main dataset

IN_JSON="$DATA_DIR/gsm8k/main_500.json"

for TYPE in language; do
  OUT_JSON="$DATA_DIR/gsm8k/main_500_prxed_${TYPE}.json"

  echo "▶︎ Processing slice ($TYPE)..."

  if ! cargo gen_phrx_modchoice \
      --version-set "$TYPE" \
      --model "gemini-2.5-flash-preview-05-20" \
      "$IN_JSON" \
      "$OUT_JSON"; then
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - type $TYPE failed" >> "$WORKDIR/times.log"
  fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - further_1-2_prx_alpaca_gen_phrx finished gsm8k language" >> "$WORKDIR/times.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') - further_1-2_prx_alpaca_gen_phrx started mmlu language" >> "$WORKDIR/times.log"

### MMLU moral dataset

IN_JSON="$DATA_DIR/mmlu/moral_scenarios_500.json"

for TYPE in language; do
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

echo "$(date '+%Y-%m-%d %H:%M:%S') - further_1-2_prx_alpaca_gen_phrx finished mmlu language" >> "$WORKDIR/times.log"
echo "All slices complete."
