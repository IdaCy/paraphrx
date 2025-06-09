# Purpose: Re-phrase Alpaca JSON slices locally
# Usage:
# chmod +x a_data/preproc/hpc/gsm8k_prx_spec_char.sh
# nohup a_data/preproc/hpc/gsm8k_prx_spec_char.sh > logs/gsm8k_prx_spec_char_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
# caffeinate -dims nohup a_data/preproc/hpc/gsm8k_prx_spec_char.sh > logs/gsm8k_prx_spec_char_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
#
# ps -f -u "$USER" | grep gsm8k_prx_spec_char.sh | grep -v grep

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

echo "$(date '+%Y-%m-%d %H:%M:%S') - gsm8k_prx_spec_char started" >> "$WORKDIR/times.log"

export GOOGLE_API_KEY=""

### GSM8K main dataset
IN_JSON="$DATA_DIR/gsm8k/original/main_500.json"

#for TYPE in voice tone syntax style special_chars obstruction language length boundary extra context; do
for TYPE in spec_char language; do
  OUT_JSON="$DATA_DIR/gsm8k/main_500_prxed_${TYPE}.json"

  echo "---"
  echo "---"
  echo "---"
  echo "Processing GSM8K ($TYPE)..."

  if ! cargo gen_phrx_skipfail \
      --version-set "$TYPE" \
      --model "gemini-2.5-flash-preview-05-20" \
      --max-attempts 6 \
      "$IN_JSON" \
      "$OUT_JSON"; then
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - type $TYPE failed" >> "$WORKDIR/times.log"
  fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - gsm8k_prx_spec_char finished " >> "$WORKDIR/times.log"
echo "All slices complete."
