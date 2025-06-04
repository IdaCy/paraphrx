# Purpose: Re-phrase Alpaca JSON slices locally (CPU-only)
# Usage:
# chmod +x ./a_data/preproc/hpc/run_prx_alpaca_local_modelchoice_short_language.sh
# nohup ./a_data/preproc/hpc/run_prx_alpaca_local_modelchoice_short_language.sh >  logs/prx_modelchoice_short_language_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
#
# ps -f -u "$USER" | grep run_prx_alpaca_local_modelchoice_short_language.sh | grep -v grep

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
# language context !!

echo "$(date '+%Y-%m-%d %H:%M:%S') - short_language_2_prx_alpaca_gen_phrx started" >> "$WORKDIR/times.log"

# Optional: set your Google API key in the environment
#export GOOGLE_API_KEY=""
export GOOGLE_API_KEY="AIzaSyAHiQDQ0Zs-2wY8gNWWEu2bNW_Pt6Mdb68"

for SLICE in 3; do
  IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"

  for TYPE in language; do
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

echo "$(date '+%Y-%m-%d %H:%M:%S') - short_language_2_prx_alpaca_gen_phrx finished" >> "$WORKDIR/times.log"
echo "All slices complete."
