# Purpose: Re-phrase Alpaca JSON slices locally (CPU-only)
# Usage:
# chmod +x ./a_data/preproc/hpc/run_prx_alpaca_local_modelchoice_onextra25.sh
# nohup ./a_data/preproc/hpc/run_prx_alpaca_local_modelchoice_onextra25.sh >  logs/prx_modelchoice_$(date +%Y%m%d_%H%M%S)_onextra25.log 2>&1 & disown
#
# ps -f -u "$USER" | grep run_prx_alpaca_local_modelchoice_onextra25.sh | grep -v grep

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

echo "$(date '+%Y-%m-%d %H:%M:%S') - onextra25_prx_alpaca_gen_phrx started" >> "$WORKDIR/times.log"

export GOOGLE_API_KEY="AIzaSyDGO2Q2VtQS9oeIKOGx0ZYiqXLJyMudz3Q"

# TONE - 2
SLICE="2"
TYPE="tone"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_modchoice \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"

# VOICE - 2
SLICE="2"
TYPE="voice"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_modchoice \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"

# boundary - 5
SLICE="5"
TYPE="boundary"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_modchoice \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"

# context - 5
SLICE="5"
TYPE="context"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_modchoice \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"

# length - 5
SLICE="5"
TYPE="length"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_modchoice \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"

# obstruction - 5
SLICE="5"
TYPE="obstruction"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_modchoice \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"

# special_chars - 5
SLICE="5"
TYPE="special_chars"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_modchoice \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"
  

echo "$(date '+%Y-%m-%d %H:%M:%S') - onextra25_prx_alpaca_gen_phrx finished" >> "$WORKDIR/times.log"
echo "All slices complete."
