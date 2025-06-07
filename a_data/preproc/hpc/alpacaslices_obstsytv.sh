# Purpose: Re-phrase Alpaca JSON slices locally (CPU-only)
# Usage:
# chmod +x ./a_data/preproc/hpc/alpacaslices_obstsytv.sh
# nohup ./a_data/preproc/hpc/alpacaslices_obstsytv.sh >  logs/alpacaslices_obstsytv_$(date +%Y%m%d_%H%M%S)_onextra25.log 2>&1 & disown
# caffeinate -dims nohup a_data/preproc/hpc/alpacaslices_obstsytv.sh > logs/alpacaslices_obstsytv_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
# caffeinate -dims a_data/preproc/hpc/alpacaslices_obstsytv.sh > logs/alpacaslices_obstsytv_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
#
# ps -f -u "$USER" | grep alpacaslices_obstsytv.sh | grep -v grep

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

echo "$(date '+%Y-%m-%d %H:%M:%S') - alpacaslices_obstsytv started" >> "$WORKDIR/times.log"

export GOOGLE_API_KEY="AIzaSyBx1W8ovOHCOhfuuUcYJ-wquS_oDLxgMBc"



# SYNTAX
SLICE="2"
TYPE="syntax"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_skipfail \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"

SLICE="4"
TYPE="syntax"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_skipfail \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"


# TONE - 4
SLICE="4"
TYPE="tone"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_skipfail \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"


# VOICE - 4
SLICE="4"
TYPE="voice"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_skipfail \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"


echo "$(date '+%Y-%m-%d %H:%M:%S') - alpacaslices_obstsytv finished" >> "$WORKDIR/times.log"
echo "All slices complete."
