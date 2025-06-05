# Purpose: Re-phrase Alpaca JSON slices locally (CPU-only)
# Usage:
# chmod +x ./a_data/preproc/hpc/alpacaslices_bocoexlalesp.sh
# nohup ./a_data/preproc/hpc/alpacaslices_bocoexlalesp.sh >  logs/alpacaslices_bocoexlalesp_$(date +%Y%m%d_%H%M%S)_onextra25.log 2>&1 & disown
# caffeinate -dims nohup a_data/preproc/hpc/alpacaslices_bocoexlalesp.sh > logs/alpacaslices_bocoexlalesp_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
# caffeinate -dims a_data/preproc/hpc/alpacaslices_bocoexlalesp.sh > logs/alpacaslices_bocoexlalesp_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
#
# ps -f -u "$USER" | grep alpacaslices_bocoexlalesp.sh | grep -v grep

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

echo "$(date '+%Y-%m-%d %H:%M:%S') - alpacaslices_bocoexlalesp started" >> "$WORKDIR/times.log"

export GOOGLE_API_KEY="AIzaSyD7_c8jRdu8xwHxRTjjfJVU0slt7aAzGGI"


# EXTRA - 5
SLICE="5"
TYPE="extra"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_modchoice \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"


# LANGUAGE - 4
SLICE="4"
TYPE="language"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_modchoice \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"


# LENGTH - 4
SLICE="4"
TYPE="length"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_modchoice \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"


# STYLE
SLICE="4"
TYPE="style"
IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"
OUT_JSON="$DATA_DIR/alpaca/slice_100/${TYPE}_slice${SLICE}.json"
echo "Processing slice $SLICE ($TYPE)..."

cargo gen_phrx_modchoice \
  --version-set "$TYPE" \
  --model "gemini-2.5-flash-preview-05-20" \
  "$IN_JSON" \
  "$OUT_JSON"


echo "$(date '+%Y-%m-%d %H:%M:%S') - alpacaslices_bocoexlalesp finished" >> "$WORKDIR/times.log"
echo "All slices complete."
