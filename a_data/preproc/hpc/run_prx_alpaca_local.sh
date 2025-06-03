# Purpose: Re-phrase Alpaca JSON slices locally (CPU-only)
# Usage:
# nohup ./a_data/preproc/hpc/run_prx_alpaca_local.sh >  logs/prx_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown

set -euo pipefail

# user-configurable paths
#WORKDIR="$HOME/path/to/repo"
WORKDIR="/Users/ifc24/Develop/paraphrx"
DATA_DIR="$WORKDIR/a_data"
MANIFEST_PATH="$WORKDIR/a_data/preproc/rephras/Cargo.toml"
LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR"

cd "$WORKDIR"
# Ensure Rust env is loaded
if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1090
  . "$HOME/.cargo/env"
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') - sy-la-co_1-2_prx_alpaca_gen_phrx started" >> "$WORKDIR/times.log"

# Optional: set your Google API key in the environment
export GOOGLE_API_KEY="AIzaSyBx1W8ovOHCOhfuuUcYJ-wquS_oDLxgMBc"

for SLICE in 1 2; do
  IN_JSON="$DATA_DIR/alpaca/un_prxed/slice${SLICE}.json"

  for TYPE in syntax language context; do
    OUT_JSON="$DATA_DIR/alpaca/slice_100/prxed_${TYPE}_slice${SLICE}.json"

    echo "▶︎ Processing slice $SLICE ($TYPE)…"
    cargo run \
      --manifest-path "$MANIFEST_PATH" \
      --release -- \
      --version-set "$TYPE" \
      "$IN_JSON" \
      "$OUT_JSON"
  done
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - sy-la-co_1-2_prx_alpaca_gen_phrx finished" >> "$WORKDIR/times.log"
echo "All slices complete."
