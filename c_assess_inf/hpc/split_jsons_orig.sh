#!/opt/homebrew/bin/bash
set -euo pipefail

INPUT="a_data/alpaca/slice_500/extra.json"

TOTAL=$(jq '.[0] | keys_unsorted | map(select(startswith("instruct_"))) | length' "$INPUT")
HALF=$(( TOTAL / 2 ))

# First HALF instruct_* keys, alphabetically
mapfile -t KEEP < <(
  jq -r '.[0] | keys_unsorted[] | select(startswith("instruct_"))' "$INPUT" |
  sort | head -n "$HALF"
)

# Build the -k flags
for k in "${KEEP[@]}"; do
  KEEP_FLAGS+=("-k" "$k")
done

cargo jssplit \
    -i a_data/alpaca/slice_500/extra.json \
    -a a_data/alpaca/slice_500/extra_a.json \
    -b a_data/alpaca/slice_500/extra_b.json \
    -d prompt_count -d prompt_id -d instruction_original -d input -d output \
    $(printf -- '-k %s ' "${KEEP[@]}")
