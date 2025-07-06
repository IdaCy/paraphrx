#!/opt/homebrew/bin/bash
set -euo pipefail

INPUT="a_data/mmlu/prxed_moral_500_scenarios/extra.json"

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
    -i a_data/mmlu/prxed_moral_500_scenarios/extra.json \
    -a a_data/mmlu/prxed_moral_500_scenarios/extra_a.json \
    -b a_data/mmlu/prxed_moral_500_scenarios/extra_b.json \
    -d prompt_count -d prompt_id -d instruction_original \
    $(printf -- '-k %s ' "${KEEP[@]}")
