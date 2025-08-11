#!/usr/bin/env bash
set -uo pipefail
trap 'echo "› CTRL-C - stopping"; kill -TERM -- -$$' INT TERM
set -m

echo "$(date '+%Y-%m-%d %H:%M:%S') - robustalpaca received"

# Setup log directories
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_robust_alpaca_eval_$(date '+%Y%m%d_%H%M%S').txt"
exec >>"$MASTER_LOG" 2>&1
set -x

log_time() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >>"$HOME/times.log"
}

run_eval() {
    local idx=$1
    local api_key=$2
    local timestamp
    timestamp=$(date +%F_%T)

    log_time "robustalpaca started $idx"

    cargo robust_alpaca_eval llm-judge \
        --prompts a_data/alpaca/50k_phrxed.json \
        --answers-original c_assess_inf/output50k/gpt4_answers_1440.json \
        --answers-paraphrased f_finetune/output_inf_ft_50k/li9x_a1_notarg_inf.json \
        --output e_eval/output_robust_alpaca_eval/li9x_a1_notarg_inf_against_gpt4.json \
        --judging-model gemini-2.0-flash \
        --delay-ms 4000 \
        --api-call-max 200 \
        --api-key "$api_key" \
        --num-judge-votes 3 \
        >> "$LOG_DIR/robalev_${idx}_${timestamp}.log" 2>&1

    log_time "robustalpaca finished $idx"
}

# Sequential runs with different API keys
run_eval 1 "xxxxxxxxxxxxxx"
run_eval 2 "xxxxxxxxxxxxxx"

exit 0
