#!/bin/bash -l
#SBATCH --job-name=test_ft_ppl
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH -o /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_test_ft_ppl.out
#SBATCH -e /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_test_ft_ppl.err
set -euo pipefail

echo "$(date) - test_ft_ppl job started on $(hostname)"
echo "$(date) - test_ft_ppl job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

# Workspace + caches on scratch
cd /scratch_dgxl/ifc24/proj/paraphrx
mkdir -p logs tmp hf_cache pip_cache xdg_cache

export TMPDIR=$PWD/tmp
export HF_HOME=$PWD/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=$PWD/pip_cache
export XDG_CACHE_HOME=$PWD/xdg_cache
export TORCH_HOME=$XDG_CACHE_HOME/torch
export TORCHINDUCTOR_CACHE_DIR=$TORCH_HOME/inductor
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
mkdir -p "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Python environment
if [ ! -d venv ]; then
    echo "Creating venv ..."
    python3.12 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip --quiet
python -m pip install torch torchvision torchaudio \
                       --index-url https://download.pytorch.org/whl/cu121 --quiet
python -m pip install "transformers>=4.40.0" accelerate tqdm --quiet
python -m pip install huggingface_hub sentencepiece "protobuf==3.20.*" hf_transfer --quiet

python -m pip install peft --quiet

echo "Packages ready."

# Hugging Face token
source /scratch_dgxl/ifc24/.hf_token

# Paths & arguments
RUN_SCRIPT="e_eval/src/perplexity_after_ft.py"
INPUT_JSON="b_tests/data/alpaca.json"
MODEL_NAME="gemma-2-2b-it"
MODEL_OWNER="google"
MODEL_ID=${MODEL_OWNER}/${MODEL_NAME}

OUTPUT_JSON="b_tests/data/alpaca_with_perplexity_5.json"
mkdir -p "$(dirname "$OUTPUT_JSON")"

# Run
srun python "$RUN_SCRIPT" \
     "$INPUT_JSON" "$OUTPUT_JSON" \
     --model "$MODEL_ID" \
     --batch 16 \
     --quant none \
     --adapter f_finetune/outputs/alpaca/all_layers/outputs_buckets_1-1/final=ppl_ft_buckets_1-1_all_layers_alpaca \
     --adapter f_finetune/outputs/alpaca/all_layers/outputs_buckets_1-2/final=ppl_ft_buckets_1-2_all_layers_alpaca \
     --device cuda:0 \
     --debug-extra 2 --debug_tokens

echo "$(date) - test_ft_ppl job finished OK"
echo "$(date) - test_ft_ppl job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
