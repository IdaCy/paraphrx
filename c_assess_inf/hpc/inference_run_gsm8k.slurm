#!/bin/bash -l
#SBATCH --job-name=gk_g9_length_2_inf
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH -o /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_gk_g9_length_2_inf.out
#SBATCH -e /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_gk_g9_length_2_inf.err
set -euo pipefail

echo "$(date) – gk_g9_length_2_inf job started on $(hostname)"
echo "$(date) – gk_g9_length_2_inf job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

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

# Python environment (light venv)
if [ ! -d venv ]; then
    echo "Creating venv …"
    python3.12 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip --quiet
python -m pip install wheel --quiet
python -m pip install torch torchvision torchaudio \
                       --index-url https://download.pytorch.org/whl/cu121 --quiet
python -m pip install "transformers>=4.40.0" accelerate tqdm --quiet
python -m pip install huggingface_hub sentencepiece "protobuf==3.20.*" hf_transfer --quiet
python -m pip install bitsandbytes --quiet
python -m pip uninstall -y flash-attn >/dev/null 2>&1 || true
echo "Packages ready."

# Speedier model downloads when the cache is cold
export HF_HUB_ENABLE_HF_TRANSFER=1

# Hugging Face token
#source ~/.hf_token
source /scratch_dgxl/ifc24/.hf_token

# Paths & arguments
RUN_SCRIPT="c_assess_inf/src/inference_run_sped.py"
INPUT_JSON="a_data/gsm8k/prxed_main_500/length.json"
MODEL_NAME=gemma-2-9b-it
MODEL_OWNER=google
MODEL_ID=${MODEL_OWNER}/${MODEL_NAME}
OUTPUT_JSON=c_assess_inf/output/gsm8k/${MODEL_NAME}/length_2.json
mkdir -p "$(dirname "$OUTPUT_JSON")"

# Run
srun python "$RUN_SCRIPT" \
     "$INPUT_JSON" \
     "$OUTPUT_JSON" \
     --model "$MODEL_ID" \
     --batch 16 \
     --quant 4bit \
     --log_every 180 \
     --type gsm8k_g9_length_2 \
     --temperature 0 \
     --max_tokens 128 \
     --device cuda:0

echo "$(date) – gk_g9_length_2_inf job finished OK"
echo "$(date) – gk_g9_length_2_inf job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
