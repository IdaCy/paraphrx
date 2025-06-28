#!/bin/bash -l
#SBATCH --job-name=a_q2_voice_inf
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=23:50:00
#SBATCH -o /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_a_q2_voice_inf.out
#SBATCH -e /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_a_q2_voice_inf.err
set -euo pipefail

echo "$(date) – a_q2_voice_inf job started on $(hostname)"
echo "$(date) – a_q2_voice_inf job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

echo "$(date) – a_q2_voice_inf initial setup"

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

# needed: voice style speci_char voice
echo "$(date) – a_q2_voice_inf initial setup OK"
echo "$(date) – a_q2_voice_inf voice start"

# for all of them
RUN_SCRIPT="c_assess_inf/src/inference_run_sped_3.py"
MODEL_NAME="Qwen2.5-3B-Instruct"
MODEL_OWNER="Qwen"
MODEL_ID=${MODEL_OWNER}/${MODEL_NAME}

## voice

# Paths & arguments
INPUT_JSON="a_data/alpaca/slice_500/voice.json"
OUTPUT_JSON=c_assess_inf/output/alpaca_prxed/${MODEL_NAME}/voice.json
mkdir -p "$(dirname "$OUTPUT_JSON")"

# Run
srun python "$RUN_SCRIPT" \
     "$INPUT_JSON" \
     "$OUTPUT_JSON" \
     --model "$MODEL_ID" \
     --batch 16 \
     --quant 4bit \
     --log_every 200 \
     --type alpaca_q2_voice \
     --temperature 0 \
     --max_tokens 128

## STYLE
echo "$(date) – a_q2_voice_inf voice end"

echo "$(date) – a_q2_voice_inf job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
