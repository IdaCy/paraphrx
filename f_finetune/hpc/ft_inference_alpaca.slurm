#!/bin/bash -l
#SBATCH --job-name=buckets5_alpaca_ft_inf
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH -o /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_buckets5_alpaca_ft_inf.out
#SBATCH -e /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_buckets5_alpaca_ft_inf.err
set -euo pipefail

echo "$(date) – buckets5_alpaca_ft_inf job started on $(hostname)"
echo "$(date) – buckets5_alpaca_ft_inf job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

echo "$(date) – buckets5_alpaca_ft_inf initial setup"

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
python -m pip install torch \
                       --index-url https://download.pytorch.org/whl/cu121 --quiet
python -m pip install "transformers>=4.40.0" accelerate tqdm --quiet
python -m pip install huggingface_hub sentencepiece "protobuf==3.20.*" --quiet
python -m pip install datasets peft --quiet
python -m pip uninstall -y flash-attn >/dev/null 2>&1 || true
echo "Packages ready."

# Speedier model downloads when the cache is cold
export HF_HUB_ENABLE_HF_TRANSFER=1

# needed: voice style speci_char voice
echo "$(date) – buckets5_alpaca_ft_inf initial setup OK"
echo "$(date) – buckets5_alpaca_ft_inf voice start"

# Paths & arguments
RUN_SCRIPT="f_finetune/src/ft_inference_alpaca.py"
DATA_PATHS="f_finetune/data/alpaca_gemma-2-2b-it.json"
OUTPUT_JSON="f_finetune/outputs/inference_buckets/inference_buckets5.json"
mkdir -p "$(dirname "$OUTPUT_JSON")"

# Run
srun python "$RUN_SCRIPT" \
  --data_paths "$DATA_PATHS" \
  --base_model_path f_finetune/model \
  --lora_path       f_finetune/outputs/buckets5/final \
  --output_json "$OUTPUT_JSON" \
  --buckets 1-5 \
  --batch 16 \
  --max_tokens 256

echo "$(date) – buckets5_alpaca_ft_inf job finished OK"
echo "$(date) – buckets5_alpaca_ft_inf job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
