#!/bin/bash -l
#SBATCH --job-name=lo5_1-3_lr1e-4_ga16_sas300_ft_inf
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH -o /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_lo5_1-3_lr1e-4_ga16_sas300_ft_inf.out
#SBATCH -e /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_lo5_1-3_lr1e-4_ga16_sas300_ft_inf.err
set -euo pipefail

echo "$(date) – lo5_1-3_lr1e-4_ga16_sas300_ft_inf job started on $(hostname)"
echo "$(date) – lo5_1-3_lr1e-4_ga16_sas300_ft_inf job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

echo "$(date) – lo5_1-3_lr1e-4_ga16_sas300_ft_inf initial setup"

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
python -m pip install datasets peft wandb --quiet
python -m pip uninstall -y flash-attn >/dev/null 2>&1 || true
echo "Packages ready."

# Speedier model downloads when the cache is cold
export HF_HUB_ENABLE_HF_TRANSFER=1

source /scratch_dgxl/ifc24/.hf_token || true
[ -f /scratch_dgxl/ifc24/.wandb_key ] && source /scratch_dgxl/ifc24/.wandb_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_lora}

# silence tokenizer warnings
export TOKENIZERS_PARALLELISM=false

echo "$(date) – lo5_1-3_lr1e-4_ga16_sas300_ft_inf initial setup OK"

# Paths & arguments
RUN_SCRIPT="f_finetune/src/ft_inference_all_data.py"
BASE_MODEL_PATH="f_finetune/model"
FT_PATH="f_finetune/outputs_4/alpaca_specific_layers/outputs_buckets_1-3_lr1e-4_ga16_sas300"
OUTPUT_JSON="$FT_PATH/inf_results.json"
LORA_PATH="$FT_PATH/final"
mkdir -p "$(dirname "$OUTPUT_JSON")"

# Run
srun python "$RUN_SCRIPT" \
  --data_paths \
  "f_finetune/data/output_splits_alpaca/buckets_1-3_test.json" \
  --datasets "alpaca" \
  --base_model_path "$BASE_MODEL_PATH" \
  --lora_path "$LORA_PATH" \
  --output_json "$OUTPUT_JSON" \
  --buckets 1-3 \
  --batch 16 \
  --max_tokens 256 \
  --wandb_project "paraphrx_lora"

echo "$(date) – lo5_1-3_lr1e-4_ga16_sas300_ft_inf job finished OK"
echo "$(date) – lo5_1-3_lr1e-4_ga16_sas300_ft_inf job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
