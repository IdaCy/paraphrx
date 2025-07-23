#!/bin/bash -l
#SBATCH --job-name=o6_1-2_lowLR_fullFT_inf_ft
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%x_%j_o6_1-2_lowLR_fullFT_inf_ft.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%x_%j_o6_1-2_lowLR_fullFT_inf_ft.err

set -euo pipefail
echo "$(date) – o6_1-2_lowLR_fullFT_inf_ft job started on $(hostname)"
echo "$(date) – o6_1-2_lowLR_fullFT_inf_ft job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"

mkdir -p logs

export TMPDIR=$PROJ/tmp_${SLURM_JOB_ID}
export HF_HOME=$PROJ/hf_cache_${SLURM_JOB_ID}
export PIP_CACHE_DIR=$PROJ/pip_cache_${SLURM_JOB_ID}
export XDG_CACHE_HOME=$PROJ/xdg_cache_${SLURM_JOB_ID}

export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export PYTHONNOUSERSITE=1
export PYTHONPATH=
export HOME=$TMPDIR
export PATH=$HOME/.local/bin:$PATH

cleanup () {
  echo "$(date) – cleaning scratch for $SLURM_JOB_ID"
  rm -rf "$TMPDIR" "$HF_HOME" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"
}
# run cleanup on:
#   EXIT  – normal completion
#   ERR   – any command that triggers `set -e`
#   INT   – Ctrl‑C from interactive dev
#   TERM  – scancel / pre‑emption
trap cleanup EXIT ERR INT TERM

mkdir -p "$TMPDIR" "$HF_HOME" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"
#tmp_${SLURM_JOB_ID} hf_cache_${SLURM_JOB_ID} pip_cache_${SLURM_JOB_ID} xdg_cache_${SLURM_JOB_ID}\

# Pick an available Python-3 binary (prefers 3.11 → 3.8)
for P in python3.11 python3.10 python3.9 python3; do
  if command -v $P >/dev/null 2>&1; then PYBIN=$(command -v $P); break; fi
done
[ -z "${PYBIN:-}" ] && { echo "No python3 on PATH"; exit 1; }
echo "Using $($PYBIN -V)"

# install standalone Miniconda in TMPDIR and activate
CONDADIR="$TMPDIR/conda"
ENVDIR="$TMPDIR/conda_env_${SLURM_JOB_ID}"
if [ ! -f "$CONDADIR/bin/conda" ]; then
  wget -qO "$TMPDIR/miniconda.sh" \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$TMPDIR/miniconda.sh" -b -p "$CONDADIR"
  rm "$TMPDIR/miniconda.sh"
fi
export PATH="$CONDADIR/bin:$PATH"

# auto-accept Terms of Service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# remove any old env so create is clean
rm -rf "$ENVDIR"

# create a fresh env with just Python and pip
conda create -y -q -p "$ENVDIR" python=3.10 pip
source "$CONDADIR/bin/activate" "$ENVDIR"

# upgrade pip/setuptools/wheel (no cache)
pip install --no-cache-dir --upgrade pip setuptools wheel -q

# **Install PyTorch GPU wheel via pip (no cache)**
pip install --no-cache-dir --prefer-binary \
  torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# install the rest of stack (including bitsandbytes ≥0.43.2), no cache
pip install --no-cache-dir --prefer-binary \
  "transformers==4.45.0" \
  "peft==0.11.1" \
  "datasets==2.19.0" \
  "sentencepiece" \
  "protobuf==3.20.*" \
  "wandb"
pip uninstall -y torchaudio torchvision -q || true

python - <<'PY'
import torch, transformers, pathlib, importlib.util, numpy as np
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available(), torch.version.cuda)
print("Transformers:", transformers.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY

source /scratch_root/ifc24/.hf_token || true
[ -f /scratch_root/ifc24/.wandb_key ] && source /scratch_root/ifc24/.wandb_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_lora}

export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

RUN_SCRIPT="f_finetune/src/ft_inference_all_data.py"
BASE_MODEL_PATH="f_finetune/model"
FT_PATH="f_finetune/outputs_6/all_data_all_layers/1-2_lowLR_fullFT"
OUTPUT_JSON="$FT_PATH/inf_results.json"
LORA_PATH="$FT_PATH/final"
mkdir -p "$(dirname "$OUTPUT_JSON")"

# Run
echo "$(date) – o6_1-2_lowLR_fullFT_inf_ft initial setup OK"
srun python "$RUN_SCRIPT" \
  --data_paths \
  "f_finetune/data/output_splits_alpaca/buckets_1-2_test.json" \
  "f_finetune/data/output_splits_gsm8k/buckets_1-2_test.json" \
  "f_finetune/data/output_splits_mmlu/buckets_1-2_test.json" \
  --datasets "alpaca" "gsm8k" "mmlu" \
  --base_model_path "$BASE_MODEL_PATH" \
  --lora_path "$LORA_PATH" \
  --output_json "$OUTPUT_JSON" \
  --buckets 1-2 \
  --batch 16 \
  --max_tokens 256 \
  --wandb_project "paraphrx_lora"

echo "$(date) – o6_1-2_lowLR_fullFT_inf_ft job itself finished OK"

rm -rf "$TMPDIR" \
       "$HF_HOME" \
       "$PIP_CACHE_DIR" \
       "$XDG_CACHE_HOME"

echo "$(date) – o6_1-2_lowLR_fullFT_inf_ft job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) – o6_1-2_lowLR_fullFT_inf_ft job finished & stuff cleaned OK"
