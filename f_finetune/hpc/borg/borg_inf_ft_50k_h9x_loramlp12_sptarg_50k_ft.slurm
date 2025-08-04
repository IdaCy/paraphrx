#!/bin/bash -l
#SBATCH --job-name=h9x_loramlp12_sptarg_50k_ft
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%x_%j_h9x_loramlp12_sptarg_50k_ft.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%x_%j_h9x_loramlp12_sptarg_50k_ft.err

set -euo pipefail
echo "$(date) – h9x_loramlp12_sptarg_50k_ft job started on $(hostname)"
echo "$(date) – h9x_loramlp12_sptarg_50k_ft job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"

mkdir -p logs f_finetune/output_inf_ft_50k

export SCRATCH=/scratch_root/ifc24
export TMPDIR="$SCRATCH/tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
echo "Using scratch-backed TMPDIR=$TMPDIR"

# Shared HF cache
export HF_HOME="$PROJ/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME"

# Redirect Triton autotune cache
export TRITON_CACHE_DIR="$TMPDIR/triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

export PIP_CACHE_DIR="$TMPDIR/pip_cache_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/xdg_cache_${SLURM_JOB_ID}"
export WANDB_DIR="$TMPDIR/wandb_cache_${SLURM_JOB_ID}"
export WANDB_DATA_DIR="$TMPDIR/wandb_data_${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1
export PYTHONPATH=
export PATH=$HOME/.local/bin:$PATH

cleanup () {
  echo "$(date) – cleaning up only my directories for $SLURM_JOB_ID"
  rm -rf \
    "$PIP_CACHE_DIR" \
    "$XDG_CACHE_HOME" \
    "$WANDB_DIR" \
    "$TRITON_CACHE_DIR"
}
trap cleanup EXIT ERR INT TERM

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR"

# Pick an available Python-3 binary (prefers 3.11 → 3.8)
for P in python3.11 python3.10 python3.9 python3; do
  if command -v $P &>/dev/null; then PYBIN=$(command -v $P); break; fi
done
[ -z "${PYBIN:-}" ] && { echo "No python3 on PATH"; exit 1; }
echo "Using $($PYBIN -V)"

export PYTHON_PKGS="$TMPDIR/python-packages"
mkdir -p "$PYTHON_PKGS"
export PYTHONPATH="$PYTHON_PKGS:$PYTHONPATH"

curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$TMPDIR/get-pip.py"

# Install pip into your PYTHON_PKGS target area
$PYBIN "$TMPDIR/get-pip.py" --target "$PYTHON_PKGS" --no-cache-dir

# install pip & wheel (keep your pinned setuptools)
$PYBIN -m pip install --upgrade pip wheel \
  --no-cache-dir --target "$PYTHON_PKGS"

$PYBIN -m pip install --upgrade setuptools \
  --no-cache-dir --target "$PYTHON_PKGS"

# install everything from wheels only (no source builds)
$PYBIN -m pip install --prefer-binary --only-binary=:all: \
  torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121 \
  transformers==4.45.0 \
  peft==0.11.1 \
  bitsandbytes==0.43.2 \
  datasets==2.19.0 \
  numpy==1.26.4 \
  wandb \
  matplotlib \
  --no-cache-dir \
  --target "$PYTHON_PKGS"

$PYBIN - <<'PY'
import torch, transformers, pathlib, importlib.util, numpy as np
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available(), torch.version.cuda)
print("Bits&Bytes  :", __import__("bitsandbytes").__version__)
print("Transformers:", transformers.__version__)
print("NumPy       :", np.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY

source /scratch_root/ifc24/.hf_token || true
[ -f /scratch_root/ifc24/.wandb_idacasp_key ] && source /scratch_root/ifc24/.wandb_idacasp_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_lora}

export TOKENIZERS_PARALLELISM=false

RUN_SCRIPT="f_finetune/src/ft_inference_all_data.py"
BASE_MODEL_PATH="f_finetune/model"
FT_PATH="f_finetune/outputs_6/all_data_all_layers/1-2_lowLR_fullFT"
OUTPUT_JSON="$FT_PATH/inf_results.json"
LORA_PATH="$FT_PATH/final"
mkdir -p "$(dirname "$OUTPUT_JSON")"

# sbatch --dependency=afternotok:1143 c_assess_inf/hpc/borg_inf_acts.slurm

# Run
echo "$(date) – h9x_loramlp12_sptarg_50k_ft initial setup OK"
RUN_SCRIPT="f_finetune/src/inf_ft_50k.py"
srun $PYBIN "$RUN_SCRIPT" \
  --data_path a_data/alpaca/50k_phrxed.json \
  --base_model_path f_finetune/model \
  --lora_path f_finetune/outputs/h9x_loramlp12_sptarg_50k_ft \
  --output_json f_finetune/output_inf_ft_50k/h9x_loramlp12_sptarg_50k_ft.json \
  --log_name h9x_loramlp12_sptarg_50k_ft \
  --val_pct 0.05 --test_pct 0.05 \
  --batch 8 \
  --max_tokens 128 \
  --quant 4bit \
  --wandb_project paraphrx_inference \
  --max_samples 500 \
  $WANDB_FLAG

echo "$(date) – h9x_loramlp12_sptarg_50k_ft job itself finished OK"

rm -rf "$TMPDIR" \
       "$HF_HOME" \
       "$PIP_CACHE_DIR" \
       "$XDG_CACHE_HOME"

echo "$(date) – h9x_loramlp12_sptarg_50k_ft job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) – h9x_loramlp12_sptarg_50k_ft job finished & stuff cleaned OK"
