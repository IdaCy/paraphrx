#!/bin/bash -l
#SBATCH --job-name=l_ese_inference_with_adapter_11l1
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch_dgxl/ifc24/proj/paraphrx/logs/%j_%x.out
#SBATCH --error=/scratch_dgxl/ifc24/proj/paraphrx/logs/%j_%x.err

set -euo pipefail
echo "$(date) - ese_inference_with_adapter_11l1 job started on $(hostname)"
echo "$(date) - ese_inference_with_adapter_11l1 job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_dgxl/ifc24/proj/paraphrx
cd "$PROJ"
mkdir -p logs

export TMPDIR="${SLURM_LOCALTMP:-/tmp}"

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
  echo "$(date) - cleaning up only my directories for $SLURM_JOB_ID"
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
# --- NEW: Force a clean directory for packages ---
echo "Forcing clean package directory at $PYTHON_PKGS"
rm -rf "$PYTHON_PKGS"
mkdir -p "$PYTHON_PKGS"
# --- END NEW ---
export PYTHONPATH="$PYTHON_PKGS:$PYTHONPATH"

# DEEP-SEED compatible metadata libs
$PYBIN -m pip install --upgrade "importlib_metadata>=5.2.0" \
                                   "setuptools==65.5.1" \
                                   "testresources==2.0.1" \
    --no-cache-dir --target "$PYTHON_PKGS"

$PYBIN -m pip uninstall -y deepspeed

# install pip & wheel (keep your pinned setuptools)
$PYBIN -m pip install --upgrade pip wheel \
  --no-cache-dir --target "$PYTHON_PKGS"

# deepspeed to a wheelable version!
$PYBIN -m pip install --prefer-binary \
  torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121 \
  "transformers==4.46.3" "accelerate>=0.33.0" "tokenizers>=0.19.1" \
  peft==0.11.1 \
  "bitsandbytes>=0.43.2" \
  mpi4py \
  deepspeed \
  datasets==2.19.0 \
  numpy==1.24.4 \
  sentencepiece \
  loguru \
  protobuf==3.20.* \
  wandb \
  matplotlib \
  --no-cache-dir \
  --target "$PYTHON_PKGS"

$PYBIN -m pip uninstall -y deepspeed

$PYBIN - <<'PY'
import torch, transformers, pathlib, importlib.util, numpy as np
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available(), torch.version.cuda)
print("Bits&Bytes  :", __import__("bitsandbytes").__version__)
print("Transformers:", transformers.__version__)
print("NumPy       :", np.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY

source /scratch_dgxl/ifc24/.hf_token || true
[ -f /scratch_dgxl/ifc24/.wandb_idacasp_key ] && source /scratch_dgxl/ifc24/.wandb_idacasp_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_ft_50k}

export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

echo "$(date) - ese_inference_with_adapter_11l1 initial setup OK"
RUN_SCRIPT="f_finetune/src/inf.py"

srun "$PYBIN" h_rae/src/data_prep/inference_with_adapter.py \
  --dataset_jsonl h_rae/data/rae_official/RobustAlpacaEval.jsonl \
  --base_model f_finetune/model \
  --lora_path f_finetune/outputs_alternat/alta11l1/cp_checkpoint14000_bestsofar \
  --merge_lora \
  --save_merged_to f_finetune/outputs_alternat/alta11l1/cp_checkpoint14000_bestsofar/merged \
  --decode greedy --repetition_penalty 1.10 \
  --batch_size 256 \
  --max_new_tokens 265 \
  --out_json h_rae/data/rae_official/RobustAlpacaEval_aws_inference_with_adapter_11l1_officialrae_5.jsonl

echo "$(date) - ese_inference_with_adapter_11l1 job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) - ese_inference_with_adapter_11l1 job finished OK"
