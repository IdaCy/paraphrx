#!/bin/bash -l
#SBATCH --job-name=alta42_rae_rae
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%j_%x.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%j_%x.err

set -euo pipefail
echo "$(date) - alta42_rae_rae job started on $(hostname)"
echo "$(date) - alta42_rae_rae job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"

mkdir -p logs f_finetune/outputs_alternat/alta42_rae_rae

export SCRATCH=/scratch_root/ifc24
export TMPDIR="$SCRATCH/tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
echo "Using scratch-backed TMPDIR=$TMPDIR"

export HF_HOME="$PROJ/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME"
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
  echo "$(date) - cleaning up ENTIRE temp directory for $SLURM_JOB_ID: $TMPDIR"
  rm -rf "$TMPDIR"
}
trap cleanup EXIT ERR INT TERM

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR"

for P in python3.11 python3.10 python3.9 python3; do
  if command -v $P &>/dev/null; then PYBIN=$(command -v $P); break; fi
done
[ -z "${PYBIN:-}" ] && { echo "No python3 on PATH"; exit 1; }
echo "Using $($PYBIN -V)"
export PYTHON_PKGS="$TMPDIR/python-packages"
mkdir -p "$PYTHON_PKGS"
export PYTHONPATH="$PYTHON_PKGS:$PYTHONPATH"
curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$TMPDIR/get-pip.py"
$PYBIN "$TMPDIR/get-pip.py" --target "$PYTHON_PKGS" --no-cache-dir
$PYBIN -m pip install --upgrade "importlib_metadata>=5.2.0" "testresources==2.0.1" --no-cache-dir --target "$PYTHON_PKGS"
$PYBIN -m pip install --upgrade pip wheel --no-cache-dir --target "$PYTHON_PKGS"
$PYBIN -m pip install --upgrade setuptools --no-cache-dir --target "$PYTHON_PKGS"
# clean, deterministic versions
$PYBIN -m pip install --upgrade --force-reinstall --no-cache-dir --target "$PYTHON_PKGS" \
  "numpy==1.26.4" "accelerate<0.33"
$PYBIN -m pip uninstall -y bitsandbytes || true

# now install the rest; keep accelerate pinned so resolver won't bump it
$PYBIN -m pip install --prefer-binary --only-binary=:all: \
  torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121 \
  transformers==4.45.0 \
  safetensors==0.4.3 \
  peft==0.11.1 \
  "datasets==2.19.0" \
  "accelerate<0.33" \
  --no-cache-dir --target "$PYTHON_PKGS"

$PYBIN - <<'PY'
import torch, transformers, pathlib, importlib.util, numpy as np, safetensors
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available(), torch.version.cuda)
print("Transformers:", transformers.__version__)
print("Safetensors :", safetensors.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY
source /scratch_root/ifc24/.hf_token || true
[ -f /scratch_root/ifc24/.wandb_idacasp_key ] && source /scratch_root/ifc24/.wandb_idacasp_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrax_ft_nolap}
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

echo "$(date) - alta42_rae_rae initial setup OK"

RUN_SCRIPT="f_finetune/src/inf_alternative_setup.py"
OUTPUT_DIR="f_finetune/outputs_alternat/alta42"
MODEL_PATH="f_finetune/model"

WANDB_PROJECT="paraphrx_ft_lap"
RUN_NAME="inf_alta4_alpaca"

# sbatch --dependency=afterany:1799:1794 f_finetune/hpc/borg/borg_ft_lap_9xreal_stable5.slurm

srun "$PYBIN" "$RUN_SCRIPT" \
  --data_path h_rae/data/rae_official/RobustAlpacaEval_converted.json \
  --first_min_style_count 1 \
  --require_content_score 1 \
  --base_model_path "f_finetune/model" \
  --lora_path "$OUTPUT_DIR/final" \
  --merge_lora \
  --base_model_path "f_finetune/model" \
  --output_json "h_rae/data/rae_official/alta42_rae.json" \
  --val_pct 1.0 --test_pct 0.0 --split val \
  --batch 256

echo "$(date) - alta42_rae_rae job itself finished OK"

echo "$(date) - alta42_rae_rae job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) - alta42_rae_rae job finished & stuff cleaned OK"
