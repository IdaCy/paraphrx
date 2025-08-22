#!/bin/bash -l
#SBATCH --job-name=safetensors_download
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%x_%j_download.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%x_%j_download.err

set -euo pipefail
echo "$(date) - safetensors_download job started on $(hostname)"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"

mkdir -p logs

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

for P in python3.11 python3.10 python3.9 python3; do
  if command -v $P &>/dev/null; then PYBIN=$(command -v $P); break; fi
done
[ -z "${PYBIN:-}" ] && { echo "No python3 on PATH"; exit 1; }
echo "Using $($PYBIN -V)"

export PYTHON_PKGS="$TMPDIR/python-packages"
mkdir -p "$PYTHON_PKGS"
export PYTHONPATH="$PYTHON_PKGS:$PYTHONPATH"

# Install pip packages locally (no cache, no global install)
curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$TMPDIR/get-pip.py"
$PYBIN "$TMPDIR/get-pip.py" --target "$PYTHON_PKGS" --no-cache-dir
$PYBIN -m pip install --upgrade pip wheel setuptools --no-cache-dir --target "$PYTHON_PKGS"
$PYBIN -m pip install --prefer-binary --only-binary=:all: wandb --no-cache-dir --target "$PYTHON_PKGS"
[ -f /scratch_root/ifc24/.wandb_idacasp_key ] && source /scratch_root/ifc24/.wandb_idacasp_key

echo "$(date) - safetensors_download initial setup OK"

python3 e_eval/src/sync_big_file.py download \
        alpaca_500_327paraphrases \
        e_eval/data/alpaca_gemma-2-2b-it.json

echo "$(date) - safetensors_download part 2 finished OK"

echo "$(date) - safetensors_download job finished OK"
