#!/bin/bash -l
#SBATCH --job-name=com_9x_a1_notarg
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%x_%j_com_9x_a1_notarg.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%x_%j_com_9x_a1_notarg.err

set -euo pipefail
echo "$(date) – com_9x_a1_notarg job started on $(hostname)"
echo "$(date) – com_9x_a1_notarg job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"

mkdir -p logs f_finetune/outputs/cp_9x_a1_notarg_best/mass_20test

export SCRATCH=/scratch_root/ifc24
export TMPDIR="$SCRATCH/tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
echo "Using scratch-backed TMPDIR=$TMPDIR"

export MPLCONFIGDIR="$TMPDIR/matplotlib_config"
mkdir -p "$MPLCONFIGDIR"

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
  echo "$(date) – cleaning up ENTIRE temp directory for $SLURM_JOB_ID: $TMPDIR"
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
$PYBIN -m pip install --prefer-binary --only-binary=:all: \
  torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121 \
  transformers==4.45.0 \
  safetensors==0.4.3 \
  peft==0.11.1 \
  bitsandbytes==0.43.2 \
  mpi4py \
  datasets==2.19.0 \
  numpy==1.26.4 \
  sentencepiece \
  loguru \
  protobuf==3.20.* \
  matplotlib \
  seaborn \
  scipy \
  --no-cache-dir \
  --target "$PYTHON_PKGS"
$PYBIN - <<'PY'
import torch, transformers, pathlib, importlib.util, numpy as np, safetensors
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available(), torch.version.cuda)
print("Bits&Bytes  :", __import__("bitsandbytes").__version__)
print("Transformers:", transformers.__version__)
print("Safetensors :", safetensors.__version__)
print("NumPy       :", np.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY
source /scratch_root/ifc24/.hf_token || true
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

echo "$(date) – com_9x_a1_notarg initial setup OK"
RUN_SCRIPT="f_finetune/src/compare_activat_sim.py"
TOKENIZED_DATA="f_finetune/outputs_lap/cached_tokenized_data_9x"

############### IS LIMITED ################
############### IS LIMITED ################
############### IS LIMITED ################
############### IS LIMITED ################
############### IS LIMITED ################
############### IS LIMITED ################
############### IS LIMITED ################
srun $PYBIN "$RUN_SCRIPT" \
  --run_mode aggregate \
  --data_split test \
  --paraphrase_keys "instruct_joke" "instruct_sardonic" \
  --base_model_path "f_finetune/model" \
  --ft_model_path "f_finetune/outputs/cp_9x_a1_notarg_best/final" \
  --prompts_json_path c_assess_inf/output50k/answers.json \
  --output_dir "f_finetune/outputs/cp_9x_a1_notarg_best/mass_20test" \
  --limit 20

echo "$(date) – com_9x_a1_notarg job itself finished OK"

echo "$(date) – com_9x_a1_notarg job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) – com_9x_a1_notarg job finished & stuff cleaned OK"
