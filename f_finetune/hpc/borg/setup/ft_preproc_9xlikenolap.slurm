#!/bin/bash -l
#SBATCH --job-name=1xdata_likenolap
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%x_%j_1xdata_likenolap.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%x_%j_1xdata_likenolap.err

set -euo pipefail
echo "$(date) – 1xdata_likenolap job started on $(hostname)"
echo "$(date) – 1xdata_likenolap job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"

mkdir -p logs

export SCRATCH=/scratch_root/ifc24
export TMPDIR="$SCRATCH/tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
echo "Using scratch-backed TMPDIR=$TMPDIR"

# Shared HF cache
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
  echo "$(date) – cleaning up only my directories for $SLURM_JOB_ID"
  rm -rf \
    "$PIP_CACHE_DIR" \
    "$XDG_CACHE_HOME" \
    "$WANDB_DIR" \
    "$TRITON_CACHE_DIR"
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

$PYBIN -m pip install --upgrade "importlib_metadata>=5.2.0" \
                                "testresources==2.0.1" \
    --no-cache-dir --target "$PYTHON_PKGS"

$PYBIN -m pip install --upgrade pip wheel \
  --no-cache-dir --target "$PYTHON_PKGS"

$PYBIN -m pip install --upgrade setuptools \
  --no-cache-dir --target "$PYTHON_PKGS"

$PYBIN -m pip install --prefer-binary --only-binary=:all: \
  torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cpu \
  transformers==4.41.2 \
  datasets==2.19.0 \
  numpy==1.26.4 \
  sentencepiece \
  loguru \
  protobuf==3.20.* \
  --no-cache-dir \
  --target "$PYTHON_PKGS"

$PYBIN - <<'PY'
import torch, transformers, pathlib, importlib.util, numpy as np
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available()) # Will print False, which is correct
print("Transformers:", transformers.__version__)
print("NumPy       :", np.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY

source /scratch_root/ifc24/.hf_token || true
[ -f /scratch_root/ifc24/.wandb_idacasp_key ] && source /scratch_root/ifc24/.wandb_idacasp_key

export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

# Run
echo "$(date) – 1xdata_likenolap initial setup OK"
RUN_SCRIPT="f_finetune/src/ft_preprocess_data.py"
srun $PYBIN "$RUN_SCRIPT" \
  --prompts_path a_data/alpaca/50k_phrxed.json \
  --instruct_types \
    instruct_formal_memo \
    instruct_joke \
    instruct_sardonic \
    instruct_future_tense \
    instruct_coord_to_subord \
    instruct_dramatic \
    instruct_polite_request \
    instruct_one_typo_punctuation \
  --model_path f_finetune/model \
  --batch_size 1024 \
  --log_file f_finetune/data/1xdata_likenolap.log \
  --max_answer_tokens 512 \
  --output_path f_finetune/data/tokenized_data_1xlikenolap

echo "$(date) – 1xdata_likenolap job itself finished OK"

rm -rf "$TMPDIR" \
       "$PIP_CACHE_DIR" \
       "$XDG_CACHE_HOME"

echo "$(date) – 1xdata_likenolap job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) – 1xdata_likenolap job finished & stuff cleaned OK"
