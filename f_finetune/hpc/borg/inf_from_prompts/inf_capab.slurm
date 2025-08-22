#!/bin/bash -l
#SBATCH --job-name=i_capab_great_nolap_lpr9x_a1
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%x_%j_ft_6layer.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%x_%j_ft_6layer.err

set -euo pipefail
echo "$(date) - i_capab_great_nolap_lpr9x_a1 job started on $(hostname)"
echo "$(date) - i_capab_great_nolap_lpr9x_a1 job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"

mkdir -p logs f_finetune/outputs_great_nolap/i_capab_great_nolap_lpr9x_a1

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
import torch, transformers, pathlib, importlib.util, numpy as np, safetensors
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available(), torch.version.cuda)
print("Bits&Bytes  :", __import__("bitsandbytes").__version__)
print("Transformers:", transformers.__version__)
print("NumPy       :", np.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY
source /scratch_root/ifc24/.hf_token || true
[ -f /scratch_root/ifc24/.wandb_idacasp_key ] && source /scratch_root/ifc24/.wandb_idacasp_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_ft_nolap_inf}
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

echo "$(date) - i_capab_great_nolap_lpr9x_a1 initial setup OK"
RUN_SCRIPT="f_finetune/src/inf_capability.py"

srun $PYBIN "$RUN_SCRIPT" \
    --data_path a_data/alpaca/50k_phrxed.json \
    --base_model_path "f_finetune/outputs_great_nolap/lpr9x_a1_notarg_50k_ft/final" \
    --output_json "f_finetune/outputs_great_nolap/lpr9x_a1_notarg_50k_ft/answers20.json" \
    --log_name "i_capab_great_nolap_lpr9x_a1" \
    --batch 32 \
    --quant 4bit \
    --max_samples 20 \
    --mmlu_moral \
    --mmlu \
    --mmlu_subjects college_mathematics global_facts \
    --gsm8k \
    --daily_dialog \
  $WANDB_FLAG

echo "$(date) - i_capab_great_nolap_lpr9x_a1 job itself finished OK"

echo "$(date) - i_capab_great_nolap_lpr9x_a1 job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) - i_capab_great_nolap_lpr9x_a1 job finished & stuff cleaned OK"
