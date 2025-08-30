#!/bin/bash -l
#SBATCH --job-name=plot
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%j_%x.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%j_%x.err

set -euo pipefail
echo "$(date) - plot job started on $(hostname)"
echo "$(date) - plot job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

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
  echo "$(date) - cleaning up for $SLURM_JOB_ID (TMPDIR=$TMPDIR)"
  if [[ -n "${TMPDIR:-}" && "$TMPDIR" == "$SCRATCH/tmp/"* ]]; then
    rm -rf "$TMPDIR"
  fi
  if [[ -n "${HF_HOME:-}" && "$HF_HOME" == "$PROJ/hf_cache" ]]; then
    rm -rf "$HF_HOME"
  fi
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

# install everything from wheels only (no source builds)
$PYBIN -m pip install --prefer-binary --only-binary=:all: \
  numpy==1.26.4 \
  pandas \
  scipy \
  seaborn \
  matplotlib \
  jsonlines \
  scikit-learn \
  --no-cache-dir \
  --target "$PYTHON_PKGS"

export TOKENIZERS_PARALLELISM=false

echo "$(date) – plot job started on $(hostname)"

  python3 b_tests/graphics/plot_ac_caus_results.py \
  --outdir b_tests/graphics/ac \
    f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2_preneurips/ac_cause_I_DIAG_mlp_only \
    f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2_preneurips/ac_cause_D_attn_only_14_18_22_last \
    f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2_preneurips/ac_cause_H4_noblock_noL22 \
    f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2_preneurips/ac_L_opt_mid_attn_mlp_noBlock_noL22_last \
    f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2_preneurips/ac_cause_G_OPT_l6_plus_lateATTN_last_K4

echo "$(date) – plot job finished OK"
