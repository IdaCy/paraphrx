#!/bin/bash -l
#SBATCH --job-name=inf_lap
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%j_%x.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%j_%x.err

set -euo pipefail
echo "$(date) - inf_lap job started on $(hostname)"
echo "$(date) - inf_lap job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"

mkdir -p logs f_finetune/outputs_ft_2/bigpipe

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
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_inf_2}
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

echo "$(date) - inf_lap initial setup OK"
RUN_SCRIPT="f_finetune/src/ft_2.py"
BASE_MODEL="f_finetune/model"

echo "$(date) - stage 1 starting"

PROMPTS_SECOND="a_data/alpaca/50k_phrxed_nofirst500.json"
PROMPTS_FIRST="a_data/alpaca/paraphrases_500.json"
STYLES_SECOND="a_data/alpaca/selections/50k_phrxed_nofirst500.json"
STYLES_FIRST="a_data/alpaca/selections/paraphrases_500.json"
ANSWERS_SECOND="c_assess_inf/output50k/answers.json"

echo "$(date) - Stage 1 (SFT) starting"

srun $PYBIN "$RUN_SCRIPT" \
  --data_paths "$PROMPTS_SECOND" "$PROMPTS_FIRST" \
  --styles_paths "$STYLES_SECOND" "$STYLES_FIRST" \
  --model_path "$BASE_MODEL" \
  --output_dir f_finetune/outputs_ft_2/bigpipe/stage1_sft \
  --run_name s1_sft_consistency \
  --bf16 \
  --consistency_lambda 0.05 \
  --rebalance_by_style \
  --batch_size 16 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --num_epochs 3 \
  $WANDB_FLAG

# Get best checkpoint dir (adjust if you know the ID)
CKPT_DIR=$(ls -d f_finetune/outputs_ft_2/bigpipe/stage1_sft/checkpoint-* | sort -V | tail -n1)

echo "$(date) - Stage 1.5 (GroupDRO) starting"

srun $PYBIN "$RUN_SCRIPT" \
  --data_paths "$PROMPTS_SECOND" "$PROMPTS_FIRST" \
  --styles_paths "$STYLES_SECOND" "$STYLES_FIRST" \
  --model_path "$BASE_MODEL" \
  --resume_checkpoint "$CKPT_DIR" \
  --output_dir f_finetune/outputs_ft_2/bigpipe/stage1p5_groupdro \
  --run_name s1p5_groupdro \
  --bf16 \
  --use_groupdro --groupdro_eta 0.05 \
  --batch_size 16 --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 --num_epochs 1 \
  $WANDB_FLAG

# Optional Stage 2 (ORPO) — needs answers JSON and only ONE prompts JSON (SECOND)
echo "$(date) - Stage 2 (ORPO) starting"

CKPT_DIR2=$(ls -d f_finetune/outputs_ft_2/bigpipe/stage1p5_groupdro/checkpoint-* | sort -V | tail -n1)

srun $PYBIN "$RUN_SCRIPT" \
  --stage orpo \
  --data_paths "$PROMPTS_SECOND" \
  --styles_paths "$STYLES_SECOND" \
  --answers_path "$ANSWERS_SECOND" \
  --model_path "$BASE_MODEL" \
  --resume_checkpoint "$CKPT_DIR2" \
  --output_dir f_finetune/outputs_ft_2/bigpipe/stage2_orpo \
  --run_name s2_orpo \
  --bf16 \
  --orpo_beta 0.1 \
  --batch_size 8 --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 --num_epochs 1 \
  $WANDB_FLAG

echo "$(date) - stage 3 completed"

echo "$(date) - inf_lap job itself finished OK"

echo "$(date) - inf_lap job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) - inf_lap job finished & stuff cleaned OK"
