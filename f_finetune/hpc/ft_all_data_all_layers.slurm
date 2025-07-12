#!/bin/bash -l
#SBATCH --job-name=1-4_all_layers_ft_all_data
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_1-4_all_layers_ft_all_data.out
#SBATCH --error=/scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_1-4_all_layers_ft_all_data.err

set -euo pipefail
echo "$(date) – 1-4_all_layers_ft_all_data job started on $(hostname)"
echo "$(date) – 1-4_all_layers_ft_all_data job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_dgxl/ifc24/proj/paraphrx
cd "$PROJ"
mkdir -p logs tmp hf_cache pip_cache xdg_cache f_finetune/outputs/all_data_all_layers/outputs_buckets_1-4
export TMPDIR=$PROJ/tmp
export HF_HOME=$PROJ/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export PIP_CACHE_DIR=$PROJ/pip_cache
export XDG_CACHE_HOME=$PROJ/xdg_cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=
export HOME=$TMPDIR
export PATH=$HOME/.local/bin:$PATH

# Pick an available Python-3 binary (prefers 3.11 - as 3.8)
for P in python3.11 python3.10 python3.9 python3; do
  if command -v $P >/dev/null 2>&1; then PYBIN=$(command -v $P); break; fi
done
[ -z "${PYBIN:-}" ] && { echo "No python3 on PATH"; exit 1; }
echo "Using $($PYBIN -V)"

# create fresh venv in node-local scratch
VENV_DIR="$TMPDIR/venv_$SLURM_JOB_ID"
rm -rf "$VENV_DIR"
$PYBIN -m pip install --user virtualenv -q
virtualenv -p "$PYBIN" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel -q

python -m pip install --prefer-binary \
  torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121 \
  transformers==4.45.0 \
  peft==0.11.1 \
  bitsandbytes==0.42.0 \
  datasets==2.19.0 \
  numpy==1.24.4 sentencepiece loguru protobuf==3.20.* wandb testresources -q
python -m pip uninstall -y torchaudio torchvision -q || true

python - <<'PY'
import torch, transformers, pathlib, importlib.util, numpy as np
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available(), torch.version.cuda)
print("Bits&Bytes  :", __import__("bitsandbytes").__version__)
print("Transformers:", transformers.__version__)
print("NumPy       :", np.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY

source /scratch_dgxl/ifc24/.hf_token || true
[ -f /scratch_dgxl/ifc24/.wandb_key ] && source /scratch_dgxl/ifc24/.wandb_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx}

RUN_SCRIPT="f_finetune/src/finetuning.py"

srun python "$RUN_SCRIPT" \
  --data_paths \
  "f_finetune/data/output_splits_alpaca/buckets_1-4_train.json" \
  "f_finetune/data/output_splits_gsm8k/buckets_1-4_train.json" \
  "f_finetune/data/output_splits_mmlu/buckets_1-4_train.json" \
  --output_dir "f_finetune/outputs/all_data_all_layers/outputs_buckets_1-4" \
  --run_name buckets_1-4_all_layers \
  --buckets 1-4 \
  --bf16 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --save_steps 200 \
  $WANDB_FLAG

echo "$(date) – 1-4_all_layers_ft_all_data job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) – 1-4_all_layers_ft_all_data job finished OK"
