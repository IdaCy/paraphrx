#!/bin/bash -l
#SBATCH --job-name=ft_buckets_1-5
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_ft_buckets_1-5.out
#SBATCH --error=/scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_ft_buckets_1-5.err

set -euo pipefail
echo "$(date) – ft_buckets_1-5 job started on $(hostname)"
echo "$(date) – ft_buckets_1-5 job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_dgxl/ifc24/proj/paraphrx
cd "$PROJ"
mkdir -p logs tmp hf_cache pip_cache xdg_cache f_finetune/outputs/alpaca/all_layers/outputs_buckets_1-5
export TMPDIR=$PROJ/tmp
export HF_HOME=$PROJ/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export PIP_CACHE_DIR=$PROJ/pip_cache
export XDG_CACHE_HOME=$PROJ/xdg_cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=

# Pick an available Python-3 binary (prefers 3.12 -as 3.8)
for P in python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v $P >/dev/null 2>&1; then PYBIN=$(command -v $P); break; fi
done
[ -z "${PYBIN:-}" ] && { echo "No python3 on PATH"; exit 1; }
echo "Using $($PYBIN -V)"

# create fresh venv
rm -rf venv
$PYBIN -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel -q

python -m pip install --prefer-binary \
  torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121 \
  transformers==4.45.0 \
  peft==0.11.1 \
  bitsandbytes==0.44.1 \
  datasets==2.19.0 \
  numpy==1.26.4 sentencepiece loguru protobuf==3.20.* wandb -q
python -m pip uninstall -y torchaudio torchvision -q || true

python - <<'PY'
import torch, transformers, platform, pathlib, importlib.util, numpy as np
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
INPUT_JSON="f_finetune/data/output_splits/buckets_1-5_train.json"

srun python "$RUN_SCRIPT" \
  --data_paths "$INPUT_JSON" \
  --output_dir f_finetune/outputs/alpaca/all_layers/outputs_buckets_1-5 \
  --run_name buckets_1-5 \
  --buckets 1-5 \
  --bf16 \
  $WANDB_FLAG

echo "$(date) – ft_buckets_1-5 job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) – ft_buckets_1-5 job finished OK"
