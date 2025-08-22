#!/bin/bash -l
#SBATCH --job-name=gemma_download
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%x_%j_gemma_download.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%x_%j_gemma_download.err

set -euo pipefail
echo "$(date) – gemma_download job started on $(hostname)"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"

export TMPDIR=$PROJ/tmp
export HOME=$TMPDIR
export PATH=$HOME/.local/bin:$PATH

# Pick an available Python‐3 binary
for P in python3.11 python3.10 python3.9 python3; do
  if command -v $P &>/dev/null; then PYBIN=$(command -v $P); break; fi
done
[ -z "${PYBIN:-}" ] && { echo "No python3 on PATH"; exit 1; }
echo "Using $($PYBIN -V)"

# install standalone Miniconda into TMPDIR if needed
CONDADIR="$TMPDIR/conda"
ENVDIR="$TMPDIR/env"
if [ ! -f "$CONDADIR/bin/conda" ]; then
  wget -qO "$TMPDIR/miniconda.sh" \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$TMPDIR/miniconda.sh" -b -p "$CONDADIR"
  rm "$TMPDIR/miniconda.sh"
fi
export PATH="$CONDADIR/bin:$PATH"

# accept conda TOS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# create and activate a fresh env
conda create -y -q -p "$ENVDIR" python=3.10 pip
source "$CONDADIR/bin/activate" "$ENVDIR"

# upgrade pip & install huggingface_hub
pip install --upgrade pip setuptools wheel -q
pip install huggingface_hub -q

# run the download script
python f_finetune/src/gemma_download.py

echo "$(date) – gemma_download job finished OK"
