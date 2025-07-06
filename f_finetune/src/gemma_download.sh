#!/usr/bin/env bash
set -euo pipefail

# Ensure we can call pip via python3 -m pip
if ! python3 -m pip --version &>/dev/null; then
  echo "-> Bootstrapping pip in user space..."
  curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python3 get-pip.py --user
  rm get-pip.py
fi

# Ensure ~/.local/bin is on PATH so 'pip' and installed scripts work
export PATH="$HOME/.local/bin:$PATH"

# Install / upgrade huggingface_hub into user site
echo "-> Installing huggingface_hub..."
python3 -m pip install --user --upgrade huggingface_hub

# Now download the model -using HF token
echo "-> Downloading google/gemma-2-2b-t into f_finetune/model..."
python3 - <<'PYCODE'
import os
from huggingface_hub import login, snapshot_download

# Log in with token (expects HF_TOKEN in environment)
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("Please set HF_TOKEN environment variable before running this script.")
login(token=hf_token)

# Ensure output dir exists
outdir = "f_finetune/model"
os.makedirs(outdir, exist_ok=True)

snapshot_download(
    repo_id="google/gemma-2-2b-it",
    local_dir=outdir,
    resume_download=True,
)

print("Model ready in f_finetune/model")
PYCODE
