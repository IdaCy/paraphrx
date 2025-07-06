#!/usr/bin/env python3
from huggingface_hub import snapshot_download

from huggingface_hub import login; login(token="xxx")

# Download everything under google/gemma-2-2b-t into f_finetune/model
snapshot_download(
    repo_id="google/gemma-2-2b-t",
    local_dir="f_finetune/model",
    force_download=False,   # True to re-download even if cached
    resume_download=True,   # continue partial downloads
)

print("Model downloaded to f_finetune/model")
