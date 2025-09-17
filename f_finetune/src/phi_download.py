#!/usr/bin/env python3
from huggingface_hub import snapshot_download, login

login(token="xxx")

# Download everything under microsoft/Phi-3.5-mini-instruct into f_finetune/model/phi  35
snapshot_download(
    repo_id="microsoft/Phi-3.5-mini-instruct",
    local_dir="f_finetune/model/phi35",
    force_download=False,   # True = always re-download
    resume_download=True,   # resume if partial download
)

print("Model downloaded to f_finetune/model/phi35")
