import sys
import os
import wandb
"""
python sync_big_file.py upload path/to/thefile.bin the_artifact_name

rm -rf ~/.cache/wandb
rm -rf ~/.cache
export WANDB_DIR=/scratch_dgxl/ifc24/proj/paraphrx/wandb_tmp
export WANDB_ARTIFACT_DIR=/scratch_dgxl/ifc24/proj/paraphrx/wandb_tmp
    python3 e_eval/src/sync_big_file.py upload \
        f_finetune/outputs/l9x_a1_notarg_50k_ft/final/model-00001-of-00002.safetensors \
        safetensors00002
    python3 e_eval/src/sync_big_file.py upload \
        f_finetune/outputs/l9x_a1_notarg_50k_ft/final/model-00002-of-00002.safetensors \
        safetensors00001
    python3 e_eval/src/sync_big_file.py upload \
        f_finetune/outputs/l9x_a1_notarg_50k_ft/final/tokenizer.model \
        tokenizer_model
    python3 e_eval/src/sync_big_file.py upload \
        f_finetune/outputs/l9x_a1_notarg_50k_ft/final/training_args.bin \
        tokenizer_training_args_bin
    python3 e_eval/src/sync_big_file.py upload \
        a_data/alpaca/50k_phrxed.json \
        prompts_50k

    python3 e_eval/src/sync_big_file.py upload_folder \
        f_finetune/outputs/l9x_a1_notarg_50k_ft \
        9x_a1_notarg_best

    python3 e_eval/src/sync_big_file.py upload_folder \
        f_finetune/outputs_lap/tokenized_data_9x_wstyle_wcount \
        tokenized_data_9x_wstyle_wcount

    python3 e_eval/src/sync_big_file.py upload_folder \
        f_finetune/data/tokenized_data_9x_pr_aw \
        tokenized_data_9x_pr_aw

    python3 e_eval/src/sync_big_file.py upload \
        c_assess_inf/output50k/answers.json \
        answers_50k

    srun "$PYBIN" e_eval/src/sync_big_file.py upload_folder \
        f_finetune/data/tokenized_real9x_preproc \
        tokenized_real9x_preproc

    python3 e_eval/src/sync_big_file.py upload_folder \
        f_finetune/outputs_great_nolap/lpr9x_a1_notarg_50k_ft \
        lpr9x_a1_notarg_50k_ft

    python3 e_eval/src/sync_big_file.py upload_folder \
        f_finetune/outputs_aw/l9x_a1_notarg_50k_ft \
        cp_l9x_a1_notarg_50k_ft

    python3 e_eval/src/sync_big_file.py upload_folder \
        f_finetune/outputs_lap_pr/lpr9x_full_stable5_lap \
        lpr9x_full_stable5_lap

    python3 e_eval/src/sync_big_file.py upload_folder \
        f_finetune/data/tokenized_data_9x_pr_aw \
        tokenized_data_no9xbutall_pr_aw

    python3 e_eval/src/sync_big_file.py upload_folder \
        f_finetune/data/tokenized_real9x_output_preproc \
        tokenized_real9x_output_preproc

    python3 e_eval/src/sync_big_file.py upload_folder \
        f_finetune/data/tokenized_real1x_output_preproc \
        tokenized_real1x_output_preproc

    python3 e_eval/src/sync_big_file.py upload_folder \
        f_finetune/data \
        f_finetune_data

# UPLOADED AND NOT DOWNLOADED YET
    python3 e_eval/src/sync_big_file.py upload_folder \
        f_finetune/outputs_lap_pr/opt_all_topa \
        opt_all_topa


python sync_big_file.py download the_artifact_name /path/to/save
python3 e_eval/src/sync_big_file.py download \
    safetensors00002 \
    f_finetune/outputs/cp_9x_a1_notarg_best/final/model-00001-of-00002.safetensors
python3 e_eval/src/sync_big_file.py download \
    safetensors00001 \
    f_finetune/outputs/cp_9x_a1_notarg_best/final/model-00002-of-00002.safetensors
python3 e_eval/src/sync_big_file.py download \
    tokenizer_model \
    f_finetune/outputs/cp_9x_a1_notarg_best/final/tokenizer.model
python3 e_eval/src/sync_big_file.py download \
    tokenizer_training_args_bin \
    f_finetune/outputs/cp_9x_a1_notarg_best/final/training_args.bin
python3 e_eval/src/sync_big_file.py download \
    prompts_50k \
    a_data/alpaca/50k_phrxed.json

python3 e_eval/src/sync_big_file.py download_folder \
    tokenized_data_9x_wstyle \
    f_finetune/outputs_lap/tokenized_data_9x_wstyle

python3 e_eval/src/sync_big_file.py download_folder \
    9x_a1_notarg_best \
    f_finetune/outputs/l9x_a1_notarg_50k_ft

python3 e_eval/src/sync_big_file.py download \
    answers_50k \
    c_assess_inf/output50k/answers.json

python3 e_eval/src/sync_big_file.py download_folder \
        tokenized_real9x_preproc \
        f_finetune/data/tokenized_real9x_preproc

python3 e_eval/src/sync_big_file.py download_folder \
        lpr9x_a1_notarg_50k_ft \
        f_finetune/outputs_great_nolap/lpr9x_a1_notarg_50k_ft

python3 e_eval/src/sync_big_file.py download_folder \
        cp_l9x_a1_notarg_50k_ft \
        f_finetune/outputs_aw/l9x_a1_notarg_50k_ft

python3 e_eval/src/sync_big_file.py download_folder \
        lpr9x_full_stable5_lap \
        f_finetune/outputs_lap_pr/lpr9x_full_stable5_lap

python3 e_eval/src/sync_big_file.py download_folder \
        tokenized_data_no9xbutall_pr_aw \
        f_finetune/data/tokenized_data_9x_pr_aw

python3 e_eval/src/sync_big_file.py download_folder \
        tokenized_real9x_output_preproc \
        f_finetune/data/tokenized_real9x_output_preproc

python3 e_eval/src/sync_big_file.py download_folder \
        tokenized_real1x_output_preproc \
        f_finetune/data/tokenized_real1x_output_preproc

        
  Upload file:        python sync_big_file.py upload <file_path> <artifact_name>
  Download file:      python sync_big_file.py download <artifact_name> <output_dir>
  Upload folder:      python sync_big_file.py upload_folder <folder_path> <artifact_name>
  Download folder:    python sync_big_file.py download_folder <artifact_name> <output_dir>
"""

def upload_file(file_path, artifact_name, project="my-project"):
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        sys.exit(1)

    wandb.init(project=project)
    artifact = wandb.Artifact(artifact_name, type="dataset")
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"✓ Uploaded {file_path} as artifact '{artifact_name}'")


def download_file(artifact_name, output_dir, project="my-project"):
    os.makedirs(output_dir, exist_ok=True)
    wandb.init(project=project)
    artifact = wandb.use_artifact(f"{artifact_name}:latest")
    artifact_dir = artifact.download(output_dir)
    wandb.finish()
    print(f"✓ Downloaded artifact '{artifact_name}' to {artifact_dir}")


def upload_folder(folder_path, artifact_name, project="my-project"):
    if not os.path.exists(folder_path):
        print(f"✗ Folder not found: {folder_path}")
        sys.exit(1)
    if not os.path.isdir(folder_path):
        print(f"✗ Path is not a folder: {folder_path}")
        sys.exit(1)

    wandb.init(project=project)
    artifact = wandb.Artifact(artifact_name, type="dataset")
    artifact.add_dir(folder_path)
    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"✓ Uploaded folder {folder_path} as artifact '{artifact_name}'")


def download_folder(artifact_name, output_dir, project="my-project"):
    os.makedirs(output_dir, exist_ok=True)
    wandb.init(project=project)
    artifact = wandb.use_artifact(f"{artifact_name}:latest")
    artifact_dir = artifact.download(output_dir)
    wandb.finish()
    print(f"✓ Downloaded folder artifact '{artifact_name}' to {artifact_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Upload file:     python sync_big_file.py upload <file_path> <artifact_name>")
        print("  Download file:   python sync_big_file.py download <artifact_name> <output_dir>")
        print("  Upload folder:   python sync_big_file.py upload_folder <folder_path> <artifact_name>")
        print("  Download folder: python sync_big_file.py download_folder <artifact_name> <output_dir>")
        sys.exit(1)

    action = sys.argv[1].lower()

    if action == "upload":
        if len(sys.argv) != 4:
            print("Usage: python sync_big_file.py upload <file_path> <artifact_name>")
            sys.exit(1)
        upload_file(sys.argv[2], sys.argv[3])

    elif action == "download":
        if len(sys.argv) != 4:
            print("Usage: python sync_big_file.py download <artifact_name> <output_dir>")
            sys.exit(1)
        download_file(sys.argv[2], sys.argv[3])

    elif action == "upload_folder":
        if len(sys.argv) != 4:
            print("Usage: python sync_big_file.py upload_folder <folder_path> <artifact_name>")
            sys.exit(1)
        upload_folder(sys.argv[2], sys.argv[3])

    elif action == "download_folder":
        if len(sys.argv) != 4:
            print("Usage: python sync_big_file.py download_folder <artifact_name> <output_dir>")
            sys.exit(1)
        download_folder(sys.argv[2], sys.argv[3])

    else:
        print("✗ Invalid action. Use 'upload', 'download', 'upload_folder', or 'download_folder'.")
        sys.exit(1)
