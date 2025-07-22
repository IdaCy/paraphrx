import wandb
import os
from glob import glob


def main():
    # Start a “prep-data” run
    run = wandb.init(
        project="paraphrx_lora",
        job_type="prepare_data",
        name="upload-all-jsons"
    )

    # For each JSON in e_eval/data, log it as an Artifact
    data_dir = "e_eval/data"
    for path in glob(os.path.join(data_dir, "*.json")):
        fname = os.path.basename(path)
        # e.g. "alpaca_gemma-2-2b-it.json"
        art_name = fname.replace(".json", "")  # artifact name
        art = wandb.Artifact(
            name=art_name,
            type="dataset",
            description=f"{fname} from paraphrx project"
        )
        art.add_file(path)
        run.log_artifact(art)
        print(f"✔ uploaded {fname} as {art_name}:latest")

    run.finish()
    print("All JSONs uploaded.")


if __name__ == "__main__":
    main()
