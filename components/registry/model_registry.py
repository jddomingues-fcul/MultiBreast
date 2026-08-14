import os

import wandb


def download_model_checkpoint(
    wandb_path: str, out_folder: str = "model_checkpoints"
) -> str:
    run_id = wandb_path.split("/")[-1]
    api = wandb.Api()
    run = api.run(wandb_path)

    for elem in run.files():
        if elem.name.lower() == "ckpt.pt":
            elem.download(
                root=os.path.join(out_folder, run_id), replace=False, exist_ok=True
            )
            break

    return os.path.join(out_folder, run_id, "ckpt.pt")
