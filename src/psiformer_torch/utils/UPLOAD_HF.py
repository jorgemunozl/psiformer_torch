from __future__ import annotations
import os
from huggingface_hub import HfApi, snapshot_download

from psiformer_torch.config import Train_Config


def upload_checkpoints(
    repo_id: str = Train_Config.repo_id,
    folder_path: str = "checkpoints",
    repo_type: str = "model",
    private: bool = False,
) -> str:
    api = HfApi(token=os.getenv("HF_TOKEN"))
    url = api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        repo_type=repo_type,
        path_in_repo=folder_path,
    )
    return str(url)


def main(mode=''):
    if mode == "upload":
        upload_url = upload_checkpoints()
        print(f"Checkpoints uploaded to: {upload_url}")
    else:
        snapshot_download(
            repo_id=Train_Config.repo_id,
            local_dir=Train_Config.checkpoint_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.pth"],   # only .pth
        )


if __name__ == "__main__":
    main()
