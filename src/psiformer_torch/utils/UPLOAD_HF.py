from __future__ import annotations

import os
from huggingface_hub import HfApi



def upload_checkpoints(
    repo_id: str = REPO_ID,
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


if __name__ == "__main__":
    url = upload_checkpoints()
    print("Repo ready:", url)
