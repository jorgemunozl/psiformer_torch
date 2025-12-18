from huggingface_hub import HfApi


REPO_ID = "jorgemunozl/psiformer_torch"


api = HfApi()
api.upload_folder(
    repo_id=REPO_ID,
    folder_path="checkpoints",
    repo_type="model",
    path_in_repo="checkpoints",
)
