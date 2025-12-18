from huggingface_hub import HfApi
import os


REPO_ID = "jorgemunozl/psiformer_torch"

#REPO_ID = "your-username/my-model-weights"
REPO_TYPE = "model"
PRIVATE = False

api = HfApi(token=os.getenv("HF_TOKEN"))  # optional; if None, uses huggingface-cli cached login
url = api.create_repo(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    private=PRIVATE,
    exist_ok=True,  # don't fail if it already exists
)

api.upload_folder(
    repo_id=REPO_ID,
    folder_path="checkpoints",
    repo_type="model",
    path_in_repo="checkpoints",
)

print("Repo ready:", url)
