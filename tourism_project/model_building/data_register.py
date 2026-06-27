
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError
import os

# Set Hugging Face token from environment variables
# Remove hardcoded token for security; rely on environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define Hugging Face repository details for the dataset
repo_id = "vinitksingh/tourismdataset"
repo_type = "dataset"

# Check if the dataset repository exists on Hugging Face, create it if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Hugging Face Dataset Repo '{repo_id}' already exists. Using it.")
except HfHubHTTPError:
    print(f"Hugging Face Dataset Repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Hugging Face Dataset Repo '{repo_id}' created.")

# Upload the data folder to the Hugging Face repository
# Use the absolute path for the data folder to ensure it's found correctly.
api.upload_folder(
    folder_path="/content/tourism_project/data", # Changed to absolute path
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Data from '/content/tourism_project/data' uploaded to Hugging Face Dataset Hub '{repo_id}'.")
