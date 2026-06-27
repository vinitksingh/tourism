
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError
import os

# Set Hugging Face token from environment variables for secure authentication.
# It's crucial to load the token from an environment variable (e.g., HF_TOKEN) 
# rather than hardcoding it directly in the script for security best practices.
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define Hugging Face repository details for the dataset.
repo_id = "vinitksingh/tourismdataset"
repo_type = "dataset"

# Check if the dataset repository exists on Hugging Face Hub. 
# If it doesn't exist, create a new public dataset repository.
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Hugging Face Dataset Repo '{repo_id}' already exists. Using it.")
except HfHubHTTPError:
    print(f"Hugging Face Dataset Repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Hugging Face Dataset Repo '{repo_id}' created.")

# Upload the local data folder (containing `tourism.csv`) to the Hugging Face repository.
# The `folder_path` is relative to the current working directory, ensuring compatibility 
# within GitHub Actions workflows.
api.upload_folder(
    folder_path="tourism_project/data", # Local folder containing the dataset files
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Data from 'tourism_project/data' uploaded to Hugging Face Dataset Hub '{repo_id}'.")
