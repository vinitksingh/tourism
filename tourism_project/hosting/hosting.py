
from huggingface_hub import HfApi
import os

# Set Hugging Face token from environment variables for secure authentication
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define the Hugging Face Space repository ID for deployment
space_repo_id = "vinitksingh/tourism-prediction-app-space"

# Upload the deployment folder to the Hugging Face Space.
# This includes the Streamlit app, Dockerfile, and requirements.
api.upload_folder(
    folder_path="tourism_project/deployment", # Local folder containing Streamlit app and dependencies
    repo_id=space_repo_id,                   # Target Hugging Face Space repository
    repo_type="space",                       # Type of repository is 'space' for Streamlit apps
    path_in_repo="",                         # Upload to the root of the space
)
print(f"Deployment files uploaded to Hugging Face Space '{space_repo_id}'.")
