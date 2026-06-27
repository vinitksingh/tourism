
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/vinitksingh/tourismdataset/tourism.csv"

# Load the dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier 'CustomerID'
df.drop(columns=['CustomerID'], inplace=True)

# Encode the categorical 'TypeofContact' column to numerical representation
label_encoder = LabelEncoder()
df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])

# Define the target column
target_col = 'ProdTaken'

# Split into features (X) and target (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save the split datasets to CSV files locally
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

# List of files to upload
files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Upload the split datasets to the Hugging Face dataset repository
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # Upload just the filename
        repo_id="vinitksingh/tourismdataset",
        repo_type="dataset",
    )
print("Split datasets uploaded to Hugging Face.")
