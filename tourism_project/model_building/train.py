
# This script trains an XGBoost model to predict customer purchases,
# logs experiments with MLflow, and uploads the trained model to Hugging Face Hub.

# --- Import Libraries ---
import pandas as pd
import sklearn
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
import joblib
import mlflow
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# --- Hugging Face Configuration ---
# Initialize Hugging Face API with token from environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define Hugging Face dataset repository ID
DATASET_REPO_ID = "vinitksingh/tourismdataset"

# --- MLflow Configuration ---
# Set the MLflow tracking URI from environment variable, default to local if not set
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("MLOps_experiment")

# --- Load Preprocessed Data ---
# Load the preprocessed datasets directly from Hugging Face
Xtrain = pd.read_csv(hf_hub_download(repo_id=DATASET_REPO_ID, filename="Xtrain.csv", repo_type="dataset"))
Xtest = pd.read_csv(hf_hub_download(repo_id=DATASET_REPO_ID, filename="Xtest.csv", repo_type="dataset")).squeeze()
ytrain = pd.read_csv(hf_hub_download(repo_id=DATASET_REPO_ID, filename="ytrain.csv", repo_type="dataset")).squeeze()
ytest = pd.read_csv(hf_hub_download(repo_id=DATASET_REPO_ID, filename="ytest.csv", repo_type="dataset")).squeeze()

print("Preprocessed datasets loaded successfully from Hugging Face.")

# --- Feature Definition ---
# Define numeric and categorical features for preprocessing
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'TypeofContact'
]
categorical_features = ['Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched']

# --- Class Weight Calculation ---
# Calculate class weight to handle class imbalance in the target variable
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Calculated class weight for imbalance handling: {class_weight}")

# --- Preprocessing Pipeline ---
# Define preprocessing steps: scale numeric features and one-hot encode categorical features
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# --- Model Definition ---
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Create a pipeline combining preprocessor and XGBoost model
model_pipeline = make_pipeline(preprocessor, xgb_model)

# --- MLflow Experiment Tracking and Model Training ---
with mlflow.start_run():
    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores to MLflow
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log the best parameters from GridSearchCV to the main MLflow run
    mlflow.log_params(grid_search.best_params_)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Define classification threshold
    classification_threshold = 0.45

    # Make predictions on training and testing data using the best model
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Generate classification reports for training and testing data
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log performance metrics to MLflow
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })
    print("Model training and evaluation metrics logged to MLflow.")

    # --- Model Export and Upload to Hugging Face ---
    # Save the best model locally
    model_save_path = "tourism_project/model_building/tourism_model_v1.joblib"
    joblib.dump(best_model, model_save_path)

    # Log the model artifact to MLflow
    mlflow.log_artifact(model_save_path, artifact_path="model")
    print(f"Model saved locally and logged as MLflow artifact at: {model_save_path}")

    # Define Hugging Face model repository details
    HF_MODEL_REPO_ID = "vinitksingh/tourism-package-model"
    HF_MODEL_FILENAME = "tourism_model_v1.joblib"

    # Create model repository on Hugging Face if it doesn't exist
    try:
        api.repo_info(repo_id=HF_MODEL_REPO_ID, repo_type="model")
        print(f"Hugging Face Model Repo '{HF_MODEL_REPO_ID}' already exists. Using it.")
    except HfHubHTTPError:
        print(f"Hugging Face Model Repo '{HF_MODEL_REPO_ID}' not found. Creating new repo...")
        create_repo(repo_id=HF_MODEL_REPO_ID, repo_type="model", private=False)
        print(f"Hugging Face Model Repo '{HF_MODEL_REPO_ID}' created.")

    # Upload the trained model file to Hugging Face Model Hub
    api.upload_file(
        path_or_fileobj=model_save_path,
        path_in_repo=HF_MODEL_FILENAME,
        repo_id=HF_MODEL_REPO_ID,
        repo_type="model",
    )
    print(f"Model '{HF_MODEL_FILENAME}' uploaded to Hugging Face Model Hub '{HF_MODEL_REPO_ID}'.")
