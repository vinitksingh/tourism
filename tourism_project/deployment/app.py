import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
# Define the Hugging Face model repository ID where the trained model is stored.
MODEL_REPO_ID = "vinitksingh/tourism-package-model"
MODEL_FILENAME = "tourism_model_v1.joblib"

# --- Load the Model ---
# Use Streamlit's caching mechanism to load the model only once for performance.
@st.cache_resource
def load_model():
    try:
        # Download the model file from Hugging Face Hub.
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        # Load the pre-trained model using joblib.
        model = joblib.load(model_path)
        return model
    except Exception as e:
        # Display an error message if model loading fails and stop the application.
        st.error(f"Error loading model from Hugging Face: {e}")
        st.stop() # Stop the app if model can't be loaded

# Load the model globally when the app starts.
model = load_model()

# --- Streamlit UI for Tourism Package Prediction ---
st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase a new Wellness Tourism Package based on their profile and interaction data.
Please enter the customer details below to get a prediction.
""")

# --- User Input Fields ---
st.header("Customer Information")

# Organize input fields into two columns for a cleaner layout.
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Customer's age.")
    # Dropdown for Type of Contact, with clear options.
    typeofcontact_options = ["Company Invited", "Self Inquiry"]
    typeofcontact_selected = st.selectbox("Type of Contact", typeofcontact_options, help="Method of contact with the customer.")
    citytier = st.selectbox("City Tier", [1, 2, 3], help="City category (Tier 1 > Tier 2 > Tier 3).")
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=0, max_value=10, value=1, help="Total number of people accompanying the customer.")
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5], help="Preferred hotel rating by the customer.")
    num_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=20, value=5, help="Average number of trips the customer takes annually.")
    # Toggle for Passport ownership, displaying Yes/No.
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Whether the customer holds a valid passport.")

with col2:
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, help="Customer's satisfaction with the sales pitch.")
    # Toggle for Car ownership, displaying Yes/No.
    own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Whether the customer owns a car.")
    num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0, help="Number of children below age 5 accompanying the customer.")
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=50000.0, step=1000.0, help="Gross monthly income of the customer.")
    # Dropdown for Occupation, including an 'Other' option for flexibility.
    occupation_options = ["Salaried", "Small Business", "Large Business", "Free Lancer", "Other"]
    occupation = st.selectbox("Occupation", occupation_options, help="Customer's occupation.")
    gender_options = ["Male", "Female"]
    gender = st.selectbox("Gender", gender_options, help="Gender of the customer.")
    marital_status_options = ["Single", "Married", "Divorced"]
    marital_status = st.selectbox("Marital Status", marital_status_options, help="Marital status of the customer.")
    # Dropdown for Designation, including an 'Other' option.
    designation_options = ["Executive", "Manager", "Senior Manager", "AVP", "VP", "President", "Other"]
    designation = st.selectbox("Designation", designation_options, help="Customer's designation in their current organization.")
    # Dropdown for Product Pitched, including an 'Other' option.
    product_pitched_options = ["Basic", "Deluxe", "Standard", "Super Deluxe", "King", "Other"]
    product_pitched = st.selectbox("Product Pitched", product_pitched_options, help="The type of product pitched to the customer.")
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=10, help="Duration of the sales pitch delivered.")

# --- Convert categorical inputs to numerical if necessary before passing to model ---
# TypeofContact: Label Encoding (Company Invited: 0, Self Inquiry: 1 based on original train.py logic)
typeofcontact_encoded = 0 if typeofcontact_selected == "Company Invited" else 1

# Create a Pandas DataFrame from the user inputs, ensuring column names and order
# match the features the model was trained on.
input_df = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofcontact_encoded,
    'CityTier': citytier,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfPersonVisiting': num_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'MonthlyIncome': monthly_income,
    'Occupation': occupation,
    'Gender': gender,
    'MaritalStatus': marital_status,
    'Designation': designation,
    'ProductPitched': product_pitched
}])

# --- Prediction ---
# Trigger prediction when the 'Predict Purchase' button is clicked.
if st.button("Predict Purchase"):
    try:
        # The loaded model (a pipeline including preprocessor and XGBoost) expects
        # raw features and handles its own encoding/scaling internally.
        prediction_proba = model.predict_proba(input_df)[:, 1]
        classification_threshold = 0.45 # Use the same threshold as in training for consistency.
        prediction = (prediction_proba >= classification_threshold).astype(int)[0]

        # Display the prediction result to the user.
        result = "Customer is likely to purchase the package!" if prediction == 1 else "Customer is unlikely to purchase the package."
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success(f"**{result}** (Probability: {prediction_proba[0]:.2f})")
        else:
            st.info(f"**{result}** (Probability: {prediction_proba[0]:.2f})")
    except Exception as e:
        # Catch and display any errors during the prediction process.
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input values are valid and the model is correctly loaded.")
