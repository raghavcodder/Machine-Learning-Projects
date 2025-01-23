# First code is for predictive model using random forest

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import pickle


# Used to load the datasets provided
df_dev = pd.read_csv("dev_data_to_be_shared.csv") 
df_val = pd.read_csv("validation_data_to_be_shared.csv")  

# This is the code used for cleaning anf preprocessing
df_dev_cleaned = df_dev.dropna(how='all')
df_val_cleaned = df_val.dropna(how='all')

df_dev_cleaned.fillna(df_dev_cleaned.mean(), inplace=True)  
df_dev_cleaned.fillna(df_dev_cleaned.mode().iloc[0], inplace=True)  

# Defining variables for training
X_dev = df_dev_cleaned.drop(columns=['bad_flag', 'account_number'])  
y_dev = df_dev_cleaned['bad_flag']

X_val = df_val_cleaned.drop(columns=['account_number'])

# Train-test split for development data that was givenn to us in the document
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight("balanced", classes=np.unique(y_dev), y=y_dev)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Random Forest Classifier model
rf = RandomForestClassifier(random_state=42, class_weight="balanced")

# Hyperparameter tuning using GridSearchCV as stated in the documenation
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 4],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and best model
print(f"Best Parameters: {grid_search.best_params_}")
best_rf_model = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_rf_model.predict(X_test)
y_pred_proba_best = best_rf_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_best)}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_best)}")
print("Classification Report:\n", classification_report(y_test, y_pred_best))

# Saving the random forest model for streamlit implementation
joblib.dump(best_rf_model, "optimized_random_forest_model_main.pkl")


model_features = best_rf_model.feature_names_in_

X_val = df_val.filter(items=model_features)

# Check for missing features in validation data
missing_features = set(model_features) - set(X_val.columns)
if missing_features:
    raise ValueError(f"Missing features in validation data: {missing_features}")

# Apply the model to the validation data
val_probs = best_rf_model.predict_proba(X_val)[:, 1]
df_val["predicted_probability"] = val_probs

# Extract relevant attribute categories
onus_cols = [col for col in df_val.columns if col.startswith("onus_attributes")]
transaction_cols = [col for col in df_val.columns if col.startswith("transaction_attribute")]
bureau_cols = [col for col in df_val.columns if col.startswith("bureau")]
bureau_enquiry_cols = [col for col in df_val.columns if col.startswith("bureau_enquiry")]

# Normalize each category to a range of 0–1
scaler = MinMaxScaler()

if onus_cols:
    df_val[onus_cols] = scaler.fit_transform(df_val[onus_cols])
if transaction_cols:
    df_val[transaction_cols] = scaler.fit_transform(df_val[transaction_cols])
if bureau_cols:
    df_val[bureau_cols] = scaler.fit_transform(df_val[bureau_cols])
if bureau_enquiry_cols:
    df_val[bureau_enquiry_cols] = scaler.fit_transform(df_val[bureau_enquiry_cols])

# Function created to calculate our unique FinWell score
def calculate_finwell_score(row):
    # Weights (adjust as needed)
    weight_prob = 0.5  # Higher weight for inverse_prob to ibtain values from 300 to 900
    weight_onus = 0.2
    weight_transaction = 0.2
    weight_bureau = 0.05
    weight_enquiry = 0.05

    # Calculate intermediate scores, handle missing groups
    onus_score = row[onus_cols].mean() if onus_cols else 0
    transaction_score = row[transaction_cols].mean() if transaction_cols else 0
    bureau_score = row[bureau_cols].mean() if bureau_cols else 0
    enquiry_score = row[bureau_enquiry_cols].mean() if bureau_enquiry_cols else 0

    # Normalize predicted_probability and invert it
    inverse_prob = 1 - row["predicted_probability"]
  
    # Weighted sum
    raw_score = (
        weight_prob * inverse_prob +
        weight_onus * onus_score +
        weight_transaction * transaction_score +
        weight_bureau * bureau_score +
        weight_enquiry * enquiry_score
    )

    # Ensure raw score is within 0–1
    raw_score = max(0, min(raw_score, 1))

    # Scale raw score to desired range (300–900)
    scaled_score = 300 + (raw_score * 600)  
    return scaled_score

df_val["finwell_score"] = df_val.apply(calculate_finwell_score, axis=1)

# Define risk categories
def assign_risk_category(score):
    if score < 500:
        return "High Risk"
    elif 500 <= score < 700:
        return "Moderate Risk"
    else:
        return "Low Risk"

df_val["risk_category"] = df_val["finwell_score"].apply(assign_risk_category)

# Save results to a CSV file
df_val[["account_number", "predicted_probability", "finwell_score", "risk_category"]].to_csv(
    "finwell_predictions_validation_new1.csv", index=False
)

print("Results saved to 'finwell_predictions_validation.csv'")


# This is the streamlit code 

import streamlit as st
import pandas as pd
import joblib as jb
from sklearn.preprocessing import MinMaxScaler

# Apply custom styles
st.markdown(
    """
    <style>
    /* Set background color to black */
    .stApp {
        background-color: black;
        color: white;
    }

    /* Set the style of text inputs and buttons */
    input, button {
        color: black;
        font-size: 16px;
    }

    /* Customize DataFrame display */
    .dataframe {
        background-color: black;
        color: white;
    }

    /* Add a border around all containers */
    div[data-testid="stVerticalBlock"] {
        border: 1px solid white;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and introduction
st.title("Random Forest Model with FinWell Score")
st.write(
    """
    Welcome to the FinWell Score Predictor app! Upload your validation data and calculate
    FinWell Scores and Risk Categories with our Random Forest model.
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload Validation CSV", type=["csv"])

if uploaded_file is not None:
    # Load the model and uploaded data
    model = jb.load("optimized_random_forest_model_main.pkl")
    df_val = pd.read_csv(uploaded_file)

    st.write("Data Preview:")
    st.dataframe(df_val.head())  # Show first few rows

    # Feature alignment
    model_features = model.feature_names_in_
    X_val = df_val.filter(items=model_features)

    # Check for missing features in the validation dataset
    missing_features = set(model_features) - set(X_val.columns)
    if missing_features:
        st.error(f"Missing features in validation data: {missing_features}")
    else:
        # Predict probabilities
        val_probs = model.predict_proba(X_val)[:, 1]
        df_val["predicted_probability"] = val_probs

        # Feature normalization
        onus_cols = [col for col in df_val.columns if col.startswith("onus_attributes")]
        transaction_cols = [col for col in df_val.columns if col.startswith("transaction_attribute")]
        bureau_cols = [col for col in df_val.columns if col.startswith("bureau")]
        bureau_enquiry_cols = [col for col in df_val.columns if col.startswith("bureau_enquiry")]

        scaler = MinMaxScaler()
        if onus_cols:
            df_val[onus_cols] = scaler.fit_transform(df_val[onus_cols])
        if transaction_cols:
            df_val[transaction_cols] = scaler.fit_transform(df_val[transaction_cols])
        if bureau_cols:
            df_val[bureau_cols] = scaler.fit_transform(df_val[bureau_cols])
        if bureau_enquiry_cols:
            df_val[bureau_enquiry_cols] = scaler.fit_transform(df_val[bureau_enquiry_cols])

        # FinWell score calculation
        def calculate_finwell_score(row):
            weight_prob = 0.5
            weight_onus = 0.2
            weight_transaction = 0.2
            weight_bureau = 0.05
            weight_enquiry = 0.05

            onus_score = row[onus_cols].mean() if onus_cols else 0
            transaction_score = row[transaction_cols].mean() if transaction_cols else 0
            bureau_score = row[bureau_cols].mean() if bureau_cols else 0
            enquiry_score = row[bureau_enquiry_cols].mean() if bureau_enquiry_cols else 0

            inverse_prob = 1 - row["predicted_probability"]

            raw_score = (
                weight_prob * inverse_prob +
                weight_onus * onus_score +
                weight_transaction * transaction_score +
                weight_bureau * bureau_score +
                weight_enquiry * enquiry_score
            )
            raw_score = max(0, min(raw_score, 1))
            scaled_score = 300 + (raw_score * 600)
            return scaled_score

        df_val["finwell_score"] = df_val.apply(calculate_finwell_score, axis=1)

        # Risk categorization
        def assign_risk_category(score):
            if score < 500:
                return "High Risk"
            elif 500 <= score < 700:
                return "Moderate Risk"
            else:
                return "Low Risk"

        df_val["risk_category"] = df_val["finwell_score"].apply(assign_risk_category)

        # Display results
        st.write("Results:")
        st.dataframe(df_val[["account_number", "predicted_probability", "finwell_score", "risk_category"]].head())

        # Save results option
        if st.button("Save Results"):
            output_file = "finwell_predictions_validation_new.csv"
            df_val[["account_number", "predicted_probability", "finwell_score", "risk_category"]].to_csv(output_file, index=False)
            st.success(f"Results saved to {output_file}")
