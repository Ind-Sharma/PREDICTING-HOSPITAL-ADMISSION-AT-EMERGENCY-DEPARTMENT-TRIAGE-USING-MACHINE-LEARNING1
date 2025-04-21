import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# App title
st.set_page_config(page_title="Hospital Admission Prediction Dashboard", layout="wide")
st.title("\U0001F3E5 Hospital Admission Prediction")

# Description
st.markdown("""
This dashboard uses pre-trained machine learning models to predict hospital admission outcomes based on triage data.

**To use the dashboard:**
1. Upload a `.csv` or `.csv.gz` file containing triage data.
2. The dashboard will preprocess the data.
3. Predictions from Logistic Regression, XGBoost, and Random Forest models will be displayed side-by-side.
4. A final decision column will show the majority vote among the three models.
""")

# Upload data for prediction
st.subheader("\U0001F4E4 Upload Data for Prediction")
pred_file = st.file_uploader("Upload a CSV or CSV.GZ file", type=["csv", "gz"], key="pred")

# Load pre-saved models and training columns
with open("logreg_model.pkl", "rb") as f:
    logreg_model = pickle.load(f)
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("training_columns.pkl", "rb") as f:
    training_columns = pickle.load(f)

if pred_file:
    if pred_file.name.endswith(".gz"):
        with gzip.open(pred_file, 'rt') as f:
            input_df = pd.read_csv(f)
    else:
        input_df = pd.read_csv(pred_file)

    st.subheader("\U0001F4CB Uploaded Data Preview")
    st.dataframe(input_df)

    # Preprocessing: Encode + Impute + Scale
    input_df_encoded = pd.get_dummies(input_df)
    missing_cols = set(training_columns) - set(input_df_encoded.columns)
    for col in missing_cols:
        input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[training_columns]  # align columns

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(input_df_encoded)
    X_scaled = scaler.fit_transform(X_imputed)

    # Predictions
    logreg_preds = logreg_model.predict(X_scaled)
    xgb_preds = xgb_model.predict(X_scaled)
    rf_preds = rf_model.predict(X_scaled)

    def decode_preds(pred):
        return ["Admit" if p == 1 else "Not Admit" for p in pred]

    logreg_decoded = decode_preds(logreg_preds)
    xgb_decoded = decode_preds(xgb_preds)
    rf_decoded = decode_preds(rf_preds)

    # Majority vote
    final_preds = []
    for l, x, r in zip(logreg_preds, xgb_preds, rf_preds):
        vote = l + x + r
        final_preds.append("Admit" if vote >= 2 else "Not Admit")

    # Display results
    result_df = pd.DataFrame({
        "Logistic Regression": logreg_decoded,
        "XGBoost": xgb_decoded,
        "Random Forest": rf_decoded,
        "Final Decision (Majority Vote)": final_preds
    })

    st.subheader("\U0001F9E0 Model Predictions")
    st.dataframe(result_df)
