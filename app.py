import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load pre-trained model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")  # feature list used during training

st.title("Churn Prediction App")
st.divider()
st.write("Please enter the values and hit the predict button")
st.divider()

# Sidebar inputs
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Enter Monthly Charges", min_value=10, max_value=200, value=30)
TotalCharges = st.number_input("Enter Total Charges", min_value=10, max_value=9000)

st.divider()

# -----------------------------
# Create input dictionary
# -----------------------------
input_dict = {
    'gender': [gender],
    'SeniorCitizen': [1 if SeniorCitizen == "Yes" else 0],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
}

input_df = pd.DataFrame(input_dict)

# -----------------------------
# Preprocessing (same as training)
# -----------------------------
input_df['gender'] = input_df['gender'].map({'Female': 1, 'Male': 0})
# Replace 'No internet service' / 'No phone service' with 'No'
service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    input_df[col] = input_df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

# Binary columns: convert Yes/No to 1/0
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies', 'PaperlessBilling']
for col in binary_cols:
    input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

# One-hot encode remaining categorical variables
input_df_encoded = pd.get_dummies(input_df)

# Align with training columns
input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)

# Scale the numeric columns
input_df_scaled = scaler.transform(input_df_encoded)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_df_scaled)[0]
    prob = model.predict_proba(input_df_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The customer is likely to churn. (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ The customer is likely to stay. (Confidence: {prob:.2f})")

churn_result = "Yes" if prediction == 1 else "No"

st.subheader("🔍 Prediction Result")
st.write(f"**Churn Prediction:** `{churn_result}`")
st.write(f"**Confidence:** `{prob:.2f}`")

