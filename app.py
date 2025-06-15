import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")  # Or replace with .sav or .pkl

st.title("📉 Customer Churn Prediction")
st.write("This app predicts whether a customer will churn based on input features.")

# Sample inputs — replace with your actual features
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

# You can add more fields based on your model's features

# Preprocess inputs — this must match your training preprocessing
gender = 1 if gender == "Male" else 0
senior = 1 if senior == "Yes" else 0

# Arrange features in the right order as model expects
features = np.array([[gender, senior, tenure, monthly_charges]])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn. (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Customer is likely to stay. (Confidence: {prob:.2f})")
