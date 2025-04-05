import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np

# Load your model
MODEL_PATH = "/Users/raghavsharma/Desktop/mlProjects/creditCardFraudDetection/app/xgboost_fraud_model.json"
model = xgb.Booster()
model.load_model(MODEL_PATH)

st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details below to predict if it's fraudulent.")

# Define your feature columns (as per training)
predictors = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']

# Collect user input
user_input = {}
for feature in predictors:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Prepare input for prediction
input_df = pd.DataFrame([user_input])
dinput = xgb.DMatrix(input_df[predictors])

# Predict
if st.button("🚀 Predict"):
    y_pred = model.predict(dinput)[0]
    result = "⚠️ Fraudulent" if y_pred > 0.5 else "✅ Legitimate"
    st.subheader(f"Prediction: {result}")
    st.metric("Fraud Probability", f"{y_pred:.4f}")
