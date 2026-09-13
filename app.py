"""
Customer Churn Prediction - Streamlit Web App
------------------------------------------------
Run locally with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------------------
# Load trained model + preprocessing objects
# -----------------------------------------------------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer is likely to **churn (cancel their subscription)** "
    "based on their account details. Fill in the details below and click **Predict**."
)

st.divider()

# -----------------------------------------------------------------------
# Input form
# -----------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months with company)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])

with col2:
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])

st.divider()

# -----------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------
if st.button("🔮 Predict Churn", use_container_width=True):
    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "TechSupport": tech_support,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
    }

    input_df = pd.DataFrame([input_dict])

    # Apply the same label encoding used during training
    categorical_cols = [
        "Contract", "InternetService", "TechSupport", "PaperlessBilling",
        "PaymentMethod", "Partner", "Dependents"
    ]
    for col in categorical_cols:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn** (probability: {probability:.1%})")
    else:
        st.success(f"✅ This customer is **likely to stay** (churn probability: {probability:.1%})")

    st.progress(min(int(probability * 100), 100))

    st.caption(
        "This is a demo model trained on synthetic data designed to mimic real telecom "
        "churn patterns. In production, this would be trained and validated on the "
        "company's real customer data."
    )

st.divider()
st.caption("Built with Python, Scikit-learn, and Streamlit · [View project on GitHub](#)")
