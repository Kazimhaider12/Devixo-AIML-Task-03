"""
Bonus Deliverable: Interactive Customer Churn Prediction Interface (Streamlit)
Run with: streamlit run app.py
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Ensure src/ is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from data_preprocessing import engineer_features

MODEL_PATH = os.path.join("models", "best_model.joblib")

st.set_page_config(
    page_title="Telco Churn Predictor | Devixo Week 3",
    page_icon="📡",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0F172A; color: #F8FAFC; }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("📡 Customer Churn Prediction Engine")
st.caption("Devixo Solutions AI/ML Internship — Week 3 End-to-End ML Project | Kazim Haider Syed")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Please run `python src/train.py` first to generate `models/best_model.joblib`.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.sidebar.header("Customer Profile")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

st.sidebar.header("Account & Billing")
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=75.0, step=1.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(monthly_charges * max(tenure, 1)), step=10.0)

st.sidebar.header("Subscribed Services")
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multi_lines = "No phone service" if phone == "No" else st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])
internet = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

if internet == "No":
    sec = backup = prot = tech = tv = movies = "No internet service"
else:
    sec = st.sidebar.selectbox("Online Security", ["No", "Yes"])
    backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
    prot = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
    tech = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
    tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
    movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])

raw_input = pd.DataFrame([{
    "customerID": "LIVE-0001",
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multi_lines,
    "InternetService": internet,
    "OnlineSecurity": sec,
    "OnlineBackup": backup,
    "DeviceProtection": prot,
    "TechSupport": tech,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}])

features = engineer_features(raw_input)
churn_prob = float(model.predict_proba(features)[0, 1])
churn_pred = int(churn_prob >= 0.50)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Churn Probability", f"{churn_prob * 100:.1f}%")
with col2:
    risk = "🔴 High Risk" if churn_prob >= 0.60 else ("🟡 Medium Risk" if churn_prob >= 0.35 else "🟢 Low Risk")
    st.metric("Risk Tier", risk)
with col3:
    st.metric("Prediction", "Likely to Churn" if churn_pred == 1 else "Likely to Retain")

st.progress(min(max(churn_prob, 0.0), 1.0))

st.subheader("Recommended Retention Actions")
tips = []
if contract == "Month-to-month":
    tips.append("Offer a discounted 1-year or 2-year contract upgrade with a loyalty rebate.")
if internet == "Fiber optic" and sec == "No":
    tips.append("Bundle complimentary Online Security or Tech Support for 3 months to increase stickiness.")
if payment == "Electronic check":
    tips.append("Incentivize switching to automatic bank transfer or credit card billing.")
if tenure < 12:
    tips.append("Trigger an early-lifecycle onboarding check-in call during the first 12 months.")
if not tips:
    tips.append("Customer profile is stable; maintain periodic engagement and referral incentives.")

for t in tips:
    st.markdown(f"- {t}")
