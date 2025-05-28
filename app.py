import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and scaler
model = pickle.load(open('logistic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# App title
st.set_page_config(page_title="👁️ Eye Cancer Risk Predictor", layout="centered")
st.title("👁️ Eye Cancer Risk Prediction")
st.markdown("""
### 📋 Please provide the following information below:
""")

# Input fields
age = st.slider("🎂 Age", 1, 100, 30)
gender = st.selectbox("⚧️ Gender", ["Male", "Female"])
cancer_type = st.selectbox("🧬 Cancer Type", ["Type 1", "Type 2", "Type 3"])
laterality = st.selectbox("👁️ Eye Affected", ["Left", "Right", "Bilateral"])
stage = st.selectbox("📊 Stage at Diagnosis", ["Stage I", "Stage II", "Stage III"])
treatment = st.selectbox("💊 Treatment Type", ["Surgery", "Radiation", "Chemotherapy"])
surgery_status = st.radio("🔪 Surgery Done?", ["Yes", "No"])
radiation_therapy = st.radio("☢️ Radiation Therapy", ["Yes", "No"])
chemotherapy = st.radio("🧪 Chemotherapy", ["Yes", "No"])
outcome = st.selectbox("📈 Outcome Status", ["Recovered", "Stable", "Deceased"])
survival_time = st.slider("📆 Survival Time (Months)", 0, 120, 24)
family_history = st.radio("👨‍👩‍👦 Family History", ["Yes", "No"])
country = st.selectbox("🌍 Country", ["Bangladesh", "India", "USA", "UK"])

# Encoding text input into numerical
gender_val = 1 if gender == "Male" else 0
cancer_val = {"Type 1": 0, "Type 2": 1, "Type 3": 2}[cancer_type]
laterality_val = {"Left": 0, "Right": 1, "Bilateral": 2}[laterality]
stage_val = {"Stage I": 0, "Stage II": 1, "Stage III": 2}[stage]
treatment_val = {"Surgery": 0, "Radiation": 1, "Chemotherapy": 2}[treatment]
surgery_val = 1 if surgery_status == "Yes" else 0
radiation_val = 1 if radiation_therapy == "Yes" else 0
chemo_val = 1 if chemotherapy == "Yes" else 0
outcome_val = {"Recovered": 0, "Stable": 1, "Deceased": 2}[outcome]
family_val = 1 if family_history == "Yes" else 0
country_val = {"Bangladesh": 0, "India": 1, "USA": 2, "UK": 3}[country]

# Final feature set
features = np.array([[age, gender_val, cancer_val, laterality_val, stage_val,
                      treatment_val, surgery_val, radiation_val, chemo_val,
                      outcome_val, survival_time, family_val, country_val]])

# Scale the features
features_scaled = scaler.transform(features)

# Predict
if st.button("🔮 Predict"):
    prediction = model.predict(features_scaled)[0]
    risk_level = ["🟢 Low Risk", "🟡 Medium Risk", "🔴 High Risk"]
    st.markdown(f"## Result: {risk_level[prediction]}")
    st.success("✅ Prediction successful")

    st.markdown("""
    ### ℹ️ What This Means:
    - 🟢 Low Risk: Minimal concern
    - 🟡 Medium Risk: Moderate attention recommended
    - 🔴 High Risk: Immediate medical attention may be required
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Asif Iqbal")
