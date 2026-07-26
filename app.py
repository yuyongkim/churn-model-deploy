import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load model and encoder
with open("churn_rf_healthy_meals.pkl", "rb") as f:
    model = pickle.load(f)

with open("churn_encoder_healthy_meals.pkl", "rb") as f:
    encoder = pickle.load(f)

# UI
st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

age = st.slider("Age", 18, 80, 35)
income_level = st.radio("Income Level", ["Low", "Medium", "High", "Very High"])
education = st.radio("Education", ["Graduate", "High School", "Other", "Post-Graduate"])
device_type = st.radio("Device Type", ["Desktop-only", "Mobile-only", "Multi-device"])
tech_comfort_score = st.slider("Tech Comfort Score", 1, 10, 5)

if st.button("Predict"):
    raw = pd.DataFrame([{
        'INCOME_LEVEL': income_level,
        'EDUCATION':    education,
        'DEVICE_TYPE':  device_type,
    }])

    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    numeric_df = pd.DataFrame([{
        'AGE':                    age,
        'TECH_COMFORT_SCORE':     tech_comfort_score,
        'NUM_SESSIONS':           0,
        'TOTAL_SESSION_DURATION': 0,
        'NUM_ACTIVE_DAYS':        0,
    }])

    input_df = pd.concat([numeric_df.reset_index(drop=True),
                          encoded_df.reset_index(drop=True)], axis=1)
    
    # Match exact training column order
    expected_cols = ['AGE', 'TECH_COMFORT_SCORE', 'NUM_SESSIONS', 
                     'TOTAL_SESSION_DURATION', 'NUM_ACTIVE_DAYS'] + \
                    list(encoder.get_feature_names_out())
    input_df = input_df[expected_cols]

    probability = model.predict_proba(input_df)[0][1]
    risk = "Low" if probability >= 0.6 else "Medium" if probability >= 0.4 else "High"

    st.metric("Renewal Probability", f"{probability:.1%}")
    st.metric("Churn Risk", risk)
    st.progress(probability)
