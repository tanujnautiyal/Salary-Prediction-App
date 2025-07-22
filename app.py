import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved components
model = joblib.load('xgboost_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')
scaled_columns = joblib.load('scaled_columns.pkl')  # ['age', 'capital-gain', ...]

st.sidebar.markdown("**Project by Tanuj Nautiyal**  \nModel: XGBoost  \nData: UCI Adult Dataset")
def local_css(css):
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
st.markdown("## 📊 Income Prediction App")
st.markdown("Use machine learning to predict whether a person earns **>50K or ≤50K USD/year** based on their profile.")

dark_css = """
    .stButton>button {
        color: white;
        background-color: #00AAFF;
        padding: 0.6em 1.2em;
        border-radius: 0.5em;
        border: none;
    }
    .stTextInput>div>div>input {
        background-color: #20242F;
        color: white;
    }
    .stSelectbox>div>div {
        background-color: #20242F;
        color: white;
    }
"""

local_css(dark_css)

# --- Streamlit input form ---
def user_input():
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100)
        workclass = st.selectbox("Workclass", options=label_encoders['workclass'].classes_)
        education = st.selectbox("Education", options=label_encoders['education'].classes_)
        marital_status = st.selectbox("Marital Status", options=label_encoders['marital-status'].classes_)
        occupation = st.selectbox("Occupation", options=label_encoders['occupation'].classes_)

    with col2:
        relationship = st.selectbox("Relationship", options=label_encoders['relationship'].classes_)
        race = st.selectbox("Race", options=label_encoders['race'].classes_)
        gender = st.selectbox("Gender", options=label_encoders['gender'].classes_)
        capital_gain = st.number_input("Capital Gain", min_value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0)
        hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100)

    return pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'education': [education],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
    })


df_input = user_input()

# --- Apply label encoding only on known categorical columns ---
for col in label_encoders:
    if col in df_input.columns:
        df_input[col] = label_encoders[col].transform(df_input[col])

# --- Add missing numerical columns with default values (not for encoding!) ---
defaults = {
    'educational-num': 10,
    'fnlwgt': 100000
}
for col, val in defaults.items():
    if col not in df_input.columns:
        df_input[col] = val

# --- Ensure all scaled columns exist ---
# Load trained scaler and column list
scaler = joblib.load('scaler.pkl')
scaled_columns = joblib.load('scaled_columns.pkl')

import pandas as pd

# Load scaler and the feature list
scaler = joblib.load('scaler.pkl')
scaled_columns = joblib.load('scaled_columns.pkl')

# Ensure missing columns are filled (e.g., if user left any blank)
for col in scaled_columns:
    if col not in df_input.columns:
        df_input[col] = 0  # or mean used during training

# Ensure correct order and column names
to_scale = df_input[scaled_columns].copy()


# Now transform safely
df_input[scaled_columns] = scaler.transform(to_scale)



# --- Final check: add any remaining required features ---
for col in feature_columns:
    if col not in df_input.columns:
        df_input[col] = 0  # catch-all safeguard

# --- Reorder to match training set ---
df_input = df_input[feature_columns]

# --- Predict ---
if st.button("Predict"):
    with st.spinner("Analyzing..."):
        prediction = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][prediction]

        if prediction == 1:
            st.success(f"💰 **Prediction: >50K USD** (Confidence: {prob:.2%})")
        else:
            st.info(f"📉 **Prediction: ≤50K USD** (Confidence: {prob:.2%})")
import matplotlib.pyplot as plt

if st.checkbox("Show Feature Importances"):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 4))
    plt.barh(np.array(feature_columns)[sorted_idx], importances[sorted_idx])
    st.pyplot(plt)


with st.expander("See Encoded Input Data"):
    st.dataframe(df_input)

