# Salary-Prediction-App

# 💼 Salary Prediction Web App

This is a Streamlit web application that predicts whether a person earns more than 50K per year based on demographic and work-related attributes. The model is trained using the UCI Adult Income dataset.

---

## 🚀 Demo

👉 [Live App on Streamlit](https://salary-prediction-app-ml.streamlit.app/)

---

## 📌 Features

- Predict salary category: `>50K` or `<=50K`
- User-friendly UI built with **Streamlit**
- Uses **XGBoost** model trained on cleaned and encoded data
- Automatic **label encoding** for categorical inputs
- **StandardScaler** used for numerical features
- Realtime prediction on user inputs

---

## 📂 Files in this Repo

| File/Folder          | Description |
|----------------------|-------------|
| `app.py`             | Main Streamlit app script |
| `xgboost_model.pkl`  | Trained XGBoost model |
| `scaler.pkl`         | StandardScaler used to scale numeric inputs |
| `label_encoders.pkl` | LabelEncoders for each categorical column |
| `feature_columns.pkl`| List of all final model features |
| `scaled_columns.pkl` | List of scaled numeric columns |
| `requirements.txt`   | Required libraries for Streamlit Cloud |

---

## 🧪 Input Features

- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Gender
- Capital Gain
- Capital Loss
- Hours per Week

---

## 🧠 Model Details

- Model: `XGBoost Classifier`
- Dataset: UCI Adult Income
- Preprocessing:
  - Label encoding of categorical variables
  - Scaling of numerical variables using StandardScaler
- Target Variable: `Income (>50K or <=50K)`

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
