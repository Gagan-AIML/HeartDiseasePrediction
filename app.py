import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------
# Load and prepare model
# ------------------------
@st.cache_resource
def train_model():
    # Load dataset
    heart_data = pd.read_csv("heart_disease_data.csv")

    X = heart_data.drop(columns="target", axis=1)
    y = heart_data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    return model, train_acc, test_acc


model, train_acc, test_acc = train_model()

# ------------------------
# Streamlit UI
# ------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to check the risk of heart disease.")

st.sidebar.header("Model Info")
st.sidebar.write(f"**Training Accuracy:** {train_acc:.2f}")
st.sidebar.write(f"**Testing Accuracy:** {test_acc:.2f}")

# Collect user input
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (cp)", options=[0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate (thalach)", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of ST segment (slope)", options=[0, 1, 2])
ca = st.selectbox("Number of major vessels (ca)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

# Prediction
if st.button("Predict"):
    input_data = np.array([age, sex, cp, trestbps, chol, fbs,
                           restecg, thalach, exang, oldpeak,
                           slope, ca, thal]).reshape(1, -1)
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The person is likely to have heart disease.")
    else:
        st.success("✅ The person is unlikely to have heart disease.")

