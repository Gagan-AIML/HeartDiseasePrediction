# app.py

import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("heart_model.pkl", "rb"))

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below:")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol Level", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression (Oldpeak)", value=1.0)
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0-2)", [0, 1, 2])

if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            ca, thal]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Predicted Probability of Heart Disease: {round(probability*100,2)}%")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")