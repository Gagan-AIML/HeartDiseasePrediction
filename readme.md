# ❤️ Heart Disease Prediction using Logistic Regression

## 📌 Project Overview
This project predicts the likelihood of heart disease using patient medical attributes.

The goal is to build a simple, interpretable, and generalized machine learning model that can assist in early risk detection.

## 🚀 Tech Stack
- Python
- Scikit-learn
- Pandas
- Streamlit
- Logistic Regression

## 📊 Dataset Features
The model was trained on 12 medical features:

- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Resting ECG (restecg)
- Maximum Heart Rate (thalach)
- Exercise Induced Angina (exang)
- ST Depression (oldpeak)
- Number of Major Vessels (ca)
- Thalassemia (thal)

Target:
- 0 = No Heart Disease
- 1 = Heart Disease

## 🧠 Model Used
- Logistic Regression
- No over-complex tuning
- Focus on generalization and interpretability

## 📈 Model Performance

- Training Accuracy: ~84%
- Test Accuracy: ~80%
- Evaluated using:
  - Accuracy
  - Confusion Matrix
  - Classification Report

The model was kept simple to avoid overfitting on a small dataset.

## 🎯 Why Logistic Regression?

- Works well on small structured datasets
- Interpretable coefficients
- Lower overfitting risk compared to tree-based models
- Provides probability scores for risk assessment

## 🖥️ Streamlit App

An interactive web app allows users to input patient details and receive:
- Predicted risk
- Probability score

To run locally:

```bash
streamlit run app.py