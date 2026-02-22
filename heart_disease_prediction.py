# train.py (Simple Generalized Version)

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
heart_data = pd.read_csv("heart_disease_data.csv")

# Drop 'slope'
heart_data = heart_data.drop(columns=["slope"])

# Features & Target
X = heart_data.drop(columns="target", axis=1)
y = heart_data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

# Simple Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Evaluation
print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy:", accuracy_score(y_test, test_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, test_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, test_pred))

# Save model
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as heart_model.pkl")