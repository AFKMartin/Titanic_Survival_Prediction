import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "titanic_xgb_92_17.pkl")
DATA_PATH = os.path.join(BASE_DIR, "../data/processed_titanic.csv")  # adjust if needed

# Load the model
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# Load the processed CSV
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"CSV not found: {DATA_PATH}")
data = pd.read_csv(DATA_PATH)

# Split into features and target
y = data["Survived"]
x = data.drop(columns=["Survived"])

# Split into train/test to mimic your notebook
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)

# Test accuracy
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {acc*100:.2f}%")

# Confusion matrix
conmat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conmat)

# Single sample prediction (make sure order matches model features)
sample = [3, 0, 22, 1, 0, False, True, 1, 0, 0, 1]  # Example passenger
pred = model.predict([sample])[0]
proba = model.predict_proba([sample])[0][1] if hasattr(model, "predict_proba") else None

print("\nSingle sample prediction:")
print("Prediction:", "Survived" if pred == 1 else "Not Survived")
if proba is not None:
    print(f"Survival probability: {proba*100:.2f}%")

# TODO this alredy all done, use the pkl to develop the backend 