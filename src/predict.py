"""
predict.py
----------
Run predictions using trained model
(UCI Pima Indians Diabetes Dataset)
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np


# -----------------------------
# Load trained artifacts
# -----------------------------
print("Started")
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


def predict_diabetes(input_data):
    """
    input_data: list or array of 8 values in correct order
    returns: prediction, probability
    """

    input_array = np.array(input_data).reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability


# -----------------------------
# Run directly (CLI usage)
# -----------------------------
if __name__ == "__main__":
    print("🩺 Diabetes Risk Prediction")
    print("----------------------------")

    user_input = []

    for feature in FEATURE_NAMES:
        value = float(input(f"Enter {feature}: "))
        user_input.append(value)

    pred, prob = predict_diabetes(user_input)

    print("\n📊 Prediction Result")
    if pred == 1:
        print("⚠️ High Risk of Diabetes")
    else:
        print("✅ Low Risk of Diabetes")

    print(f"Risk Probability: {prob:.2f}")
