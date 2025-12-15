import os
from flask import Flask, render_template, request
import joblib
import numpy as np

# -----------------------------
# Fix template path explicitly
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load model & scaler
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = [float(request.form[f]) for f in FEATURES]
    input_array = np.array(input_data).reshape(1, -1)

    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    risk = "High Risk" if prediction == 1 else "Low Risk"

    return render_template(
        "index.html",
        prediction=risk,
        probability=f"{probability:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
