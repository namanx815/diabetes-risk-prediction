"""
train.py
--------
Model training script for
UCI Pima Indians Diabetes Dataset
"""

import warnings
warnings.filterwarnings("ignore")
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Import preprocessing pipeline
from src.preprocessing import preprocess_pipeline


def train_model():
    """
    Train XGBoost model on preprocessed data
    """

    # Run preprocessing
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        filepath="data\diabetes.csv"
    )

    # Initialize model
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n📊 Model Evaluation Results")
    print("----------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save trained model
    joblib.dump(model, "model.pkl")
    print("\n✅ Model saved as model.pkl")


# Run training
if __name__ == "__main__":
    print("🚀 Training started...")
    train_model()
    print("🎉 Training completed successfully!")
