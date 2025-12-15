"""
preprocessing.py
----------------
Preprocessing and feature engineering for
UCI Pima Indians Diabetes Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("preprocessing.py started")

# Columns where 0 represents missing values
ZERO_AS_MISSING_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV file
    """
    df = pd.read_csv("data\diabetes.csv")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace invalid zero values with median
    """
    df = df.copy()
    for col in ZERO_AS_MISSING_COLS:
        df[col] = df[col].replace(0, df[col].median())
    return df


def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into features (X) and target (y)
    """
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y


def scale_features(X_train, X_test, save_scaler: bool = True):
    """
    Scale features using StandardScaler
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        joblib.dump(scaler, "scaler.pkl")

    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Complete preprocessing pipeline:
    - Load data
    - Handle missing values
    - Train-test split
    - Feature scaling
    """

    # Load
    df = load_data(filepath)

    # Clean
    df = handle_missing_values(df)

    # Split features & target
    X, y = split_features_target(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


# Run directly (for testing)
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        filepath="data/diabetes.csv"
    )

    print("Preprocessing completed successfully!")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
