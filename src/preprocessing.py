"""
Data Preprocessing Pipeline

Handles encoding of categorical variables, feature scaling, train/test
splitting, and class imbalance via SMOTE. Produces clean tensors ready
for the PyTorch model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path


class DataPreprocessor:
    """End-to-end preprocessing for insurance claims data."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_cols = ["claim_type", "region"]
        self.numeric_cols = [
            "policyholder_age", "policy_tenure_months", "num_prior_claims",
            "claim_amount", "police_report_filed", "witnesses_present",
            "vehicle_age_years", "vehicle_value", "days_to_report",
            "claim_hour", "claim_to_value_ratio"
        ]

    def fit_transform(self, df, target_col="is_fraud", test_size=0.2,
                      apply_smote=True, random_state=42):
        """
        Full preprocessing pipeline: encode, scale, split, and balance.

        Returns
        -------
        dict with keys: X_train, X_test, y_train, y_test, feature_names
        """
        df = df.copy()

        # Encode categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Separate features and target
        feature_cols = self.numeric_cols + self.categorical_cols
        X = df[feature_cols].values
        y = df[target_col].values
        self.feature_names = feature_cols

        # Train-test split (stratified to preserve fraud ratio)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y
        )

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Handle class imbalance with SMOTE on training set only
        if apply_smote:
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"SMOTE applied: training set resampled to {len(y_train)} samples")
            print(f"  Class distribution: {np.bincount(y_train)}")

        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        print(f"Features: {len(self.feature_names)}")

        return {
            "X_train": X_train.astype(np.float32),
            "X_test": X_test.astype(np.float32),
            "y_train": y_train.astype(np.float32),
            "y_test": y_test.astype(np.float32),
            "feature_names": self.feature_names,
        }

    def transform(self, df):
        """Transform new data using fitted encoders and scaler."""
        df = df.copy()
        for col in self.categorical_cols:
            df[col] = self.label_encoders[col].transform(df[col])

        feature_cols = self.numeric_cols + self.categorical_cols
        X = df[feature_cols].values
        X = self.scaler.transform(X)
        return X.astype(np.float32)

    def save(self, path="models/preprocessor.pkl"):
        """Save preprocessor state for inference."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "feature_names": self.feature_names,
            "categorical_cols": self.categorical_cols,
            "numeric_cols": self.numeric_cols,
        }, path)
        print(f"Preprocessor saved to {path}")

    def load(self, path="models/preprocessor.pkl"):
        """Load saved preprocessor state."""
        state = joblib.load(path)
        self.scaler = state["scaler"]
        self.label_encoders = state["label_encoders"]
        self.feature_names = state["feature_names"]
        self.categorical_cols = state["categorical_cols"]
        self.numeric_cols = state["numeric_cols"]
        return self
