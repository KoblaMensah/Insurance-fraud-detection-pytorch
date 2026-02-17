"""
Unit tests for the fraud detection pipeline.

Validates data generation, preprocessing, model architecture,
and prediction outputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from src.data_generator import generate_claims_data
from src.preprocessing import DataPreprocessor
from src.model import FraudDetector
from src.dataset import ClaimsDataset, create_dataloaders


def test_data_generation():
    """Test that synthetic data has expected shape and properties."""
    df = generate_claims_data(n_samples=100, fraud_rate=0.15)

    assert len(df) == 100, "Expected 100 samples"
    assert "is_fraud" in df.columns, "Missing target column"
    assert df["is_fraud"].isin([0, 1]).all(), "Target must be binary"
    assert df["claim_amount"].min() >= 500, "Claim amount below minimum"
    assert len(df.columns) == 14, f"Expected 14 columns, got {len(df.columns)}"

    print("test_data_generation PASSED")


def test_preprocessing():
    """Test preprocessing output shapes and types."""
    df = generate_claims_data(n_samples=200, fraud_rate=0.15)
    preprocessor = DataPreprocessor()
    data = preprocessor.fit_transform(df, apply_smote=False)

    assert data["X_train"].dtype == np.float32, "Features should be float32"
    assert data["y_train"].dtype == np.float32, "Labels should be float32"
    assert data["X_train"].shape[1] == len(data["feature_names"])
    assert len(data["feature_names"]) == 13, "Expected 13 features"

    print("test_preprocessing PASSED")


def test_model_forward():
    """Test that model produces correct output shape."""
    input_dim = 13
    model = FraudDetector(input_dim=input_dim)

    batch = torch.randn(16, input_dim)
    output = model(batch)

    assert output.shape == (16,), f"Expected shape (16,), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"

    print("test_model_forward PASSED")


def test_model_predict_proba():
    """Test that predict_proba returns values between 0 and 1."""
    model = FraudDetector(input_dim=13)
    batch = torch.randn(8, 13)

    probs = model.predict_proba(batch)

    assert (probs >= 0).all() and (probs <= 1).all(), \
        "Probabilities must be between 0 and 1"
    assert probs.shape == (8,), f"Expected shape (8,), got {probs.shape}"

    print("test_model_predict_proba PASSED")


def test_dataset_and_dataloader():
    """Test PyTorch Dataset and DataLoader creation."""
    X = np.random.randn(50, 13).astype(np.float32)
    y = np.random.randint(0, 2, 50).astype(np.float32)

    dataset = ClaimsDataset(X, y)
    assert len(dataset) == 50

    sample_x, sample_y = dataset[0]
    assert sample_x.shape == (13,)
    assert sample_y.shape == ()

    print("test_dataset_and_dataloader PASSED")


if __name__ == "__main__":
    test_data_generation()
    test_preprocessing()
    test_model_forward()
    test_model_predict_proba()
    test_dataset_and_dataloader()
    print("\nAll tests passed!")
