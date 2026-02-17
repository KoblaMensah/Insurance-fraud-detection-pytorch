"""
Neural Network Architecture for Fraud Detection

A feedforward network with batch normalization and dropout for
regularization. The architecture is intentionally moderate (3 hidden
layers) — deep enough to capture nonlinear patterns in tabular data,
but not so deep that it overfits on a ~5000 sample dataset.
"""

import torch
import torch.nn as nn


class FraudDetector(nn.Module):
    """
    Feedforward neural network for binary fraud classification.

    Architecture: Input -> 64 -> 32 -> 16 -> 1
    Each hidden layer uses BatchNorm + ReLU + Dropout.
    """

    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1: input -> 64
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 2: 64 -> 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Layer 3: 32 -> 16
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),

            # Output layer
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

    def predict_proba(self, x):
        """Return fraud probability (sigmoid applied)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def get_model_summary(model, input_dim):
    """Print a summary of model architecture and parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nFraudDetector Architecture")
    print(f"{'='*45}")
    print(f"Input dimension:    {input_dim}")
    print(f"Hidden layers:      64 -> 32 -> 16")
    print(f"Output:             1 (binary)")
    print(f"Total parameters:   {total_params:,}")
    print(f"Trainable params:   {trainable_params:,}")
    print(f"{'='*45}")

    return {"total_params": total_params, "trainable_params": trainable_params}
