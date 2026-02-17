"""
Training Loop with Early Stopping

Trains the FraudDetector model using BCE loss and Adam optimizer.
Includes early stopping to prevent overfitting and saves the best
model checkpoint based on validation loss.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


def train_model(model, train_loader, test_loader, epochs=100, lr=1e-3,
                patience=10, model_path="models/fraud_detector.pt"):
    """
    Train the fraud detection model.

    Parameters
    ----------
    model : FraudDetector
        The neural network to train.
    train_loader : DataLoader
        Training data.
    test_loader : DataLoader
        Validation/test data.
    epochs : int
        Maximum number of training epochs.
    lr : float
        Learning rate for Adam optimizer.
    patience : int
        Early stopping patience (epochs without improvement).
    model_path : str
        Path to save the best model checkpoint.

    Returns
    -------
    dict with training history (losses and metrics per epoch).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on: {device}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ---- Training phase ----
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # ---- Validation phase ----
        model.eval()
        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_accuracy = correct / total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_accuracy)

        scheduler.step(avg_val_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.4f} | "
                  f"LR: {current_lr:.6f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save best model
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break

    # Load best model weights
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"\nBest model saved to {model_path} "
          f"(val_loss: {best_val_loss:.4f})")

    return history
