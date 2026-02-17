"""
Model Evaluation

Computes classification metrics and generates visualizations.
For fraud detection, we emphasize recall and F1 over accuracy
because missing a fraudulent claim is more costly than a false alarm.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from pathlib import Path


def evaluate_model(model, test_loader, threshold=0.5):
    """
    Evaluate the trained model on test data.

    Returns
    -------
    dict with predictions, probabilities, and all metrics.
    """
    device = next(model.parameters()).device
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs),
    }

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"  {name:>12s}: {value:.4f}")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Legitimate", "Fraudulent"]
    ))

    return {
        "metrics": metrics,
        "probabilities": all_probs,
        "predictions": all_preds,
        "labels": all_labels,
    }


def plot_training_history(history, save_path="models/training_history.png"):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train Loss", color="#2196F3")
    axes[0].plot(history["val_loss"], label="Val Loss", color="#FF5722")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(history["val_accuracy"], label="Val Accuracy", color="#4CAF50")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(labels, predictions, save_path="models/confusion_matrix.png"):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Fraudulent"],
        yticklabels=["Legitimate", "Fraudulent"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(labels, probabilities, save_path="models/roc_curve.png"):
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {save_path}")
