"""
Model Explainability — SHAP and Integrated Gradients

Provides two complementary approaches to understanding model predictions:
  1. SHAP (SHapley Additive exPlanations): Model-agnostic, game-theory
     based feature importance. Shows global and per-prediction contributions.
  2. Integrated Gradients (via Captum): PyTorch-native gradient-based
     attribution. Fast and exact for neural networks.

Both methods answer: "Which features drove this prediction?"
This is critical in insurance — regulators and auditors need to understand
why a claim was flagged as fraudulent.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
from captum.attr import IntegratedGradients
from pathlib import Path


def compute_shap_values(model, X_background, X_explain, feature_names):
    """
    Compute SHAP values for the model predictions.

    Parameters
    ----------
    model : FraudDetector
        Trained PyTorch model.
    X_background : np.ndarray
        Background dataset for SHAP (subset of training data, ~100 samples).
    X_explain : np.ndarray
        Samples to explain.
    feature_names : list
        Names of input features.

    Returns
    -------
    shap.Explanation object with computed SHAP values.
    """
    device = next(model.parameters()).device
    model.eval()

    def model_predict(x):
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32).to(device)
            logits = model(tensor)
            return torch.sigmoid(logits).cpu().numpy()

    explainer = shap.KernelExplainer(model_predict, X_background)
    shap_values = explainer.shap_values(X_explain)

    return shap_values


def plot_shap_summary(shap_values, X_explain, feature_names,
                      save_path="models/shap_summary.png"):
    """Generate and save SHAP summary (beeswarm) plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_explain,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {save_path}")


def plot_shap_bar(shap_values, feature_names,
                  save_path="models/shap_feature_importance.png"):
    """Generate and save SHAP bar plot showing mean feature importance."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        mean_abs_shap[sorted_idx],
        color="#2196F3"
    )
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Feature Importance (SHAP)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP feature importance plot saved to {save_path}")


def compute_integrated_gradients(model, X_sample, feature_names, baseline=None):
    """
    Compute Integrated Gradients attributions using Captum.

    Parameters
    ----------
    model : FraudDetector
        Trained PyTorch model.
    X_sample : np.ndarray
        Single sample or batch to explain (n_samples, n_features).
    feature_names : list
        Feature names for labeling.
    baseline : np.ndarray, optional
        Baseline input (defaults to zeros).

    Returns
    -------
    np.ndarray of attribution scores per feature.
    """
    device = next(model.parameters()).device
    model.eval()

    input_tensor = torch.tensor(X_sample, dtype=torch.float32).to(device)
    input_tensor.requires_grad_(True)

    if baseline is not None:
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32).to(device)
    else:
        baseline_tensor = torch.zeros_like(input_tensor).to(device)

    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_tensor, baselines=baseline_tensor)

    return attributions.detach().cpu().numpy()


def plot_integrated_gradients(attributions, feature_names, sample_idx=0,
                              save_path="models/integrated_gradients.png"):
    """Plot Integrated Gradients attributions for a single prediction."""
    if attributions.ndim > 1:
        attr = attributions[sample_idx]
    else:
        attr = attributions

    sorted_idx = np.argsort(np.abs(attr))
    colors = ["#FF5722" if v > 0 else "#2196F3" for v in attr[sorted_idx]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        attr[sorted_idx],
        color=colors
    )
    ax.set_xlabel("Attribution Score")
    ax.set_title("Integrated Gradients — Feature Attributions")
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Integrated Gradients plot saved to {save_path}")
