"""
Main Pipeline — End-to-End Fraud Detection

Run this script to execute the full pipeline:
  1. Generate synthetic data
  2. Preprocess and split
  3. Train the model
  4. Evaluate performance
  5. Generate explanations

Usage:
    python run_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generator import generate_claims_data, save_dataset
from src.preprocessing import DataPreprocessor
from src.dataset import create_dataloaders
from src.model import FraudDetector, get_model_summary
from src.train import train_model
from src.evaluate import (
    evaluate_model, plot_training_history,
    plot_confusion_matrix, plot_roc_curve
)
from src.explain import (
    compute_integrated_gradients, plot_integrated_gradients
)


def main():
    print("=" * 60)
    print("INSURANCE FRAUD DETECTION PIPELINE")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/5] Generating synthetic claims data...")
    df = generate_claims_data(n_samples=5000, fraud_rate=0.12)
    save_dataset(df)
    print(f"  Fraud rate: {df['is_fraud'].mean():.1%}")

    # Step 2: Preprocess
    print("\n[2/5] Preprocessing data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.fit_transform(df)
    preprocessor.save()
    train_loader, test_loader = create_dataloaders(data, batch_size=64)

    # Step 3: Train model
    print("\n[3/5] Training FraudDetector model...")
    input_dim = data["X_train"].shape[1]
    model = FraudDetector(input_dim=input_dim, dropout_rate=0.3)
    get_model_summary(model, input_dim)

    history = train_model(
        model, train_loader, test_loader,
        epochs=100, lr=1e-3, patience=15
    )

    # Step 4: Evaluate
    print("\n[4/5] Evaluating model...")
    results = evaluate_model(model, test_loader)
    plot_training_history(history)
    plot_confusion_matrix(results["labels"], results["predictions"])
    plot_roc_curve(results["labels"], results["probabilities"])

    # Step 5: Explainability
    print("\n[5/5] Computing feature attributions...")
    # Integrated Gradients (fast, no background data needed)
    sample_data = data["X_test"][:10]
    attributions = compute_integrated_gradients(
        model, sample_data, data["feature_names"]
    )
    plot_integrated_gradients(attributions, data["feature_names"])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nKey metrics:")
    for metric, value in results["metrics"].items():
        print(f"  {metric:>12s}: {value:.4f}")
    print(f"\nArtifacts saved to models/")
    print(f"Run the dashboard: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
