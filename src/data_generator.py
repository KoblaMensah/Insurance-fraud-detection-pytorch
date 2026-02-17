"""
Synthetic Insurance Claims Data Generator

Generates realistic insurance claims data with controllable fraud rates.
Features are designed to mirror real-world patterns observed in auto
insurance fraud — for example, newer policyholders and claims without
police reports correlate with higher fraud likelihood.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_claims_data(n_samples=5000, fraud_rate=0.12, random_state=42):
    """
    Generate synthetic insurance claims dataset.

    Parameters
    ----------
    n_samples : int
        Number of claims to generate.
    fraud_rate : float
        Proportion of fraudulent claims (0.0 to 1.0).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Generated claims data with features and fraud label.
    """
    rng = np.random.RandomState(random_state)

    # Determine fraud labels first — we'll bias features based on this
    is_fraud = rng.binomial(1, fraud_rate, n_samples)

    # ---- Policyholder features ----
    age = np.where(
        is_fraud,
        rng.normal(32, 6, n_samples),   # Fraudsters tend younger
        rng.normal(45, 12, n_samples)
    ).clip(18, 80).astype(int)

    policy_tenure_months = np.where(
        is_fraud,
        rng.exponential(8, n_samples),   # Shorter tenure
        rng.exponential(36, n_samples)
    ).clip(1, 240).astype(int)

    num_prior_claims = np.where(
        is_fraud,
        rng.poisson(2.5, n_samples),
        rng.poisson(0.8, n_samples)
    ).clip(0, 15)

    # ---- Claim features ----
    claim_amount = np.where(
        is_fraud,
        rng.lognormal(9.5, 0.8, n_samples),  # Higher amounts
        rng.lognormal(8.2, 1.0, n_samples)
    ).clip(500, 150000).round(2)

    police_report_filed = np.where(
        is_fraud,
        rng.binomial(1, 0.25, n_samples),  # Less likely to file
        rng.binomial(1, 0.72, n_samples)
    )

    witnesses_present = np.where(
        is_fraud,
        rng.binomial(1, 0.15, n_samples),
        rng.binomial(1, 0.55, n_samples)
    )

    # ---- Vehicle features ----
    vehicle_age_years = np.where(
        is_fraud,
        rng.normal(12, 4, n_samples),   # Older vehicles
        rng.normal(6, 3, n_samples)
    ).clip(0, 25).astype(int)

    vehicle_value = np.where(
        is_fraud,
        rng.lognormal(9.0, 0.6, n_samples),
        rng.lognormal(9.8, 0.5, n_samples)
    ).clip(2000, 80000).round(2)

    # ---- Categorical features ----
    claim_types = ["Collision", "Theft", "Fire", "Vandalism", "Other"]
    fraud_claim_probs = [0.15, 0.35, 0.25, 0.15, 0.10]
    legit_claim_probs = [0.45, 0.15, 0.08, 0.12, 0.20]

    claim_type = np.where(
        is_fraud,
        rng.choice(claim_types, n_samples, p=fraud_claim_probs),
        rng.choice(claim_types, n_samples, p=legit_claim_probs)
    )

    regions = ["Urban", "Suburban", "Rural"]
    region = rng.choice(regions, n_samples, p=[0.45, 0.35, 0.20])

    # ---- Time features ----
    days_to_report = np.where(
        is_fraud,
        rng.exponential(18, n_samples),   # Slower reporting
        rng.exponential(4, n_samples)
    ).clip(0, 90).astype(int)

    claim_hour = np.where(
        is_fraud,
        rng.normal(2, 4, n_samples) % 24,   # Late night claims
        rng.normal(14, 5, n_samples) % 24
    ).astype(int)

    # ---- Derived features ----
    claim_to_value_ratio = (claim_amount / vehicle_value).round(4)

    # ---- Assemble DataFrame ----
    df = pd.DataFrame({
        "policyholder_age": age,
        "policy_tenure_months": policy_tenure_months,
        "num_prior_claims": num_prior_claims,
        "claim_amount": claim_amount,
        "claim_type": claim_type,
        "police_report_filed": police_report_filed,
        "witnesses_present": witnesses_present,
        "vehicle_age_years": vehicle_age_years,
        "vehicle_value": vehicle_value,
        "region": region,
        "days_to_report": days_to_report,
        "claim_hour": claim_hour,
        "claim_to_value_ratio": claim_to_value_ratio,
        "is_fraud": is_fraud,
    })

    return df


def save_dataset(df, output_dir="data/raw", filename="insurance_claims.csv"):
    """Save generated dataset to CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath} ({len(df)} records)")
    return filepath


if __name__ == "__main__":
    print("Generating synthetic insurance claims data...")
    df = generate_claims_data(n_samples=5000, fraud_rate=0.12)

    print(f"\nDataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.1%}")
    print(f"\nFeature summary:")
    print(df.describe().round(2))
    print(f"\nClaim type distribution:\n{df['claim_type'].value_counts()}")

    save_dataset(df)
