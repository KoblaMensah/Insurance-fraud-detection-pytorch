# Insurance Fraud Detection with PyTorch

A deep learning-based system for detecting fraudulent insurance claims. Built with PyTorch, this project covers the full ML lifecycle: data preprocessing, model training, evaluation, explainability (SHAP and Integrated Gradients), and an interactive Streamlit dashboard.

**[Live Demo](https://insurance-fraud-detection-pytorch-kobla-mensah.streamlit.app)** — Try the deployed dashboard

## Features

- **Synthetic Data Generation** — Realistic insurance claims data with configurable fraud rates and domain-informed feature distributions
- **Automated Preprocessing** — Categorical encoding, feature scaling, stratified splitting, and SMOTE oversampling for class imbalance
- **PyTorch Neural Network** — 3-layer feedforward architecture with batch normalization, dropout, and early stopping
- **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1, AUC-ROC with confusion matrix and ROC curve visualizations
- **Dual Explainability** — SHAP (model-agnostic) and Integrated Gradients (gradient-based) for transparent predictions
- **Interactive Dashboard** — [Deployed Streamlit app](https://insurance-fraud-detection-pytorch-kobla-mensah.streamlit.app) with real-time fraud scoring, risk gauge, and feature attribution plots

## Quick Start

```bash
# Clone the repository
git clone https://github.com/KoblaMensah/Insurance-fraud-detection-pytorch.git
cd Insurance-fraud-detection-pytorch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python run_pipeline.py

# Launch the dashboard
streamlit run app/dashboard.py
```

## Project Structure

```
├── run_pipeline.py            # Main entry point — end-to-end execution
├── src/
│   ├── data_generator.py      # Synthetic claims data generation
│   ├── preprocessing.py       # Encoding, scaling, SMOTE, train/test split
│   ├── dataset.py             # PyTorch Dataset and DataLoader
│   ├── model.py               # FraudDetector neural network
│   ├── train.py               # Training loop with early stopping
│   ├── evaluate.py            # Metrics and visualization
│   └── explain.py             # SHAP and Integrated Gradients
├── app/
│   └── dashboard.py           # Streamlit interactive dashboard
├── tests/
│   └── test_model.py          # Unit tests
├── PRD.md                     # Product requirements document
└── documentation.txt          # Full technical documentation
```

## Model Architecture

```
Input (13 features)
    │
    ├── Linear(13, 64) + BatchNorm + ReLU + Dropout(0.3)
    ├── Linear(64, 32) + BatchNorm + ReLU + Dropout(0.3)
    ├── Linear(32, 16) + BatchNorm + ReLU + Dropout(0.15)
    └── Linear(16, 1) → Sigmoid → Fraud Probability
```

The network is intentionally moderate — deep enough to capture nonlinear patterns in tabular data, but not so deep that it overfits on a 5,000-sample dataset. Batch normalization stabilizes training, and dropout prevents over-reliance on any single feature path.

## Dataset Features

| Feature | Description |
|---------|-------------|
| policyholder_age | Age of the policyholder |
| policy_tenure_months | Duration of the insurance policy |
| num_prior_claims | Number of previously filed claims |
| claim_amount | Dollar amount of the current claim |
| claim_type | Collision, Theft, Fire, Vandalism, or Other |
| police_report_filed | Whether a police report was filed |
| witnesses_present | Whether witnesses were present |
| vehicle_age_years | Age of the insured vehicle |
| vehicle_value | Estimated market value of the vehicle |
| region | Urban, Suburban, or Rural |
| days_to_report | Days between incident and claim filing |
| claim_hour | Hour of day when the incident occurred |
| claim_to_value_ratio | Claim amount relative to vehicle value |

## Explainability

The system provides two complementary explanation methods:

**SHAP (SHapley Additive exPlanations)** — Game-theory based approach that assigns each feature a contribution to the prediction. Model-agnostic and widely recognized in industry.

**Integrated Gradients** — PyTorch-native gradient-based attribution via Facebook's Captum library. Faster than SHAP and satisfies key mathematical axioms (sensitivity and implementation invariance).

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Data Processing | pandas, scikit-learn |
| Class Balancing | imbalanced-learn (SMOTE) |
| Explainability | SHAP, Captum |
| Dashboard | Streamlit, Plotly |
| Visualization | matplotlib, seaborn |
| Deployment | Streamlit Community Cloud |

## Live Dashboard

The fraud detection dashboard is deployed and publicly accessible:

**https://insurance-fraud-detection-pytorch-kobla-mensah.streamlit.app**

Enter claim details in the sidebar, click **Analyze Claim**, and get an instant fraud risk score with feature-level explanations showing which factors drove the prediction.

## Running Tests

```bash
python tests/test_model.py
```

## Documentation

- [PRD.md](PRD.md) — Product requirements, objectives, and technical architecture
- [documentation.txt](documentation.txt) — Full technical documentation with design rationale

## License

MIT
