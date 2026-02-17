# Product Requirements Document (PRD)
## Insurance Fraud Detection System

### 1. Overview

**Project:** Deep Learning-Based Insurance Fraud Detection
**Author:** Kobla Mensah
**Date:** February 2026
**Status:** Active Development

This system uses a PyTorch neural network to identify potentially fraudulent insurance claims. It combines data preprocessing, supervised deep learning, model explainability, and an interactive dashboard to deliver actionable fraud predictions.

### 2. Problem Statement

Insurance fraud costs the U.S. insurance industry over $80 billion annually (FBI estimates). Manual review of claims is time-consuming, inconsistent, and fails to detect sophisticated fraud patterns. There is a clear need for automated, explainable fraud detection systems that can flag suspicious claims while providing transparent reasoning for each decision.

### 3. Objectives

| Objective | Success Metric |
|-----------|---------------|
| Detect fraudulent claims with high recall | Recall > 80% on test set |
| Minimize false positives | Precision > 70% |
| Provide explainable predictions | SHAP and IG attributions for every prediction |
| Interactive demo for stakeholders | Streamlit dashboard with real-time predictions |
| Reproducible results | End-to-end pipeline with fixed random seeds |

### 4. Target Users

- **Insurance Claims Adjusters:** Use fraud scores to prioritize manual reviews.
- **Compliance Officers:** Review explainability outputs to ensure fair decision-making.
- **Data Science Teams:** Extend or retrain the model with production data.
- **Stakeholders/Executives:** Use the dashboard for high-level fraud analytics.

### 5. Functional Requirements

#### 5.1 Data Pipeline
- Generate synthetic insurance claims data with configurable fraud rates
- Support for 13 features across policyholder, claim, vehicle, and temporal categories
- Automated categorical encoding, feature scaling, and train/test splitting
- SMOTE oversampling to handle class imbalance in training data

#### 5.2 Model
- PyTorch feedforward neural network with batch normalization and dropout
- Binary classification with BCE loss and sigmoid output
- Early stopping and learning rate scheduling to prevent overfitting
- Model checkpoint saving and loading for deployment

#### 5.3 Evaluation
- Standard classification metrics: accuracy, precision, recall, F1, AUC-ROC
- Confusion matrix and ROC curve visualizations
- Training/validation loss curves for convergence analysis

#### 5.4 Explainability
- SHAP (SHapley Additive exPlanations) for global feature importance
- Integrated Gradients (Captum) for per-prediction attributions
- Visual attribution plots for each prediction

#### 5.5 Dashboard
- Streamlit web interface with sidebar input controls
- Real-time fraud probability with risk gauge visualization
- Per-prediction feature attribution chart
- Dataset exploration with distribution visualizations

### 6. Non-Functional Requirements

- **Reproducibility:** All random operations seeded for deterministic results
- **Modularity:** Each component (data, model, eval, explain) is independently testable
- **Performance:** Prediction latency under 100ms per claim
- **Portability:** Runs on CPU; GPU optional for training acceleration

### 7. Technical Architecture

```
[Data Generation] → [Preprocessing] → [PyTorch DataLoader]
                                            ↓
                                      [FraudDetector NN]
                                            ↓
                                  [Evaluation + Metrics]
                                            ↓
                              [SHAP / Integrated Gradients]
                                            ↓
                                  [Streamlit Dashboard]
```

### 8. Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Deep Learning | PyTorch | Industry standard, dynamic computation graphs, research-friendly |
| Data Processing | pandas, scikit-learn | Mature ecosystem, well-documented |
| Class Balancing | imbalanced-learn (SMOTE) | Proven technique for imbalanced datasets |
| Explainability | SHAP, Captum | Complementary approaches (model-agnostic + gradient-based) |
| Dashboard | Streamlit | Rapid prototyping, Python-native, low overhead |
| Visualization | matplotlib, seaborn, plotly | Static (reports) + interactive (dashboard) |

### 9. Data Schema

| Feature | Type | Description |
|---------|------|-------------|
| policyholder_age | int | Age of policyholder (18-80) |
| policy_tenure_months | int | Months since policy started |
| num_prior_claims | int | Number of previous claims filed |
| claim_amount | float | Dollar amount of current claim |
| claim_type | categorical | Collision, Theft, Fire, Vandalism, Other |
| police_report_filed | binary | Whether a police report was filed |
| witnesses_present | binary | Whether witnesses were present |
| vehicle_age_years | int | Age of the insured vehicle |
| vehicle_value | float | Estimated value of vehicle |
| region | categorical | Urban, Suburban, Rural |
| days_to_report | int | Days between incident and claim filing |
| claim_hour | int | Hour of day when incident occurred |
| claim_to_value_ratio | float | Derived: claim_amount / vehicle_value |
| is_fraud | binary | Target variable (0 = legitimate, 1 = fraud) |

### 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Synthetic data may not capture real fraud patterns | Model may underperform on production data | Designed features based on published fraud research; architecture supports retraining on real data |
| Class imbalance (12% fraud) | Model biased toward majority class | SMOTE oversampling + stratified splitting |
| Neural network as black box | Lack of trust from stakeholders | Dual explainability (SHAP + Integrated Gradients) |
| Overfitting on small dataset | Poor generalization | Dropout, batch normalization, early stopping, weight decay |

### 11. Future Enhancements

- Integration with real claims databases via API endpoints
- Temporal fraud pattern detection using LSTM/Transformer layers
- Automated retraining pipeline with MLflow tracking
- Role-based access control for the dashboard
- A/B testing framework for model comparison
