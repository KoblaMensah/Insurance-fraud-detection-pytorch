"""
Streamlit Dashboard for Insurance Fraud Detection

Interactive web application that lets users:
  - Input claim details and get a fraud prediction
  - View model confidence and risk factors
  - Explore feature importance via Integrated Gradients
  - Review overall model performance metrics

Run with: streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.model import FraudDetector
from src.preprocessing import DataPreprocessor
from src.data_generator import generate_claims_data
from captum.attr import IntegratedGradients


# ---- Page config ----
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("Insurance Fraud Detection System")
st.markdown("*Deep learning-powered fraud detection with explainable predictions*")


@st.cache_resource
def load_model():
    """Load the trained model and preprocessor."""
    model_path = project_root / "models" / "fraud_detector.pt"
    preprocessor_path = project_root / "models" / "preprocessor.pkl"

    if not model_path.exists() or not preprocessor_path.exists():
        st.error(
            "Model not found. Run `python run_pipeline.py` first to train the model."
        )
        st.stop()

    preprocessor = DataPreprocessor()
    preprocessor.load(str(preprocessor_path))

    input_dim = len(preprocessor.feature_names)
    model = FraudDetector(input_dim=input_dim)
    model.load_state_dict(torch.load(str(model_path), weights_only=True))
    model.eval()

    return model, preprocessor


def compute_attributions(model, input_tensor):
    """Compute Integrated Gradients for a single input."""
    model.eval()
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    baseline = torch.zeros_like(input_tensor)
    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_tensor, baselines=baseline)
    return attributions.detach().numpy().flatten()


# ---- Load model ----
model, preprocessor = load_model()

# ---- Sidebar: Claim Input ----
st.sidebar.header("Enter Claim Details")

policyholder_age = st.sidebar.slider("Policyholder Age", 18, 80, 35)
policy_tenure_months = st.sidebar.slider("Policy Tenure (months)", 1, 240, 24)
num_prior_claims = st.sidebar.slider("Number of Prior Claims", 0, 15, 1)
claim_amount = st.sidebar.number_input("Claim Amount ($)", 500, 150000, 15000)
claim_type = st.sidebar.selectbox(
    "Claim Type", ["Collision", "Theft", "Fire", "Vandalism", "Other"]
)
police_report_filed = st.sidebar.selectbox("Police Report Filed?", [1, 0],
                                            format_func=lambda x: "Yes" if x else "No")
witnesses_present = st.sidebar.selectbox("Witnesses Present?", [1, 0],
                                          format_func=lambda x: "Yes" if x else "No")
vehicle_age_years = st.sidebar.slider("Vehicle Age (years)", 0, 25, 5)
vehicle_value = st.sidebar.number_input("Vehicle Value ($)", 2000, 80000, 25000)
region = st.sidebar.selectbox("Region", ["Urban", "Suburban", "Rural"])
days_to_report = st.sidebar.slider("Days to Report Claim", 0, 90, 3)
claim_hour = st.sidebar.slider("Hour of Incident (24h)", 0, 23, 14)

# ---- Build input DataFrame ----
input_data = pd.DataFrame([{
    "policyholder_age": policyholder_age,
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
    "claim_to_value_ratio": round(claim_amount / vehicle_value, 4),
}])

# ---- Predict ----
if st.sidebar.button("Analyze Claim", type="primary"):
    X_input = preprocessor.transform(input_data)
    input_tensor = torch.tensor(X_input, dtype=torch.float32)

    with torch.no_grad():
        logits = model(input_tensor)
        fraud_prob = torch.sigmoid(logits).item()

    is_fraud = fraud_prob >= 0.5

    # ---- Results ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Fraud Probability", f"{fraud_prob:.1%}")
    with col2:
        st.metric("Risk Level",
                   "HIGH" if fraud_prob > 0.7 else "MEDIUM" if fraud_prob > 0.4 else "LOW")
    with col3:
        st.metric("Prediction", "FRAUDULENT" if is_fraud else "LEGITIMATE")

    # Risk gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fraud_prob * 100,
        title={"text": "Fraud Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#FF5722" if is_fraud else "#4CAF50"},
            "steps": [
                {"range": [0, 40], "color": "#E8F5E9"},
                {"range": [40, 70], "color": "#FFF3E0"},
                {"range": [70, 100], "color": "#FFEBEE"},
            ],
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ---- Explainability ----
    st.subheader("Feature Attributions (Integrated Gradients)")
    st.markdown(
        "Shows which features contributed most to this prediction. "
        "Red bars push toward fraud, blue bars push toward legitimate."
    )

    attributions = compute_attributions(model, input_tensor)
    feature_names = preprocessor.feature_names

    attr_df = pd.DataFrame({
        "Feature": feature_names,
        "Attribution": attributions
    }).sort_values("Attribution", key=abs, ascending=True)

    colors = ["#FF5722" if v > 0 else "#2196F3" for v in attr_df["Attribution"]]

    fig_attr = go.Figure(go.Bar(
        x=attr_df["Attribution"],
        y=attr_df["Feature"],
        orientation="h",
        marker_color=colors
    ))
    fig_attr.update_layout(
        title="Feature Attributions for This Prediction",
        xaxis_title="Attribution Score",
        height=400
    )
    st.plotly_chart(fig_attr, use_container_width=True)

    # ---- Input summary ----
    st.subheader("Claim Summary")
    st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

else:
    # ---- Default view: Dataset overview ----
    st.info("Enter claim details in the sidebar and click **Analyze Claim** to get a prediction.")

    st.subheader("Sample Data Overview")
    sample_df = generate_claims_data(n_samples=500, fraud_rate=0.12)

    col1, col2 = st.columns(2)

    with col1:
        fig_dist = px.histogram(
            sample_df, x="claim_amount", color="is_fraud",
            nbins=40, title="Claim Amount Distribution by Fraud Status",
            labels={"is_fraud": "Is Fraud"},
            color_discrete_map={0: "#2196F3", 1: "#FF5722"}
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        fig_type = px.histogram(
            sample_df, x="claim_type", color="is_fraud",
            title="Claim Type Distribution",
            labels={"is_fraud": "Is Fraud"},
            color_discrete_map={0: "#2196F3", 1: "#FF5722"}
        )
        st.plotly_chart(fig_type, use_container_width=True)

    st.dataframe(sample_df.head(20), use_container_width=True)
