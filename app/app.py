
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Tensile Fabric Analysis",
    page_icon="🧵",
    layout="wide"
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = PROJECT_DIR / "results" / "final_mechanical_features.csv"
PREDICTION_FILE = PROJECT_DIR / "results" / "final_cluster_predictions.csv"
MODEL_FILE = PROJECT_DIR / "model" / "final_tensile_cluster_model.pkl"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    features = pd.read_csv(FEATURE_FILE)
    predictions = pd.read_csv(PREDICTION_FILE)
    return features.merge(predictions, on="Sample")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

try:
    data = load_data()
    model_bundle = load_model()
except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.stop()

# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title("🧵 Tensile Fabric Analysis")
st.caption("PCL Non-Woven Fabric | Mechanical Analysis and ML Clustering")

st.write(
    "Explore tensile test measurements, mechanical behaviour, "
    "machine-learning clusters, anomaly patterns and rupture analysis."
)

st.info(
    "Cluster 0 — Lower Mechanical Response,  "
    "Cluster 1 — Higher Mechanical Response"
)

st.divider()

# --------------------------------------------------
# KEY METRICS
# --------------------------------------------------

total_samples = len(data)
cluster_0 = int((data["Cluster"] == 0).sum())
cluster_1 = int((data["Cluster"] == 1).sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Samples", total_samples)
col2.metric("Cluster 0", cluster_0)
col3.metric("Cluster 1", cluster_1)
col4.metric(
    "Silhouette Score",
    f'{model_bundle["silhouette_score"]:.4f}'
)

st.divider()

# --------------------------------------------------
# CLUSTER DISTRIBUTION
# --------------------------------------------------

st.subheader("Cluster Distribution")

distribution = (
    data["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Samples")
)

st.bar_chart(
    distribution,
    x="Cluster",
    y="Samples"
)

# --------------------------------------------------
# MECHANICAL PROPERTY OVERVIEW
# --------------------------------------------------

st.subheader("Mechanical Property Overview")

property_columns = [
    "Maximum_Stress_MPa",
    "Maximum_Strain_Percent",
    "Energy_Absorption_MJ_per_m3",
    "Largest_Post_Peak_Drop_Percent"
]

available_columns = [
    col for col in property_columns if col in data.columns
]

st.dataframe(
    data[["Sample", "Cluster"] + available_columns],
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Tensile Fabric Quality Detection | Exploratory "
    "Mechanical Analysis and Unsupervised Machine Learning"
)

