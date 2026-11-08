
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="📊",
    layout="wide"
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
FEATURE_FILE = PROJECT_DIR / "results" / "final_mechanical_features.csv"
PREDICTION_FILE = PROJECT_DIR / "results" / "final_cluster_predictions.csv"

# Load data
@st.cache_data
def load_data():
    features = pd.read_csv(FEATURE_FILE)
    predictions = pd.read_csv(PREDICTION_FILE)
    return features.merge(predictions, on="Sample")

try:
    data = load_data()
except FileNotFoundError as error:
    st.error(f"Required file not found: {error}")
    st.stop()

# Page heading
st.title("📊 Dataset Explorer")
st.write("Explore all tensile test samples and their mechanical properties.")

# Sample selection
selected_sample = st.selectbox(
    "Select a tensile test sample",
    data["Sample"].tolist()
)

sample = data[data["Sample"] == selected_sample].iloc[0]

# Main metrics
st.divider()
st.subheader(f"Sample Analysis: {selected_sample}")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Maximum Stress",
    f'{sample["Maximum_Stress_MPa"]:.4f} MPa'
)

col2.metric(
    "Maximum Strain",
    f'{sample["Maximum_Strain_Percent"]:.2f}%'
)

col3.metric(
    "Initial Stiffness",
    f'{sample["Initial_Stiffness_MPa_per_Strain"]:.4f}'
)

col4.metric(
    "Energy Absorption",
    f'{sample["Energy_Absorption_MJ_per_m3"]:.4f}'
)

# Cluster information
st.subheader("Machine Learning Cluster")

cluster = int(sample["Cluster"])

st.success(f"Assigned Cluster: {cluster}")

st.caption(
    "Cluster membership represents a relative mechanical pattern, "
    "Cluster 0 — Lower Mechanical Response, "
    "Cluster 1 — Higher Mechanical Response"
)

# Feature details
st.subheader("Mechanical Properties")

display_columns = [
    "Maximum_Force_N",
    "Maximum_Stress_MPa",
    "Strain_At_Max_Stress_Percent",
    "Maximum_Strain_Percent",
    "Initial_Stiffness_MPa_per_Strain",
    "Energy_Absorption_MJ_per_m3",
    "Largest_Post_Peak_Drop_MPa",
    "Largest_Post_Peak_Drop_Percent",
    "Force_Collapse_Detected",
    "Force_Collapse_Displacement_mm"
]

available_columns = [
    column for column in display_columns
    if column in data.columns
]

property_table = pd.DataFrame({
    "Property": available_columns,
    "Value": [sample[column] for column in available_columns]
})

st.dataframe(
    property_table,
    use_container_width=True,
    hide_index=True
)

# Complete dataset
st.divider()
st.subheader("Complete Dataset")

st.dataframe(
    data,
    use_container_width=True,
    hide_index=True
)

# Download
st.download_button(
    "Download Dataset CSV",
    data.to_csv(index=False).encode("utf-8"),
    file_name="tensile_fabric_dataset.csv",
    mime="text/csv"
)
