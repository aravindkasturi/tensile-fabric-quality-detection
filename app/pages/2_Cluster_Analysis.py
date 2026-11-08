
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cluster Analysis",
    page_icon="🔬",
    layout="wide"
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_DIR / "results" / "final_mechanical_features.csv"
)

PREDICTION_FILE = (
    PROJECT_DIR / "results" / "final_cluster_predictions.csv"
)

GRAPH_FILE = (
    PROJECT_DIR / "results" / "final_analysis_graphs"
    / "cluster_profiles.png"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

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

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔬 Cluster Analysis")

st.write(
    "Explore the two mechanical behaviour groups identified "
    "by the K-Means clustering model."
)

st.info(
    "Cluster 0 — Lower Mechanical Response.   "
    " Cluster 1 — Higher Mechanical Response"
)

# --------------------------------------------------
# CLUSTER COUNTS
# --------------------------------------------------

cluster_0 = data[data["Cluster"] == 0]
cluster_1 = data[data["Cluster"] == 1]

col1, col2, col3 = st.columns(3)

col1.metric("Total Samples", len(data))
col2.metric("Cluster 0", len(cluster_0))
col3.metric("Cluster 1", len(cluster_1))

# --------------------------------------------------
# SAMPLE ASSIGNMENTS
# --------------------------------------------------

st.divider()
st.subheader("Sample Cluster Assignments")

st.dataframe(
    data[["Sample", "Cluster"]],
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# CLUSTER PROFILES
# --------------------------------------------------

st.divider()
st.subheader("Average Mechanical Properties")

features = [
    "Maximum_Stress_MPa",
    "Maximum_Strain_Percent",
    "Strain_At_Max_Stress_Percent",
    "Initial_Stiffness_MPa_per_Strain",
    "Energy_Absorption_MJ_per_m3",
    "Largest_Post_Peak_Drop_Percent"
]

available_features = [
    feature for feature in features
    if feature in data.columns
]

profile = (
    data.groupby("Cluster")[available_features]
    .mean()
    .T
)

st.dataframe(
    profile.round(4),
    use_container_width=True
)

# --------------------------------------------------
# CLUSTER COMPARISON CHART
# --------------------------------------------------

st.subheader("Cluster Feature Comparison")

st.bar_chart(profile)

# --------------------------------------------------
# EXISTING CLUSTER PROFILE GRAPH
# --------------------------------------------------

st.divider()
st.subheader("Standardized Cluster Profiles")

if GRAPH_FILE.exists():
    st.image(
        str(GRAPH_FILE),
        caption="Mechanical feature profiles - Final four-feature model",
        use_container_width=True
    )
else:
    st.warning("Cluster profile graph not found.")
# --------------------------------------------------
# INDIVIDUAL CLUSTER DETAILS
# --------------------------------------------------

st.divider()
st.subheader("Explore Individual Clusters")

selected_cluster = st.selectbox(
    "Select Cluster",
    [0, 1]
)

selected_data = data[
    data["Cluster"] == selected_cluster
]

st.write(f"Samples belonging to Cluster {selected_cluster}")

st.dataframe(
    selected_data[
        ["Sample"] + available_features
    ],
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

st.download_button(
    "Download Cluster Analysis CSV",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="tensile_cluster_analysis.csv",
    mime="text/csv"
)
