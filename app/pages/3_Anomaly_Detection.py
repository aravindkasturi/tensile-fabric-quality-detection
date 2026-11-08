
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🚨",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parents[2]

ANOMALY_FILE = (
    PROJECT_DIR / "results" / "anomaly_detection_results.csv"
)

GRAPH_DIR = PROJECT_DIR / "results" / "eda_graphs"

@st.cache_data
def load_data():
    return pd.read_csv(ANOMALY_FILE)

st.title("🚨 Anomaly Detection")

st.write(
    "Identify samples that show statistically unusual "
    "mechanical patterns using Isolation Forest."
)

st.info(
    "A potential anomaly is not necessarily a defective or BAD sample. "
    "It only indicates unusual behaviour relative to this dataset."
)

try:
    data = load_data()
except FileNotFoundError:
    st.error("Anomaly detection results were not found.")
    st.stop()

if not {"Sample", "Anomaly_Status", "Anomaly_Score"}.issubset(data.columns):
    st.error("The anomaly results file has an unexpected structure.")
    st.stop()

data["Anomaly_Score"] = pd.to_numeric(
    data["Anomaly_Score"], errors="coerce"
)

potential = data[data["Anomaly_Status"] == "Potential Anomaly"]
typical = data[data["Anomaly_Status"] == "Typical Pattern"]

# Metrics
st.divider()

c1, c2, c3 = st.columns(3)

c1.metric("Total Samples", len(data))
c2.metric("Typical Pattern", len(typical))
c3.metric("Potential Anomalies", len(potential))

# Results table
st.divider()
st.subheader("Anomaly Detection Results")

st.dataframe(
    data.sort_values("Anomaly_Score"),
    use_container_width=True,
    hide_index=True
)

# Selected sample
st.subheader("Individual Sample Analysis")

selected_sample = st.selectbox(
    "Select Sample",
    data["Sample"].tolist()
)

sample = data[data["Sample"] == selected_sample].iloc[0]

col1, col2 = st.columns(2)

col1.metric("Anomaly Status", sample["Anomaly_Status"])
col2.metric("Anomaly Score", f'{sample["Anomaly_Score"]:.6f}')

if sample["Anomaly_Status"] == "Potential Anomaly":
    st.warning("This sample requires further investigation.")
else:
    st.success("This sample follows the typical pattern identified.")

# Graphs
st.divider()
st.subheader("Anomaly Visualizations")

for graph_name, caption in [
    ("anomaly_scores.png", "Isolation Forest anomaly scores"),
    ("anomaly_distribution.png", "Anomaly distribution")
]:
    graph_path = GRAPH_DIR / graph_name

    if graph_path.exists():
        st.image(
            str(graph_path),
            caption=caption,
            use_container_width=True
        )
    else:
        st.warning(f"{graph_name} not found.")

# Download
st.download_button(
    "Download Anomaly Report",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="anomaly_detection_report.csv",
    mime="text/csv"
)
