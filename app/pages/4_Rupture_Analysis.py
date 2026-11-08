
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Rupture Analysis",
    page_icon="💥",
    layout="wide"
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

RUPTURE_FILE = (
    PROJECT_DIR / "results" / "rupture_detection_results.csv"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(RUPTURE_FILE)

st.title("💥 Rupture Analysis")

st.write(
    "Analyse possible rupture behaviour using force-drop "
    "measurements from the tensile tests."
)

st.info(
    "The automated detector identifies possible rupture patterns. "
    "These results do not independently confirm physical fracture."
)

try:
    data = load_data()
except FileNotFoundError:
    st.error("Rupture detection results were not found.")
    st.stop()

required_columns = [
    "Sample",
    "Rupture_Detected",
    "Rupture_Displacement_mm",
    "Force_Before_Rupture_N",
    "Force_After_Rupture_N",
    "Force_Drop_Percent"
]

missing = [
    col for col in required_columns
    if col not in data.columns
]

if missing:
    st.error(f"Missing columns in rupture results: {missing}")
    st.stop()

# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def is_rupture(value):
    return str(value).strip().lower() in [
        "yes", "true", "1"
    ]

data["Rupture_Detected"] = data["Rupture_Detected"].apply(
    is_rupture
)

numeric_columns = [
    "Rupture_Displacement_mm",
    "Force_Before_Rupture_N",
    "Force_After_Rupture_N",
    "Force_Drop_Percent"
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

data["Force_Drop_Percent"] = (
    data["Force_Drop_Percent"].clip(0, 100)
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

rupture_count = int(data["Rupture_Detected"].sum())
no_rupture_count = len(data) - rupture_count

st.divider()

c1, c2, c3 = st.columns(3)

c1.metric("Total Samples", len(data))
c2.metric("Possible Rupture Detected", rupture_count)
c3.metric("No Rupture Detected", no_rupture_count)

# --------------------------------------------------
# RUPTURE RESULTS
# --------------------------------------------------

st.divider()
st.subheader("Sample-Level Rupture Results")

st.dataframe(
    data,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# RUPTURE DISTRIBUTION
# --------------------------------------------------

st.subheader("Rupture Detection Distribution")

distribution = pd.DataFrame({
    "Detection Status": [
        "Possible Rupture",
        "No Rupture Detected"
    ],
    "Samples": [
        rupture_count,
        no_rupture_count
    ]
})

st.bar_chart(
    distribution,
    x="Detection Status",
    y="Samples"
)

# --------------------------------------------------
# INDIVIDUAL SAMPLE ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("Individual Rupture Analysis")

selected_sample = st.selectbox(
    "Select Sample",
    data["Sample"].tolist()
)

sample = data[
    data["Sample"] == selected_sample
].iloc[0]

if sample["Rupture_Detected"]:
    st.warning("Possible rupture detected by the automated rule.")
else:
    st.success("No rupture detected by the current automated rule.")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Rupture Displacement",
    (
        f'{sample["Rupture_Displacement_mm"]:.3f} mm'
        if pd.notna(sample["Rupture_Displacement_mm"])
        else "Not detected"
    )
)

col2.metric(
    "Force Before Rupture",
    (
        f'{sample["Force_Before_Rupture_N"]:.4f} N'
        if pd.notna(sample["Force_Before_Rupture_N"])
        else "Not available"
    )
)

col3.metric(
    "Force Drop",
    (
        f'{sample["Force_Drop_Percent"]:.2f}%'
        if pd.notna(sample["Force_Drop_Percent"])
        else "Not available"
    )
)

st.caption(
    "The current detection rule uses a sustained force drop "
    "of at least 80% over five readings."
)

# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

st.divider()

st.download_button(
    "Download Rupture Analysis CSV",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="rupture_analysis_report.csv",
    mime="text/csv"
)
