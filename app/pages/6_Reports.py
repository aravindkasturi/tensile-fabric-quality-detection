
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Reports",
    page_icon="📑",
    layout="wide"
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_DIR / "results"

FEATURE_FILE = RESULTS_DIR / "final_mechanical_features.csv"
CLUSTER_FILE = RESULTS_DIR / "final_cluster_predictions.csv"
COMPARISON_FILE = RESULTS_DIR / "sample_comparative_assessment.csv"

GRAPH_DIRS = [
    RESULTS_DIR / "graphs",
    RESULTS_DIR / "eda_graphs",
    RESULTS_DIR / "final_analysis_graphs"
]

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📑 Reports and Downloads")

st.write(
    "Access, review and download tensile fabric analysis "
    "results, cluster assignments and generated visualizations."
)

# --------------------------------------------------
# DATA REPORTS
# --------------------------------------------------

st.header("Available Data Reports")

report_files = [
    ("Final Mechanical Features", FEATURE_FILE),
    ("Final Cluster Predictions", CLUSTER_FILE),
    ("Comparative Sample Assessment", COMPARISON_FILE),
    ("Anomaly Detection Results", RESULTS_DIR / "anomaly_detection_results.csv"),
    ("Rupture Detection Results", RESULTS_DIR / "rupture_detection_results.csv")
]

for report_name, report_path in report_files:

    with st.container(border=True):

        col1, col2 = st.columns([3, 1])

        col1.write(f"**{report_name}**")

        if report_path.exists():

            report_data = pd.read_csv(report_path)

            col1.caption(
                f"{len(report_data)} records | {report_path.name}"
            )

            col2.download_button(
                "Download CSV",
                data=report_data.to_csv(index=False).encode("utf-8"),
                file_name=report_path.name,
                mime="text/csv",
                key=f"download_{report_path.name}"
            )

        else:
            col1.warning("Report not available.")

# --------------------------------------------------
# VISUAL REPORTS
# --------------------------------------------------

st.divider()
st.header("Generated Visualizations")

image_extensions = [".png", ".jpg", ".jpeg"]

for graph_dir in GRAPH_DIRS:

    if not graph_dir.exists():
        continue

    graph_files = sorted(
        path for path in graph_dir.iterdir()
        if path.suffix.lower() in image_extensions
    )

    if graph_files:

        st.subheader(graph_dir.name.replace("_", " ").title())

        for graph_file in graph_files:

            with st.expander(graph_file.stem.replace("_", " ").title()):

                st.image(
                    str(graph_file),
                    use_container_width=True
                )

# --------------------------------------------------
# PROJECT NOTE
# --------------------------------------------------

st.divider()

st.info(
    "Cluster assignments are exploratory mechanical groupings. "
    "They are not validated GOOD/BAD quality classifications."
)

st.caption(
    "Tensile Fabric Quality Detection | Reports and Analysis"
)
