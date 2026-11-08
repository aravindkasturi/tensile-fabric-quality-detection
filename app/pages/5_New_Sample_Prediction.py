
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="New Sample Prediction",
    page_icon="🧪",
    layout="wide"
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_DIR / "model" / "final_tensile_cluster_model.pkl"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

try:
    model_bundle = load_model()
except FileNotFoundError:
    st.error("Final clustering model not found.")
    st.stop()

pipeline = model_bundle["pipeline"]
feature_names = model_bundle["features"]

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🧪 New Sample Cluster Prediction")

st.write(
    "Upload a new tensile test CSV to calculate mechanical "
    "properties and predict its machine-learning cluster."
)

st.info(
    "The model assigns Cluster 0 or Cluster 1 based on "
    "relative mechanical behaviour. These are not GOOD/BAD labels."
)

# --------------------------------------------------
# SAMPLE DIMENSIONS
# --------------------------------------------------

st.sidebar.header("Sample Dimensions")

gauge_length = st.sidebar.number_input(
    "Gauge Length (mm)",
    min_value=0.01,
    value=10.0,
    step=0.5
)

sample_width = st.sidebar.number_input(
    "Sample Width (mm)",
    min_value=0.01,
    value=4.7,
    step=0.1
)

sample_thickness = st.sidebar.number_input(
    "Sample Thickness (mm)",
    min_value=0.01,
    value=1.0,
    step=0.1
)

cross_sectional_area = sample_width * sample_thickness

st.sidebar.info(
    f"Cross-sectional Area: {cross_sectional_area:.3f} mm²"
)

# --------------------------------------------------
# UPLOAD CSV
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Tensile Test CSV",
    type=["csv"],
    help="CSV must contain Force_N and Displacement_mm."
)

if uploaded_file is None:
    st.warning("Upload a tensile test CSV to begin.")
    st.stop()

# --------------------------------------------------
# PROCESS CSV
# --------------------------------------------------

try:
    raw_data = pd.read_csv(uploaded_file)

    required_columns = [
        "Force_N",
        "Displacement_mm"
    ]

    missing = [
        col for col in required_columns
        if col not in raw_data.columns
    ]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    for col in required_columns:
        raw_data[col] = pd.to_numeric(
            raw_data[col],
            errors="coerce"
        )

    raw_data = raw_data.dropna(
        subset=required_columns
    ).copy()

    if len(raw_data) < 10:
        st.error("At least 10 valid measurement rows are required.")
        st.stop()

    if not np.isfinite(raw_data[required_columns].to_numpy()).all():
        st.error("CSV contains invalid measurement values.")
        st.stop()

    # Engineering measurements
    raw_data["Engineering_Strain_Percent"] = (
        raw_data["Displacement_mm"] / gauge_length
    ) * 100

    raw_data["Engineering_Stress_MPa"] = (
        raw_data["Force_N"] / cross_sectional_area
    )

    # Keep original measurements and create tensile-only stress
    raw_data["Tensile_Stress_MPa"] = (
        raw_data["Engineering_Stress_MPa"].clip(lower=0)
    )

    raw_data = raw_data.sort_values(
        "Engineering_Strain_Percent"
    ).reset_index(drop=True)

    strain = raw_data["Engineering_Strain_Percent"]
    stress = raw_data["Tensile_Stress_MPa"]

    # --------------------------------------------------
    # EXTRACT MODEL FEATURES
    # --------------------------------------------------

    maximum_stress = float(stress.max())

    maximum_strain = float(strain.max())

    if maximum_stress <= 0:
        st.error("No positive tensile stress was found.")
        st.stop()

    peak_index = int(stress.idxmax())

    # Strain at maximum stress
    strain_at_peak = float(strain.iloc[peak_index])

    # Initial stiffness: 5%-20% strain
    stiffness_region = raw_data[
        (raw_data["Engineering_Strain_Percent"] >= 5) &
        (raw_data["Engineering_Strain_Percent"] <= 20)
    ]

    if len(stiffness_region) < 3:
        st.error(
            "Insufficient measurements in the 5%-20% strain "
            "region to calculate initial stiffness."
        )
        st.stop()

    stiffness_strain = (
        stiffness_region["Engineering_Strain_Percent"].to_numpy()
        / 100
    )

    stiffness_stress = (
        stiffness_region["Tensile_Stress_MPa"].to_numpy()
    )

    initial_stiffness = float(
        np.polyfit(
            stiffness_strain,
            stiffness_stress,
            1
        )[0]
    )

    # Energy absorption:
    # Average stress at duplicate strain coordinates
    energy_data = (
        raw_data.groupby(
            "Engineering_Strain_Percent",
            as_index=False
        )["Tensile_Stress_MPa"]
        .mean()
        .sort_values("Engineering_Strain_Percent")
    )

    energy_absorption = float(
        np.trapezoid(
            energy_data["Tensile_Stress_MPa"].to_numpy(),
            energy_data["Engineering_Strain_Percent"].to_numpy() / 100
        )
    )

    # Largest post-peak stress drop
    post_peak_stress = stress.iloc[peak_index:]

    minimum_post_peak_stress = max(
        0.0,
        float(post_peak_stress.min())
    )

    post_peak_drop_percent = float(
        (
            (maximum_stress - minimum_post_peak_stress)
            / maximum_stress
        ) * 100
    )

    post_peak_drop_percent = float(
        np.clip(post_peak_drop_percent, 0, 100)
    )

    # --------------------------------------------------
    # BUILD MODEL INPUT
    # --------------------------------------------------

    feature_values = {
        "Maximum_Stress_MPa": maximum_stress,
        "Maximum_Strain_Percent": maximum_strain,
        "Energy_Absorption_MJ_per_m3": energy_absorption,
        "Largest_Post_Peak_Drop_Percent": post_peak_drop_percent
    }

    model_input = pd.DataFrame(
        [[feature_values[name] for name in feature_names]],
        columns=feature_names
    )

    # Predict cluster
    predicted_cluster = int(
        pipeline.predict(model_input)[0]
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.divider()
    st.header("Prediction Result")

    if predicted_cluster == 0:
        st.success("Predicted Cluster: 0")
    else:
        st.info("Predicted Cluster: 1")

    st.caption(
        "This is an exploratory cluster assignment, "
        "Cluster 1 — Higher Mechanical Response",
        "Cluster 0 — Lower Mechanical Response"
    )

    # --------------------------------------------------
    # MECHANICAL METRICS
    # --------------------------------------------------

    st.subheader("Calculated Mechanical Properties")

    c1, c2 = st.columns(2)

    c1.metric(
        "Maximum Stress",
        f"{maximum_stress:.4f} MPa"
    )

    c2.metric(
        "Maximum Strain",
        f"{maximum_strain:.2f}%"
    )

    c3, c4 = st.columns(2)

    c3.metric(
        "Energy Absorption",
        f"{energy_absorption:.4f} MJ/m³"
    )

    c4.metric(
        "Post-Peak Drop",
        f"{post_peak_drop_percent:.2f}%"
    )

    st.metric(
        "Initial Stiffness",
        f"{initial_stiffness:.4f}"
    )

    # --------------------------------------------------
    # MODEL INPUT TABLE
    # --------------------------------------------------

    st.subheader("Four Features Supplied to the Model")

    st.dataframe(
        model_input,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # STRESS-STRAIN GRAPH
    # --------------------------------------------------

    st.divider()
    st.subheader("Stress-Strain Curve")

    chart_data = raw_data.set_index(
        "Engineering_Strain_Percent"
    )[["Engineering_Stress_MPa"]]

    st.line_chart(chart_data)

    # --------------------------------------------------
    # RAW DATA
    # --------------------------------------------------

    with st.expander("View Uploaded Measurements"):
        st.dataframe(
            raw_data,
            use_container_width=True,
            hide_index=True
        )

except Exception as error:
    st.error(f"Unable to process the uploaded CSV: {error}")
