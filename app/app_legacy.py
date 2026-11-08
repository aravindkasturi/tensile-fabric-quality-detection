
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Tensile Fabric Quality Analysis",
    page_icon="🧵",
    layout="wide"
)


# ==========================================================
# 2. PROJECT PATHS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = PROJECT_DIR / "results" / "final_mechanical_features.csv"
GRAPH_DIR = PROJECT_DIR / "results" / "graphs"
EDA_GRAPH_DIR = PROJECT_DIR / "results" / "eda_graphs"
MODEL_FILE = PROJECT_DIR / "model" / "final_tensile_cluster_model.pkl"

ANOMALY_FILE = PROJECT_DIR / "results" / "anomaly_detection_results.csv"
RUPTURE_FILE = PROJECT_DIR / "results" / "rupture_detection_results.csv"
COMPARISON_FILE = (
    PROJECT_DIR / "results" / "sample_comparative_assessment.csv"
)


# ==========================================================
# 3. DATA LOADING
# ==========================================================

@st.cache_data
def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


features = load_csv(FEATURE_FILE)
# Load final clustering model
if not MODEL_FILE.exists():
    st.error("Final clustering model not found.")
    st.stop()

model_bundle = joblib.load(MODEL_FILE)

cluster_pipeline = model_bundle["pipeline"]
cluster_features = model_bundle["features"]

# Assign clusters to existing samples
features["Cluster"] = cluster_pipeline.predict(
    features[cluster_features]
)
anomalies = load_csv(ANOMALY_FILE)
ruptures = load_csv(RUPTURE_FILE)
comparison = load_csv(COMPARISON_FILE)


# ==========================================================
# 4. HELPER FUNCTIONS
# ==========================================================

def format_value(value, decimals=4, suffix=""):
    if pd.isna(value):
        return "Not available"
    return f"{value:.{decimals}f}{suffix}"


def calculate_initial_stiffness(df):
    region = df[
        (df["Engineering_Strain_Percent"] >= 5) &
        (df["Engineering_Strain_Percent"] <= 20)
    ]

    if len(region) < 3:
        return np.nan

    strain = region["Engineering_Strain_Percent"].to_numpy() / 100
    stress = region["Engineering_Stress_MPa"].to_numpy()

    slope, _ = np.polyfit(strain, stress, 1)
    return float(slope)


def calculate_post_peak_features(df, peak_index, peak_stress):
    after_peak = df.loc[peak_index:].copy()

    if len(after_peak) <= 1 or peak_stress <= 0:
        return np.nan, False

    minimum_stress = max(
        0.0,
        float(after_peak["Engineering_Stress_MPa"].min())
    )

    drop_percent = (
        (peak_stress - minimum_stress) / peak_stress
    ) * 100

    low_stress = (
        after_peak["Engineering_Stress_MPa"].to_numpy()
        <= peak_stress * 0.20
    )

    required_points = 5

    candidate_detected = any(
        low_stress[i:i + required_points].all()
        for i in range(len(low_stress) - required_points + 1)
    )

    return float(drop_percent), bool(candidate_detected)


def is_rupture(value):
    return str(value).strip().lower() in ["yes", "true", "1"]


# ==========================================================
# 5. HEADER
# ==========================================================

st.title("🧵 Tensile Fabric Quality Analysis")

st.write(
    "Analyse tensile test measurements, stress-strain behaviour, "
    "mechanical properties, rupture behaviour and anomaly patterns."
)

st.caption(
    "Prototype status: results provide mechanical analysis and "
    "exploratory screening. GOOD/BAD quality classification "
    "has not yet been validated."
)

st.divider()

# ==========================================================
# CLUSTER CLASSIFICATION SUMMARY
# ==========================================================

st.header("Machine Learning Cluster Classification")

cluster_0_count = int((features["Cluster"] == 0).sum())
cluster_1_count = int((features["Cluster"] == 1).sum())

c1, c2, c3 = st.columns(3)

c1.metric("Total Samples", len(features))
c2.metric("Cluster 0", cluster_0_count)
c3.metric("Cluster 1", cluster_1_count)

st.info(
    "Samples are grouped using the trained K-Means model. "
    "Cluster 0 and Cluster 1 represent relative mechanical "
    "patterns"
)

st.subheader("Sample Cluster Assignments")

st.dataframe(
    features[["Sample", "Cluster"]],
    use_container_width=True,
    hide_index=True
)

st.divider()


# ==========================================================
# 6. DATA VALIDATION
# ==========================================================

if features.empty:
    st.error(
        "Mechanical feature dataset not found or empty.\n\n"
        "Run `python extract_features.py` from the project root."
    )
    st.stop()

if "Sample" not in features.columns:
    st.error("The mechanical feature file must contain a Sample column.")
    st.stop()

features["Sample"] = features["Sample"].astype(str)


# ==========================================================
# 7. SIDEBAR - SAMPLE DIMENSIONS
# ==========================================================

st.sidebar.header("Sample Dimensions")

gauge_length = st.sidebar.number_input(
    "Gauge Length (mm)",
    min_value=0.01,
    value=10.0,
    step=0.5,
    format="%.2f"
)

sample_width = st.sidebar.number_input(
    "Sample Width (mm)",
    min_value=0.01,
    value=4.7,
    step=0.1,
    format="%.2f"
)

sample_thickness = st.sidebar.number_input(
    "Sample Thickness (mm)",
    min_value=0.01,
    value=1.0,
    step=0.1,
    format="%.2f"
)

cross_sectional_area = sample_width * sample_thickness

st.sidebar.info(
    f"Cross-sectional Area: {cross_sectional_area:.2f} mm²"
)


# ==========================================================
# 8. SIDEBAR - UPLOAD NEW TEST
# ==========================================================

st.sidebar.divider()
st.sidebar.header("Upload New Test Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload tensile test CSV",
    type=["csv"],
    help="CSV must contain Force_N and Displacement_mm."
)

if uploaded_file is not None:

    try:
        uploaded_data = pd.read_csv(uploaded_file)

        required_columns = ["Force_N", "Displacement_mm"]

        missing_columns = [
            column for column in required_columns
            if column not in uploaded_data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing columns: " + ", ".join(missing_columns)
            )

        for column in required_columns:
            uploaded_data[column] = pd.to_numeric(
                uploaded_data[column],
                errors="coerce"
            )

        uploaded_data = uploaded_data.dropna(
            subset=required_columns
        ).copy()

        if len(uploaded_data) < 5:
            raise ValueError(
                "At least five valid measurement rows are required."
            )

        uploaded_data["Engineering_Stress_MPa"] = (
            uploaded_data["Force_N"] / cross_sectional_area
        )

        uploaded_data["Engineering_Strain_Percent"] = (
            uploaded_data["Displacement_mm"] / gauge_length
        ) * 100

        uploaded_data = uploaded_data.sort_values(
            "Engineering_Strain_Percent"
        ).reset_index(drop=True)

        peak_index = uploaded_data["Engineering_Stress_MPa"].idxmax()

        maximum_stress = float(
            uploaded_data.loc[
                peak_index, "Engineering_Stress_MPa"
            ]
        )

        maximum_strain = float(
            uploaded_data["Engineering_Strain_Percent"].max()
        )

        maximum_force = float(uploaded_data["Force_N"].max())

        maximum_displacement = float(
            uploaded_data["Displacement_mm"].max()
        )

        strain_at_peak = float(
            uploaded_data.loc[
                peak_index, "Engineering_Strain_Percent"
            ]
        )

        initial_stiffness = calculate_initial_stiffness(
            uploaded_data
        )

        strain_decimal = (
            uploaded_data["Engineering_Strain_Percent"].to_numpy()
            / 100
        )

        stress_values = (
            uploaded_data["Engineering_Stress_MPa"].to_numpy()
        )

        energy_absorption = float(
            np.trapezoid(stress_values, strain_decimal)
        )

        post_peak_drop, candidate_failure = (
            calculate_post_peak_features(
                uploaded_data,
                peak_index,
                maximum_stress
            )
        )

        st.sidebar.success(
            "CSV uploaded and processed successfully."
        )

        st.sidebar.dataframe(
            uploaded_data.head(),
            use_container_width=True
        )

        st.divider()
        st.header("Uploaded Sample Analysis")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Maximum Stress",
            f"{maximum_stress:.4f} MPa"
        )

        c2.metric(
            "Maximum Strain",
            f"{maximum_strain:.2f}%"
        )

        c3.metric(
            "Initial Stiffness",
            format_value(initial_stiffness)
        )

        c4.metric(
            "Energy Absorption",
            f"{energy_absorption:.4f} MJ/m³"
        )

        st.subheader("Uploaded Sample Stress-Strain Curve")

        st.line_chart(
            uploaded_data.set_index(
                "Engineering_Strain_Percent"
            )[["Engineering_Stress_MPa"]]
        )

        st.subheader("Uploaded Sample Mechanical Properties")

        uploaded_properties = pd.DataFrame({
            "Property": [
                "Maximum Force (N)",
                "Maximum Stress (MPa)",
                "Maximum Strain (%)",
                "Strain at Maximum Stress (%)",
                "Maximum Displacement (mm)",
                "Initial Stiffness",
                "Energy Absorption (MJ/m³)",
                "Largest Post-Peak Drop (%)",
                "Gauge Length (mm)",
                "Cross-sectional Area (mm²)"
            ],
            "Value": [
                maximum_force,
                maximum_stress,
                maximum_strain,
                strain_at_peak,
                maximum_displacement,
                initial_stiffness,
                energy_absorption,
                post_peak_drop,
                gauge_length,
                cross_sectional_area
            ]
        })

        st.dataframe(
            uploaded_properties,
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Uploaded Sample Post-Peak Behaviour")

        if pd.notna(post_peak_drop):
            st.metric(
                "Largest Post-Peak Drop",
                f"{post_peak_drop:.2f}%"
            )

        if candidate_failure:
            st.warning(
                "Candidate failure pattern detected. "
                "This is an exploratory signal, not confirmed fracture."
            )
        else:
            st.info(
                "No candidate failure pattern detected "
                "using the current screening rule."
            )

    except Exception as error:
        st.sidebar.error(f"Unable to process CSV: {error}")


# ==========================================================
# 9. SAMPLE SELECTION
# ==========================================================

st.sidebar.divider()
st.sidebar.header("Sample Selection")

sample_list = features["Sample"].dropna().unique().tolist()

if not sample_list:
    st.error("No sample records were found.")
    st.stop()

selected_sample = st.sidebar.selectbox(
    "Choose a tensile test sample",
    sample_list
)

sample_data = features[
    features["Sample"] == selected_sample
].iloc[0]


# ==========================================================
# 10. ANOMALY DETECTION
# ==========================================================

st.divider()
st.header("Anomaly Detection Overview")

anomaly_ready = (
    not anomalies.empty and
    {"Sample", "Anomaly_Status", "Anomaly_Score"}.issubset(
        anomalies.columns
    )
)

if anomaly_ready:

    anomalies["Sample"] = anomalies["Sample"].astype(str)

    anomalies["Anomaly_Score"] = pd.to_numeric(
        anomalies["Anomaly_Score"],
        errors="coerce"
    )

    anomalies = anomalies.drop_duplicates(
        "Sample",
        keep="last"
    )

    potential_count = int(
        (anomalies["Anomaly_Status"] == "Potential Anomaly").sum()
    )

    typical_count = int(
        (anomalies["Anomaly_Status"] == "Typical Pattern").sum()
    )

    a1, a2, a3 = st.columns(3)

    a1.metric("Samples Analysed", anomalies["Sample"].nunique())
    a2.metric("Typical Pattern", typical_count)
    a3.metric("Potential Anomalies", potential_count)

    st.info(
        "A potential anomaly is statistically unusual relative to "
        "the samples used to fit the current Isolation Forest. "
        "It is not proof of poor quality or a defect."
    )

    st.subheader("Sample-Level Anomaly Report")

    anomaly_report = anomalies[
        ["Sample", "Anomaly_Status", "Anomaly_Score"]
    ].sort_values(
        "Anomaly_Score",
        ascending=True
    )

    st.dataframe(
        anomaly_report,
        hide_index=True,
        use_container_width=True
    )

    selected_anomaly = anomalies[
        anomalies["Sample"] == selected_sample
    ]

    st.subheader(f"Anomaly Result: {selected_sample}")

    if selected_anomaly.empty:

        st.warning("No anomaly result is available for this sample.")

    else:

        selected_status = selected_anomaly.iloc[0]["Anomaly_Status"]
        selected_score = selected_anomaly.iloc[0]["Anomaly_Score"]

        x1, x2 = st.columns(2)

        x1.metric("Anomaly Status", str(selected_status))

        x2.metric(
            "Anomaly Score",
            format_value(selected_score, 6)
        )

        if selected_status == "Potential Anomaly":

            st.warning(
                "This sample is flagged for further investigation. "
                "Do not label it BAD based on anomaly detection alone."
            )

        else:

            st.success(
                "This sample follows the typical pattern identified "
                "in the current dataset. This does not certify it as GOOD."
            )

    st.subheader("Anomaly Score Visualization")

    anomaly_score_graph = EDA_GRAPH_DIR / "anomaly_scores.png"

    if anomaly_score_graph.exists():

        st.image(
            str(anomaly_score_graph),
            caption="Isolation Forest anomaly scores.",
            use_container_width=True
        )

    else:

        st.warning(
            "Anomaly score graph not found. "
            "Run `python visualize_anomalies.py`."
        )

    st.subheader("Anomaly Distribution")

    anomaly_distribution_graph = (
        EDA_GRAPH_DIR / "anomaly_distribution.png"
    )

    if anomaly_distribution_graph.exists():

        st.image(
            str(anomaly_distribution_graph),
            caption="Distribution of typical and potential anomaly samples.",
            use_container_width=True
        )

    else:

        st.warning("Anomaly distribution graph not found.")

else:

    st.warning(
        "Anomaly results are not available. "
        "Run `python anomaly_detection.py`."
    )


# ==========================================================
# 11. RUPTURE DETECTION
# ==========================================================

st.divider()
st.header("Rupture Detection Overview")

rupture_ready = (
    not ruptures.empty and
    {
        "Sample",
        "Rupture_Detected",
        "Rupture_Displacement_mm",
        "Force_Before_Rupture_N",
        "Force_After_Rupture_N",
        "Force_Drop_Percent"
    }.issubset(ruptures.columns)
)

if rupture_ready:

    ruptures["Sample"] = ruptures["Sample"].astype(str)

    for column in [
        "Rupture_Displacement_mm",
        "Force_Before_Rupture_N",
        "Force_After_Rupture_N",
        "Force_Drop_Percent"
    ]:

        ruptures[column] = pd.to_numeric(
            ruptures[column],
            errors="coerce"
        )

    ruptures = ruptures.drop_duplicates(
        "Sample",
        keep="last"
    )

    rupture_yes = ruptures["Rupture_Detected"].apply(is_rupture)

    rupture_count = int(rupture_yes.sum())
    no_rupture_count = int(len(ruptures) - rupture_count)

    r1, r2, r3 = st.columns(3)

    r1.metric("Samples with Possible Rupture", rupture_count)
    r2.metric("No Rupture Detected", no_rupture_count)
    r3.metric("Samples Analysed", ruptures["Sample"].nunique())

    st.info(
        "The detector flags sustained force drops of at least 80% "
        "from the preceding peak over five readings. These are "
        "automated possible-rupture detections."
    )

    rupture_report = ruptures.copy()

    rupture_report["Force_Drop_Percent"] = (
        rupture_report["Force_Drop_Percent"].clip(0, 100)
    )

    st.subheader("Sample-Level Rupture Report")

    st.dataframe(
        rupture_report[
            [
                "Sample",
                "Rupture_Detected",
                "Rupture_Displacement_mm",
                "Force_Before_Rupture_N",
                "Force_After_Rupture_N",
                "Force_Drop_Percent"
            ]
        ],
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Rupture Detection Distribution")

    rupture_chart = pd.DataFrame(
        {"Count": [rupture_count, no_rupture_count]},
        index=["Possible Rupture", "No Rupture Detected"]
    )

    st.bar_chart(rupture_chart)

else:

    st.warning(
        "Rupture results are unavailable. "
        "Run `python detect_rupture.py`."
    )


# ==========================================================
# 12. SELECTED SAMPLE MECHANICAL ANALYSIS
# ==========================================================

st.divider()
st.header(f"Existing Sample Analysis: {selected_sample}")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Maximum Stress",
    format_value(
        sample_data.get("Maximum_Stress_MPa"),
        4,
        " MPa"
    )
)

col2.metric(
    "Maximum Strain",
    format_value(
        sample_data.get("Maximum_Strain_Percent"),
        2,
        "%"
    )
)

col3.metric(
    "Initial Stiffness",
    format_value(
        sample_data.get("Initial_Stiffness_MPa_per_Strain"),
        4
    )
)

col4.metric(
    "Energy Absorption",
    format_value(
        sample_data.get("Energy_Absorption_MJ_per_m3"),
        4,
        " MJ/m³"
    )
)


# ==========================================================
# 13. STRESS-STRAIN GRAPH
# ==========================================================

st.divider()
st.subheader("Stress-Strain Curve")

graph_file = GRAPH_DIR / f"{selected_sample}_stress_strain.png"

if graph_file.exists():

    st.image(
        str(graph_file),
        caption=f"Stress-Strain Curve - {selected_sample}",
        use_container_width=True
    )

else:

    st.warning(f"Stress-strain graph not found: {graph_file}")


# ==========================================================
# 14. POST-PEAK BEHAVIOUR
# ==========================================================

st.subheader("Post-Peak Behaviour")

drop_percent = sample_data.get(
    "Largest_Post_Peak_Drop_Percent",
    np.nan
)

failure_detected = sample_data.get(
    "Candidate_Failure_Detected",
    False
)

p1, p2 = st.columns(2)

p1.metric(
    "Largest Post-Peak Drop",
    format_value(drop_percent, 2, "%")
)

if bool(failure_detected):

    p2.warning(
        "Candidate failure pattern detected. "
        "This is exploratory, not confirmed physical fracture."
    )

else:

    p2.info(
        "No candidate failure pattern detected "
        "using the current screening rule."
    )


# ==========================================================
# 15. SELECTED SAMPLE RUPTURE DETAILS
# ==========================================================

st.divider()
st.subheader("Rupture Detection Result")

if rupture_ready:

    selected_rupture = ruptures[
        ruptures["Sample"] == selected_sample
    ]

    if selected_rupture.empty:

        st.warning("No rupture result is available for this sample.")

    else:

        rupture_row = selected_rupture.iloc[0]

        if is_rupture(rupture_row["Rupture_Detected"]):

            st.warning(
                "Possible rupture detected by the automated "
                "force-drop rule. This is not a verified quality verdict."
            )

        else:

            st.success(
                "No rupture detected by the current automated rule. "
                "This does not prove the sample is defect-free."
            )

        rr1, rr2, rr3 = st.columns(3)

        rr1.metric(
            "Rupture Displacement",
            format_value(
                rupture_row["Rupture_Displacement_mm"],
                3,
                " mm"
            )
        )

        rr2.metric(
            "Force Before Rupture",
            format_value(
                rupture_row["Force_Before_Rupture_N"],
                4,
                " N"
            )
        )

        force_drop = rupture_row["Force_Drop_Percent"]

        if pd.notna(force_drop):
            force_drop = float(np.clip(force_drop, 0, 100))

        rr3.metric(
            "Force Drop",
            format_value(force_drop, 2, "%")
        )

        st.caption(
            "Force after rupture: "
            + format_value(
                rupture_row["Force_After_Rupture_N"],
                4,
                " N"
            )
        )

else:

    st.info("Rupture results are not available.")


# ==========================================================
# 16. MECHANICAL PROPERTY TABLE
# ==========================================================

st.divider()
st.subheader("Mechanical Properties")

display_columns = [
    "Maximum_Force_N",
    "Maximum_Stress_MPa",
    "Strain_At_Max_Stress_Percent",
    "Maximum_Displacement_mm",
    "Maximum_Strain_Percent",
    "Initial_Stiffness_MPa_per_Strain",
    "Energy_Absorption_MJ_per_m3",
    "Largest_Post_Peak_Drop_Percent"
]

available_columns = [
    column for column in display_columns
    if column in features.columns
]

property_table = pd.DataFrame({
    "Property": available_columns,
    "Value": [
        sample_data[column]
        for column in available_columns
    ]
})

st.dataframe(
    property_table,
    hide_index=True,
    use_container_width=True
)


# ==========================================================
# 17. ALL SAMPLES COMPARISON
# ==========================================================

st.divider()
st.subheader("All Samples Comparison")

comparison_columns = [
    "Sample",
    "Maximum_Stress_MPa",
    "Maximum_Strain_Percent",
    "Energy_Absorption_MJ_per_m3"
]

comparison_columns = [
    column for column in comparison_columns
    if column in features.columns
]

comparison_table = features[comparison_columns].copy()

if anomaly_ready:

    comparison_table = comparison_table.merge(
        anomalies[
            ["Sample", "Anomaly_Status", "Anomaly_Score"]
        ],
        on="Sample",
        how="left"
    )

if rupture_ready:

    comparison_table = comparison_table.merge(
        ruptures[
            [
                "Sample",
                "Rupture_Detected",
                "Rupture_Displacement_mm",
                "Force_Drop_Percent"
            ]
        ],
        on="Sample",
        how="left"
    )

    comparison_table["Force_Drop_Percent"] = (
        comparison_table["Force_Drop_Percent"].clip(0, 100)
    )

st.dataframe(
    comparison_table,
    hide_index=True,
    use_container_width=True
)


# ==========================================================
# 18. COMPARATIVE SAMPLE ASSESSMENT
# ==========================================================

st.divider()
st.header("Comparative Sample Assessment")

if comparison.empty:

    st.warning(
        "Comparative assessment file not found.\n\n"
        "Run `python compare_samples.py` from the project root."
    )

else:

    if "Sample" not in comparison.columns:

        st.error(
            "Comparative assessment file must contain a Sample column."
        )

    else:

        comparison["Sample"] = comparison["Sample"].astype(str)

        st.write(
            "Compare the mechanical properties of all tested samples. "
            "Rank 1 represents the highest measured value for that property."
        )

        st.info(
            "These are relative mechanical-property rankings, not "
            "GOOD/BAD classifications. Higher values are not necessarily "
            "better for every fabric application."
        )

        # --------------------------------------------------
        # FULL COMPARATIVE TABLE
        # --------------------------------------------------

        st.subheader("Complete Comparative Assessment")

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True
        )

        # --------------------------------------------------
        # DOWNLOAD REPORT
        # --------------------------------------------------

        csv_data = comparison.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Comparative Assessment CSV",
            data=csv_data,
            file_name="sample_comparative_assessment.csv",
            mime="text/csv",
            use_container_width=True
        )

        # --------------------------------------------------
        # RANKING CHARTS
        # --------------------------------------------------

        ranking_charts = [
            (
                "Maximum Stress Ranking",
                "Maximum_Stress_MPa",
                "Maximum_Stress_Rank",
                "Maximum Stress (MPa)"
            ),
            (
                "Maximum Strain Ranking",
                "Maximum_Strain_Percent",
                "Maximum_Strain_Rank",
                "Maximum Strain (%)"
            ),
            (
                "Initial Stiffness Ranking",
                "Initial_Stiffness_MPa_per_Strain",
                "Initial_Stiffness_Rank",
                "Initial Stiffness"
            ),
            (
                "Energy Absorption Ranking",
                "Energy_Absorption_MJ_per_m3",
                "Energy_Absorption_Rank",
                "Energy Absorption (MJ/m³)"
            )
        ]

        for title, value_column, rank_column, axis_label in ranking_charts:

            if value_column in comparison.columns:

                chart_data = comparison[
                    ["Sample", value_column]
                ].copy()

                chart_data[value_column] = pd.to_numeric(
                    chart_data[value_column],
                    errors="coerce"
                )

                chart_data = chart_data.dropna(
                    subset=[value_column]
                ).sort_values(
                    value_column,
                    ascending=False
                )

                st.subheader(title)

                if not chart_data.empty:

                    st.bar_chart(
                        chart_data.set_index("Sample")[[value_column]]
                    )

                    if rank_column in comparison.columns:

                        st.caption(
                            "Ranking: 1 = highest measured value. "
                            "This is a descriptive comparison only."
                        )

        # --------------------------------------------------
        # SELECTED SAMPLE COMPARATIVE POSITION
        # --------------------------------------------------

        st.subheader(
            f"Comparative Position: {selected_sample}"
        )

        selected_comparison = comparison[
            comparison["Sample"] == selected_sample
        ]

        if selected_comparison.empty:

            st.warning(
                "Comparative assessment is not available for this sample."
            )

        else:

            selected_row = selected_comparison.iloc[0]

            position_columns = [
                ("Maximum Stress", "Maximum_Stress_Rank"),
                ("Maximum Strain", "Maximum_Strain_Rank"),
                ("Initial Stiffness", "Initial_Stiffness_Rank"),
                ("Energy Absorption", "Energy_Absorption_Rank")
            ]

            position_columns = [
                item for item in position_columns
                if item[1] in comparison.columns
            ]

            if position_columns:

                position_cols = st.columns(len(position_columns))

                for col, (label, rank_column) in zip(
                    position_cols,
                    position_columns
                ):

                    rank = selected_row[rank_column]

                    col.metric(
                        label + " Rank",
                        (
                            str(int(rank))
                            if pd.notna(rank)
                            else "N/A"
                        )
                    )

        # --------------------------------------------------
        # COMPARATIVE TABLE DOWNLOAD NOTE
        # --------------------------------------------------

        st.caption(
            "The report combines measured mechanical properties, "
            "relative rankings, anomaly results and rupture screening."
        )


# ==========================================================
# 19. VISUAL COMPARISON
# ==========================================================

st.divider()
st.header("Visual Comparison of All Samples")

if "Maximum_Stress_MPa" in features.columns:

    st.subheader("Maximum Stress Comparison")

    stress_chart = features[
        ["Sample", "Maximum_Stress_MPa"]
    ].set_index("Sample")

    st.bar_chart(stress_chart)

if "Energy_Absorption_MJ_per_m3" in features.columns:

    st.subheader("Energy Absorption Comparison")

    energy_chart = features[
        ["Sample", "Energy_Absorption_MJ_per_m3"]
    ].set_index("Sample")

    st.bar_chart(energy_chart)

st.subheader("Combined Stress-Strain Curves")

combined_graph_path = (
    GRAPH_DIR / "all_samples_comparison.png"
)

if combined_graph_path.exists():

    st.image(
        str(combined_graph_path),
        caption="Stress-Strain Comparison Across All Samples",
        use_container_width=True
    )

else:

    st.warning(
        "Combined stress-strain graph was not found."
    )


# ==========================================================
# 20. FOOTER
# ==========================================================

st.divider()

st.caption(
    "Tensile Fabric Quality Detection Project | "
    "Mechanical Analysis, Rupture Detection and Anomaly Screening"
)

st.caption(
    "GOOD/BAD classification is not yet validated. "
    "Mechanical rankings are relative comparisons. "
    "Anomaly detection and rupture detection are screening tools."
)
