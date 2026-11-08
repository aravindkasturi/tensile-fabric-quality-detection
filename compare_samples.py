
from pathlib import Path
import pandas as pd

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

FEATURE_FILE = PROJECT_DIR / "results" / "mechanical_features.csv"
RUPTURE_FILE = PROJECT_DIR / "results" / "rupture_detection_results.csv"
ANOMALY_FILE = PROJECT_DIR / "results" / "anomaly_detection_results.csv"

OUTPUT_DIR = PROJECT_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / "sample_comparative_assessment.csv"


# ==========================================================
# LOAD DATA
# ==========================================================

if not FEATURE_FILE.exists():
    raise FileNotFoundError("mechanical_features.csv not found.")

features = pd.read_csv(FEATURE_FILE)

if "Sample" not in features.columns:
    raise ValueError("Sample column is missing from mechanical_features.csv")

features["Sample"] = features["Sample"].astype(str)

print("Loaded mechanical features:", len(features), "samples")


# ==========================================================
# MERGE RUPTURE RESULTS
# ==========================================================

if RUPTURE_FILE.exists():
    rupture = pd.read_csv(RUPTURE_FILE)
    rupture["Sample"] = rupture["Sample"].astype(str)

    rupture_columns = [
        "Sample",
        "Rupture_Detected",
        "Rupture_Displacement_mm",
        "Force_Drop_Percent"
    ]

    available = [
        column for column in rupture_columns
        if column in rupture.columns
    ]

    features = features.merge(
        rupture[available].drop_duplicates("Sample"),
        on="Sample",
        how="left"
    )

    print("Rupture results merged.")
else:
    print("Warning: Rupture results not found.")


# ==========================================================
# MERGE ANOMALY RESULTS
# ==========================================================

if ANOMALY_FILE.exists():
    anomaly = pd.read_csv(ANOMALY_FILE)
    anomaly["Sample"] = anomaly["Sample"].astype(str)

    anomaly_columns = [
        "Sample",
        "Anomaly_Status",
        "Anomaly_Score"
    ]

    available = [
        column for column in anomaly_columns
        if column in anomaly.columns
    ]

    features = features.merge(
        anomaly[available].drop_duplicates("Sample"),
        on="Sample",
        how="left"
    )

    print("Anomaly results merged.")
else:
    print("Warning: Anomaly results not found.")


# ==========================================================
# MECHANICAL PROPERTY RANKINGS
# ==========================================================

# Higher values are ranked first for descriptive comparison.
# These rankings do not represent product quality.

ranking_properties = {
    "Maximum_Stress_MPa": "Maximum_Stress_Rank",
    "Maximum_Strain_Percent": "Maximum_Strain_Rank",
    "Initial_Stiffness_MPa_per_Strain": "Initial_Stiffness_Rank",
    "Energy_Absorption_MJ_per_m3": "Energy_Absorption_Rank"
}

for column, rank_column in ranking_properties.items():

    if column in features.columns:
        features[column] = pd.to_numeric(
            features[column], errors="coerce"
        )

        features[rank_column] = (
            features[column]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )


# ==========================================================
# PREPARE FINAL REPORT
# ==========================================================

display_columns = [
    "Sample",
    "Maximum_Force_N",
    "Maximum_Stress_MPa",
    "Maximum_Stress_Rank",
    "Maximum_Strain_Percent",
    "Maximum_Strain_Rank",
    "Initial_Stiffness_MPa_per_Strain",
    "Initial_Stiffness_Rank",
    "Energy_Absorption_MJ_per_m3",
    "Energy_Absorption_Rank",
    "Rupture_Detected",
    "Rupture_Displacement_mm",
    "Force_Drop_Percent",
    "Anomaly_Status",
    "Anomaly_Score"
]

available_columns = [
    column for column in display_columns
    if column in features.columns
]

report = features[available_columns].copy()

if "Force_Drop_Percent" in report.columns:
    report["Force_Drop_Percent"] = (
        report["Force_Drop_Percent"].clip(lower=0, upper=100)
    )

# Sort by maximum tensile stress for convenient comparison.
if "Maximum_Stress_MPa" in report.columns:
    report = report.sort_values(
        "Maximum_Stress_MPa",
        ascending=False,
        na_position="last"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report.to_csv(OUTPUT_FILE, index=False)


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print("\n" + "=" * 65)
print("TENSILE FABRIC - COMPARATIVE ASSESSMENT")
print("=" * 65)

print(report.to_string(index=False))

print("\n" + "=" * 65)
print("REPORT SUMMARY")
print("=" * 65)

print("Total samples:", len(report))

if "Maximum_Stress_MPa" in report.columns:
    print(
        "Highest maximum stress:",
        report.iloc[0]["Sample"]
    )

if "Rupture_Detected" in report.columns:
    rupture_count = (
        report["Rupture_Detected"]
        .astype(str)
        .str.lower()
        .isin(["yes", "true", "1"])
        .sum()
    )

    print("Possible ruptures detected:", int(rupture_count))

print("\nSaved report:")
print(OUTPUT_FILE)

print("\nNOTE:")
print(
    "These are relative mechanical-property rankings, "
    "not validated GOOD/BAD quality classifications."
)
