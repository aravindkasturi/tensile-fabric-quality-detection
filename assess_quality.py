
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

CRITERIA_FILE = PROJECT_DIR / "data" / "quality_criteria_template.csv"
FEATURE_FILE = PROJECT_DIR / "results" / "mechanical_features.csv"
RUPTURE_FILE = PROJECT_DIR / "results" / "rupture_detection_results.csv"
OUTPUT_FILE = PROJECT_DIR / "results" / "quality_assessment_results.csv"


# ==========================================================
# PARAMETER MAPPING
# ==========================================================

PARAMETER_MAPPING = {
    "Maximum_Stress": "Maximum_Stress_MPa",
    "Maximum_Strain": "Maximum_Strain_Percent",
    "Initial_Stiffness": "Initial_Stiffness_MPa_per_Strain",
    "Energy_Absorption": "Energy_Absorption_MJ_per_m3",
    "Rupture_Displacement": "Rupture_Displacement_mm",
    "Force_Drop_Percent": "Force_Drop_Percent"
}


# ==========================================================
# LOAD DATA
# ==========================================================

for file in [CRITERIA_FILE, FEATURE_FILE]:
    if not file.exists():
        raise FileNotFoundError(f"Required file not found: {file}")

criteria = pd.read_csv(CRITERIA_FILE)
features = pd.read_csv(FEATURE_FILE)

if "Sample" not in features.columns:
    raise ValueError("Mechanical features must contain Sample column.")

if not {"Parameter", "Minimum_Acceptable",
        "Maximum_Acceptable", "Acceptance_Direction",
        "Specification_Source"}.issubset(criteria.columns):
    raise ValueError("Quality criteria CSV has missing columns.")

features["Sample"] = features["Sample"].astype(str)


# ==========================================================
# MERGE RUPTURE DATA
# ==========================================================

if RUPTURE_FILE.exists():
    rupture = pd.read_csv(RUPTURE_FILE)

    if "Sample" in rupture.columns:
        rupture["Sample"] = rupture["Sample"].astype(str)

        features = features.merge(
            rupture[
                [
                    column for column in [
                        "Sample",
                        "Rupture_Detected",
                        "Rupture_Displacement_mm",
                        "Force_Drop_Percent"
                    ]
                    if column in rupture.columns
                ]
            ].drop_duplicates("Sample"),
            on="Sample",
            how="left"
        )


# ==========================================================
# PREPARE ACCEPTANCE CRITERIA
# ==========================================================

criteria["Parameter"] = criteria["Parameter"].astype(str)

criteria_lookup = {}

for _, row in criteria.iterrows():
    parameter = row["Parameter"]

    criteria_lookup[parameter] = {
        "minimum": pd.to_numeric(
            pd.Series([row["Minimum_Acceptable"]]),
            errors="coerce"
        ).iloc[0],

        "maximum": pd.to_numeric(
            pd.Series([row["Maximum_Acceptable"]]),
            errors="coerce"
        ).iloc[0],

        "direction": str(
            row["Acceptance_Direction"]
        ).strip().lower(),

        "source": str(
            row["Specification_Source"]
        ).strip()
    }


# ==========================================================
# EVALUATE ONE PROPERTY
# ==========================================================

def evaluate_property(value, parameter):

    criterion = criteria_lookup.get(parameter)

    if criterion is None:
        return "REVIEW"

    minimum = criterion["minimum"]
    maximum = criterion["maximum"]
    direction = criterion["direction"]
    source = criterion["source"]

    # Criteria must be supported by a documented specification.
    if not source or source.lower() == "nan":
        return "REVIEW"

    if pd.isna(value):
        return "REVIEW"

    if direction == "minimum":
        if pd.isna(minimum):
            return "REVIEW"
        return "PASS" if value >= minimum else "FAIL"

    if direction == "maximum":
        if pd.isna(maximum):
            return "REVIEW"
        return "PASS" if value <= maximum else "FAIL"

    if direction == "range":
        if pd.isna(minimum) or pd.isna(maximum):
            return "REVIEW"
        if minimum > maximum:
            return "REVIEW"
        return "PASS" if minimum <= value <= maximum else "FAIL"

    return "REVIEW"


# ==========================================================
# EVALUATE ALL SAMPLES
# ==========================================================

assessment_rows = []

for _, sample in features.iterrows():

    result = {
        "Sample": sample["Sample"]
    }

    property_results = []

    for parameter, column in PARAMETER_MAPPING.items():

        if column not in features.columns:
            value = np.nan
        else:
            value = pd.to_numeric(
                pd.Series([sample[column]]),
                errors="coerce"
            ).iloc[0]

        status = evaluate_property(value, parameter)

        result[parameter + "_Value"] = value
        result[parameter + "_Status"] = status

        property_results.append(status)

    # ------------------------------------------------------
    # FINAL ASSESSMENT
    # ------------------------------------------------------

    if "FAIL" in property_results:
        final_status = "BAD"

    elif "REVIEW" in property_results:
        final_status = "REVIEW"

    else:
        final_status = "GOOD"

    result["Final_Assessment"] = final_status

    assessment_rows.append(result)


# ==========================================================
# SAVE RESULTS
# ==========================================================

assessment = pd.DataFrame(assessment_rows)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

assessment.to_csv(OUTPUT_FILE, index=False)


# ==========================================================
# DISPLAY SUMMARY
# ==========================================================

print("\n" + "=" * 65)
print("TENSILE FABRIC QUALITY ASSESSMENT")
print("=" * 65)

print(assessment.to_string(index=False))

print("\nFINAL ASSESSMENT SUMMARY")

print(
    assessment["Final_Assessment"]
    .value_counts()
    .reindex(["GOOD", "BAD", "REVIEW"], fill_value=0)
    .to_string()
)

print("\nSaved:", OUTPUT_FILE)

print(
    "\nNote: REVIEW means the available acceptance criteria "
    "or measurement evidence is insufficient."
)
