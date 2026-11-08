
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("results")

SAMPLES = [
    "Test093",
    "Test094",
    "Test095",
    "Test097",
    "Test098",
    "Test099",
    "Test100",
    "Test101",
    "Test102",
    "Test103",
]


# --------------------------------------------------
# INITIAL STIFFNESS
# --------------------------------------------------

def calculate_initial_stiffness(df):
    """
    Estimate initial stiffness using linear regression
    over the 5%-20% engineering strain interval.

    Strain is converted to decimal form.
    Result unit: MPa per unit strain.
    """

    region = df[
        (df["Strain_Percent"] >= 5) &
        (df["Strain_Percent"] <= 20)
    ]

    if len(region) < 3:
        return np.nan

    strain = region["Strain_Percent"].to_numpy() / 100
    stress = region["Stress_MPa"].to_numpy()

    slope, _ = np.polyfit(strain, stress, 1)

    return float(slope)


# --------------------------------------------------
# POST-PEAK ANALYSIS
# --------------------------------------------------

def calculate_post_peak_features(df, peak_index, peak_stress):

    after_peak = df.loc[peak_index:].copy()

    if len(after_peak) <= 1 or peak_stress <= 0:
        return np.nan, np.nan, np.nan, False

    # Lowest stress recorded after the peak
    minimum_post_peak_stress = max(
        0.0,
        float(after_peak["Stress_MPa"].min())
    )

    largest_drop = max(
        0.0,
        peak_stress - minimum_post_peak_stress
    )

    drop_percent = (
        largest_drop / peak_stress
    ) * 100

    # Candidate failure indicator:
    # Stress falls to 20% or less of peak and stays
    # there for at least five consecutive measurements.
    threshold = peak_stress * 0.20

    low_stress = (
        after_peak["Stress_MPa"].to_numpy() <= threshold
    )

    candidate_failure_strain = np.nan
    candidate_detected = False

    required_consecutive_points = 5

    for i in range(
        len(low_stress) - required_consecutive_points + 1
    ):
        window = low_stress[
            i:i + required_consecutive_points
        ]

        if window.all():
            candidate_failure_strain = float(
                after_peak.iloc[i]["Strain_Percent"]
            )
            candidate_detected = True
            break

    return (
        largest_drop,
        drop_percent,
        candidate_failure_strain,
        candidate_detected
    )


# --------------------------------------------------
# PROCESS ONE SAMPLE
# --------------------------------------------------

def extract_sample_features(sample):

    file_path = DATA_DIR / f"{sample}_processed.csv"

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)

    required_columns = [
        "Strain_Percent",
        "Stress_MPa",
        "Force_N",
        "Displacement_mm",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing column {column} in {sample}"
            )

    # Clean and sort measurements
    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=["Strain_Percent", "Stress_MPa"]
    )

    df = df.sort_values(
        "Strain_Percent"
    ).reset_index(drop=True)

    if len(df) < 5:
        raise ValueError(
            f"Not enough valid measurements in {sample}"
        )

    # Peak stress
    peak_index = df["Stress_MPa"].idxmax()

    peak_stress = float(
        df.loc[peak_index, "Stress_MPa"]
    )

    strain_at_peak = float(
        df.loc[peak_index, "Strain_Percent"]
    )

    maximum_strain = float(
        df["Strain_Percent"].max()
    )

    maximum_force = float(
        df["Force_N"].max()
    )

    maximum_displacement = float(
        df["Displacement_mm"].max()
    )

    # Initial stiffness
    initial_stiffness = calculate_initial_stiffness(df)

    # Energy absorption: integrate stress against decimal strain
    strain_decimal = (
        df["Strain_Percent"].to_numpy() / 100
    )

    stress_values = df["Stress_MPa"].to_numpy()

    energy_absorption = float(
        np.trapezoid(
            stress_values,
            strain_decimal
        )
    )

    # Post-peak analysis
    (
        largest_post_peak_drop,
        largest_post_peak_drop_percent,
        candidate_failure_strain,
        candidate_failure_detected
    ) = calculate_post_peak_features(
        df,
        peak_index,
        peak_stress
    )

    return {
        "Sample": sample,
        "Measurement_Points": len(df),
        "Maximum_Force_N": maximum_force,
        "Maximum_Stress_MPa": peak_stress,
        "Strain_At_Max_Stress_Percent": strain_at_peak,
        "Maximum_Displacement_mm": maximum_displacement,
        "Maximum_Strain_Percent": maximum_strain,
        "Initial_Stiffness_MPa_per_Strain": initial_stiffness,
        "Energy_Absorption_MJ_per_m3": energy_absorption,
        "Largest_Post_Peak_Drop_MPa": largest_post_peak_drop,
        "Largest_Post_Peak_Drop_Percent": largest_post_peak_drop_percent,
        "Candidate_Failure_Strain_Percent": candidate_failure_strain,
        "Candidate_Failure_Detected": candidate_failure_detected,
    }


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_rows = []

    for sample in SAMPLES:

        print(f"\nExtracting features: {sample}")

        try:
            features = extract_sample_features(sample)

            feature_rows.append(features)

            print(
                f"  Maximum stress: "
                f"{features['Maximum_Stress_MPa']:.4f} MPa"
            )

            print(
                f"  Maximum strain: "
                f"{features['Maximum_Strain_Percent']:.2f}%"
            )

            print(
                f"  Candidate failure detected: "
                f"{features['Candidate_Failure_Detected']}"
            )

        except Exception as error:
            print(f"  ERROR: {error}")

    if not feature_rows:
        print("\nNo features were extracted.")
        return

    feature_df = pd.DataFrame(feature_rows)

    output_file = (
        OUTPUT_DIR / "mechanical_features.csv"
    )

    feature_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("MECHANICAL FEATURE EXTRACTION COMPLETED")
    print("=" * 60)

    print(f"Successfully processed: {len(feature_rows)} / 10")
    print(f"Saved feature dataset: {output_file}")

    print("\nFEATURE SUMMARY")

    print(
        feature_df[
            [
                "Sample",
                "Maximum_Stress_MPa",
                "Initial_Stiffness_MPa_per_Strain",
                "Energy_Absorption_MJ_per_m3",
                "Largest_Post_Peak_Drop_Percent",
                "Candidate_Failure_Detected",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
