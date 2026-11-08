
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
GRAPHS_DIR = RESULTS_DIR / "final_analysis_graphs"

FAILURE_THRESHOLD = 0.20
CONSECUTIVE_POINTS = 5

REQUIRED_COLUMNS = [
    "Displacement_mm",
    "Force_N",
    "Strain_Percent",
    "Stress_MPa"
]


# ==========================================================
# INITIAL STIFFNESS
# ==========================================================

def calculate_initial_stiffness(df):

    region = df[
        (df["Strain_Percent"] >= 5) &
        (df["Strain_Percent"] <= 20)
    ]

    if len(region) < 3:
        return np.nan

    strain = region["Strain_Percent"].to_numpy() / 100
    stress = region["Tensile_Stress_MPa"].to_numpy()

    slope, _ = np.polyfit(strain, stress, 1)

    return float(slope)


# ==========================================================
# RUPTURE / FORCE COLLAPSE ANALYSIS
# ==========================================================

def detect_force_collapse(df, peak_index, peak_force):

    after_peak = df.loc[peak_index:].copy()

    if len(after_peak) < CONSECUTIVE_POINTS:
        return False, np.nan, np.nan

    threshold = peak_force * FAILURE_THRESHOLD

    below_threshold = (
        after_peak["Force_N"].to_numpy() <= threshold
    )

    collapse_displacement = np.nan
    force_after_collapse = np.nan

    for i in range(
        len(below_threshold) - CONSECUTIVE_POINTS + 1
    ):

        window = below_threshold[
            i:i + CONSECUTIVE_POINTS
        ]

        if window.all():

            collapse_displacement = float(
                after_peak.iloc[i]["Displacement_mm"]
            )

            force_after_collapse = float(
                after_peak.iloc[i]["Force_N"]
            )

            return (
                True,
                collapse_displacement,
                force_after_collapse
            )

    return False, np.nan, np.nan


# ==========================================================
# PROCESS ONE SAMPLE
# ==========================================================

def analyse_sample(file_path):

    sample_name = file_path.stem.replace("_processed", "")

    df = pd.read_csv(file_path)

    for column in REQUIRED_COLUMNS:

        if column not in df.columns:
            raise ValueError(
                f"{column} missing in {file_path.name}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()

    if len(df) < 10:
        raise ValueError("Insufficient valid measurements")

    # Sort measurements by displacement
    df = df.sort_values(
        "Displacement_mm"
    ).reset_index(drop=True)

    # Preserve original measurements.
    # Do not overwrite Force_N or Stress_MPa.

    df["Tensile_Force_N"] = df["Force_N"].clip(lower=0)

    df["Tensile_Stress_MPa"] = df["Stress_MPa"].clip(lower=0)

    # Maximum tensile properties
    peak_index = df["Tensile_Stress_MPa"].idxmax()

    peak_stress = float(
        df.loc[peak_index, "Tensile_Stress_MPa"]
    )

    peak_force = float(
        df.loc[peak_index, "Force_N"]
    )

    strain_at_peak = float(
        df.loc[peak_index, "Strain_Percent"]
    )

    maximum_strain = float(
        df["Strain_Percent"].max()
    )

    maximum_force = float(
        df["Tensile_Force_N"].max()
    )

    maximum_displacement = float(
        df["Displacement_mm"].max()
    )

    # Initial stiffness
    initial_stiffness = calculate_initial_stiffness(df)

    # Energy absorption:
    # Integrate non-negative tensile stress against
    # engineering strain, in decimal form.
    strain_decimal = (
        df["Strain_Percent"].to_numpy() / 100
    )

    tensile_stress = (
        df["Tensile_Stress_MPa"].to_numpy()
    )

    # Remove duplicate strain coordinates before integration.
    integration_data = pd.DataFrame({
        "Strain": strain_decimal,
        "Stress": tensile_stress
    })

    integration_data = (
        integration_data
        .groupby("Strain", as_index=False)["Stress"]
        .mean()
        .sort_values("Strain")
    )

    energy_absorption = float(
        np.trapezoid(
            integration_data["Stress"].to_numpy(),
            integration_data["Strain"].to_numpy()
        )
    )

    # Post-peak drop
    post_peak = df.loc[peak_index:]

    minimum_post_peak_stress = float(
        post_peak["Tensile_Stress_MPa"].min()
    )

    largest_drop = max(
        0.0,
        peak_stress - minimum_post_peak_stress
    )

    drop_percent = (
        largest_drop / peak_stress * 100
        if peak_stress > 0 else np.nan
    )

    drop_percent = min(drop_percent, 100.0)

    # Force-collapse detection uses original force readings.
    collapse_detected, collapse_displacement, force_after = (
        detect_force_collapse(
            df,
            peak_index,
            peak_force
        )
    )

    return {
        "Sample": sample_name,
        "Measurement_Points": len(df),

        "Maximum_Force_N": maximum_force,
        "Maximum_Stress_MPa": peak_stress,

        "Strain_At_Max_Stress_Percent": strain_at_peak,
        "Maximum_Strain_Percent": maximum_strain,

        "Initial_Stiffness_MPa_per_Strain": initial_stiffness,

        "Energy_Absorption_MJ_per_m3": energy_absorption,

        "Largest_Post_Peak_Drop_MPa": largest_drop,
        "Largest_Post_Peak_Drop_Percent": drop_percent,

        "Force_Collapse_Detected": collapse_detected,
        "Force_Collapse_Displacement_mm": collapse_displacement,
        "Force_After_Collapse_N": force_after,

        "Negative_Force_Readings": int(
            (df["Force_N"] < 0).sum()
        )
    }


# ==========================================================
# GRAPH GENERATION
# ==========================================================

def create_comparison_graphs():

    files = sorted(
        DATA_DIR.glob("*_processed.csv")
    )

    # Force-displacement comparison
    plt.figure(figsize=(12, 7))

    for file_path in files:

        df = pd.read_csv(file_path)

        plt.plot(
            df["Displacement_mm"],
            df["Force_N"],
            label=file_path.stem.replace("_processed", "")
        )

    plt.title("Force vs Displacement - All Samples")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Original Force (N)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GRAPHS_DIR / "all_samples_force_displacement.png",
        dpi=200
    )

    plt.close()

    # Stress-strain comparison
    plt.figure(figsize=(12, 7))

    for file_path in files:

        df = pd.read_csv(file_path)

        plt.plot(
            df["Strain_Percent"],
            df["Stress_MPa"].clip(lower=0),
            label=file_path.stem.replace("_processed", "")
        )

    plt.title("Tensile Stress vs Strain - All Samples")
    plt.xlabel("Engineering Strain (%)")
    plt.ylabel("Non-negative Tensile Stress (MPa)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GRAPHS_DIR / "all_samples_stress_strain.png",
        dpi=200
    )

    plt.close()


# ==========================================================
# MAIN
# ==========================================================

def main():

    RESULTS_DIR.mkdir(exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        DATA_DIR.glob("*_processed.csv")
    )

    if not files:
        print("No processed datasets found.")
        return

    results = []

    print("=" * 65)
    print("FINAL TENSILE FEATURE ANALYSIS")
    print("=" * 65)

    for file_path in files:

        print(f"\nAnalysing {file_path.name}...")

        try:

            result = analyse_sample(file_path)

            results.append(result)

            print(
                f"  Maximum stress: "
                f"{result['Maximum_Stress_MPa']:.4f} MPa"
            )

            print(
                f"  Energy absorption: "
                f"{result['Energy_Absorption_MJ_per_m3']:.4f} MJ/m3"
            )

            print(
                f"  Force collapse detected: "
                f"{result['Force_Collapse_Detected']}"
            )

        except Exception as error:

            print(f"  ERROR: {error}")

    if not results:
        print("\nNo samples successfully analysed.")
        return

    result_df = pd.DataFrame(results)

    output_file = (
        RESULTS_DIR / "final_mechanical_features.csv"
    )

    result_df.to_csv(
        output_file,
        index=False
    )

    create_comparison_graphs()

    print("\n" + "=" * 65)
    print("FINAL ANALYSIS COMPLETED")
    print("=" * 65)

    print(
        f"Successfully analysed: {len(results)} / {len(files)}"
    )

    print(f"\nFeature dataset: {output_file}")

    print(f"Graphs saved in: {GRAPHS_DIR}")

    print("\nCONSOLIDATED FEATURE TABLE")

    print(
        result_df.to_string(index=False)
    )


if __name__ == "__main__":
    main()
