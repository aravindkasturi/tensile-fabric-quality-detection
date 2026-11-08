
from pathlib import Path
import re

import numpy as np
import pandas as pd


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

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

THICKNESS_MM = 1.0


# --------------------------------------------------
# READ GAUGE LENGTH AND WIDTH FROM TST FILE
# --------------------------------------------------

def read_sample_dimensions(settings_file):

    content = settings_file.read_text(
        encoding="utf-8",
        errors="replace"
    )

    gauge_match = re.search(
        r"SampleSizeX_um\s+([\d.]+)",
        content
    )

    width_match = re.search(
        r"SampleSizeY_um\s+([\d.]+)",
        content
    )

    if not gauge_match or not width_match:
        raise ValueError(
            f"Could not find sample dimensions in {settings_file.name}"
        )

    gauge_length_mm = float(gauge_match.group(1)) / 1000
    width_mm = float(width_match.group(1)) / 1000

    return gauge_length_mm, width_mm


# --------------------------------------------------
# FIND THE CORRECT MEASUREMENT CSV
# --------------------------------------------------

def find_measurement_csv(sample_folder, sample_name):

    expected_file = sample_folder / f"{sample_name}Data.csv"

    if expected_file.exists():
        return expected_file

    # Fallback: select a CSV whose name starts with sample name
    candidates = list(
        sample_folder.glob(f"{sample_name}*.csv")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No measurement CSV found for {sample_name}"
        )

    return candidates[0]


# --------------------------------------------------
# PROCESS ONE SAMPLE
# --------------------------------------------------

def process_sample(sample_name):

    print(f"\nProcessing {sample_name}...")

    sample_folder = DATA_DIR / sample_name

    if not sample_folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {sample_folder}"
        )

    settings_file = sample_folder / f"{sample_name}.tst"

    if not settings_file.exists():
        raise FileNotFoundError(
            f"Settings file not found: {settings_file}"
        )

    csv_file = find_measurement_csv(
        sample_folder,
        sample_name
    )

    # Read dimensions
    gauge_length_mm, width_mm = read_sample_dimensions(
        settings_file
    )

    area_mm2 = width_mm * THICKNESS_MM

    # Read measurement data
    df = pd.read_csv(csv_file)

    required_columns = [
        "Displacement_mm",
        "Force_N"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"{column} missing in {csv_file.name}"
            )

    # Convert measurements to numeric values
    df["Displacement_mm"] = pd.to_numeric(
        df["Displacement_mm"],
        errors="coerce"
    )

    df["Force_N"] = pd.to_numeric(
        df["Force_N"],
        errors="coerce"
    )

    # Remove invalid measurements
    df = df.dropna(
        subset=["Displacement_mm", "Force_N"]
    ).copy()

    if df.empty:
        raise ValueError(
            f"No valid measurement rows in {csv_file.name}"
        )

    # Engineering calculations
    df["Strain"] = (
        df["Displacement_mm"] / gauge_length_mm
    )

    df["Strain_Percent"] = df["Strain"] * 100

    df["Stress_MPa"] = df["Force_N"] / area_mm2

    # Extract mechanical properties
    max_stress_index = df["Stress_MPa"].idxmax()

    max_stress = float(
        df.loc[max_stress_index, "Stress_MPa"]
    )

    strain_at_max_stress = float(
        df.loc[max_stress_index, "Strain_Percent"]
    )

    max_strain = float(
        df["Strain_Percent"].max()
    )

    max_force = float(df["Force_N"].max())

    max_displacement = float(
        df["Displacement_mm"].max()
    )

    # Area under engineering stress-strain curve
    auc = float(
        np.trapezoid(
            df["Stress_MPa"].to_numpy(),
            df["Strain"].to_numpy()
        )
    )

    # Save processed measurements
    output_file = (
        PROCESSED_DIR / f"{sample_name}_processed.csv"
    )

    df.to_csv(output_file, index=False)

    # Prepare sample summary
    summary = {
        "Sample": sample_name,
        "Measurement_File": csv_file.name,
        "Measurement_Points": len(df),
        "Gauge_Length_mm": gauge_length_mm,
        "Width_mm": width_mm,
        "Thickness_mm": THICKNESS_MM,
        "Area_mm2": area_mm2,
        "Maximum_Force_N": max_force,
        "Maximum_Stress_MPa": max_stress,
        "Strain_At_Max_Stress_Percent": strain_at_max_stress,
        "Maximum_Displacement_mm": max_displacement,
        "Maximum_Strain_Percent": max_strain,
        "Stress_Strain_AUC_MPa": auc,
    }

    print(f"  CSV: {csv_file.name}")
    print(f"  Measurement points: {len(df)}")
    print(f"  Gauge length: {gauge_length_mm} mm")
    print(f"  Width: {width_mm} mm")
    print(f"  Maximum stress: {max_stress:.4f} MPa")
    print(f"  Maximum strain: {max_strain:.4f}%")

    return summary


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_summaries = []

    for sample_name in SAMPLES:

        try:
            summary = process_sample(sample_name)
            all_summaries.append(summary)

        except Exception as error:
            print(f"ERROR processing {sample_name}: {error}")

    if not all_summaries:
        print("\nNo samples were processed successfully.")
        return

    summary_df = pd.DataFrame(all_summaries)

    summary_file = (
        RESULTS_DIR / "tensile_quality_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("DATA PROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"Successfully processed: "
        f"{len(all_summaries)} / {len(SAMPLES)} samples"
    )

    print(f"\nSummary saved to: {summary_file}")
    print(f"Processed data saved to: {PROCESSED_DIR}")

    print("\nMECHANICAL PROPERTY SUMMARY")
    print(
        summary_df[
            [
                "Sample",
                "Maximum_Stress_MPa",
                "Maximum_Strain_Percent",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
