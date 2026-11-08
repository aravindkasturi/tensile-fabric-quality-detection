
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
RESULTS_DIR = PROJECT_DIR / "results"
GRAPH_DIR = RESULTS_DIR / "graphs"

SAMPLES = [
    "Test093", "Test094", "Test095",
    "Test097", "Test098", "Test099",
    "Test100", "Test101", "Test102", "Test103"
]

THICKNESS_MM = 1.0


# ==========================================
# READ SPECIMEN DIMENSIONS
# ==========================================

def read_dimensions(settings_file):

    content = settings_file.read_text(
        encoding="utf-8",
        errors="replace"
    )

    x_match = re.search(
        r"SampleSizeX_um\s+([\d.]+)", content
    )

    y_match = re.search(
        r"SampleSizeY_um\s+([\d.]+)", content
    )

    if not x_match or not y_match:
        raise ValueError(
            f"Cannot read dimensions: {settings_file}"
        )

    gauge_length = float(x_match.group(1)) / 1000
    width = float(y_match.group(1)) / 1000

    return gauge_length, width


# ==========================================
# LOAD ONE SAMPLE
# ==========================================

def load_sample(sample):

    folder = RAW_DIR / sample

    settings_file = folder / f"{sample}.tst"
    csv_file = folder / f"{sample}Data.csv"

    if not settings_file.exists():
        raise FileNotFoundError(settings_file)

    if not csv_file.exists():
        raise FileNotFoundError(csv_file)

    gauge_length, width = read_dimensions(settings_file)

    df = pd.read_csv(csv_file)

    required = ["Displacement_mm", "Force_N"]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"{column} missing in {sample}"
            )

        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(subset=required).copy()

    df = df.sort_values(
        "Displacement_mm"
    ).reset_index(drop=True)

    area = width * THICKNESS_MM

    df["Stress_MPa"] = df["Force_N"] / area

    df["Strain_Percent"] = (
        df["Displacement_mm"] / gauge_length
    ) * 100

    peak_index = df["Force_N"].idxmax()

    summary = {
        "Sample": sample,
        "Gauge_Length_mm": gauge_length,
        "Width_mm": width,
        "Thickness_mm": THICKNESS_MM,
        "Area_mm2": area,
        "Measurement_Points": len(df),
        "Maximum_Force_N": df["Force_N"].max(),
        "Maximum_Displacement_mm": df["Displacement_mm"].max(),
        "Displacement_At_Max_Force_mm":
            df.loc[peak_index, "Displacement_mm"],
        "Final_Force_N": df["Force_N"].iloc[-1],
        "Maximum_Stress_MPa": df["Stress_MPa"].max(),
        "Maximum_Strain_Percent": df["Strain_Percent"].max()
    }

    return df, summary


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():

    RESULTS_DIR.mkdir(exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    loaded_samples = {}
    summaries = []

    for sample in SAMPLES:

        print(f"\nLoading {sample}...")

        df, summary = load_sample(sample)

        loaded_samples[sample] = df
        summaries.append(summary)

        print(
            f"  Maximum force: "
            f"{summary['Maximum_Force_N']:.4f} N"
        )

        print(
            f"  Gauge length: "
            f"{summary['Gauge_Length_mm']:.2f} mm"
        )

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        RESULTS_DIR / "raw_curve_comparison_summary.csv",
        index=False
    )

    # ======================================
    # GRAPH 1: ALL FORCE-DISPLACEMENT CURVES
    # ======================================

    fig, ax = plt.subplots(figsize=(12, 7))

    for sample, df in loaded_samples.items():

        ax.plot(
            df["Displacement_mm"],
            df["Force_N"],
            label=sample,
            linewidth=1.7
        )

    ax.set_title("All PCL Samples: Force vs Displacement")
    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Force (N)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR / "all_samples_force_displacement.png",
        dpi=180
    )

    plt.close(fig)

    # ======================================
    # GRAPH 2: COMMON DISPLACEMENT RANGE
    # ======================================

    common_start = max(
        df["Displacement_mm"].min()
        for df in loaded_samples.values()
    )

    common_end = min(
        df["Displacement_mm"].max()
        for df in loaded_samples.values()
    )

    grid = np.linspace(
        common_start,
        common_end,
        250
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    interpolated_results = []

    for sample, df in loaded_samples.items():

        x = df["Displacement_mm"].to_numpy()
        y = df["Force_N"].to_numpy()

        unique_x, indices = np.unique(
            x, return_index=True
        )

        unique_y = y[indices]

        interpolated_force = np.interp(
            grid, unique_x, unique_y
        )

        ax.plot(
            grid,
            interpolated_force,
            label=sample,
            linewidth=1.7
        )

        interpolated_results.append(
            pd.DataFrame({
                "Sample": sample,
                "Displacement_mm": grid,
                "Interpolated_Force_N": interpolated_force
            })
        )

    ax.set_title(
        "Force vs Displacement: Shared Range Only"
    )

    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Interpolated Force (N)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    fig.tight_layout()

    fig.savefig(
        GRAPH_DIR /
        "all_samples_force_displacement_common_range.png",
        dpi=180
    )

    plt.close(fig)

    # ======================================
    # SAVE RESULTS
    # ======================================

    common_df = pd.concat(
        interpolated_results,
        ignore_index=True
    )

    common_df.to_csv(
        RESULTS_DIR / "common_displacement_force_comparison.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("RAW TENSILE CURVE COMPARISON COMPLETED")
    print("=" * 60)

    print("\nSUMMARY:")
    print(summary_df.to_string(index=False))

    print(
        f"\nCommon displacement range: "
        f"{common_start:.3f} to {common_end:.3f} mm"
    )

    print("\nGenerated files:")
    print("results/raw_curve_comparison_summary.csv")
    print("results/common_displacement_force_comparison.csv")
    print("results/graphs/all_samples_force_displacement.png")
    print(
        "results/graphs/"
        "all_samples_force_displacement_common_range.png"
    )

    print(
        "\nNote: These results describe mechanical behaviour. "
        "They do not establish GOOD/BAD quality."
    )


if __name__ == "__main__":
    main()
