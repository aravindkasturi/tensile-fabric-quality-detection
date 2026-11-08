

from pathlib import Path

import matplotlib

# Use a non-interactive backend.
# This avoids requiring Tkinter.
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/graphs")

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


def create_graphs():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    combined_fig, combined_ax = plt.subplots(figsize=(12, 7))

    successful_samples = 0

    for sample in SAMPLES:

        file_path = DATA_DIR / f"{sample}_processed.csv"

        if not file_path.exists():
            print(f"Missing processed file: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Sort measurements by strain
        df = df.sort_values("Strain_Percent")

        strain = df["Strain_Percent"]
        stress = df["Stress_MPa"]

        # Individual graph
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(strain, stress, linewidth=2)

        ax.set_title(f"Stress-Strain Curve - {sample}")
        ax.set_xlabel("Engineering Strain (%)")
        ax.set_ylabel("Engineering Stress (MPa)")

        ax.grid(True, alpha=0.3)

        max_index = stress.idxmax()

        ax.scatter(
            strain.loc[max_index],
            stress.loc[max_index],
            marker="o",
            s=60,
            label="Maximum stress"
        )

        ax.legend()

        fig.tight_layout()

        output_path = RESULTS_DIR / f"{sample}_stress_strain.png"

        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        # Add curve to combined graph
        combined_ax.plot(
            strain,
            stress,
            linewidth=1.8,
            label=sample
        )

        successful_samples += 1

        print(f"Created graph: {output_path}")

    # Combined graph
    combined_ax.set_title("Stress-Strain Comparison - All Samples")
    combined_ax.set_xlabel("Engineering Strain (%)")
    combined_ax.set_ylabel("Engineering Stress (MPa)")
    combined_ax.grid(True, alpha=0.3)
    combined_ax.legend(loc="best", fontsize=9)

    combined_fig.tight_layout()

    combined_path = RESULTS_DIR / "all_samples_comparison.png"

    combined_fig.savefig(combined_path, dpi=180)
    plt.close(combined_fig)

    print("\n" + "=" * 55)
    print("GRAPH GENERATION COMPLETED")
    print("=" * 55)

    print(f"Individual graphs: {successful_samples}")
    print(f"Combined graph: {combined_path}")


if __name__ == "__main__":
    create_graphs()
