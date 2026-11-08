
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR / "results" / "anomaly_detection_results.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR / "results" / "eda_graphs"
)


# --------------------------------------------------
# LOAD ANOMALY RESULTS
# --------------------------------------------------

def main():

    print("\nLoading anomaly detection results...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Sort samples by anomaly score
    df = df.sort_values(
        "Anomaly_Score"
    )

    # --------------------------------------------------
    # 1. ANOMALY SCORE BAR CHART
    # --------------------------------------------------

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=df,
        x="Sample",
        y="Anomaly_Score",
        hue="Anomaly_Status",
        dodge=False
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    plt.title("Isolation Forest - Anomaly Scores")
    plt.xlabel("Tensile Test Sample")
    plt.ylabel("Anomaly Score")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "anomaly_scores.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # 2. ANOMALY STATUS DISTRIBUTION
    # --------------------------------------------------

    status_counts = (
        df["Anomaly_Status"]
        .value_counts()
    )

    plt.figure(figsize=(8, 5))

    plt.pie(
        status_counts.values,
        labels=status_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Anomaly Detection Distribution")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "anomaly_distribution.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------

    print("\nANOMALY VISUALIZATION COMPLETED")

    print("\nGenerated graphs:")

    print(
        OUTPUT_DIR / "anomaly_scores.png"
    )

    print(
        OUTPUT_DIR / "anomaly_distribution.png"
    )


if __name__ == "__main__":
    main()
