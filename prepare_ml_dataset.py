
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR / "results" / "mechanical_features.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_FILE = (
    OUTPUT_DIR / "ml_features.csv"
)


# --------------------------------------------------
# SELECT FEATURES
# --------------------------------------------------

SELECTED_FEATURES = [
    "Maximum_Stress_MPa",
    "Strain_At_Max_Stress_Percent",
    "Maximum_Strain_Percent",
    "Initial_Stiffness_MPa_per_Strain",
    "Energy_Absorption_MJ_per_m3",
    "Largest_Post_Peak_Drop_Percent",
]


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def main():

    print("\nLoading mechanical feature dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Original dataset shape: {df.shape}")


    # --------------------------------------------------
    # VALIDATE FEATURES
    # --------------------------------------------------

    missing_columns = [
        column
        for column in SELECTED_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )


    # --------------------------------------------------
    # CREATE ML FEATURE DATASET
    # --------------------------------------------------

    ml_df = df[
        ["Sample"] + SELECTED_FEATURES
    ].copy()


    # --------------------------------------------------
    # HANDLE MISSING VALUES
    # --------------------------------------------------

    print("\nMissing values before cleaning:")

    print(ml_df.isnull().sum())

    # Do not replace missing values with invented values.
    # Remove rows only if selected ML features are missing.

    ml_df = ml_df.dropna(
        subset=SELECTED_FEATURES
    )


    # --------------------------------------------------
    # CHECK INFINITE VALUES
    # --------------------------------------------------

    numeric_features = ml_df[
        SELECTED_FEATURES
    ]

    ml_df[SELECTED_FEATURES] = (
        numeric_features.replace(
            [float("inf"), float("-inf")],
            pd.NA
        )
    )

    ml_df = ml_df.dropna(
        subset=SELECTED_FEATURES
    )


    # --------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ml_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("ML FEATURE PREPARATION COMPLETED")
    print("=" * 60)

    print(f"\nFinal dataset shape: {ml_df.shape}")

    print("\nSelected features:")

    for feature in SELECTED_FEATURES:
        print(f"- {feature}")

    print("\nPrepared dataset:")

    print(ml_df.to_string(index=False))

    print(f"\nSaved to: {OUTPUT_FILE}")

    print("\nIMPORTANT:")
    print("This is an unlabelled feature dataset.")
    print("No GOOD/BAD labels have been created.")


if __name__ == "__main__":
    main()
