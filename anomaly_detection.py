
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -----------------------------------------
# PROJECT PATHS
# -----------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" / "ml_features.csv"
)

OUTPUT_DIR = PROJECT_DIR / "results"

MODEL_DIR = PROJECT_DIR / "model"


# -----------------------------------------
# SELECT FEATURES
# -----------------------------------------

FEATURE_COLUMNS = [
    "Maximum_Stress_MPa",
    "Strain_At_Max_Stress_Percent",
    "Maximum_Strain_Percent",
    "Initial_Stiffness_MPa_per_Strain",
    "Energy_Absorption_MJ_per_m3",
    "Largest_Post_Peak_Drop_Percent",
]


# -----------------------------------------
# MAIN FUNCTION
# -----------------------------------------

def main():

    print("=" * 55)
    print("TENSILE FABRIC - ANOMALY DETECTION")
    print("=" * 55)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\nLoading ML dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("Dataset shape:", df.shape)

    # Validate columns
    missing = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    # Prepare features
    X = df[FEATURE_COLUMNS].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if X.isnull().any().any():
        raise ValueError(
            "Missing or invalid numerical values found."
        )

    # Build model pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "isolation_forest",
            IsolationForest(
                n_estimators=200,
                contamination=0.10,
                random_state=42
            )
        )
    ])

    # Train model
    print("\nTraining Isolation Forest...")

    model.fit(X)

    # Predict anomalies
    predictions = model.predict(X)

    # Calculate anomaly scores
    scores = model.decision_function(X)

    # Add results
    results = df.copy()

    results["Anomaly_Prediction"] = predictions

    results["Anomaly_Status"] = [
        "Potential Anomaly" if value == -1
        else "Typical Pattern"
        for value in predictions
    ]

    results["Anomaly_Score"] = scores

    # Save results
    output_file = (
        OUTPUT_DIR / "anomaly_detection_results.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    # Save model
    model_file = (
        MODEL_DIR / "anomaly_detection_model.pkl"
    )

    joblib.dump(
        model,
        model_file
    )

    # Display results
    print("\nANOMALY DETECTION RESULTS")
    print("-" * 55)

    print(
        results[
            [
                "Sample",
                "Anomaly_Status",
                "Anomaly_Score"
            ]
        ].to_string(index=False)
    )

    anomaly_count = sum(predictions == -1)

    print("\nTotal samples:", len(results))
    print("Potential anomalies:", anomaly_count)
    print("Typical patterns:", len(results) - anomaly_count)

    print("\nSaved results:", output_file)
    print("Saved model:", model_file)

    print("\n" + "=" * 55)
    print("ANOMALY DETECTION COMPLETED")
    print("=" * 55)

    print(
        "\nIMPORTANT: An anomaly is not automatically a BAD sample."
    )


if __name__ == "__main__":
    main()
