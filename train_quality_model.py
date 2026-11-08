
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_DIR / "results" / "mechanical_features.csv"

OUTPUT_DIR = PROJECT_DIR / "model"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_FILE = OUTPUT_DIR / "quality_clustering_model.pkl"
RESULT_FILE = RESULTS_DIR / "quality_predictions.csv"


# ==========================================================
# SELECTED FEATURES
# ==========================================================

FEATURE_COLUMNS = [
    "Maximum_Stress_MPa",
    "Initial_Stiffness_MPa_per_Strain",
    "Energy_Absorption_MJ_per_m3"
]


# ==========================================================
# LOAD DATA
# ==========================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Mechanical feature file not found: {INPUT_FILE}"
    )

data = pd.read_csv(INPUT_FILE)

if "Sample" not in data.columns:
    raise ValueError("Sample column is missing.")

missing_columns = [
    column for column in FEATURE_COLUMNS
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        "Missing required features: " + ", ".join(missing_columns)
    )

data["Sample"] = data["Sample"].astype(str)

X = data[FEATURE_COLUMNS].apply(
    pd.to_numeric,
    errors="coerce"
)

if X.isnull().any().any():
    raise ValueError(
        "Missing or invalid feature values found. "
        "Please clean the mechanical feature dataset first."
    )

if len(X) < 3:
    raise ValueError("At least three samples are required.")

if len(X.drop_duplicates()) < 2:
    raise ValueError("At least two distinct feature patterns are required.")


# ==========================================================
# BUILD MACHINE LEARNING PIPELINE
# ==========================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "kmeans",
        KMeans(
            n_clusters=2,
            random_state=42,
            n_init=20
        )
    )
])


# ==========================================================
# TRAIN CLUSTERING MODEL
# ==========================================================

cluster_ids = pipeline.fit_predict(X)


# ==========================================================
# ASSIGN RELATIVE GOOD / BAD NAMES
# ==========================================================

scaled_features = pipeline.named_steps["scaler"].transform(X)

cluster_means = {}

for cluster_id in [0, 1]:

    cluster_rows = scaled_features[
        cluster_ids == cluster_id
    ]

    cluster_means[cluster_id] = float(
        cluster_rows.mean()
    )

# Higher average standardized mechanical values
# are designated GOOD for this prototype.
good_cluster = max(
    cluster_means,
    key=cluster_means.get
)

predicted_labels = [
    "GOOD" if cluster_id == good_cluster else "BAD"
    for cluster_id in cluster_ids
]


# ==========================================================
# PREPARE PREDICTION RESULTS
# ==========================================================

results = data[["Sample"]].copy()

for column in FEATURE_COLUMNS:
    results[column] = X[column]

results["Cluster_ID"] = cluster_ids
results["Predicted_Quality"] = predicted_labels

results["Assessment_Type"] = (
    "Unsupervised relative performance grouping"
)


# ==========================================================
# MODEL EVALUATION
# ==========================================================

if len(set(cluster_ids)) == 2 and len(X) > 2:

    score = silhouette_score(
        pipeline.named_steps["scaler"].transform(X),
        cluster_ids
    )

else:
    score = np.nan


# ==========================================================
# SAVE MODEL AND RESULTS
# ==========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "good_cluster": int(good_cluster),
        "cluster_means": cluster_means
    },
    MODEL_FILE
)

results.to_csv(
    RESULT_FILE,
    index=False
)


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print("\n" + "=" * 65)
print("TENSILE FABRIC QUALITY CLUSTERING MODEL")
print("=" * 65)

print("\nFeatures used:")

for column in FEATURE_COLUMNS:
    print("-", column)

print("\nCluster average standardized performance:")

for cluster_id, mean_value in cluster_means.items():

    label = (
        "GOOD"
        if cluster_id == good_cluster
        else "BAD"
    )

    print(
        f"Cluster {cluster_id} ({label}): {mean_value:.4f}"
    )

print("\nSAMPLE-LEVEL PREDICTIONS")

print(results.to_string(index=False))

print("\nPREDICTION SUMMARY")

print(
    results["Predicted_Quality"]
    .value_counts()
    .reindex(["GOOD", "BAD"], fill_value=0)
    .to_string()
)

print("\nSilhouette Score:", round(score, 4) if pd.notna(score) else "N/A")

print("\nModel saved:", MODEL_FILE)
print("Predictions saved:", RESULT_FILE)

print(
    "\nIMPORTANT: GOOD/BAD here means higher/lower relative "
    "mechanical performance within this small dataset. "
    "It is not validated product quality."
)
