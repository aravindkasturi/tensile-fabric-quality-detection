import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Paths
INPUT_FILE = "results/final_mechanical_features.csv"
MODEL_DIR = "model"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Select four mechanical features
FEATURES = [
    "Maximum_Stress_MPa",
    "Maximum_Strain_Percent",
    "Energy_Absorption_MJ_per_m3",
    "Largest_Post_Peak_Drop_Percent"
]

X = df[FEATURES].copy()

if X.isnull().any().any():
    raise ValueError("Missing values found in selected features.")

# Create ML pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=2,
        random_state=42,
        n_init=20
    ))
])

# Train model
cluster_labels = pipeline.fit_predict(X)

# Calculate silhouette score
score = silhouette_score(
    pipeline.named_steps["scaler"].transform(X),
    cluster_labels
)

# Save model and feature information
model_bundle = {
    "pipeline": pipeline,
    "features": FEATURES,
    "silhouette_score": score,
    "training_samples": len(df)
}

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_tensile_cluster_model.pkl"
)

joblib.dump(model_bundle, MODEL_PATH)

# Save training predictions
predictions = pd.DataFrame({
    "Sample": df["Sample"],
    "Cluster": cluster_labels
})

PREDICTION_PATH = os.path.join(
    RESULTS_DIR,
    "final_cluster_predictions.csv"
)

predictions.to_csv(PREDICTION_PATH, index=False)

# Display results
print("\n========== FINAL MODEL TRAINING ==========")
print(f"Training samples: {len(df)}")
print(f"Features used: {len(FEATURES)}")
print(f"Silhouette Score: {score:.4f}")

print("\nCluster distribution:")
print(predictions["Cluster"].value_counts().sort_index())

print("\nSample predictions:")
print(predictions.to_string(index=False))

print("\nModel saved:", MODEL_PATH)
print("Predictions saved:", PREDICTION_PATH)