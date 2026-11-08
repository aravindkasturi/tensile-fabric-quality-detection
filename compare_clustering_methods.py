import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# Project paths
INPUT_FILE = "results/final_mechanical_features.csv"
OUTPUT_DIR = "results"

# Load the final feature dataset
df = pd.read_csv(INPUT_FILE)

# Define feature sets
features_4 = [
    "Maximum_Stress_MPa",
    "Maximum_Strain_Percent",
    "Energy_Absorption_MJ_per_m3",
    "Largest_Post_Peak_Drop_Percent"
]

features_6 = [
    "Maximum_Stress_MPa",
    "Maximum_Strain_Percent",
    "Energy_Absorption_MJ_per_m3",
    "Largest_Post_Peak_Drop_Percent",
    "Strain_At_Max_Stress_Percent",
    "Initial_Stiffness_MPa_per_Strain"
]

feature_sets = {
    "4_Features": features_4,
    "6_Features": features_6
}

methods = {
    "KMeans": KMeans(
        n_clusters=2,
        random_state=42,
        n_init=20
    ),
    "Hierarchical": AgglomerativeClustering(
        n_clusters=2,
        linkage="ward"
    )
}

metrics = []
all_labels = []

for feature_set_name, selected_features in feature_sets.items():

    print(f"\nAnalyzing {feature_set_name}...")

    # Select features
    X = df[selected_features].copy()

    # Check for missing values
    if X.isnull().any().any():
        raise ValueError(
            f"Missing values found in {feature_set_name}"
        )

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for method_name, model in methods.items():

        # Fit clustering model
        cluster_labels = model.fit_predict(X_scaled)

        # Calculate silhouette score
        score = silhouette_score(
            X_scaled,
            cluster_labels
        )

        # Record metrics
        metrics.append({
            "Feature_Set": feature_set_name,
            "Method": method_name,
            "Silhouette_Score": score
        })

        # Record sample-level cluster assignments
        for sample, cluster in zip(
            df["Sample"],
            cluster_labels
        ):
            all_labels.append({
                "Sample": sample,
                "Feature_Set": feature_set_name,
                "Method": method_name,
                "Cluster": int(cluster)
            })

        print(
            f"{method_name}: "
            f"Silhouette Score = {score:.4f}"
        )


# Save comparison metrics
metrics_df = pd.DataFrame(metrics)

metrics_path = os.path.join(
    OUTPUT_DIR,
    "clustering_comparison_metrics.csv"
)

metrics_df.to_csv(metrics_path, index=False)


# Save cluster assignments
labels_df = pd.DataFrame(all_labels)

labels_path = os.path.join(
    OUTPUT_DIR,
    "clustering_comparison_labels.csv"
)

labels_df.to_csv(labels_path, index=False)


print("\n========== COMPARISON SUMMARY ==========")
print(metrics_df.to_string(index=False))

print("\nFiles saved successfully:")
print(metrics_path)
print(labels_path)