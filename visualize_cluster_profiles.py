
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load final mechanical features and final model predictions
features = pd.read_csv("results/final_mechanical_features.csv")
predictions = pd.read_csv("results/final_cluster_predictions.csv")

# Merge using the final model's actual cluster assignments
data = features.merge(predictions, on="Sample")

# These are the exact four features used by the final model
columns = [
    "Maximum_Stress_MPa",
    "Maximum_Strain_Percent",
    "Energy_Absorption_MJ_per_m3",
    "Largest_Post_Peak_Drop_Percent"
]

# Calculate average values for each final cluster
profile = data.groupby("Cluster")[columns].mean()

# Standardize feature averages for visualization only
overall_mean = data[columns].mean()
overall_std = data[columns].std(ddof=0).replace(0, 1)

profile_standardized = (profile - overall_mean) / overall_std

# Create graph
plt.figure(figsize=(12, 7))

for cluster in profile_standardized.index:
    plt.plot(
        profile_standardized.columns,
        profile_standardized.loc[cluster],
        marker="o",
        linewidth=2,
        label=f"Cluster {cluster}"
    )

plt.axhline(0, linestyle="--", linewidth=1)
plt.xticks(rotation=25, ha="right")
plt.ylabel("Standardized feature average")
plt.title("Mechanical Feature Profiles - Final Four-Feature Model")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save graph
output_dir = "results/final_analysis_graphs"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "cluster_profiles.png")

plt.savefig(output_path, dpi=300)
plt.close()

print("Final cluster profile graph created successfully.")
print(f"Saved to: {output_path}")

print("\nFinal cluster assignments:")
print(data[["Sample", "Cluster"]].sort_values("Sample").to_string(index=False))
