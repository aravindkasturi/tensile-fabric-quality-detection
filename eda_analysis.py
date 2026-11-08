
from pathlib import Path

import pandas as pd
import numpy as np


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

FEATURE_FILE = (
    PROJECT_DIR / "results" / "mechanical_features.csv"
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(FEATURE_FILE)


# --------------------------------------------------
# BASIC DATASET INFORMATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("TENSILE FABRIC DATASET - EDA")
print("=" * 60)

print("\n1. DATASET SHAPE")
print(df.shape)

print("\n2. COLUMN NAMES")
print(df.columns.tolist())

print("\n3. FIRST FIVE RECORDS")
print(df.head())

print("\n4. DATA TYPES")
print(df.dtypes)

print("\n5. MISSING VALUES")
print(df.isnull().sum())

print("\n6. DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n7. NUMERICAL SUMMARY")
print(df.describe().round(4))

print("\n" + "=" * 60)
print("BASIC EDA COMPLETED")
print("=" * 60)


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# EDA VISUALIZATIONS
# --------------------------------------------------

GRAPH_DIR = PROJECT_DIR / "results" / "eda_graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# 1. Maximum Stress Distribution

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Sample",
    y="Maximum_Stress_MPa"
)

plt.title("Maximum Stress Across Samples")
plt.xlabel("Sample")
plt.ylabel("Maximum Stress (MPa)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "maximum_stress.png",
    dpi=300
)

plt.close()


# 2. Energy Absorption Comparison

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Sample",
    y="Energy_Absorption_MJ_per_m3"
)

plt.title("Energy Absorption Across Samples")
plt.xlabel("Sample")
plt.ylabel("Energy Absorption (MJ/m³)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "energy_absorption.png",
    dpi=300
)

plt.close()


# 3. Correlation Heatmap

numerical_df = df.select_dtypes(
    include=["number"]
)

correlation = numerical_df.corr()

plt.figure(figsize=(13, 10))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Mechanical Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.close()


print("\nEDA VISUALIZATIONS COMPLETED")
print(f"Graphs saved in: {GRAPH_DIR}")
