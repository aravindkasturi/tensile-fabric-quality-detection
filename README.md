
# Tensile Fabric Quality Detection

## 1. Project Overview

Tensile Fabric Quality Detection is an exploratory machine-learning project designed to analyse the mechanical behaviour of PCL non-woven fabric using tensile test data.

The system processes tensile force and displacement measurements, extracts important mechanical properties, identifies unusual patterns, detects possible force-collapse events, and groups samples using unsupervised machine learning.

The application provides an interactive dashboard built with Streamlit.

## 2. Project Objectives

- Analyse tensile test measurements from 10 PCL fabric samples.
- Calculate engineering stress and strain.
- Extract important mechanical properties.
- Compare mechanical behaviour across samples.
- Group samples into two clusters using K-Means.
- Identify potential anomalies using Isolation Forest.
- Detect possible rupture or force-collapse events.
- Predict the cluster of an uploaded tensile test sample.
- Generate visual reports and downloadable CSV files.

## 3. Technologies Used

- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- VS Code

## 4. Dataset

The project contains 10 tensile test samples:

Test093, Test094, Test095, Test097, Test098, Test099, Test100, Test101, Test102 and Test103.

All samples belong to the same PCL non-woven fabric batch.

The measured sample thickness is 1 mm.

The sample dimensions and tensile test settings are used during mechanical property calculations.

## 5. Mechanical Feature Extraction

The following four features are used by the final clustering model:

1. Maximum Stress (MPa)
2. Maximum Strain (%)
3. Energy Absorption (MJ/m³)
4. Largest Post-Peak Drop (%)

Additional mechanical properties are calculated and displayed in the application, including:

- Maximum Force
- Strain at Maximum Stress
- Initial Stiffness
- Candidate Failure Strain
- Force Collapse Information

## 6. Machine Learning Methods

### K-Means Clustering

A StandardScaler and K-Means pipeline is used to group the 10 samples into two clusters.

Final model configuration:

- Number of clusters: 2
- Random State: 42
- Number of Initializations: 20
- Silhouette Score: 0.3831

### Cluster Interpretation

**Cluster 0 – Lower Mechanical Response**

- Lower average maximum stress
- Lower average maximum strain
- Lower average energy absorption
- Smaller average post-peak drop

**Cluster 1 – Higher Mechanical Response**

- Higher average maximum stress
- Higher average maximum strain
- Higher average energy absorption
- Larger average post-peak drop

These clusters represent relative mechanical behaviour in the analysed dataset.

### Anomaly Detection

Isolation Forest is used to identify samples with unusual mechanical patterns.

The anomaly detection model uses StandardScaler and Isolation Forest.

### Rupture Detection

A threshold-based force-collapse detection method identifies possible rupture events using a sustained force drop.

This is an automated indicator and does not independently confirm physical specimen rupture.

## 7. Streamlit Application

The application contains seven pages:

1. Dashboard – Overall project summary and cluster distribution.
2. Dataset Explorer – Sample-level mechanical properties and dataset exploration.
3. Cluster Analysis – Cluster assignments and mechanical profile comparisons.
4. Anomaly Detection – Potential anomaly identification and score visualizations.
5. Rupture Analysis – Possible force-collapse event analysis.
6. New Sample Prediction – Upload tensile test data and predict its cluster.
7. Reports – Download CSV reports and view generated graphs.

## 8. Project Structure

```text
Tensile_Fabric_Quality_Detection/
│
├── app/
│   ├── app.py
│   └── pages/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── quality_criteria_template.csv
│
├── model/
│   ├── final_tensile_cluster_model.pkl
│   ├── quality_clustering_model.pkl
│   └── anomaly_detection_model.pkl
│
├── results/
│   ├── graphs/
│   ├── eda_graphs/
│   ├── final_analysis_graphs/
│   ├── final_mechanical_features.csv
│   ├── final_cluster_predictions.csv
│   ├── anomaly_detection_results.csv
│   └── rupture_detection_results.csv
│
├── process_datasets.py
├── extract_features.py
├── eda_analysis.py
├── train_final_model.py
├── visualize_cluster_profiles.py
├── detect_rupture.py
├── compare_samples.py
│
├── requirements.txt
└── README.md
```

## 9. Installation and Execution

### Step 1: Clone or download the project

Open the project directory in VS Code.

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Streamlit application

```bash
streamlit run app/app.py
```

The application will open in your browser.

## 10. New Sample Prediction

To predict the cluster of a new tensile test sample:

1. Open New Sample Prediction.
2. Enter the sample gauge length.
3. Enter the sample width.
4. Enter the sample thickness.
5. Upload the tensile test CSV file.
6. Ensure the CSV contains `Force_N` and `Displacement_mm` columns.
7. View the calculated mechanical properties.
8. View the predicted cluster.

The dimensions must match the actual tensile test specimen.

## 11. Limitations

- The current dataset contains only 10 samples from one material batch.
- The clustering model is exploratory and uses unsupervised learning.
- The silhouette score is a clustering evaluation metric, not classification accuracy.
- Cluster assignments do not establish verified material quality.
- The model has limited evidence for generalization to unseen samples or different material batches.
- Rupture detection identifies possible force-collapse events and does not independently confirm physical rupture.
- Maximum strain may be influenced by the displacement limits used in individual tensile tests.

## 12. Conclusion

This project demonstrates an exploratory machine-learning workflow for tensile fabric analysis.

It combines mechanical feature extraction, exploratory data analysis, clustering, anomaly detection, force-collapse analysis and interactive visualization in a single Streamlit application.
