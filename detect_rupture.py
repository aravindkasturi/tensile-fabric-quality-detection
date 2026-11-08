
import os
import glob
import numpy as np
import pandas as pd

INPUT_DIR = "data/processed"
OUTPUT_DIR = "results"

# Rupture detection settings
DROP_THRESHOLD = 0.80       # At least 30% force reduction
SUSTAINED_POINTS = 5        # Drop must persist for 3 readings
ROLLING_WINDOW = 5          # Smooth small fluctuations

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []

files = sorted(glob.glob(
    os.path.join(INPUT_DIR, "*_processed.csv")
))

if not files:
    raise FileNotFoundError("No processed CSV files found.")

for file in files:
    sample = os.path.basename(file).replace("_processed.csv", "")
    df = pd.read_csv(file)

    required = ["Force_N", "Displacement_mm"]

    if not all(col in df.columns for col in required):
        print(f"Skipping {sample}: required columns missing.")
        continue

    df = df[required].apply(pd.to_numeric, errors="coerce").dropna()
    df = df.reset_index(drop=True)

    if len(df) < ROLLING_WINDOW + SUSTAINED_POINTS:
        print(f"Skipping {sample}: insufficient readings.")
        continue

    force = df["Force_N"].rolling(
        ROLLING_WINDOW, center=True, min_periods=1
    ).median().to_numpy()

    displacement = df["Displacement_mm"].to_numpy()

    peak_force = 0.0
    rupture_found = False
    rupture_index = None

    for i in range(ROLLING_WINDOW, len(force) - SUSTAINED_POINTS):
        peak_force = max(peak_force, force[i - 1])

        if peak_force <= 0:
            continue

        drop_fraction = (peak_force - force[i]) / peak_force

        if drop_fraction >= DROP_THRESHOLD:
            following = force[i:i + SUSTAINED_POINTS]

            if np.all(following <= peak_force * (1 - DROP_THRESHOLD)):
                rupture_found = True
                rupture_index = i
                break

    if rupture_found:
        results.append({
            "Sample": sample,
            "Rupture_Detected": "Yes",
            "Rupture_Displacement_mm": round(
                displacement[rupture_index], 3
            ),
            "Force_Before_Rupture_N": round(peak_force, 4),
            "Force_After_Rupture_N": round(
                force[rupture_index], 4
            ),
            "Force_Drop_Percent": round(
                (peak_force - force[rupture_index])
                / peak_force * 100, 2
            )
        })

        print(f"{sample}: Possible rupture detected")
    else:
        results.append({
            "Sample": sample,
            "Rupture_Detected": "No",
            "Rupture_Displacement_mm": np.nan,
            "Force_Before_Rupture_N": np.nan,
            "Force_After_Rupture_N": np.nan,
            "Force_Drop_Percent": np.nan
        })

        print(f"{sample}: No rupture detected")

output = pd.DataFrame(results)

output.to_csv(
    os.path.join(OUTPUT_DIR, "rupture_detection_results.csv"),
    index=False
)

print("\nRUPTURE DETECTION SUMMARY")
print(output.to_string(index=False))
print("\nSaved: results/rupture_detection_results.csv")
