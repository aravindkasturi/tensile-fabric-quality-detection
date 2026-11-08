
from pathlib import Path
import pandas as pd

# Project data directory
DATA_DIR = Path("data/raw")

# Expected sample folders
SAMPLES = [
    "Test093",
    "Test094",
    "Test095",
    "Test097",
    "Test098",
    "Test099",
    "Test100",
    "Test101",
    "Test102",
    "Test103",
]


def inspect_sample(sample_name):
    sample_folder = DATA_DIR / sample_name

    print("\n" + "=" * 65)
    print(f"INSPECTING: {sample_name}")
    print("=" * 65)

    if not sample_folder.exists():
        print("ERROR: Sample folder not found!")
        return

    # Find CSV and TST files
    csv_files = list(sample_folder.glob("*.csv"))
    tst_files = list(sample_folder.glob("*.tst"))

    print("\nCSV files:")
    for file in csv_files:
        print(f"  - {file.name}")

    print("\nTST settings files:")
    for file in tst_files:
        print(f"  - {file.name}")

    # Inspect measurement CSV
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            print(f"\nMeasurement file: {csv_file.name}")
            print(f"Rows: {df.shape[0]}")
            print(f"Columns: {df.shape[1]}")
            print("Column names:")
            print(df.columns.tolist())
            print("\nFirst 5 rows:")
            print(df.head().to_string(index=False))

        except Exception as error:
            print(f"Could not read {csv_file.name}: {error}")

    # Display test settings
    for tst_file in tst_files:
        try:
            print(f"\nSettings preview: {tst_file.name}")
            content = tst_file.read_text(
                encoding="utf-8",
                errors="replace"
            )
            print(content[:2000])

        except Exception as error:
            print(f"Could not read {tst_file.name}: {error}")


if __name__ == "__main__":
    for sample in SAMPLES:
        inspect_sample(sample)

    print("\n" + "=" * 65)
    print("DATASET INSPECTION COMPLETED")
    print("=" * 65)
