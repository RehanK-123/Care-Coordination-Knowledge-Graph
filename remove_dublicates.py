"""
Remove Duplicates from CSV Files
=================================
Reads all CSV files in the output/ directory, removes identical duplicate rows,
and overwrites the cleaned files back to the same location.
"""

import pandas as pd
import os
import glob

# Path to CSV files
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def remove_duplicates():
    """Read all CSVs, remove duplicate rows, and save back."""
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))

    if not csv_files:
        print(f"❌ No CSV files found in {OUTPUT_DIR}")
        return

    print("=" * 60)
    print("  DUPLICATE REMOVAL REPORT")
    print("=" * 60)
    print()

    total_removed = 0

    for filepath in sorted(csv_files):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        original_count = len(df)

        # Remove exact duplicate rows (keep first occurrence)
        df_clean = df.drop_duplicates()
        clean_count = len(df_clean)
        duplicates = original_count - clean_count

        if duplicates > 0:
            # Overwrite the file with cleaned data
            df_clean.to_csv(filepath, index=False)
            status = f"🧹 Removed {duplicates} duplicates"
        else:
            status = "✅ No duplicates"

        print(f"  {filename:<25} {original_count:>8} → {clean_count:>8}  {status}")
        total_removed += duplicates

    print()
    print("-" * 60)
    print(f"  Total duplicates removed: {total_removed}")
    print("=" * 60)


if __name__ == "__main__":
    remove_duplicates()
