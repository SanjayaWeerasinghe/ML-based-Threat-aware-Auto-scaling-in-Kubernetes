"""
CIC-IDS2018 Dataset Exploration
Analyzes the dataset structure, labels, features, and quality
"""

import pandas as pd
import numpy as np
import os
from glob import glob
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CIC-IDS2018 DATASET EXPLORATION")
print("=" * 80)

# Dataset path (from src/data_processing/v2/ go up 3 levels to project root)
dataset_dir = "../../../datasets/cic-ids2018"
csv_files = sorted(glob(os.path.join(dataset_dir, "*.csv")))

print(f"\n1. Found {len(csv_files)} CSV files:")
print("-" * 80)

total_samples = 0
all_labels = {}
file_info = []

# Analyze each file
for i, csv_file in enumerate(csv_files, 1):
    filename = os.path.basename(csv_file)
    file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB

    print(f"\n[{i}/{len(csv_files)}] Analyzing: {filename}")
    print(f"    File size: {file_size:.2f} MB")

    try:
        # Read first few rows to get structure
        df_sample = pd.read_csv(csv_file, nrows=1000)

        # Count total rows (efficiently)
        with open(csv_file, 'r') as f:
            row_count = sum(1 for line in f) - 1  # Subtract header

        print(f"    Total rows: {row_count:,}")
        print(f"    Columns: {len(df_sample.columns)}")
        total_samples += row_count

        # Analyze labels in this file
        print(f"    Reading all labels...")
        df_labels = pd.read_csv(csv_file, usecols=['Label'])
        label_counts = df_labels['Label'].value_counts()

        print(f"    Label distribution:")
        for label, count in label_counts.items():
            print(f"      - {label:30s}: {count:10,} ({count/row_count*100:5.2f}%)")
            all_labels[label] = all_labels.get(label, 0) + count

        # Store file info
        file_info.append({
            'filename': filename,
            'size_mb': file_size,
            'rows': row_count,
            'labels': dict(label_counts)
        })

    except Exception as e:
        print(f"    ERROR: {e}")
        continue

print("\n" + "=" * 80)
print("2. OVERALL DATASET SUMMARY")
print("=" * 80)

print(f"\nTotal samples across all files: {total_samples:,}")
print(f"\nLabel distribution (entire dataset):")
print("-" * 80)

# Sort labels: Benign first, then attacks alphabetically
sorted_labels = sorted(all_labels.items(), key=lambda x: (x[0] != 'Benign', x[0]))

for label, count in sorted_labels:
    percentage = (count / total_samples) * 100
    print(f"  {label:30s}: {count:12,} ({percentage:6.2f}%)")

# Calculate benign vs attack ratio
benign_count = all_labels.get('Benign', 0)
attack_count = total_samples - benign_count

if total_samples > 0:
    print(f"\n{'Benign (Normal)':30s}: {benign_count:12,} ({benign_count/total_samples*100:6.2f}%)")
    print(f"{'Attack (All types)':30s}: {attack_count:12,} ({attack_count/total_samples*100:6.2f}%)")
else:
    print("\n⚠️ ERROR: No data found! Check if CSV files exist in the dataset directory.")
    print(f"Looking in: {os.path.abspath(dataset_dir)}")
    exit(1)

print("\n" + "=" * 80)
print("3. FEATURE ANALYSIS")
print("=" * 80)

# Load one file to analyze features
print(f"\nLoading sample from: {os.path.basename(csv_files[0])}")
df_sample = pd.read_csv(csv_files[0], nrows=10000)

print(f"\nTotal features: {len(df_sample.columns)}")
print(f"Features (excluding Label): {len(df_sample.columns) - 1}")

print(f"\nColumn names:")
for i, col in enumerate(df_sample.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nData types:")
print(df_sample.dtypes.value_counts())

# Check for missing values
print(f"\nMissing values check (first 10,000 rows):")
missing = df_sample.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values found")
else:
    print(f"  Columns with missing values:")
    for col, count in missing[missing > 0].items():
        print(f"    - {col}: {count} ({count/len(df_sample)*100:.2f}%)")

# Check for infinite values
print(f"\nInfinite values check:")
numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
inf_count = 0
for col in numeric_cols:
    inf_in_col = np.isinf(df_sample[col]).sum()
    if inf_in_col > 0:
        inf_count += inf_in_col
        print(f"  - {col}: {inf_in_col} infinite values")

if inf_count == 0:
    print("  ✓ No infinite values found")

# Basic statistics
print(f"\nBasic statistics (sample):")
print(df_sample.describe().T[['mean', 'std', 'min', 'max']])

print("\n" + "=" * 80)
print("4. DATA QUALITY SUMMARY")
print("=" * 80)

print(f"""
✓ Total Files:        {len(csv_files)}
✓ Total Samples:      {total_samples:,}
✓ Benign Samples:     {benign_count:,} ({benign_count/total_samples*100:.2f}%)
✓ Attack Samples:     {attack_count:,} ({attack_count/total_samples*100:.2f}%)
✓ Attack Types:       {len([l for l in all_labels.keys() if l != 'Benign'])}
✓ Features:           {len(df_sample.columns) - 1} (excluding Label)
""")

print("\n" + "=" * 80)
print("5. RECOMMENDATIONS FOR V2 PROCESSING")
print("=" * 80)

print("""
Based on this analysis:

1. DATA SPLITTING STRATEGY:
   - Training set: Use ONLY 'Benign' traffic (anomaly detection approach)
   - Test set: Mix of 'Benign' + all attack types
   - Recommended split: 80% Benign for training, 20% Benign + all attacks for testing

2. FEATURE ENGINEERING:
   - CIC-IDS2018 already has 79 pre-extracted features (no manual extraction needed!)
   - Consider feature selection to reduce dimensionality
   - Handle infinite values if found
   - Standardize/normalize features before training

3. LABEL HANDLING:
   - Binary mapping: 'Benign' = 1 (normal), everything else = -1 (anomaly)
   - Alternative: Keep specific attack types for detailed evaluation

4. DATA SIZE CONSIDERATIONS:
   - Dataset is LARGE (16M+ samples)
   - Consider sampling for initial prototyping
   - Use chunked processing for full dataset training

5. ATTACK DIVERSITY:
   - Multiple attack types provide robust testing
   - Can evaluate model performance per attack category
""")

print("\n" + "=" * 80)
print("✅ EXPLORATION COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print("  1. Run feature_selection.py to reduce dimensionality")
print("  2. Run split_dataset.py to create train/test splits")
print("  3. Train models with v2 training scripts")
