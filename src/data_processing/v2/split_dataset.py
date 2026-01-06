"""
CIC-IDS2018 Dataset Splitter
Splits the dataset into training and test sets for anomaly detection

Training: ONLY Benign traffic (for anomaly detection)
Testing: Mix of Benign + All attack types
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from glob import glob
import numpy as np

print("=" * 80)
print("CIC-IDS2018 DATASET PREPARATION")
print("=" * 80)

# Configuration
DATASET_DIR = "../../../datasets/cic-ids2018"
OUTPUT_DIR = "../../../datasets/cic-ids2018/processed/splits"
TRAIN_TEST_SPLIT = 0.8  # 80% for training, 20% for testing
RANDOM_STATE = 42

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"   Output directory: {OUTPUT_DIR}")

print(f"\n1. Loading dataset files from: {DATASET_DIR}")
csv_files = sorted(glob(os.path.join(DATASET_DIR, "*.csv")))
print(f"   Found {len(csv_files)} CSV files")

# Load all data
print("\n2. Loading and combining all CSV files...")
print("   This may take several minutes for 16M+ rows...")

all_data = []
total_rows = 0

for i, csv_file in enumerate(csv_files, 1):
    filename = os.path.basename(csv_file)
    print(f"   [{i}/{len(csv_files)}] Loading {filename}...", end=" ")

    try:
        # Read CSV
        df = pd.read_csv(csv_file, low_memory=False)
        rows_before = len(df)

        # Clean: Remove rows where Label is literally "Label" (header rows that leaked)
        df = df[df['Label'] != 'Label']
        rows_after = len(df)

        if rows_before != rows_after:
            print(f"(cleaned {rows_before - rows_after} header rows)", end=" ")

        all_data.append(df)
        total_rows += len(df)
        print(f"✓ {len(df):,} rows")

    except Exception as e:
        print(f"✗ ERROR: {e}")
        continue

# Combine all dataframes
print(f"\n3. Combining all data...")
df_combined = pd.concat(all_data, ignore_index=True)
print(f"   Total records: {len(df_combined):,}")
print(f"   Total columns: {df_combined.shape[1]}")

# Handle column inconsistencies (some files have 84 cols, some 80)
print(f"\n4. Standardizing columns...")
# Get the most common set of columns (80 features)
# We'll keep only the 80 standard columns
standard_columns = [
    'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts',
    'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
    'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
    'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
    'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min',
    'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt',
    'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
    'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
    'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg',
    'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
    'Idle Max', 'Idle Min', 'Label'
]

# Keep only standard columns that exist in the dataframe
available_cols = [col for col in standard_columns if col in df_combined.columns]
df_combined = df_combined[available_cols]
print(f"   Standardized to {len(available_cols)} columns")

# Check class distribution
print("\n5. Class Distribution:")
label_counts = df_combined['Label'].value_counts()
for label, count in label_counts.items():
    print(f"   - {label:30s}: {count:10,} ({count/len(df_combined)*100:5.2f}%)")

# Separate Benign and Attack traffic
print("\n6. Separating Benign and Attack traffic...")
benign_data = df_combined[df_combined['Label'] == 'Benign'].copy()
attack_data = df_combined[df_combined['Label'] != 'Benign'].copy()

print(f"   Benign traffic: {len(benign_data):,} samples")
print(f"   Attack traffic: {len(attack_data):,} samples")

# Split Benign data: 80% training, 20% testing
print(f"\n7. Splitting Benign data ({TRAIN_TEST_SPLIT*100}% train, {(1-TRAIN_TEST_SPLIT)*100}% test)...")
benign_train, benign_test = train_test_split(
    benign_data,
    test_size=(1-TRAIN_TEST_SPLIT),
    random_state=RANDOM_STATE
)

print(f"   Training Set (Benign only): {len(benign_train):,} samples")
print(f"   Test Set (Benign portion):  {len(benign_test):,} samples")

# Test set: Combine benign_test + all attacks
print(f"\n8. Creating Test Set (Benign + All Attacks)...")
test_data = pd.concat([benign_test, attack_data], ignore_index=True)

# Shuffle test data
test_data = test_data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"   Test Set Total: {len(test_data):,} samples")
print(f"   - Benign:  {len(benign_test):,} ({len(benign_test)/len(test_data)*100:.2f}%)")
print(f"   - Attack:  {len(attack_data):,} ({len(attack_data)/len(test_data)*100:.2f}%)")

# Handle data quality issues
print(f"\n9. Handling data quality issues...")

# Replace infinite values with NaN
print(f"   Replacing infinite values with NaN...")
benign_train = benign_train.replace([np.inf, -np.inf], np.nan)
test_data = test_data.replace([np.inf, -np.inf], np.nan)

# Drop rows with NaN in critical columns (or fill with median)
# For now, let's fill with 0 (can be improved with better imputation)
print(f"   Filling NaN values with 0...")
benign_train = benign_train.fillna(0)
test_data = test_data.fillna(0)

print(f"   ✓ Data quality issues handled")

# Save the splits
print(f"\n10. Saving splits to: {OUTPUT_DIR}")

train_path = os.path.join(OUTPUT_DIR, "training_benign.csv")
test_path = os.path.join(OUTPUT_DIR, "testing_mixed.csv")

print(f"   Saving training set...")
benign_train.to_csv(train_path, index=False)
print(f"   ✓ Training set saved to: {train_path}")

print(f"   Saving test set...")
test_data.to_csv(test_path, index=False)
print(f"   ✓ Test set saved to: {test_path}")

# Save test labels separately (for evaluation)
test_labels_path = os.path.join(OUTPUT_DIR, "test_labels.csv")
test_labels = test_data['Label']
test_labels.to_csv(test_labels_path, index=False, header=True)
print(f"   ✓ Test labels saved to: {test_labels_path}")

print("\n" + "=" * 80)
print("✅ DATASET PREPARATION COMPLETE!")
print("=" * 80)

print(f"\nDataset Statistics:")
print(f"├── Training (Benign only):  {len(benign_train):,} samples")
print(f"└── Testing (Mixed):         {len(test_data):,} samples")
print(f"    ├── Benign:              {len(benign_test):,} ({len(benign_test)/len(test_data)*100:.1f}%)")
print(f"    └── Attack:              {len(attack_data):,} ({len(attack_data)/len(test_data)*100:.1f}%)")

print(f"\nAttack Types in Test Set:")
attack_labels = test_data[test_data['Label'] != 'Benign']['Label'].value_counts()
for label, count in attack_labels.items():
    print(f"  - {label:30s}: {count:8,}")

print(f"\nFiles created:")
print(f"  ✓ {train_path}")
print(f"  ✓ {test_path}")
print(f"  ✓ {test_labels_path}")

print(f"\nNext step: Run feature_selection.py to prepare features for training")
