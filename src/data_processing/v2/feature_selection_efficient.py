"""
CIC-IDS2018 Feature Selection and Preparation (Memory-Efficient Version)
Processes data in chunks to avoid memory issues
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import os

print("=" * 80)
print("CIC-IDS2018 FEATURE SELECTION (MEMORY-EFFICIENT)")
print("=" * 80)

# Paths
SPLITS_DIR = "../../../datasets/cic-ids2018/processed/splits"
FEATURES_DIR = "../../../datasets/cic-ids2018/processed/features"

TRAIN_PATH = os.path.join(SPLITS_DIR, "training_benign.csv")
TEST_PATH = os.path.join(SPLITS_DIR, "testing_mixed.csv")
TEST_LABELS_PATH = os.path.join(SPLITS_DIR, "test_labels.csv")

# Create output directory
os.makedirs(FEATURES_DIR, exist_ok=True)

CHUNK_SIZE = 100000

print(f"\n1. Analyzing data structure...")
# Read first chunk to understand columns
first_chunk = pd.read_csv(TRAIN_PATH, nrows=1000, low_memory=False)
print(f"   Total columns in data: {len(first_chunk.columns)}")

# Features to remove (non-numeric or not useful for ML)
features_to_remove = [
    'Label',  # Target variable
    'Timestamp',  # Not useful for anomaly detection
    'Dst Port',  # Too specific, high cardinality
    'Protocol',  # Categorical, will be dropped for simplicity
]

# Get columns to keep
columns_to_keep = [col for col in first_chunk.columns if col not in features_to_remove]
print(f"   Features after removing non-useful columns: {len(columns_to_keep)}")

print(f"\n2. Computing variance for feature selection...")
print(f"   Reading training data to calculate variance...")

# Calculate variance on a large sample (first 500K rows)
sample_data = pd.read_csv(TRAIN_PATH, nrows=500000, usecols=columns_to_keep, low_memory=False)
print(f"   Sample size for variance calculation: {len(sample_data):,} rows")

# Calculate variance
variances = sample_data.var()

# Find zero or near-zero variance features
low_variance_features = variances[variances < 1e-10].index.tolist()
print(f"   Features with near-zero variance: {len(low_variance_features)}")

if len(low_variance_features) > 0:
    print(f"   Removing features: {low_variance_features[:10]}{'...' if len(low_variance_features) > 10 else ''}")

# Update columns to keep
columns_to_keep = [col for col in columns_to_keep if col not in low_variance_features]
print(f"   Features after variance filter: {len(columns_to_keep)}")

print(f"\n3. Checking for highly correlated features...")
# Calculate correlation on the sample
correlation_matrix = sample_data[columns_to_keep].corr().abs()

# Find features with correlation > 0.95
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

if len(to_drop) > 0:
    print(f"   Removing {len(to_drop)} highly correlated features:")
    for col in to_drop[:10]:
        print(f"     - {col}")
    if len(to_drop) > 10:
        print(f"     ... and {len(to_drop) - 10} more")
    columns_to_keep = [col for col in columns_to_keep if col not in to_drop]
else:
    print(f"   ✓ No highly correlated features found")

print(f"   ✓ Final feature count: {len(columns_to_keep)}")

print(f"\n4. Final feature set ({len(columns_to_keep)} features):")
for i, col in enumerate(columns_to_keep, 1):
    print(f"   {i:2d}. {col}")

print(f"\n5. Processing and saving features...")

# Output paths
TRAIN_FEATURES_PATH = os.path.join(FEATURES_DIR, "train_features.csv")
TEST_FEATURES_PATH = os.path.join(FEATURES_DIR, "test_features.csv")
TEST_LABELS_COPY_PATH = os.path.join(FEATURES_DIR, "test_labels.csv")

# Process training data in chunks
print(f"\n   Processing training data in chunks...")
first_write = True
processed_rows = 0

for chunk in pd.read_csv(TRAIN_PATH, chunksize=CHUNK_SIZE, usecols=columns_to_keep, low_memory=False):
    # Write to output
    chunk.to_csv(
        TRAIN_FEATURES_PATH,
        mode='w' if first_write else 'a',
        header=first_write,
        index=False
    )
    first_write = False
    processed_rows += len(chunk)

    if processed_rows % 1000000 == 0:
        print(f"      Processed {processed_rows:,} training rows...")

print(f"   ✓ Training features saved: {TRAIN_FEATURES_PATH}")
print(f"      Total training samples: {processed_rows:,}")

# Process test data in chunks
print(f"\n   Processing test data in chunks...")
first_write = True
processed_rows = 0

for chunk in pd.read_csv(TEST_PATH, chunksize=CHUNK_SIZE, usecols=columns_to_keep, low_memory=False):
    # Write to output
    chunk.to_csv(
        TEST_FEATURES_PATH,
        mode='w' if first_write else 'a',
        header=first_write,
        index=False
    )
    first_write = False
    processed_rows += len(chunk)

    if processed_rows % 1000000 == 0:
        print(f"      Processed {processed_rows:,} test rows...")

print(f"   ✓ Test features saved: {TEST_FEATURES_PATH}")
print(f"      Total test samples: {processed_rows:,}")

# Copy test labels
print(f"\n   Copying test labels...")
test_labels = pd.read_csv(TEST_LABELS_PATH)
test_labels.to_csv(TEST_LABELS_COPY_PATH, index=False)
print(f"   ✓ Test labels copied: {TEST_LABELS_COPY_PATH}")

print("\n" + "=" * 80)
print("✅ FEATURE SELECTION COMPLETE!")
print("=" * 80)

print(f"\nFeature Summary:")
print(f"  Original features:         79")
print(f"  After removing non-useful: {len([c for c in first_chunk.columns if c not in features_to_remove])}")
print(f"  After variance filter:     {len(columns_to_keep) + len(low_variance_features)}")
print(f"  After correlation filter:  {len(columns_to_keep)}")

print(f"\nFiles created:")
print(f"  ✓ {TRAIN_FEATURES_PATH}")
print(f"  ✓ {TEST_FEATURES_PATH}")
print(f"  ✓ {TEST_LABELS_COPY_PATH}")

# Show basic stats on a sample
print(f"\nBasic statistics (sample from training data):")
sample = pd.read_csv(TRAIN_FEATURES_PATH, nrows=10000)
stats = sample.describe().T[['mean', 'std', 'min', 'max']]
print(stats)

print(f"\n" + "=" * 80)
print(f"📊 DATA READY FOR MODEL TRAINING!")
print("=" * 80)
print(f"\nYou can now train models using:")
print(f"  - Training features: {len(columns_to_keep)} features × 10.7M samples")
print(f"  - Test features:     {len(columns_to_keep)} features × 5.4M samples")
print(f"\nNext step: Create and run v2 model training scripts")
