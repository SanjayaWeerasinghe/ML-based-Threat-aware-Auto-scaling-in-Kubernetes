"""
CIC-IDS2018 Feature Selection and Preparation
Prepares the 79 features for model training by removing non-useful features
and selecting the most important ones
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import os

print("=" * 80)
print("CIC-IDS2018 FEATURE SELECTION AND PREPARATION")
print("=" * 80)

# Paths
SPLITS_DIR = "../../../datasets/cic-ids2018/processed/splits"
FEATURES_DIR = "../../../datasets/cic-ids2018/processed/features"

# Input paths (from split_dataset.py)
TRAIN_PATH = os.path.join(SPLITS_DIR, "training_benign.csv")
TEST_PATH = os.path.join(SPLITS_DIR, "testing_mixed.csv")
TEST_LABELS_PATH = os.path.join(SPLITS_DIR, "test_labels.csv")

# Create output directory
os.makedirs(FEATURES_DIR, exist_ok=True)

print(f"\n1. Loading prepared datasets...")
print(f"   Training data: {TRAIN_PATH}")
print(f"   Test data: {TEST_PATH}")

# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
test_labels = pd.read_csv(TEST_LABELS_PATH)

print(f"   ✓ Training samples: {len(train_df):,}")
print(f"   ✓ Test samples: {len(test_df):,}")
print(f"   ✓ Total features: {train_df.shape[1] - 1} (excluding Label)")

print(f"\n2. Removing non-numeric and non-useful features...")

# Features to remove (non-numeric or not useful for ML)
features_to_remove = [
    'Label',  # Target variable
    'Timestamp',  # Not useful for anomaly detection
    'Dst Port',  # Too specific, high cardinality
    'Protocol',  # Categorical, will be dropped for simplicity
]

# Remove these columns if they exist
for col in features_to_remove:
    if col in train_df.columns:
        train_df = train_df.drop(columns=[col])
        test_df = test_df.drop(columns=[col])
        print(f"   Removed: {col}")

print(f"   ✓ Features after removal: {train_df.shape[1]}")

print(f"\n3. Checking for low-variance features...")

# Remove features with zero or very low variance
selector = VarianceThreshold(threshold=0.0)  # Remove features with zero variance
train_features = selector.fit_transform(train_df)
test_features = selector.transform(test_df)

# Get selected feature names
selected_feature_mask = selector.get_support()
selected_features = train_df.columns[selected_feature_mask].tolist()

print(f"   Features removed due to zero variance: {train_df.shape[1] - len(selected_features)}")
print(f"   ✓ Remaining features: {len(selected_features)}")

# Convert back to DataFrames with selected features
train_df = pd.DataFrame(train_features, columns=selected_features)
test_df = pd.DataFrame(test_features, columns=selected_features)

print(f"\n4. Checking for highly correlated features...")

# Calculate correlation matrix
correlation_matrix = train_df.corr().abs()

# Find features with correlation > 0.95
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

if len(to_drop) > 0:
    print(f"   Removing {len(to_drop)} highly correlated features:")
    for col in to_drop:
        print(f"     - {col}")
    train_df = train_df.drop(columns=to_drop)
    test_df = test_df.drop(columns=to_drop)
else:
    print(f"   ✓ No highly correlated features found")

print(f"   ✓ Final feature count: {train_df.shape[1]}")

print(f"\n5. Final feature set:")
for i, col in enumerate(train_df.columns, 1):
    print(f"   {i:2d}. {col}")

# Save processed features
print(f"\n6. Saving processed features to: {FEATURES_DIR}")

TRAIN_FEATURES_PATH = os.path.join(FEATURES_DIR, "train_features.csv")
TEST_FEATURES_PATH = os.path.join(FEATURES_DIR, "test_features.csv")
TEST_LABELS_COPY_PATH = os.path.join(FEATURES_DIR, "test_labels.csv")

train_df.to_csv(TRAIN_FEATURES_PATH, index=False)
test_df.to_csv(TEST_FEATURES_PATH, index=False)

# Also copy test_labels to features directory for convenience
test_labels.to_csv(TEST_LABELS_COPY_PATH, index=False)

print(f"   ✓ Training features saved: {TRAIN_FEATURES_PATH}")
print(f"   ✓ Test features saved: {TEST_FEATURES_PATH}")
print(f"   ✓ Test labels copied: {TEST_LABELS_COPY_PATH}")

print("\n" + "=" * 80)
print("✅ FEATURE SELECTION COMPLETE!")
print("=" * 80)

print(f"\nFeature Summary:")
print(f"  Original features:       79")
print(f"  After non-numeric removal: ~75")
print(f"  After variance filter:   {len(selected_features)}")
print(f"  After correlation filter: {train_df.shape[1]}")
print(f"\nFinal dataset dimensions:")
print(f"  Training: {len(train_df):,} samples × {train_df.shape[1]} features")
print(f"  Testing:  {len(test_df):,} samples × {test_df.shape[1]} features")

print(f"\nFiles created:")
print(f"  ✓ {TRAIN_FEATURES_PATH}")
print(f"  ✓ {TEST_FEATURES_PATH}")
print(f"  ✓ {TEST_LABELS_PATH} (already exists)")

print(f"\nBasic statistics (training data):")
print(train_df.describe().T[['mean', 'std', 'min', 'max']])

print(f"\nNext step: Train models with v2 training scripts (src/models/v2/)")
