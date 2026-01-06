"""
CSIC 2010 Dataset Splitter
Splits the dataset into training and test sets for anomaly detection
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("=" * 60)
print("CSIC 2010 Dataset Preparation")
print("=" * 60)

# Load the dataset
data_path = "../../datasets/csic-2010/csic_database.csv"
print(f"\n1. Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

print(f"   Total records: {len(df):,}")
print(f"   Columns: {df.shape[1]}")

# Check class distribution
print("\n2. Class Distribution:")
print(df.iloc[:, 0].value_counts())

# Separate Normal and Anomalous traffic
normal_data = df[df.iloc[:, 0] == 'Normal']
anomalous_data = df[df.iloc[:, 0] == 'Anomalous']

print(f"\n3. Separated Data:")
print(f"   Normal traffic: {len(normal_data):,} samples")
print(f"   Anomalous traffic: {len(anomalous_data):,} samples")

# Split Normal data: 80% training, 20% testing
normal_train, normal_test = train_test_split(
    normal_data,
    test_size=0.2,
    random_state=42
)

print(f"\n4. Training Set (Only Normal Traffic):")
print(f"   Training samples: {len(normal_train):,}")

# Test set: Mix of normal + anomalous
test_data = pd.concat([normal_test, anomalous_data], ignore_index=True)

# Shuffle test data
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n5. Test Set (Normal + Anomalous):")
print(f"   Normal samples: {len(normal_test):,}")
print(f"   Anomalous samples: {len(anomalous_data):,}")
print(f"   Total test samples: {len(test_data):,}")

# Save the splits
output_dir = "../../datasets/csic-2010"
os.makedirs(output_dir, exist_ok=True)

train_path = os.path.join(output_dir, "training_normal.csv")
test_path = os.path.join(output_dir, "testing_mixed.csv")

print(f"\n6. Saving splits...")
normal_train.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)

print(f"   ✓ Training set saved to: {train_path}")
print(f"   ✓ Test set saved to: {test_path}")

print("\n" + "=" * 60)
print("✅ Dataset preparation complete!")
print("=" * 60)

# Show statistics
print(f"\nDataset Statistics:")
print(f"├── Training (Normal only):  {len(normal_train):,} samples")
print(f"└── Testing (Mixed):         {len(test_data):,} samples")
print(f"    ├── Normal:              {len(normal_test):,} ({len(normal_test)/len(test_data)*100:.1f}%)")
print(f"    └── Anomalous:           {len(anomalous_data):,} ({len(anomalous_data)/len(test_data)*100:.1f}%)")
