"""
CIC-IDS2018 Dataset Splitter (Memory-Efficient Version)
Processes data in chunks to avoid memory issues with 16M+ rows

Training: ONLY Benign traffic (for anomaly detection)
Testing: Mix of Benign + All attack types
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from glob import glob
import numpy as np

print("=" * 80)
print("CIC-IDS2018 DATASET PREPARATION (MEMORY-EFFICIENT)")
print("=" * 80)

# Configuration
DATASET_DIR = "../../../datasets/cic-ids2018"
OUTPUT_DIR = "../../../datasets/cic-ids2018/processed/splits"
TRAIN_TEST_SPLIT = 0.8  # 80% for training, 20% for testing
RANDOM_STATE = 42
CHUNK_SIZE = 100000  # Process 100K rows at a time

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"   Output directory: {OUTPUT_DIR}")

print(f"\n1. Loading dataset files from: {DATASET_DIR}")
csv_files = sorted(glob(os.path.join(DATASET_DIR, "*.csv")))
print(f"   Found {len(csv_files)} CSV files")

# Output file paths
TRAIN_PATH = os.path.join(OUTPUT_DIR, "training_benign.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "testing_mixed.csv")
TEST_LABELS_PATH = os.path.join(OUTPUT_DIR, "test_labels.csv")

# Initialize counters
total_benign = 0
total_attack = 0
benign_train_count = 0
benign_test_count = 0

print(f"\n2. Processing files in chunks (chunk size: {CHUNK_SIZE:,} rows)...")
print("   This approach is memory-efficient for large datasets")

# First pass: Count and separate data
print(f"\n3. Processing Benign and Attack data...")

first_write_train = True
first_write_test = True

for i, csv_file in enumerate(csv_files, 1):
    filename = os.path.basename(csv_file)
    print(f"\n   [{i}/{len(csv_files)}] Processing {filename}...")

    try:
        # Process file in chunks
        chunk_num = 0
        for chunk in pd.read_csv(csv_file, chunksize=CHUNK_SIZE, low_memory=False):
            chunk_num += 1

            # Clean: Remove header rows that leaked into data
            chunk = chunk[chunk['Label'] != 'Label']

            if len(chunk) == 0:
                continue

            # Separate benign and attack
            benign_chunk = chunk[chunk['Label'] == 'Benign'].copy()
            attack_chunk = chunk[chunk['Label'] != 'Benign'].copy()

            # Process benign data - split into train/test
            if len(benign_chunk) > 0:
                # Split this benign chunk
                if len(benign_chunk) > 1:
                    try:
                        benign_train_chunk, benign_test_chunk = train_test_split(
                            benign_chunk,
                            test_size=(1-TRAIN_TEST_SPLIT),
                            random_state=RANDOM_STATE
                        )
                    except:
                        # If split fails (too few samples), put all in training
                        benign_train_chunk = benign_chunk
                        benign_test_chunk = pd.DataFrame()
                else:
                    benign_train_chunk = benign_chunk
                    benign_test_chunk = pd.DataFrame()

                # Handle infinite values
                benign_train_chunk = benign_train_chunk.replace([np.inf, -np.inf], np.nan)
                benign_train_chunk = benign_train_chunk.fillna(0)

                # Append to training file
                benign_train_chunk.to_csv(
                    TRAIN_PATH,
                    mode='w' if first_write_train else 'a',
                    header=first_write_train,
                    index=False
                )
                first_write_train = False
                benign_train_count += len(benign_train_chunk)

                # Append test benign to test file
                if len(benign_test_chunk) > 0:
                    benign_test_chunk = benign_test_chunk.replace([np.inf, -np.inf], np.nan)
                    benign_test_chunk = benign_test_chunk.fillna(0)

                    benign_test_chunk.to_csv(
                        TEST_PATH,
                        mode='w' if first_write_test else 'a',
                        header=first_write_test,
                        index=False
                    )
                    first_write_test = False
                    benign_test_count += len(benign_test_chunk)

                total_benign += len(benign_chunk)

            # Process attack data - all goes to test
            if len(attack_chunk) > 0:
                attack_chunk = attack_chunk.replace([np.inf, -np.inf], np.nan)
                attack_chunk = attack_chunk.fillna(0)

                attack_chunk.to_csv(
                    TEST_PATH,
                    mode='a',
                    header=False,
                    index=False
                )
                total_attack += len(attack_chunk)

            # Progress update
            if chunk_num % 10 == 0:
                print(f"      Processed {chunk_num * CHUNK_SIZE:,} rows... "
                      f"(Benign: {total_benign:,}, Attack: {total_attack:,})")

        print(f"      ✓ Completed: Benign={total_benign:,}, Attack={total_attack:,}")

    except Exception as e:
        print(f"      ✗ ERROR: {e}")
        continue

total_samples = total_benign + total_attack
total_test = benign_test_count + total_attack

print("\n" + "=" * 80)
print("4. PROCESSING SUMMARY")
print("=" * 80)

print(f"\nTotal samples processed: {total_samples:,}")
print(f"\nData Distribution:")
print(f"  Benign traffic:  {total_benign:,} ({total_benign/total_samples*100:.2f}%)")
print(f"  Attack traffic:  {total_attack:,} ({total_attack/total_samples*100:.2f}%)")

print(f"\nDataset Splits:")
print(f"  Training (Benign only): {benign_train_count:,} samples")
print(f"  Testing (Mixed):        {total_test:,} samples")
print(f"    - Benign:             {benign_test_count:,} ({benign_test_count/total_test*100:.2f}%)")
print(f"    - Attack:             {total_attack:,} ({total_attack/total_test*100:.2f}%)")

# Now create test_labels.csv by reading the Label column from test file
print(f"\n5. Extracting test labels...")
print(f"   Reading test data in chunks to extract labels...")

test_labels = []
for chunk in pd.read_csv(TEST_PATH, chunksize=CHUNK_SIZE, usecols=['Label']):
    test_labels.extend(chunk['Label'].tolist())

# Save test labels
pd.DataFrame({'Label': test_labels}).to_csv(TEST_LABELS_PATH, index=False)
print(f"   ✓ Test labels saved: {TEST_LABELS_PATH}")

print("\n" + "=" * 80)
print("✅ DATASET PREPARATION COMPLETE!")
print("=" * 80)

print(f"\nFiles created:")
print(f"  ✓ {TRAIN_PATH}")
print(f"  ✓ {TEST_PATH}")
print(f"  ✓ {TEST_LABELS_PATH}")

print(f"\nTest Set Attack Distribution:")
# Read test data in chunks to count attack types
attack_counts = {}
for chunk in pd.read_csv(TEST_PATH, chunksize=CHUNK_SIZE, usecols=['Label']):
    chunk_counts = chunk[chunk['Label'] != 'Benign']['Label'].value_counts()
    for label, count in chunk_counts.items():
        attack_counts[label] = attack_counts.get(label, 0) + count

for label, count in sorted(attack_counts.items(), key=lambda x: -x[1]):
    print(f"  - {label:30s}: {count:8,}")

print(f"\nMemory-efficient processing completed!")
print(f"Next step: Run feature_selection.py to prepare features for training")
