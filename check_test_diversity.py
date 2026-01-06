"""
Quick check: Do first 500K test samples have all attack types?
"""
import pandas as pd

print("Checking attack diversity in first 500K vs full test set...\n")

# First 500K
sample_labels = pd.read_csv("datasets/cic-ids2018/processed/features/test_labels.csv", nrows=500000)
sample_dist = sample_labels.iloc[:, 0].value_counts()

print("=" * 80)
print("FIRST 500K TEST SAMPLES (current)")
print("=" * 80)
for label, count in sample_dist.items():
    print(f"{label:30s}: {count:8,} ({count/len(sample_labels)*100:5.2f}%)")
print(f"\nUnique attack types: {len(sample_dist) - 1}")  # -1 for Benign

# Full test set
print("\n" + "=" * 80)
print("FULL 5.4M TEST SAMPLES")
print("=" * 80)
full_labels = pd.read_csv("datasets/cic-ids2018/processed/features/test_labels.csv", low_memory=False)
full_dist = full_labels.iloc[:, 0].value_counts()

for label, count in full_dist.items():
    # Handle potential float labels
    label_str = str(label) if not isinstance(label, str) else label
    print(f"{label_str:30s}: {count:8,} ({count/len(full_labels)*100:5.2f}%)")

benign_count = 1 if 'Benign' in full_dist else 0
print(f"\nUnique attack types: {len(full_dist) - benign_count}")

# Compare
print("\n" + "=" * 80)
print("MISSING ATTACK TYPES IN FIRST 500K")
print("=" * 80)
missing = set(full_dist.keys()) - set(sample_dist.keys())
if missing:
    for attack in missing:
        print(f"  ❌ {attack}")
else:
    print("  ✓ All attack types present in first 500K!")
