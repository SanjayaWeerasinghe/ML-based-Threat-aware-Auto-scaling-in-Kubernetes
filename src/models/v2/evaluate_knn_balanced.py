"""
KNN Model Evaluation with Balanced Test Set
Loads pre-trained model and tests on realistic 50/50 benign/attack distribution
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import time
import os

print("=" * 80)
print("KNN MODEL EVALUATION - BALANCED TEST SET")
print("=" * 80)

# Paths
DATA_DIR = "../../../datasets/cic-ids2018/processed/features"
MODELS_DIR = "../../../results/models/v2"
RESULTS_DIR = "../../../results/v2"

# Load saved model and scaler
print("\n1. Loading pre-trained model...")
model_path = os.path.join(MODELS_DIR, "knn_model.pkl")
scaler_path = os.path.join(MODELS_DIR, "knn_scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print(f"   ✓ Model loaded: {model_path}")
print(f"   ✓ Scaler loaded: {scaler_path}")

# Load test labels to find benign and attack samples
print("\n2. Loading test data to create balanced sample...")
print("   Loading full test labels to identify Benign vs Attack samples...")

# Load all test labels (memory efficient - just one column)
all_labels = pd.read_csv(os.path.join(DATA_DIR, "test_labels.csv"))
print(f"   Total test samples available: {len(all_labels):,}")

# Count distribution
label_counts = all_labels['Label'].value_counts()
print(f"\n   Original distribution:")
for label, count in label_counts.items():
    print(f"     {label}: {count:,} ({count/len(all_labels)*100:.1f}%)")

# Find indices of benign and attack samples
benign_indices = all_labels[all_labels['Label'] == 'Benign'].index.tolist()
attack_indices = all_labels[all_labels['Label'] != 'Benign'].index.tolist()

print(f"\n   Available samples:")
print(f"     Benign: {len(benign_indices):,}")
print(f"     Attack: {len(attack_indices):,}")

# Create balanced sample (50K benign, 50K attack = 100K total)
BENIGN_SAMPLE_SIZE = 50000
ATTACK_SAMPLE_SIZE = 50000

print(f"\n3. Creating balanced test set...")
print(f"   Target: {BENIGN_SAMPLE_SIZE:,} benign + {ATTACK_SAMPLE_SIZE:,} attack = {BENIGN_SAMPLE_SIZE + ATTACK_SAMPLE_SIZE:,} total")

# Randomly sample indices
np.random.seed(42)
sampled_benign_indices = np.random.choice(benign_indices, BENIGN_SAMPLE_SIZE, replace=False)
sampled_attack_indices = np.random.choice(attack_indices, ATTACK_SAMPLE_SIZE, replace=False)

# Combine and sort indices for efficient CSV reading
all_sampled_indices = sorted(list(sampled_benign_indices) + list(sampled_attack_indices))

print(f"   ✓ Selected {len(all_sampled_indices):,} samples")
print(f"   Index range: {min(all_sampled_indices)} to {max(all_sampled_indices)}")

# Load test features for selected indices
print(f"\n4. Loading test features...")
test_features_full = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"), low_memory=False)

# Select only the sampled indices
test_features = test_features_full.iloc[all_sampled_indices].reset_index(drop=True)
test_labels = all_labels.iloc[all_sampled_indices].reset_index(drop=True)

print(f"   ✓ Loaded {len(test_features):,} samples × {test_features.shape[1]} features")

# Clean non-numeric columns (same as training)
print(f"\n5. Cleaning non-numeric columns...")
non_numeric_cols = test_features.select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print(f"   Removing {len(non_numeric_cols)} non-numeric columns...")
    test_features = test_features.drop(columns=non_numeric_cols)
print(f"   ✓ Numeric features: {test_features.shape[1]}")

# Verify balance
print(f"\n6. Verifying test set balance...")
balanced_counts = test_labels['Label'].value_counts()
print(f"   Test set distribution:")
for label, count in balanced_counts.items():
    print(f"     {label}: {count:,} ({count/len(test_labels)*100:.1f}%)")

# Handle infinity and NaN values
print(f"\n7. Handling invalid values (inf, NaN)...")
# Replace inf with NaN
test_features = test_features.replace([np.inf, -np.inf], np.nan)

# Check for problematic values
inf_count = np.isinf(test_features.values).sum()
nan_count = np.isnan(test_features.values).sum()
print(f"   Found {inf_count} infinity values")
print(f"   Found {nan_count} NaN values")

if nan_count > 0:
    print(f"   Replacing NaN values with column medians...")
    test_features = test_features.fillna(test_features.median())
    print(f"   ✓ NaN values handled")

# Cap extremely large values (above 99.9th percentile)
print(f"   Capping extreme values...")
for col in test_features.columns:
    upper_limit = test_features[col].quantile(0.999)
    lower_limit = test_features[col].quantile(0.001)
    test_features[col] = test_features[col].clip(lower=lower_limit, upper=upper_limit)
print(f"   ✓ Extreme values capped")

# Standardize features
print(f"\n8. Standardizing features...")
X_test = scaler.transform(test_features)
print(f"   ✓ Features normalized (mean=0, std=1)")

# Convert labels to binary (Benign=1, Attack=-1)
y_test_binary = test_labels['Label'].map(lambda x: 1 if x == 'Benign' else -1)

# Make predictions
print(f"\n9. Making predictions...")
start_time = time.time()
y_pred = model.predict(X_test)
prediction_time = time.time() - start_time

print(f"   ✓ Predictions completed in {prediction_time:.2f} seconds")
print(f"   Average prediction time: {(prediction_time/len(X_test))*1000:.4f} ms per sample")
print(f"   Throughput: {len(X_test)/prediction_time:.0f} samples/second")

# Calculate metrics
print(f"\n10. Evaluating model performance...")

accuracy = accuracy_score(y_test_binary, y_pred)
precision = precision_score(y_test_binary, y_pred, pos_label=-1)
recall = recall_score(y_test_binary, y_pred, pos_label=-1)
f1 = f1_score(y_test_binary, y_pred, pos_label=-1)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS (BALANCED TEST SET)")
print("=" * 80)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}% of predicted attacks are correct)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}% of actual attacks detected)")
print(f"F1-Score:  {f1:.4f} (harmonic mean)")

# Confusion Matrix
print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)
cm = confusion_matrix(y_test_binary, y_pred, labels=[1, -1])
print(f"\n                Predicted")
print(f"                Normal  Anomalous")
print(f"Actual Normal     {cm[0][0]:6d}    {cm[0][1]:6d}")
print(f"Actual Anomalous  {cm[1][0]:6d}    {cm[1][1]:6d}")

tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\nTrue Negatives (Normal correctly identified):     {tn:8d}")
print(f"False Positives (Normal flagged as Attack):       {fp:8d}")
print(f"False Negatives (Attack missed):                  {fn:8d}")
print(f"True Positives (Attack correctly detected):       {tp:8d}")
print(f"\nFalse Positive Rate: {fpr*100:.2f}%")

# Detailed classification report
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 80)
target_names = ['Benign', 'Attack']
print(classification_report(y_test_binary, y_pred, labels=[1, -1], target_names=target_names))

# Save results
print("\n11. Saving evaluation results...")
metrics = {
    'Model': 'K-Nearest Neighbors (KNN) - V2',
    'Dataset': 'CIC-IDS2018',
    'Test Set Type': 'Balanced (50/50)',
    'Test Samples': len(X_test),
    'Benign Samples': BENIGN_SAMPLE_SIZE,
    'Attack Samples': ATTACK_SAMPLE_SIZE,
    'Features': test_features.shape[1],
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'False Positive Rate': fpr,
    'Prediction Time (s)': prediction_time,
    'Avg Prediction Time (ms)': (prediction_time/len(X_test))*1000,
    'Throughput (samples/s)': len(X_test)/prediction_time,
    'True Positives': int(tp),
    'False Positives': int(fp),
    'False Negatives': int(fn),
    'True Negatives': int(tn)
}

metrics_df = pd.DataFrame([metrics])
metrics_path = os.path.join(RESULTS_DIR, "knn_metrics_balanced.csv")
metrics_df.to_csv(metrics_path, index=False)

print(f"   ✓ Metrics saved to: {metrics_path}")

print("\n" + "=" * 80)
print("✅ BALANCED EVALUATION COMPLETE!")
print("=" * 80)

print(f"\nComparison:")
print(f"  Imbalanced Test (95% attacks): 51.67% accuracy, 12.95% FPR")
print(f"  Balanced Test (50% attacks):   {accuracy*100:.2f}% accuracy, {fpr*100:.2f}% FPR")

print(f"\nKey Metrics on Balanced Test:")
print(f"  ├── Benign Detection Rate: {(tn/(tn+fp))*100:.2f}% ({tn}/{tn+fp})")
print(f"  ├── Attack Detection Rate: {(tp/(tp+fn))*100:.2f}% ({tp}/{tp+fn})")
print(f"  ├── False Positive Rate:   {fpr*100:.2f}%")
print(f"  └── Overall Accuracy:      {accuracy*100:.2f}%")
