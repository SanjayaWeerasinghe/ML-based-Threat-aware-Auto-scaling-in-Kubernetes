"""
GMM Model Evaluation with Balanced Test Set
Loads pre-trained model and tests on realistic 50/50 benign/attack distribution
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import time
import os

print("=" * 80)
print("GMM EVALUATION - BALANCED TEST SET")
print("=" * 80)

# Paths
DATA_DIR = "../../../datasets/cic-ids2018/processed/features"
MODELS_DIR = "../../../results/models/v2"
RESULTS_DIR = "../../../results/v2"

# Load saved model, scaler, and threshold
print("\n1. Loading pre-trained model...")
model_path = os.path.join(MODELS_DIR, "gmm_model.pkl")
scaler_path = os.path.join(MODELS_DIR, "gmm_scaler.pkl")
threshold_path = os.path.join(MODELS_DIR, "gmm_threshold.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
threshold = joblib.load(threshold_path)

print(f"   ✓ Model loaded: {model_path}")
print(f"   ✓ Scaler loaded: {scaler_path}")
print(f"   ✓ Threshold loaded: {threshold}")

# Load test labels to find benign and attack samples
print("\n2. Loading test data to create balanced sample...")
all_labels = pd.read_csv(os.path.join(DATA_DIR, "test_labels.csv"))
print(f"   Total test samples available: {len(all_labels):,}")

# Find indices of benign and attack samples
benign_indices = all_labels[all_labels['Label'] == 'Benign'].index.tolist()
attack_indices = all_labels[all_labels['Label'] != 'Benign'].index.tolist()

print(f"   Available: {len(benign_indices):,} benign, {len(attack_indices):,} attack")

# Create balanced sample (50K benign, 50K attack = 100K total)
BENIGN_SAMPLE_SIZE = 50000
ATTACK_SAMPLE_SIZE = 50000

print(f"\n3. Creating balanced test set ({BENIGN_SAMPLE_SIZE:,} + {ATTACK_SAMPLE_SIZE:,} = 100K)...")
np.random.seed(42)
sampled_benign_indices = np.random.choice(benign_indices, BENIGN_SAMPLE_SIZE, replace=False)
sampled_attack_indices = np.random.choice(attack_indices, ATTACK_SAMPLE_SIZE, replace=False)

all_sampled_indices = sorted(list(sampled_benign_indices) + list(sampled_attack_indices))
print(f"   ✓ Selected {len(all_sampled_indices):,} samples")

# Load test features
print(f"\n4. Loading test features...")
test_features_full = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"), low_memory=False)
test_features = test_features_full.iloc[all_sampled_indices].reset_index(drop=True)
test_labels = all_labels.iloc[all_sampled_indices].reset_index(drop=True)

print(f"   ✓ Loaded {len(test_features):,} samples × {test_features.shape[1]} features")

# Clean non-numeric columns
print(f"\n5. Cleaning data...")
non_numeric_cols = test_features.select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print(f"   Removing {len(non_numeric_cols)} non-numeric columns...")
    test_features = test_features.drop(columns=non_numeric_cols)

# Handle inf/NaN
test_features = test_features.replace([np.inf, -np.inf], np.nan)
nan_count = np.isnan(test_features.values).sum()
if nan_count > 0:
    print(f"   Replacing {nan_count} NaN values with medians...")
    test_features = test_features.fillna(test_features.median())

# Cap extreme values
for col in test_features.columns:
    upper = test_features[col].quantile(0.999)
    lower = test_features[col].quantile(0.001)
    test_features[col] = test_features[col].clip(lower=lower, upper=upper)

print(f"   ✓ Cleaned features: {test_features.shape[1]}")

# Verify balance
balanced_counts = test_labels['Label'].value_counts()
print(f"\n6. Test set distribution:")
for label, count in balanced_counts.items():
    print(f"     {label}: {count:,} ({count/len(test_labels)*100:.1f}%)")

# Standardize
print(f"\n7. Standardizing features...")
X_test = scaler.transform(test_features)
y_test_binary = test_labels['Label'].map(lambda x: 1 if x == 'Benign' else -1)

# Predict using GMM log likelihood
print(f"\n8. Making predictions...")
start_time = time.time()
log_likelihood = model.score_samples(X_test)
y_pred = np.where(log_likelihood < threshold, -1, 1)  # Below threshold = anomaly
prediction_time = time.time() - start_time

print(f"   ✓ Predictions completed in {prediction_time:.2f} seconds")
print(f"   Throughput: {len(X_test)/prediction_time:,.0f} samples/second")

# Evaluate
print(f"\n9. Evaluating performance...")

accuracy = accuracy_score(y_test_binary, y_pred)
precision = precision_score(y_test_binary, y_pred, pos_label=-1, zero_division=0)
recall = recall_score(y_test_binary, y_pred, pos_label=-1, zero_division=0)
f1 = f1_score(y_test_binary, y_pred, pos_label=-1, zero_division=0)

print("\n" + "=" * 80)
print("GMM - BALANCED TEST SET RESULTS")
print("=" * 80)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f}")

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

# Classification report
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test_binary, y_pred, labels=[1, -1], target_names=['Benign', 'Attack'], zero_division=0))

# Save results
print("\n10. Saving results...")
metrics = {
    'Model': 'GMM - V2',
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
    'Throughput (samples/s)': len(X_test)/prediction_time,
    'True Positives': int(tp),
    'False Positives': int(fp),
    'False Negatives': int(fn),
    'True Negatives': int(tn)
}

metrics_df = pd.DataFrame([metrics])
metrics_path = os.path.join(RESULTS_DIR, "gmm_metrics_balanced.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"   ✓ Saved: {metrics_path}")

print("\n" + "=" * 80)
print("✅ BALANCED EVALUATION COMPLETE!")
print("=" * 80)

print(f"\nModel Comparison (Balanced Test):")
print(f"  KNN:                 66.35% accuracy, 20.69% FPR")
print(f"  Isolation Forest:    49.06% accuracy, 5.63% FPR")
print(f"  GMM:                 {accuracy*100:.2f}% accuracy, {fpr*100:.2f}% FPR")

print(f"\nKey Metrics:")
print(f"  ├── Benign Detection: {(tn/(tn+fp))*100:.2f}%")
print(f"  ├── Attack Detection: {(tp/(tp+fn))*100:.2f}%")
print(f"  ├── FPR: {fpr*100:.2f}%")
print(f"  └── Accuracy: {accuracy*100:.2f}%")
