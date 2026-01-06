"""
Gaussian Mixture Model (GMM) Training - V3
BCCC-cPacket Cloud DDoS 2024 Dataset
Statistical anomaly detection using probability distributions
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
import os

print("=" * 80)
print("GAUSSIAN MIXTURE MODEL (GMM) V3 - BCCC-cPacket Cloud DDoS 2024")
print("=" * 80)

# Paths (from src/models/v3/ go up 3 levels to project root)
DATA_DIR = "../../../datasets/BCCC-cPacket-Cloud-DDoS-2024"
MODELS_DIR = "../../../results/models/v3"
RESULTS_DIR = "../../../results/v3"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration
RANDOM_STATE = 42
N_COMPONENTS = 1  # Single Gaussian for normal traffic
COVARIANCE_TYPE = 'full'

print("\n1. Loading BCCC dataset...")
data_file = os.path.join(DATA_DIR, "bccc-cpacket-cloud-ddos-2024-merged.parquet")
df = pd.read_parquet(data_file)

print(f"   ✓ Loaded {len(df):,} samples")
print(f"   ✓ Features: {df.shape[1]}")
print(f"\n   Class distribution:")
print(df['label'].value_counts())

# Separate features and labels
print("\n2. Preparing data for anomaly detection...")
label_col = 'label'
features = df.drop(columns=[label_col])

# Remove any remaining non-numeric columns
non_numeric_cols = features.select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print(f"   Removing {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
    features = features.drop(columns=non_numeric_cols)

print(f"   ✓ Numeric features: {features.shape[1]}")

# Split data: Train on BENIGN only
benign_mask = df[label_col] == 'Benign'
attack_mask = df[label_col] == 'Attack'
suspicious_mask = df[label_col] == 'Suspicious'

train_features = features[benign_mask].copy()
print(f"\n   Training set (Benign only): {len(train_features):,} samples")

# Create balanced test set
benign_indices = df[benign_mask].index.tolist()
attack_indices = df[attack_mask | suspicious_mask].index.tolist()

TEST_SIZE_PER_CLASS = 50000
if len(benign_indices) < TEST_SIZE_PER_CLASS:
    TEST_SIZE_PER_CLASS = len(benign_indices)
if len(attack_indices) < TEST_SIZE_PER_CLASS:
    TEST_SIZE_PER_CLASS = min(TEST_SIZE_PER_CLASS, len(attack_indices))

print(f"\n   Creating balanced test set ({TEST_SIZE_PER_CLASS:,} benign + {TEST_SIZE_PER_CLASS:,} attack)...")

np.random.seed(RANDOM_STATE)
sampled_benign_idx = np.random.choice(benign_indices, TEST_SIZE_PER_CLASS, replace=False)
sampled_attack_idx = np.random.choice(attack_indices, TEST_SIZE_PER_CLASS, replace=False)

test_indices = sorted(list(sampled_benign_idx) + list(sampled_attack_idx))
test_features = features.iloc[test_indices].reset_index(drop=True)
test_labels = df.iloc[test_indices][label_col].reset_index(drop=True)

print(f"   ✓ Test set: {len(test_features):,} samples")

# Clean data
print(f"\n3. Cleaning data...")
train_features = train_features.replace([np.inf, -np.inf], np.nan)
nan_count_train = np.isnan(train_features.values).sum()
if nan_count_train > 0:
    print(f"   Found {nan_count_train} NaN values in training, replacing with medians...")
    train_features = train_features.fillna(train_features.median())

test_features = test_features.replace([np.inf, -np.inf], np.nan)
nan_count_test = np.isnan(test_features.values).sum()
if nan_count_test > 0:
    print(f"   Found {nan_count_test} NaN values in test, replacing with medians...")
    test_features = test_features.fillna(test_features.median())

# Cap extreme values
for col in train_features.columns:
    upper = train_features[col].quantile(0.999)
    lower = train_features[col].quantile(0.001)
    train_features[col] = train_features[col].clip(lower=lower, upper=upper)
    test_features[col] = test_features[col].clip(lower=lower, upper=upper)

print(f"   ✓ Data cleaned")

# Standardize
print("\n4. Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(train_features)
X_test = scaler.transform(test_features)

print(f"   ✓ Features normalized")

y_test_binary = test_labels.map(lambda x: 1 if x == 'Benign' else -1)

print("\n5. Training Gaussian Mixture Model...")
print(f"   Algorithm: GMM")
print(f"   Components: {N_COMPONENTS} (single Gaussian for normal traffic)")
print(f"   Covariance type: {COVARIANCE_TYPE}")
print(f"   Training samples: {len(X_train):,}")

start_time = time.time()

model = GaussianMixture(
    n_components=N_COMPONENTS,
    covariance_type=COVARIANCE_TYPE,
    random_state=RANDOM_STATE,
    max_iter=100,
    verbose=0
)

model.fit(X_train)
training_time = time.time() - start_time

print(f"   ✓ Model trained in {training_time:.2f} seconds")

# Calculate threshold
print("\n6. Calculating anomaly threshold...")
train_scores = model.score_samples(X_train)
threshold = np.percentile(train_scores, 5)  # 5th percentile as threshold
print(f"   Threshold (5th percentile): {threshold:.4f}")

# Predictions
print("\n7. Making predictions on test set...")
start_time = time.time()
test_scores = model.score_samples(X_test)
y_pred = np.where(test_scores < threshold, -1, 1)  # Below threshold = anomaly
prediction_time = time.time() - start_time

print(f"   ✓ Predictions completed in {prediction_time:.2f} seconds")
print(f"   Average prediction time: {(prediction_time/len(X_test))*1000:.4f} ms per sample")
print(f"   Throughput: {len(X_test)/prediction_time:.0f} samples/second")

# Evaluate
print("\n8. Evaluating model performance...")

accuracy = accuracy_score(y_test_binary, y_pred)
precision = precision_score(y_test_binary, y_pred, pos_label=-1, zero_division=0)
recall = recall_score(y_test_binary, y_pred, pos_label=-1, zero_division=0)
f1 = f1_score(y_test_binary, y_pred, pos_label=-1, zero_division=0)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS")
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

print(f"\nTrue Negatives:  {tn:8d}")
print(f"False Positives: {fp:8d}")
print(f"False Negatives: {fn:8d}")
print(f"True Positives:  {tp:8d}")
print(f"\nFalse Positive Rate: {fpr*100:.2f}%")

# Classification report
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test_binary, y_pred, labels=[1, -1], target_names=['Benign', 'Attack'], zero_division=0))

# Save model
print("\n9. Saving trained model...")

model_path = os.path.join(MODELS_DIR, "gmm_model.pkl")
scaler_path = os.path.join(MODELS_DIR, "gmm_scaler.pkl")
threshold_path = os.path.join(MODELS_DIR, "gmm_threshold.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(threshold, threshold_path)

print(f"   ✓ Model saved to: {model_path}")
print(f"   ✓ Scaler saved to: {scaler_path}")
print(f"   ✓ Threshold saved to: {threshold_path}")

# Save metrics
metrics = {
    'Model': 'GMM - V3',
    'Dataset': 'BCCC-cPacket Cloud DDoS 2024',
    'Training Samples': len(X_train),
    'Test Samples': len(X_test),
    'Features': train_features.shape[1],
    'n_components': N_COMPONENTS,
    'covariance_type': COVARIANCE_TYPE,
    'Threshold': threshold,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'False Positive Rate': fpr,
    'Training Time (s)': training_time,
    'Prediction Time (s)': prediction_time,
    'Avg Prediction Time (ms)': (prediction_time/len(X_test))*1000,
    'Throughput (samples/s)': len(X_test)/prediction_time,
    'True Positives': int(tp),
    'False Positives': int(fp),
    'False Negatives': int(fn),
    'True Negatives': int(tn)
}

metrics_df = pd.DataFrame([metrics])
metrics_path = os.path.join(RESULTS_DIR, "gmm_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print(f"   ✓ Metrics saved to: {metrics_path}")

print("\n" + "=" * 80)
print("✅ GMM V3 TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel Summary:")
print(f"├── Training: {len(X_train):,} benign samples, {training_time:.2f}s")
print(f"├── Accuracy: {accuracy*100:.2f}%")
print(f"├── Precision: {precision*100:.2f}%")
print(f"├── Recall: {recall*100:.2f}%")
print(f"└── FPR: {fpr*100:.2f}%")

print(f"\nV2 Comparison:")
print(f"  V2: 4.42% accuracy (imbalanced test)")
print(f"  V3: {accuracy*100:.2f}% accuracy (balanced test)")
