"""
K-Nearest Neighbors (KNN) Model Training - V3
BCCC-cPacket Cloud DDoS 2024 Dataset
Anomaly detection using Local Outlier Factor
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
import os

print("=" * 80)
print("K-NEAREST NEIGHBORS (KNN) V3 - BCCC-cPacket Cloud DDoS 2024")
print("=" * 80)

# Paths (from src/models/v3/ go up 3 levels to project root)
DATA_DIR = "../../../datasets/BCCC-cPacket-Cloud-DDoS-2024"
MODELS_DIR = "../../../results/models/v3"
RESULTS_DIR = "../../../results/v3"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration
RANDOM_STATE = 42
K_NEIGHBORS = 20
CONTAMINATION = 0.05  # Expected fraction of anomalies in training set

print("\n1. Loading BCCC dataset...")
data_file = os.path.join(DATA_DIR, "bccc-cpacket-cloud-ddos-2024-merged.parquet")
df = pd.read_parquet(data_file)

print(f"   ✓ Loaded {len(df):,} samples")
print(f"   ✓ Features: {df.shape[1]}")
print(f"\n   Class distribution:")
print(df['label'].value_counts())
print(f"\n   Percentages:")
print(df['label'].value_counts(normalize=True) * 100)

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

# Split data: Train on BENIGN only, Test on all
benign_mask = df[label_col] == 'Benign'
attack_mask = df[label_col] == 'Attack'
suspicious_mask = df[label_col] == 'Suspicious'

# For anomaly detection: train ONLY on benign samples
train_features = features[benign_mask].copy()
train_labels = df[benign_mask][label_col].copy()

print(f"\n   Training set (Benign only): {len(train_features):,} samples")

# Create balanced test set: 50% Benign, 50% Attack (treat Suspicious as Attack)
benign_indices = df[benign_mask].index.tolist()
attack_indices = df[attack_mask | suspicious_mask].index.tolist()

# Sample equal amounts for balanced test
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
print(f"\n   Test set distribution:")
print(test_labels.value_counts())

# Handle infinity and NaN values
print(f"\n3. Cleaning data...")
print(f"   Handling invalid values in training data...")
train_features = train_features.replace([np.inf, -np.inf], np.nan)
nan_count_train = np.isnan(train_features.values).sum()
if nan_count_train > 0:
    print(f"   Found {nan_count_train} NaN values, replacing with column medians...")
    train_features = train_features.fillna(train_features.median())

print(f"   Handling invalid values in test data...")
test_features = test_features.replace([np.inf, -np.inf], np.nan)
nan_count_test = np.isnan(test_features.values).sum()
if nan_count_test > 0:
    print(f"   Found {nan_count_test} NaN values, replacing with column medians...")
    test_features = test_features.fillna(test_features.median())

# Cap extreme values at 99.9th percentile
print(f"   Capping extreme values...")
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

# Convert labels to binary: Benign=1, Attack/Suspicious=-1
y_test_binary = test_labels.map(lambda x: 1 if x == 'Benign' else -1)

print("\n5. Training KNN model (Local Outlier Factor)...")
print(f"   Algorithm: Local Outlier Factor")
print(f"   k-neighbors: {K_NEIGHBORS}")
print(f"   Contamination: {CONTAMINATION}")
print(f"   Training samples: {len(X_train):,}")
print(f"   ⏰ This may take 5-15 minutes...")

start_time = time.time()

model = LocalOutlierFactor(
    n_neighbors=K_NEIGHBORS,
    contamination=CONTAMINATION,
    novelty=True,  # Enable prediction on new samples
    n_jobs=-1
)

model.fit(X_train)
training_time = time.time() - start_time

print(f"   ✓ Model trained in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")

# Predictions
print("\n6. Making predictions on test set...")
start_time = time.time()
y_pred = model.predict(X_test)
prediction_time = time.time() - start_time

print(f"   ✓ Predictions completed in {prediction_time:.2f} seconds")
print(f"   Average prediction time: {(prediction_time/len(X_test))*1000:.4f} ms per sample")
print(f"   Throughput: {len(X_test)/prediction_time:.0f} samples/second")

# Evaluate
print("\n7. Evaluating model performance...")

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

print(f"\nTrue Negatives (Benign correctly identified):     {tn:8d}")
print(f"False Positives (Benign flagged as Attack):       {fp:8d}")
print(f"False Negatives (Attack missed):                  {fn:8d}")
print(f"True Positives (Attack correctly detected):       {tp:8d}")
print(f"\nFalse Positive Rate: {fpr*100:.2f}%")

# Classification report
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test_binary, y_pred, labels=[1, -1], target_names=['Benign', 'Attack'], zero_division=0))

# Save model
print("\n8. Saving trained model...")

model_path = os.path.join(MODELS_DIR, "knn_model.pkl")
scaler_path = os.path.join(MODELS_DIR, "knn_scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"   ✓ Model saved to: {model_path}")
print(f"   ✓ Scaler saved to: {scaler_path}")

# Save metrics
metrics = {
    'Model': 'KNN (LOF) - V3',
    'Dataset': 'BCCC-cPacket Cloud DDoS 2024',
    'Training Samples': len(X_train),
    'Test Samples': len(X_test),
    'Features': train_features.shape[1],
    'k_neighbors': K_NEIGHBORS,
    'contamination': CONTAMINATION,
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
metrics_path = os.path.join(RESULTS_DIR, "knn_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print(f"   ✓ Metrics saved to: {metrics_path}")

print("\n" + "=" * 80)
print("✅ KNN V3 TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel Summary:")
print(f"├── Dataset: BCCC-cPacket Cloud DDoS 2024")
print(f"├── Training: {len(X_train):,} benign samples only")
print(f"├── Features: {train_features.shape[1]} (including rate metrics)")
print(f"├── Accuracy: {accuracy*100:.2f}%")
print(f"├── Precision: {precision*100:.2f}%")
print(f"├── Recall: {recall*100:.2f}%")
print(f"└── FPR: {fpr*100:.2f}%")

print(f"\nV2 Comparison (CIC-IDS2018):")
print(f"  V2: 66.35% accuracy, 20.69% FPR")
print(f"  V3: {accuracy*100:.2f}% accuracy, {fpr*100:.2f}% FPR")

print(f"\nKey Improvements in V3:")
print(f"  ✓ 319 features (vs 40 in V2)")
print(f"  ✓ Has rate metrics (packets_rate, bytes_rate)")
print(f"  ✓ Cloud DDoS attacks (more relevant for Kubernetes)")
print(f"  ✓ Better balanced dataset (64.6% benign vs 5% in V2)")
