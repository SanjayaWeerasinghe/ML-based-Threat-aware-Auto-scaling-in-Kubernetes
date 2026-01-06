"""
Isolation Forest Model Training - V2
CIC-IDS2018 Dataset (42 features, 10.7M training samples)
Fastest training model - good for large datasets
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
import os

print("=" * 80)
print("ISOLATION FOREST V2 - CIC-IDS2018 Dataset")
print("=" * 80)

# Paths (from src/models/v2/ go up 3 levels to project root)
DATA_DIR = "../../../datasets/cic-ids2018/processed/features"
MODELS_DIR = "../../../results/models/v2"
RESULTS_DIR = "../../../results/v2"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration - Memory optimized
TRAIN_SAMPLE_SIZE = 2500000  # 2.5M samples - memory optimized
TEST_SAMPLE_SIZE = 250000    # Test on 250K samples
RANDOM_STATE = 42

print("\n1. Loading feature data...")
print(f"   Loading training data (sampling {TRAIN_SAMPLE_SIZE:,} from 10.7M)...")
train_features = pd.read_csv(
    os.path.join(DATA_DIR, "train_features.csv"),
    nrows=TRAIN_SAMPLE_SIZE,
    low_memory=False
)

print(f"   Loading test data (sampling {TEST_SAMPLE_SIZE:,} from 5.4M)...")
test_features = pd.read_csv(
    os.path.join(DATA_DIR, "test_features.csv"),
    nrows=TEST_SAMPLE_SIZE,
    low_memory=False
)
test_labels = pd.read_csv(
    os.path.join(DATA_DIR, "test_labels.csv"),
    nrows=TEST_SAMPLE_SIZE
)

print(f"   ✓ Training samples: {len(train_features):,}")
print(f"   ✓ Test samples: {len(test_features):,}")
print(f"   ✓ Raw features: {train_features.shape[1]}")

# Remove non-numeric columns (IP addresses, timestamps, etc.)
print(f"\n   Cleaning non-numeric columns...")
non_numeric_cols = train_features.select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print(f"   Removing {len(non_numeric_cols)} non-numeric columns...")
    train_features = train_features.drop(columns=non_numeric_cols)
    test_features = test_features.drop(columns=non_numeric_cols)
print(f"   ✓ Numeric features: {train_features.shape[1]}")

# Handle infinity and NaN values in training data
print(f"\n   Handling invalid values in training data...")
train_features = train_features.replace([np.inf, -np.inf], np.nan)
nan_count_train = np.isnan(train_features.values).sum()
if nan_count_train > 0:
    print(f"   Found {nan_count_train} NaN values, replacing with medians...")
    train_features = train_features.fillna(train_features.median())

# Handle infinity and NaN values in test data
print(f"   Handling invalid values in test data...")
test_features = test_features.replace([np.inf, -np.inf], np.nan)
nan_count_test = np.isnan(test_features.values).sum()
if nan_count_test > 0:
    print(f"   Found {nan_count_test} NaN values, replacing with medians...")
    test_features = test_features.fillna(test_features.median())
print(f"   ✓ Invalid values handled")

# Standardize features
print("\n2. Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(train_features)
X_test = scaler.transform(test_features)

print(f"   ✓ Features normalized")

# Convert labels
y_test_binary = test_labels.iloc[:, 0].map(lambda x: 1 if x == 'Benign' else -1)

print("\n3. Training Isolation Forest model...")
print(f"   Algorithm: Isolation Forest")
print(f"   n_estimators: 100 trees")
print(f"   contamination: auto")
print(f"   Training samples: {len(X_train):,}")

start_time = time.time()

model = IsolationForest(
    contamination='auto',
    n_estimators=100,
    max_samples='auto',
    random_state=RANDOM_STATE,
    n_jobs=-1,  # Use all CPU cores
    verbose=0
)

model.fit(X_train)
training_time = time.time() - start_time

print(f"   ✓ Model trained in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")

# Predictions
print("\n4. Making predictions on test set...")
start_time = time.time()
y_pred = model.predict(X_test)
prediction_time = time.time() - start_time

print(f"   ✓ Predictions completed in {prediction_time:.2f} seconds")
print(f"   Average prediction time: {(prediction_time/len(X_test))*1000:.4f} ms per sample")
print(f"   Throughput: {len(X_test)/prediction_time:.0f} samples/second")

# Evaluate
print("\n5. Evaluating model performance...")

accuracy = accuracy_score(y_test_binary, y_pred)
precision = precision_score(y_test_binary, y_pred, pos_label=-1)
recall = recall_score(y_test_binary, y_pred, pos_label=-1)
f1 = f1_score(y_test_binary, y_pred, pos_label=-1)

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
fpr = fp / (fp + tn)

print(f"\nTrue Negatives:  {tn:8d}")
print(f"False Positives: {fp:8d}")
print(f"False Negatives: {fn:8d}")
print(f"True Positives:  {tp:8d}")
print(f"\nFalse Positive Rate: {fpr*100:.2f}%")

# Classification report
print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test_binary, y_pred, labels=[1, -1], target_names=['Benign', 'Attack']))

# Save model
print("\n6. Saving trained model...")

model_path = os.path.join(MODELS_DIR, "isolation_forest_model.pkl")
scaler_path = os.path.join(MODELS_DIR, "isolation_forest_scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"   ✓ Model saved to: {model_path}")
print(f"   ✓ Scaler saved to: {scaler_path}")

# Save metrics
metrics = {
    'Model': 'Isolation Forest - V2',
    'Dataset': 'CIC-IDS2018',
    'Training Samples': len(X_train),
    'Test Samples': len(X_test),
    'Features': train_features.shape[1],
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
metrics_path = os.path.join(RESULTS_DIR, "isolation_forest_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print(f"   ✓ Metrics saved to: {metrics_path}")

print("\n" + "=" * 80)
print("✅ ISOLATION FOREST V2 TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel Summary:")
print(f"├── Training: {len(X_train):,} samples × {train_features.shape[1]} features")
print(f"├── Accuracy: {accuracy*100:.2f}%")
print(f"├── Recall: {recall*100:.2f}% (best for catching attacks)")
print(f"├── FPR: {fpr*100:.2f}%")
print(f"└── Speed: {(prediction_time/len(X_test))*1000:.4f} ms/sample (FASTEST!)")

print(f"\nV1 Comparison:")
print(f"  V1: 73.93% accuracy, 42.21% FPR, 0.016 ms/sample")
print(f"  V2: {accuracy*100:.2f}% accuracy, {fpr*100:.2f}% FPR, {(prediction_time/len(X_test))*1000:.4f} ms/sample")
