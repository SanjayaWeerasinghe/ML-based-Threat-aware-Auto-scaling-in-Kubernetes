"""
Autoencoder Model Training - V3
BCCC-cPacket Cloud DDoS 2024 Dataset
Neural network-based anomaly detection using reconstruction error
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
import os

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 80)
print("AUTOENCODER V3 - BCCC-cPacket Cloud DDoS 2024")
print("=" * 80)

# Paths (from src/models/v3/ go up 3 levels to project root)
DATA_DIR = "../../../datasets/BCCC-cPacket-Cloud-DDoS-2024"
MODELS_DIR = "../../../results/models/v3"
RESULTS_DIR = "../../../results/v3"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 256
PATIENCE = 5

# Set random seeds
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

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

input_dim = X_train.shape[1]

# Build autoencoder
print("\n5. Building Autoencoder architecture...")

encoding_dim = int(input_dim / 2)  # Compress to half the features

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print(f"   Architecture: {input_dim} → 128 → 64 → {encoding_dim} → 64 → 128 → {input_dim}")
print(f"   Total parameters: {autoencoder.count_params():,}")

# Train
print(f"\n6. Training Autoencoder...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Early stopping: patience={PATIENCE}")
print(f"   ⏰ This may take 10-20 minutes...")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True
)

start_time = time.time()

history = autoencoder.fit(
    X_train, X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

training_time = time.time() - start_time

print(f"   ✓ Model trained in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
print(f"   Final loss: {history.history['loss'][-1]:.6f}")

# Calculate threshold
print("\n7. Calculating reconstruction error threshold...")
train_predictions = autoencoder.predict(X_train, batch_size=BATCH_SIZE, verbose=0)
train_mse = np.mean(np.power(X_train - train_predictions, 2), axis=1)
threshold = np.percentile(train_mse, 95)  # 95th percentile

print(f"   Threshold (95th percentile): {threshold:.6f}")

# Predictions on test set
print("\n8. Making predictions on test set...")
start_time = time.time()

test_predictions = autoencoder.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
test_mse = np.mean(np.power(X_test - test_predictions, 2), axis=1)
y_pred = np.where(test_mse > threshold, -1, 1)  # Above threshold = anomaly

prediction_time = time.time() - start_time

print(f"   ✓ Predictions completed in {prediction_time:.2f} seconds")
print(f"   Average prediction time: {(prediction_time/len(X_test))*1000:.4f} ms per sample")
print(f"   Throughput: {len(X_test)/prediction_time:.0f} samples/second")

# Evaluate
print("\n9. Evaluating model performance...")

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
print("\n10. Saving trained model...")

model_path = os.path.join(MODELS_DIR, "autoencoder_model.h5")
scaler_path = os.path.join(MODELS_DIR, "autoencoder_scaler.pkl")
threshold_path = os.path.join(MODELS_DIR, "autoencoder_threshold.pkl")

autoencoder.save(model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(threshold, threshold_path)

print(f"   ✓ Model saved to: {model_path}")
print(f"   ✓ Scaler saved to: {scaler_path}")
print(f"   ✓ Threshold saved to: {threshold_path}")

# Save metrics
metrics = {
    'Model': 'Autoencoder - V3',
    'Dataset': 'BCCC-cPacket Cloud DDoS 2024',
    'Training Samples': len(X_train),
    'Test Samples': len(X_test),
    'Features': train_features.shape[1],
    'Encoding Dimension': encoding_dim,
    'Epochs Trained': len(history.history['loss']),
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
metrics_path = os.path.join(RESULTS_DIR, "autoencoder_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print(f"   ✓ Metrics saved to: {metrics_path}")

print("\n" + "=" * 80)
print("✅ AUTOENCODER V3 TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel Summary:")
print(f"├── Architecture: {input_dim}-128-64-{encoding_dim}-64-128-{input_dim}")
print(f"├── Training: {len(X_train):,} samples, {len(history.history['loss'])} epochs, {training_time:.2f}s")
print(f"├── Accuracy: {accuracy*100:.2f}%")
print(f"├── Precision: {precision*100:.2f}%")
print(f"├── Recall: {recall*100:.2f}%")
print(f"└── FPR: {fpr*100:.2f}%")

print(f"\nV2 Comparison:")
print(f"  V2: 46.89% accuracy, 4.83% FPR")
print(f"  V3: {accuracy*100:.2f}% accuracy, {fpr*100:.2f}% FPR")
