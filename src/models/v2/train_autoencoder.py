"""
Autoencoder Model Training - V2
CIC-IDS2018 Dataset
Neural network-based anomaly detection
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
print("AUTOENCODER V2 - CIC-IDS2018 Dataset")
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
EPOCHS = 50
BATCH_SIZE = 256

# Set random seeds
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

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

# Standardize
print("\n2. Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(train_features)
X_test = scaler.transform(test_features)

print(f"   ✓ Features normalized")

y_test_binary = test_labels.iloc[:, 0].map(lambda x: 1 if x == 'Benign' else -1)

input_dim = X_train.shape[1]

# Build autoencoder
print("\n3. Building Autoencoder architecture...")

encoding_dim = 21  # Compress to half the features

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print(f"   Architecture: {input_dim} → 32 → {encoding_dim} → 32 → {input_dim}")
print(f"   Total parameters: {autoencoder.count_params():,}")

# Train
print(f"\n4. Training Autoencoder...")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Early stopping: patience=5")
print(f"   ⏰ This may take 10-20 minutes...")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
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
print("\n5. Calculating reconstruction error threshold...")
train_predictions = autoencoder.predict(X_train, batch_size=BATCH_SIZE, verbose=0)
train_mse = np.mean(np.power(X_train - train_predictions, 2), axis=1)
threshold = np.percentile(train_mse, 95)  # 95th percentile

print(f"   Threshold (95th percentile): {threshold:.6f}")

# Predictions on test set
print("\n6. Making predictions on test set...")
start_time = time.time()

test_predictions = autoencoder.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
test_mse = np.mean(np.power(X_test - test_predictions, 2), axis=1)
y_pred = np.where(test_mse > threshold, -1, 1)  # Above threshold = anomaly

prediction_time = time.time() - start_time

print(f"   ✓ Predictions completed in {prediction_time:.2f} seconds")
print(f"   Average prediction time: {(prediction_time/len(X_test))*1000:.4f} ms per sample")
print(f"   Throughput: {len(X_test)/prediction_time:.0f} samples/second")

# Evaluate
print("\n7. Evaluating model performance...")

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
print("\n8. Saving trained model...")

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
    'Model': 'Autoencoder - V2',
    'Dataset': 'CIC-IDS2018',
    'Training Samples': len(X_train),
    'Test Samples': len(X_test),
    'Features': train_features.shape[1],
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'False Positive Rate': fpr,
    'Threshold': threshold,
    'Epochs Trained': len(history.history['loss']),
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
print("✅ AUTOENCODER V2 TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel Summary:")
print(f"├── Architecture: {input_dim}-32-{encoding_dim}-32-{input_dim}")
print(f"├── Training: {len(X_train):,} samples, {len(history.history['loss'])} epochs")
print(f"├── Accuracy: {accuracy*100:.2f}%")
print(f"├── Precision: {precision*100:.2f}%")
print(f"└── Recall: {recall*100:.2f}%")

print(f"\nV1 Comparison:")
print(f"  V1: 46.89% accuracy, 4.83% FPR")
print(f"  V2: {accuracy*100:.2f}% accuracy, {fpr*100:.2f}% FPR")
