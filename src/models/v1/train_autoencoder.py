"""
Autoencoder (Neural Network) Training
Deep learning model for anomaly detection using reconstruction error
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import time
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 70)
print("AUTOENCODER - Deep Learning Anomaly Detection Model Training")
print("=" * 70)

# Load feature data
print("\n1. Loading feature data...")
train_features = pd.read_csv("../../datasets/csic-2010/train_features.csv")
test_features = pd.read_csv("../../datasets/csic-2010/test_features.csv")
test_labels = pd.read_csv("../../datasets/csic-2010/test_labels.csv")

print(f"   Training samples: {len(train_features)}")
print(f"   Test samples: {len(test_features)}")
print(f"   Features: {train_features.shape[1]}")

# Standardize features (critical for neural networks)
print("\n2. Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(train_features)
X_test = scaler.transform(test_features)

print(f"   ✓ Features normalized (mean=0, std=1)")

# Convert labels to binary
y_test_binary = test_labels.iloc[:, 0].map({'Normal': 1, 'Anomalous': -1})

print("\n3. Building Autoencoder architecture...")
input_dim = X_train.shape[1]  # 11 features

# Build the Autoencoder model
# Encoder: 11 -> 8 -> 4 (compress)
# Decoder: 4 -> 8 -> 11 (reconstruct)
model = keras.Sequential([
    # Encoder
    layers.Input(shape=(input_dim,)),
    layers.Dense(8, activation='relu', name='encoder_layer1'),
    layers.Dense(4, activation='relu', name='bottleneck'),

    # Decoder
    layers.Dense(8, activation='relu', name='decoder_layer1'),
    layers.Dense(input_dim, activation='linear', name='output')
])

model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error for reconstruction
    metrics=['mae']  # Mean Absolute Error
)

print(f"\n   Model Architecture:")
print(f"   ├── Input Layer:      {input_dim} features")
print(f"   ├── Encoder Layer 1:  8 neurons (ReLU)")
print(f"   ├── Bottleneck:       4 neurons (ReLU) - compressed representation")
print(f"   ├── Decoder Layer 1:  8 neurons (ReLU)")
print(f"   └── Output Layer:     {input_dim} features (reconstructed)")
print(f"\n   Total parameters: {model.count_params():,}")

# Train the model
print("\n4. Training Autoencoder...")
print(f"   Epochs: 50")
print(f"   Batch size: 256")
print(f"   Optimizer: Adam")
print(f"   Loss function: Mean Squared Error (MSE)")

start_time = time.time()

# Train only on normal data (X_train = input and target)
history = model.fit(
    X_train, X_train,  # Input and target are the same (reconstruction)
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    verbose=0,  # Suppress training output
    shuffle=True
)

training_time = time.time() - start_time

print(f"   ✓ Model trained in {training_time:.2f} seconds")
print(f"   ✓ Final training loss: {history.history['loss'][-1]:.6f}")
print(f"   ✓ Final validation loss: {history.history['val_loss'][-1]:.6f}")

# Calculate reconstruction error on training data to set threshold
print("\n5. Calculating reconstruction error threshold...")
train_reconstructions = model.predict(X_train, verbose=0)
train_mse = np.mean(np.square(X_train - train_reconstructions), axis=1)

# Set threshold at 95th percentile of training errors
threshold = np.percentile(train_mse, 95)
print(f"   ✓ Threshold (95th percentile): {threshold:.6f}")

# Make predictions on test set
print("\n6. Making predictions on test set...")
start_time = time.time()

test_reconstructions = model.predict(X_test, verbose=0)
test_mse = np.mean(np.square(X_test - test_reconstructions), axis=1)

# If reconstruction error > threshold, it's an anomaly
y_pred = np.where(test_mse > threshold, -1, 1)

prediction_time = time.time() - start_time

print(f"   ✓ Predictions completed in {prediction_time:.2f} seconds")
print(f"   Average prediction time: {(prediction_time/len(X_test))*1000:.4f} ms per sample")
print(f"   Mean reconstruction error: {np.mean(test_mse):.6f}")
print(f"   Max reconstruction error: {np.max(test_mse):.6f}")

# Calculate metrics
print("\n7. Evaluating model performance...")

accuracy = accuracy_score(y_test_binary, y_pred)
precision = precision_score(y_test_binary, y_pred, pos_label=-1)
recall = recall_score(y_test_binary, y_pred, pos_label=-1)
f1 = f1_score(y_test_binary, y_pred, pos_label=-1)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE METRICS")
print("=" * 70)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} (of predicted anomalies, {precision*100:.2f}% are correct)")
print(f"Recall:    {recall:.4f} (detected {recall*100:.2f}% of actual anomalies)")
print(f"F1-Score:  {f1:.4f} (harmonic mean of precision & recall)")

# Confusion Matrix
print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
cm = confusion_matrix(y_test_binary, y_pred, labels=[1, -1])
print(f"\n                Predicted")
print(f"                Normal  Anomalous")
print(f"Actual Normal     {cm[0][0]:6d}    {cm[0][1]:6d}")
print(f"Actual Anomalous  {cm[1][0]:6d}    {cm[1][1]:6d}")

# Calculate True Positives, False Positives, etc.
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
print(f"\nTrue Negatives (Normal correctly identified):     {tn:6d}")
print(f"False Positives (Normal misclassified as Anomaly): {fp:6d}")
print(f"False Negatives (Anomaly missed):                  {fn:6d}")
print(f"True Positives (Anomaly correctly detected):       {tp:6d}")

# Detailed classification report
print("\n" + "=" * 70)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 70)
target_names = ['Normal', 'Anomalous']
print(classification_report(y_test_binary, y_pred, labels=[1, -1], target_names=target_names))

# Save the model
print("\n8. Saving trained model...")
model_dir = "../../results/models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "autoencoder_model.h5")
scaler_path = os.path.join(model_dir, "autoencoder_scaler.pkl")
threshold_path = os.path.join(model_dir, "autoencoder_threshold.pkl")

model.save(model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(threshold, threshold_path)

print(f"   ✓ Model saved to: {model_path}")
print(f"   ✓ Scaler saved to: {scaler_path}")
print(f"   ✓ Threshold saved to: {threshold_path}")

# Save performance metrics
metrics = {
    'Model': 'Autoencoder',
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Training Time (s)': training_time,
    'Prediction Time (s)': prediction_time,
    'Avg Prediction Time (ms)': (prediction_time/len(X_test))*1000,
    'True Positives': int(tp),
    'False Positives': int(fp),
    'False Negatives': int(fn),
    'True Negatives': int(tn),
    'Threshold': threshold
}

metrics_df = pd.DataFrame([metrics])
metrics_path = os.path.join("../../results", "autoencoder_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print(f"   ✓ Metrics saved to: {metrics_path}")

print("\n" + "=" * 70)
print("✅ AUTOENCODER MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModel Summary:")
print(f"├── Accuracy: {accuracy*100:.2f}%")
print(f"├── Anomaly Detection Rate (Recall): {recall*100:.2f}%")
print(f"├── False Positive Rate: {(fp/(fp+tn))*100:.2f}%")
print(f"└── Average Inference Time: {(prediction_time/len(X_test))*1000:.4f} ms/sample")
