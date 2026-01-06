"""
One-Class SVM Model Training
Learns a boundary around normal data for anomaly detection
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
import os

print("=" * 70)
print("ONE-CLASS SVM - Anomaly Detection Model Training")
print("=" * 70)

# Load feature data
print("\n1. Loading feature data...")
train_features = pd.read_csv("../../datasets/csic-2010/train_features.csv")
test_features = pd.read_csv("../../datasets/csic-2010/test_features.csv")
test_labels = pd.read_csv("../../datasets/csic-2010/test_labels.csv")

print(f"   Training samples: {len(train_features)}")
print(f"   Test samples: {len(test_features)}")
print(f"   Features: {train_features.shape[1]}")

# Standardize features (critical for SVM)
print("\n2. Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(train_features)
X_test = scaler.transform(test_features)

print(f"   ✓ Features normalized (mean=0, std=1)")

# Convert labels to binary (Normal=1, Anomalous=-1)
y_test_binary = test_labels.iloc[:, 0].map({'Normal': 1, 'Anomalous': -1})

print("\n3. Training One-Class SVM model...")
print(f"   Kernel: RBF (Radial Basis Function)")
print(f"   Nu: 0.1 (upper bound on fraction of outliers)")
print(f"   Gamma: scale (kernel coefficient)")

# Train One-Class SVM
# nu: An upper bound on the fraction of training errors and lower bound of support vectors
# kernel: 'rbf' for non-linear decision boundary
start_time = time.time()

model = OneClassSVM(
    kernel='rbf',      # Radial Basis Function kernel (non-linear)
    gamma='scale',     # Kernel coefficient (1 / (n_features * X.var()))
    nu=0.1,            # Fraction of outliers expected (10%)
    verbose=False
)

model.fit(X_train)
training_time = time.time() - start_time

print(f"   ✓ Model trained in {training_time:.2f} seconds")

# Make predictions
print("\n4. Making predictions on test set...")
start_time = time.time()
y_pred = model.predict(X_test)
prediction_time = time.time() - start_time

print(f"   ✓ Predictions completed in {prediction_time:.2f} seconds")
print(f"   Average prediction time: {(prediction_time/len(X_test))*1000:.4f} ms per sample")

# Calculate metrics
print("\n5. Evaluating model performance...")

accuracy = accuracy_score(y_test_binary, y_pred)
precision = precision_score(y_test_binary, y_pred, pos_label=-1)  # Anomaly is positive class
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
print("\n6. Saving trained model...")
model_dir = "../../results/models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "one_class_svm_model.pkl")
scaler_path = os.path.join(model_dir, "svm_scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"   ✓ Model saved to: {model_path}")
print(f"   ✓ Scaler saved to: {scaler_path}")

# Save performance metrics
metrics = {
    'Model': 'One-Class SVM',
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
    'True Negatives': int(tn)
}

metrics_df = pd.DataFrame([metrics])
metrics_path = os.path.join("../../results", "one_class_svm_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print(f"   ✓ Metrics saved to: {metrics_path}")

print("\n" + "=" * 70)
print("✅ ONE-CLASS SVM MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModel Summary:")
print(f"├── Accuracy: {accuracy*100:.2f}%")
print(f"├── Anomaly Detection Rate (Recall): {recall*100:.2f}%")
print(f"├── False Positive Rate: {(fp/(fp+tn))*100:.2f}%")
print(f"└── Average Inference Time: {(prediction_time/len(X_test))*1000:.4f} ms/sample")
