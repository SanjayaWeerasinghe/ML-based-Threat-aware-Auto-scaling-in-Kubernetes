"""
Gaussian Mixture Model (GMM) Training
Probabilistic model for anomaly detection using density estimation
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import time
import os

print("=" * 70)
print("GAUSSIAN MIXTURE MODEL (GMM) - Anomaly Detection Model Training")
print("=" * 70)

# Load feature data
print("\n1. Loading feature data...")
train_features = pd.read_csv("../../datasets/csic-2010/train_features.csv")
test_features = pd.read_csv("../../datasets/csic-2010/test_features.csv")
test_labels = pd.read_csv("../../datasets/csic-2010/test_labels.csv")

print(f"   Training samples: {len(train_features)}")
print(f"   Test samples: {len(test_features)}")
print(f"   Features: {train_features.shape[1]}")

# Standardize features (important for GMM)
print("\n2. Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(train_features)
X_test = scaler.transform(test_features)

print(f"   ✓ Features normalized (mean=0, std=1)")

# Convert labels to binary (Normal=1, Anomalous=-1)
y_test_binary = test_labels.iloc[:, 0].map({'Normal': 1, 'Anomalous': -1})

print("\n3. Training Gaussian Mixture Model...")
print(f"   Number of components: 3 (mixture of 3 Gaussians)")
print(f"   Covariance type: full (allows for any covariance structure)")
print(f"   Initialization: k-means++ (smart initialization)")

# Train GMM
# n_components: number of Gaussian distributions to fit
# covariance_type: 'full' allows each component to have its own covariance matrix
start_time = time.time()

model = GaussianMixture(
    n_components=3,           # Number of Gaussian components
    covariance_type='full',   # Full covariance matrix
    init_params='kmeans',     # Initialize with k-means
    max_iter=100,             # Maximum iterations
    random_state=42
)

model.fit(X_train)
training_time = time.time() - start_time

print(f"   ✓ Model trained in {training_time:.2f} seconds")
print(f"   ✓ Model converged: {model.converged_}")

# Make predictions using probability threshold
print("\n4. Making predictions on test set...")
start_time = time.time()

# Calculate log probability for each sample
log_probs = model.score_samples(X_test)

# Determine threshold using percentile of training data
train_log_probs = model.score_samples(X_train)
threshold = np.percentile(train_log_probs, 10)  # Bottom 10% considered anomalies

# Predict: if log_prob < threshold, it's an anomaly (-1), else normal (1)
y_pred = np.where(log_probs < threshold, -1, 1)

prediction_time = time.time() - start_time

print(f"   ✓ Predictions completed in {prediction_time:.2f} seconds")
print(f"   Average prediction time: {(prediction_time/len(X_test))*1000:.4f} ms per sample")
print(f"   Threshold (log probability): {threshold:.4f}")

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

model_path = os.path.join(model_dir, "gmm_model.pkl")
scaler_path = os.path.join(model_dir, "gmm_scaler.pkl")
threshold_path = os.path.join(model_dir, "gmm_threshold.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(threshold, threshold_path)

print(f"   ✓ Model saved to: {model_path}")
print(f"   ✓ Scaler saved to: {scaler_path}")
print(f"   ✓ Threshold saved to: {threshold_path}")

# Save performance metrics
metrics = {
    'Model': 'Gaussian Mixture Model',
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
metrics_path = os.path.join("../../results", "gmm_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

print(f"   ✓ Metrics saved to: {metrics_path}")

print("\n" + "=" * 70)
print("✅ GAUSSIAN MIXTURE MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModel Summary:")
print(f"├── Accuracy: {accuracy*100:.2f}%")
print(f"├── Anomaly Detection Rate (Recall): {recall*100:.2f}%")
print(f"├── False Positive Rate: {(fp/(fp+tn))*100:.2f}%")
print(f"└── Average Inference Time: {(prediction_time/len(X_test))*1000:.4f} ms/sample")
