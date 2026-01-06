# Threat-Aware Autoscaling Implementation Guide
## Step-by-Step Instructions from Setup to Model Training

---

## Table of Contents
1. [Prerequisites Installation](#1-prerequisites-installation)
2. [Project Setup](#2-project-setup)
3. [Dataset Acquisition](#3-dataset-acquisition)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Feature Extraction](#5-feature-extraction)
6. [Model Training](#6-model-training)
7. [Next Steps](#7-next-steps)

---

## 1. Prerequisites Installation

### 1.1 System Requirements
- **OS**: Ubuntu 24.04 LTS (or similar Linux distribution)
- **RAM**: Minimum 12 GB
- **Disk Space**: Minimum 10 GB free
- **CPU**: 6+ cores recommended

### 1.2 Check Installed Software

```bash
# Check Python version (should be 3.9+)
python3 --version

# Check pip
pip3 --version

# Check Git
git --version

# Check Docker
docker --version

# Check kubectl
kubectl version --client
```

### 1.3 Install Missing Prerequisites

```bash
# Update system
apt update && apt upgrade -y

# Install Python3 and pip
apt install -y python3-pip python3-venv

# Install Docker
apt install -y ca-certificates curl gnupg lsb-release
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl start docker
systemctl enable docker

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
mv kubectl /usr/local/bin/
```

---

## 2. Project Setup

### 2.1 Clone/Navigate to Project Directory

```bash
cd /root/Project
# If you cloned from GitHub:
# git clone https://github.com/YOUR_USERNAME/ML-based-Threat-aware-Auto-scaling-in-Kubernetes.git

cd ML-based-Threat-aware-Auto-scaling-in-Kubernetes
```

### 2.2 Create Project Structure

```bash
# Create necessary directories
mkdir -p datasets/csic-2010
mkdir -p src/data_processing
mkdir -p src/models
mkdir -p src/kubernetes
mkdir -p src/utils
mkdir -p notebooks
mkdir -p logs
mkdir -p results/models
mkdir -p tests
```

### 2.3 Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Your prompt should now show (venv)
```

### 2.4 Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras jupyter notebook requests tqdm scipy
```

### 2.5 Verify Installation

```bash
# Check installed packages
pip list | grep -E "numpy|pandas|scikit-learn|tensorflow|jupyter"
```

**Expected output:**
- numpy 2.4.0
- pandas 2.3.3
- scikit-learn 1.8.0
- tensorflow 2.20.0
- jupyter (various packages)

---

## 3. Dataset Acquisition

### 3.1 Download CSIC 2010 Dataset

The CSIC 2010 HTTP dataset contains 61,065 HTTP requests with normal and anomalous traffic.

**Option A: Direct Download** (if website is accessible)
```bash
cd datasets/csic-2010

wget -O normalTrafficTraining.txt http://www.isi.csic.es/dataset/normalTrafficTraining.txt
wget -O normalTrafficTest.txt http://www.isi.csic.es/dataset/normalTrafficTest.txt
wget -O anomalousTrafficTest.txt http://www.isi.csic.es/dataset/anomalousTrafficTest.txt
```

**Option B: Use Pre-downloaded CSV**
- Obtain `csic_database.csv` (29 MB)
- Place in `datasets/csic-2010/` directory

### 3.2 Verify Dataset

```bash
# Check file exists
ls -lh datasets/csic-2010/

# Check dataset size
wc -l datasets/csic-2010/csic_database.csv

# Expected: 61,066 lines (including header)
```

---

## 4. Dataset Preparation

### 4.1 Split Dataset Script

Create `src/data_processing/split_dataset.py`:

```python
"""
CSIC 2010 Dataset Splitter
Splits the dataset into training and test sets for anomaly detection
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("=" * 60)
print("CSIC 2010 Dataset Preparation")
print("=" * 60)

# Load the dataset
data_path = "../../datasets/csic-2010/csic_database.csv"
print(f"\n1. Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

print(f"   Total records: {len(df):,}")
print(f"   Columns: {df.shape[1]}")

# Check class distribution
print("\n2. Class Distribution:")
print(df.iloc[:, 0].value_counts())

# Separate Normal and Anomalous traffic
normal_data = df[df.iloc[:, 0] == 'Normal']
anomalous_data = df[df.iloc[:, 0] == 'Anomalous']

print(f"\n3. Separated Data:")
print(f"   Normal traffic: {len(normal_data):,} samples")
print(f"   Anomalous traffic: {len(anomalous_data):,} samples")

# Split Normal data: 80% training, 20% testing
normal_train, normal_test = train_test_split(
    normal_data,
    test_size=0.2,
    random_state=42
)

print(f"\n4. Training Set (Only Normal Traffic):")
print(f"   Training samples: {len(normal_train):,}")

# Test set: Mix of normal + anomalous
test_data = pd.concat([normal_test, anomalous_data], ignore_index=True)

# Shuffle test data
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n5. Test Set (Normal + Anomalous):")
print(f"   Normal samples: {len(normal_test):,}")
print(f"   Anomalous samples: {len(anomalous_data):,}")
print(f"   Total test samples: {len(test_data):,}")

# Save the splits
output_dir = "../../datasets/csic-2010"
os.makedirs(output_dir, exist_ok=True)

train_path = os.path.join(output_dir, "training_normal.csv")
test_path = os.path.join(output_dir, "testing_mixed.csv")

print(f"\n6. Saving splits...")
normal_train.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)

print(f"   ✓ Training set saved to: {train_path}")
print(f"   ✓ Test set saved to: {test_path}")

print("\n" + "=" * 60)
print("✅ Dataset preparation complete!")
print("=" * 60)
```

### 4.2 Run Dataset Split

```bash
cd src/data_processing
python split_dataset.py
```

**Expected Output:**
- Training set: 28,800 normal samples
- Test set: 32,265 samples (7,200 normal + 25,065 anomalous)

---

## 5. Feature Extraction

Feature extraction is already implemented in `src/data_processing/feature_extraction.py`.

### 5.1 Extracted Features (11 total)

**URL-based Features:**
1. `url_length` - Length of the URL
2. `num_parameters` - Number of URL parameters
3. `num_special_chars` - Count of special characters
4. `url_entropy` - Shannon entropy (randomness measure)
5. `has_suspicious_keywords` - Binary flag for SQL/XSS patterns

**HTTP Method Features:**
6. `method_get` - Binary: 1 if GET, 0 otherwise
7. `method_post` - Binary: 1 if POST, 0 otherwise

**Header/Content Features:**
8. `user_agent_length` - Length of User-Agent string
9. `content_length` - Length of request body
10. `has_cookie` - Binary: 1 if cookie present
11. `has_content_type` - Binary: 1 if content-type header present

### 5.2 Run Feature Extraction

```bash
cd src/data_processing
python feature_extraction.py
```

**Expected Output:**
- `train_features.csv` - 28,800 samples × 11 features
- `test_features.csv` - 32,265 samples × 11 features
- `test_labels.csv` - 32,265 labels

---

## 6. Model Training

All model training scripts are located in `src/models/`. Each model follows the same pattern:
1. Load feature data
2. Standardize features
3. Train model (only on normal traffic)
4. Evaluate on test set
5. Save model and metrics

### 6.1 Train All Models

```bash
cd /root/Project/ML-based-Threat-aware-Auto-scaling-in-Kubernetes/src/models

# Activate virtual environment if not already active
source ../../venv/bin/activate

# Train Model 1: Isolation Forest
python train_isolation_forest.py

# Train Model 2: One-Class SVM
python train_one_class_svm.py

# Train Model 3: K-Nearest Neighbors (KNN)
python train_knn.py

# Train Model 4: Gaussian Mixture Model (GMM)
python train_gmm.py

# Train Model 5: Autoencoder (Neural Network)
python train_autoencoder.py
```

### 6.2 Saved Model Files

After training, the following files are created in `results/models/`:

```
results/models/
├── isolation_forest_model.pkl
├── scaler.pkl
├── one_class_svm_model.pkl
├── svm_scaler.pkl
├── knn_model.pkl
├── knn_scaler.pkl
├── gmm_model.pkl
├── gmm_scaler.pkl
├── gmm_threshold.pkl
├── autoencoder_model.h5
├── autoencoder_scaler.pkl
└── autoencoder_threshold.pkl
```

### 6.3 Performance Metrics Files

```
results/
├── isolation_forest_metrics.csv
├── one_class_svm_metrics.csv
├── knn_metrics.csv
├── gmm_metrics.csv
└── autoencoder_metrics.csv
```

---

## 7. Next Steps

### 7.1 Model Comparison
Review `RESULTS_SUMMARY.md` for detailed performance comparison of all 5 models.

### 7.2 Kubernetes Integration (Future)
- Deploy best model (KNN) as a service
- Integrate with Kubernetes HPA
- Implement threat-aware autoscaling logic

### 7.3 Testing & Validation
- Test with live traffic
- Simulate DDoS attacks
- Measure resource savings

---

## Troubleshooting

### Issue: "Module not found" errors
**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall packages
pip install -r requirements.txt  # if you have one
# OR
pip install numpy pandas scikit-learn tensorflow
```

### Issue: Dataset not found
**Solution:**
```bash
# Check dataset location
ls -la datasets/csic-2010/

# Verify you're running from correct directory
pwd
# Should be: /root/Project/ML-based-Threat-aware-Auto-scaling-in-Kubernetes
```

### Issue: TensorFlow GPU warnings
**Solution:** These are normal if you don't have a GPU. The models will run on CPU.

---

## Quick Reference Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Deactivate virtual environment
deactivate

# Check Python packages
pip list

# Run all models in sequence
cd src/models
for model in train_*.py; do python $model; done

# View metrics
cat results/*.csv
```

---

**Document Created:** January 3, 2026
**Project:** ML-based Threat-aware Auto-scaling in Kubernetes
**Author:** Implementation Team
