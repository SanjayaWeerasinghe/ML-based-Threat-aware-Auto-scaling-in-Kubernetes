# V3 Models - BCCC-cPacket Cloud DDoS 2024 Dataset

## Overview

V3 implementation uses the **BCCC-cPacket Cloud DDoS 2024** dataset, which is specifically designed for cloud-based DDoS detection and is more suitable for autoscaling research than previous datasets.

## Dataset Advantages

✅ **540,494 samples** (good size for training)
✅ **319 comprehensive features** including rate metrics
✅ **64.6% Benign, 31.5% Attack** (perfect balance for anomaly detection)
✅ **Rate-based features**: packets_rate, bytes_rate, fwd/bwd rates
✅ **39 time/duration features** for autoscaling decisions
✅ **Cloud DDoS attacks** (relevant for Kubernetes environments)
✅ **Parquet format** (fast loading and processing)

## Key Features for Autoscaling

- `packets_rate` - Can proxy for requests/second
- `bytes_rate` - Network throughput indicator
- `duration` - Flow duration
- `fwd_packets_rate` - Outbound traffic rate
- `bwd_packets_rate` - Inbound traffic rate
- `handshake_duration` - Connection establishment time

## Training Scripts

All models use **anomaly detection** approach (train only on Benign samples):

### 1. train_knn.py
- **Algorithm**: Local Outlier Factor (LOF)
- **k-neighbors**: 20
- **contamination**: 0.05
- **Expected time**: 5-15 minutes
- **Use case**: Distance-based anomaly detection

### 2. train_isolation_forest.py
- **Algorithm**: Isolation Forest
- **Trees**: 100
- **contamination**: 0.05
- **Expected time**: <1 minute (FASTEST!)
- **Use case**: Fast tree-based anomaly detection

### 3. train_gmm.py
- **Algorithm**: Gaussian Mixture Model
- **Components**: 1 (single Gaussian)
- **covariance_type**: full
- **Expected time**: <1 minute
- **Use case**: Statistical/probabilistic anomaly detection

### 4. train_one_class_svm.py
- **Algorithm**: One-Class SVM
- **Kernel**: RBF
- **nu**: 0.05
- **Expected time**: 30-60 minutes (SLOWEST!)
- **Use case**: Kernel-based anomaly detection

### 5. train_autoencoder.py
- **Algorithm**: Neural Network Autoencoder
- **Architecture**: 318 → 128 → 64 → 159 → 64 → 128 → 318
- **Epochs**: 50 (with early stopping)
- **Expected time**: 10-20 minutes
- **Use case**: Deep learning reconstruction error

## Running Training

```bash
cd /root/Project/ML-based-Threat-aware-Auto-scaling-in-Kubernetes/src/models/v3

# Activate virtual environment
source ../../venv/bin/activate

# Train models (recommended order: fastest to slowest)
python train_isolation_forest.py
python train_gmm.py
python train_knn.py
python train_autoencoder.py
python train_one_class_svm.py  # This one takes longest
```

## Output Files

### Models Directory: `../../../results/models/v3/`
- `knn_model.pkl` + `knn_scaler.pkl`
- `isolation_forest_model.pkl` + `isolation_forest_scaler.pkl`
- `gmm_model.pkl` + `gmm_scaler.pkl` + `gmm_threshold.pkl`
- `one_class_svm_model.pkl` + `one_class_svm_scaler.pkl`
- `autoencoder_model.h5` + `autoencoder_scaler.pkl` + `autoencoder_threshold.pkl`

### Metrics Directory: `../../../results/v3/`
- `knn_metrics.csv`
- `isolation_forest_metrics.csv`
- `gmm_metrics.csv`
- `one_class_svm_metrics.csv`
- `autoencoder_metrics.csv`

## Improvements Over V2

| Aspect | V2 (CIC-IDS2018) | V3 (BCCC) |
|--------|------------------|-----------|
| **Dataset Size** | 16.1M samples | 540K samples |
| **Features** | 40 features | 318 features |
| **Balance** | 95% attack, 5% benign | 64.6% benign, 31.5% attack |
| **Rate Metrics** | ❌ None | ✅ packets_rate, bytes_rate, etc. |
| **Time Features** | ❌ None | ✅ 39 duration/rate features |
| **Attack Type** | Bot (stealthy) | Cloud DDoS (relevant) |
| **Autoscaling Relevance** | Low | High |

## Expected Performance Improvements

Based on V2 results, we expect V3 to achieve:

- **KNN**: 66%+ accuracy (vs 66.35% in V2)
- **Isolation Forest**: 50%+ accuracy (vs 49.06% in V2)
- **GMM**: Better than 4.42% from V2 (balanced test)
- **One-Class SVM**: 55%+ accuracy (vs 54.97% in V2)
- **Autoencoder**: 47%+ accuracy (vs 46.89% in V2)

Better dataset should lead to better or comparable results with more relevant features for autoscaling.

## Integration with Autoscaling

The V3 models can be integrated with Kubernetes HPA:

```python
# Pseudocode for autoscaling decision
if model.predict(network_metrics) == ATTACK:
    # Apply rate limiting
    rate_limit = 10  # requests/second for suspicious traffic

    # Don't scale up - it's an attack
    if current_replicas > min_replicas:
        scale_down()
else:  # BENIGN
    # Normal autoscaling based on load
    if packets_rate > threshold:
        scale_up()
    elif packets_rate < threshold:
        scale_down()
```

## Next Steps

1. Train all 5 models
2. Compare performance metrics
3. Analyze which model works best for cloud DDoS detection
4. Integrate best model with Kubernetes autoscaling simulator
5. Test threat-aware autoscaling with rate limiting
