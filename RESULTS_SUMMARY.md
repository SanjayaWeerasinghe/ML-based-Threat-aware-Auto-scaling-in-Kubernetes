# ML-Based Anomaly Detection Results Summary
## Threat-Aware Autoscaling in Kubernetes - Model Performance Comparison

---

## Executive Summary

This document presents the comprehensive evaluation results of **5 machine learning models** for HTTP traffic anomaly detection in Kubernetes autoscaling environments. All models were trained using the **CSIC 2010 dataset** (61,065 HTTP requests) following anomaly detection methodology (training only on legitimate traffic).

**Winner:** 🏆 **K-Nearest Neighbors (Local Outlier Factor)**
- **Accuracy:** 78.53%
- **Precision:** 99.13%
- **False Positive Rate:** 2.24% (best)
- **Recommended for production deployment**

---

## Dataset Information

### Dataset: CSIC 2010 HTTP Traffic Dataset

| Metric | Value |
|--------|-------|
| **Total Samples** | 61,065 HTTP requests |
| **Normal Traffic** | 36,000 samples (59%) |
| **Anomalous Traffic** | 25,065 samples (41%) |
| **Features** | 17 raw features (HTTP headers, methods, URLs, etc.) |

### Data Split

| Set | Samples | Composition |
|-----|---------|-------------|
| **Training** | 28,800 | 100% Normal (80% of normal traffic) |
| **Testing** | 32,265 | 22.3% Normal + 77.7% Anomalous |

**Rationale:** Anomaly detection models are trained ONLY on normal traffic to detect unknown attack patterns.

---

## Feature Engineering

### Extracted Features (11 Total)

#### URL-Based Features (5)
1. **url_length** - Total characters in URL
2. **num_parameters** - Count of query parameters
3. **num_special_chars** - Special characters count (`!@#$%^&*()` etc.)
4. **url_entropy** - Shannon entropy (randomness measure)
5. **has_suspicious_keywords** - Binary flag for SQL injection/XSS patterns

#### HTTP Method Features (2)
6. **method_get** - Binary indicator for GET requests
7. **method_post** - Binary indicator for POST requests

#### Header/Content Features (4)
8. **user_agent_length** - User-Agent string length
9. **content_length** - Request body size
10. **has_cookie** - Binary flag for cookie presence
11. **has_content_type** - Binary flag for Content-Type header

---

## Model Performance Results

### Overall Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | FPR | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|-----|---------------|----------------|
| **KNN (LOF)** 🏆 | **78.53%** | **99.13%** | 73.01% | **84.09%** | **2.24%** | 7.32s | 0.222 ms |
| Isolation Forest | 73.93% | 86.63% | **78.56%** | 82.40% | 42.21% | **0.80s** | **0.016 ms** |
| One-Class SVM | 54.97% | 94.22% | 44.78% | 60.71% | 9.56% | 19.53s | 0.327 ms |
| GMM | 47.74% | 92.43% | 35.65% | 51.46% | 10.17% | 1.12s | 0.004 ms |
| Autoencoder | 46.89% | 95.97% | 33.02% | 49.14% | 4.83% | 50.57s | 0.131 ms |

**Legend:**
- FPR = False Positive Rate (lower is better)
- Bold = Best in category
- 🏆 = Overall winner

---

## Detailed Model Analysis

### 1. K-Nearest Neighbors (Local Outlier Factor) - 🏆 WINNER

#### Performance Metrics
```
Accuracy:           78.53%
Precision:          99.13%  ← Almost perfect when flagging attacks
Recall:             73.01%
F1-Score:           84.09%
False Positive Rate: 2.24%  ← Lowest among all models
```

#### Confusion Matrix
```
                    Predicted
                Normal  Anomalous
Actual Normal     7,039       161
Actual Anomalous  6,765    18,300
```

#### Performance Breakdown
- **True Negatives (Normal correctly identified):** 7,039 / 7,200 (97.76%) ✅
- **False Positives (Normal flagged as anomaly):** 161 / 7,200 (2.24%) ✅
- **True Positives (Anomaly detected):** 18,300 / 25,065 (73.01%) ✅
- **False Negatives (Anomaly missed):** 6,765 / 25,065 (26.99%) ⚠️

#### Speed Metrics
- **Training Time:** 7.32 seconds
- **Inference Time:** 0.222 ms per sample
- **Throughput:** ~4,500 requests/second

#### Strengths
✅ **Best overall accuracy** (78.53%)
✅ **Highest precision** (99.13%) - When it detects an attack, it's almost always right
✅ **Lowest false positive rate** (2.24%) - Minimal disruption to legitimate users
✅ **Good recall** (73.01%) - Catches most attacks
✅ **Production-ready speed** (0.222 ms inference)

#### Weaknesses
⚠️ Misses ~27% of attacks
⚠️ Moderate training time (7.32s)

#### Recommendation
**✅ RECOMMENDED FOR PRODUCTION**
- Best balance of accuracy, precision, and user experience
- Ideal for threat-aware autoscaling where false positives hurt user experience

---

### 2. Isolation Forest - RUNNER-UP

#### Performance Metrics
```
Accuracy:           73.93%
Precision:          86.63%
Recall:             78.56%  ← Best recall among all models
F1-Score:           82.40%
False Positive Rate: 42.21%  ← High!
```

#### Confusion Matrix
```
                    Predicted
                Normal  Anomalous
Actual Normal     4,161     3,039
Actual Anomalous  5,373    19,692
```

#### Performance Breakdown
- **True Negatives:** 4,161 / 7,200 (57.79%)
- **False Positives:** 3,039 / 7,200 (42.21%) ❌
- **True Positives:** 19,692 / 25,065 (78.56%) ✅
- **False Negatives:** 5,373 / 25,065 (21.44%)

#### Speed Metrics
- **Training Time:** 0.80 seconds (FASTEST!)
- **Inference Time:** 0.016 ms per sample (FASTEST!)
- **Throughput:** ~62,500 requests/second

#### Strengths
✅ **Best recall** (78.56%) - Catches the most attacks
✅ **Fastest training** (0.80s)
✅ **Fastest inference** (0.016 ms)
✅ Good overall accuracy (73.93%)

#### Weaknesses
❌ **Very high false positive rate** (42.21%) - Blocks 42% of legitimate traffic!
❌ Lower precision (86.63%)

#### Recommendation
**⚠️ USE WITH CAUTION**
- Great for attack detection, but disrupts user experience
- Consider if security is paramount over user experience
- Good for initial rapid prototyping due to speed

---

### 3. One-Class SVM

#### Performance Metrics
```
Accuracy:           54.97%
Precision:          94.22%
Recall:             44.78%  ← Misses >50% of attacks
F1-Score:           60.71%
False Positive Rate: 9.56%
```

#### Confusion Matrix
```
                    Predicted
                Normal  Anomalous
Actual Normal     6,512       688
Actual Anomalous 13,840    11,225
```

#### Performance Breakdown
- **True Negatives:** 6,512 / 7,200 (90.44%) ✅
- **False Positives:** 688 / 7,200 (9.56%)
- **True Positives:** 11,225 / 25,065 (44.78%) ❌
- **False Negatives:** 13,840 / 25,065 (55.22%) ❌❌

#### Speed Metrics
- **Training Time:** 19.53 seconds (SLOWEST among classical ML)
- **Inference Time:** 0.327 ms per sample
- **Throughput:** ~3,058 requests/second

#### Strengths
✅ Good precision (94.22%)
✅ Low false positives (9.56%)
✅ Doesn't disrupt users much

#### Weaknesses
❌ **Misses >50% of attacks** (poor recall)
❌ Low overall accuracy (54.97%)
❌ Slow training (19.53s)
❌ Slow inference (0.327 ms)

#### Recommendation
**❌ NOT RECOMMENDED**
- Too conservative - lets most attacks through
- Defeats purpose of threat detection

---

### 4. Gaussian Mixture Model (GMM)

#### Performance Metrics
```
Accuracy:           47.74%  ← 2nd worst
Precision:          92.43%
Recall:             35.65%  ← 2nd worst
F1-Score:           51.46%
False Positive Rate: 10.17%
```

#### Confusion Matrix
```
                    Predicted
                Normal  Anomalous
Actual Normal     6,468       732
Actual Anomalous 16,129     8,936
```

#### Performance Breakdown
- **True Negatives:** 6,468 / 7,200 (89.83%)
- **False Positives:** 732 / 7,200 (10.17%)
- **True Positives:** 8,936 / 25,065 (35.65%) ❌
- **False Negatives:** 16,129 / 25,065 (64.35%) ❌❌

#### Speed Metrics
- **Training Time:** 1.12 seconds
- **Inference Time:** 0.004 ms per sample (FASTEST!)
- **Throughput:** ~250,000 requests/second

#### Strengths
✅ **Fastest inference** (0.004 ms)
✅ Fast training (1.12s)
✅ Good precision (92.43%)

#### Weaknesses
❌ **Very low accuracy** (47.74%)
❌ **Misses 64% of attacks**
❌ Not suitable for security applications

#### Recommendation
**❌ NOT RECOMMENDED**
- Speed doesn't compensate for poor detection
- Unsuitable for threat-aware systems

---

### 5. Autoencoder (Neural Network)

#### Performance Metrics
```
Accuracy:           46.89%  ← WORST
Precision:          95.97%
Recall:             33.02%  ← WORST
F1-Score:           49.14%
False Positive Rate: 4.83%
```

#### Confusion Matrix
```
                    Predicted
                Normal  Anomalous
Actual Normal     6,852       348
Actual Anomalous 16,788     8,277
```

#### Performance Breakdown
- **True Negatives:** 6,852 / 7,200 (95.17%) ✅
- **False Positives:** 348 / 7,200 (4.83%)
- **True Positives:** 8,277 / 25,065 (33.02%) ❌❌
- **False Negatives:** 16,788 / 25,065 (66.98%) ❌❌❌

#### Architecture
```
Input (11) → Dense(8) → Dense(4) → Dense(8) → Output(11)
Total Parameters: 271
Training: 50 epochs, MSE loss
Threshold: 95th percentile reconstruction error
```

#### Speed Metrics
- **Training Time:** 50.57 seconds (SLOWEST!)
- **Inference Time:** 0.131 ms per sample
- **Throughput:** ~7,633 requests/second

#### Strengths
✅ Very high precision (95.97%)
✅ Very low false positives (4.83%)
✅ Doesn't disrupt users

#### Weaknesses
❌ **Worst accuracy** (46.89%)
❌ **Worst recall** (33.02%) - Misses 67% of attacks!
❌ **Slowest training** (50.57s)
❌ Deep learning overhead not justified

#### Recommendation
**❌ NOT RECOMMENDED**
- Deep learning doesn't provide benefits for this dataset
- Too slow, too inaccurate
- Classical ML models outperform

---

## Key Insights

### 1. Best Model for Production: KNN (LOF)
- Provides the best balance of accuracy, precision, and user experience
- 99.13% precision means near-zero false alarms
- 2.24% FPR ensures minimal disruption to legitimate users
- 73% attack detection is acceptable given low false positives

### 2. Deep Learning Underperforms
- Autoencoder (neural network) performed worst overall
- Dataset may be too small or features too simple for deep learning advantage
- Classical ML models are superior for this use case

### 3. Speed vs Accuracy Trade-off
- Isolation Forest: Fastest but high false positives
- KNN: Good balance of speed and accuracy
- One-Class SVM: Slow with poor recall

### 4. Precision vs Recall Trade-off
- High precision models (SVM, Autoencoder): Miss too many attacks
- High recall models (Isolation Forest): Too many false alarms
- KNN achieves best balance

---

## Real-World Application Scenarios

### Scenario 1: Production Kubernetes Cluster (E-commerce)
**Requirement:** Minimize user disruption while detecting attacks

**Recommended Model:** 🏆 **KNN (LOF)**
- **Why:** 2.24% FPR means 97.76% of legitimate users unaffected
- **Impact:** During DDoS (10,000 req/s), only 224 legitimate requests blocked
- **Attack Detection:** Catches 73% of attacks, preventing unnecessary scaling

### Scenario 2: High-Security Environment (Banking)
**Requirement:** Maximize attack detection, tolerate false positives

**Recommended Model:** **Isolation Forest**
- **Why:** 78.56% recall (best attack detection)
- **Trade-off:** 42% false positive rate acceptable if security is paramount
- **Speed:** Fastest inference (0.016 ms) for real-time blocking

### Scenario 3: Research/Testing Environment
**Requirement:** Fast experimentation and iteration

**Recommended Model:** **Isolation Forest**
- **Why:** Fastest training (0.8s) allows rapid prototyping
- **Use Case:** Quick validation of feature engineering changes
- **Note:** Not recommended for production due to high FPR

---

## Metrics Glossary

### Accuracy
- **Definition:** (TP + TN) / Total
- **Meaning:** Overall correctness of predictions
- **Best:** KNN (78.53%)

### Precision
- **Definition:** TP / (TP + FP)
- **Meaning:** Of predicted anomalies, how many are actually anomalies
- **Best:** KNN (99.13%)

### Recall (Sensitivity)
- **Definition:** TP / (TP + FN)
- **Meaning:** Of actual anomalies, how many were detected
- **Best:** Isolation Forest (78.56%)

### F1-Score
- **Definition:** 2 × (Precision × Recall) / (Precision + Recall)
- **Meaning:** Harmonic mean of precision and recall
- **Best:** KNN (84.09%)

### False Positive Rate (FPR)
- **Definition:** FP / (FP + TN)
- **Meaning:** Of normal traffic, how much is incorrectly flagged
- **Best (Lowest):** KNN (2.24%)

---

## Conclusion

**Final Recommendation:** Deploy **K-Nearest Neighbors (LOF)** model for threat-aware autoscaling in production Kubernetes environments.

**Justification:**
1. ✅ Best overall accuracy (78.53%)
2. ✅ Highest precision (99.13%) - Trustworthy alerts
3. ✅ Lowest false positive rate (2.24%) - Minimal user impact
4. ✅ Good recall (73.01%) - Catches most attacks
5. ✅ Production-ready speed (0.222 ms inference)

**Expected Production Impact:**
- **Cost Savings:** Prevent 73% of unnecessary scaling during attacks
- **User Experience:** 97.76% of legitimate traffic flows normally
- **Security:** Block majority of DDoS and application-layer attacks
- **Performance:** Real-time detection at <1ms latency

---

**Report Generated:** January 3, 2026
**Dataset:** CSIC 2010 (61,065 samples)
**Models Evaluated:** 5
**Recommended Model:** K-Nearest Neighbors (Local Outlier Factor)
**Status:** ✅ Ready for Kubernetes Integration Phase
