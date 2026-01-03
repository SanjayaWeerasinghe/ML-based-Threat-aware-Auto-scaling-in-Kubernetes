# Threat-Aware Autoscaling Using ML-Based Anomaly Detection in Kubernetes
## Comprehensive System Design Rationale & Literature-Based Analysis

---

## Table of Contents
1. [Problem Statement & Research Context](#1-problem-statement--research-context)
2. [Literature Review & Theoretical Foundation](#2-literature-review--theoretical-foundation)
3. [System Architecture Design Rationale](#3-system-architecture-design-rationale)
4. [Technology Stack Selection & Justification](#4-technology-stack-selection--justification)
5. [Component-by-Component Deep Dive](#5-component-by-component-deep-dive)
6. [Challenge-Solution Mapping](#6-challenge-solution-mapping)
7. [ML Model Selection Rationale](#7-ml-model-selection-rationale)
8. [Integration Strategy & Data Flow](#8-integration-strategy--data-flow)
9. [Evaluation Methodology](#9-evaluation-methodology)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. Problem Statement & Research Context

### 1.1 The Core Problem

**Traditional Kubernetes Autoscaling Limitation:**
```
Traditional HPA (Horizontal Pod Autoscaler):
- Monitors: CPU, Memory, Custom Metrics
- Decision: IF (CPU > 80%) THEN scale_up()
- Blindness: Cannot distinguish legitimate traffic from attacks
```

**The Attack Scenario:**
```
DDoS Attack → High CPU/Memory → HPA Scales Up → More Resources Consumed → Cost Explosion
```

**Example Attack Impact:**
- Legitimate traffic: 1000 req/s → 5 pods needed
- DDoS attack: 50,000 req/s → HPA scales to 250 pods
- **Problem**: You're paying for resources to serve attackers!

### 1.2 Research Gap

**Existing Research:**
1. **Autoscaling Research**: Focuses on performance optimization (response time, throughput)
2. **Security Research**: Focuses on attack detection and mitigation
3. **Gap**: Limited integration between security intelligence and autoscaling decisions

**Your Research Contribution:**
- Bridges security and performance domains
- Makes autoscaling "threat-aware"
- Prevents resource waste during attacks
- Uses anomaly detection (doesn't require attack samples)

### 1.3 Why This Matters

**Cloud Cost Context:**
- AWS/GCP/Azure charge per pod/container runtime
- DDoS can cost $1000s in minutes with blind autoscaling
- Legitimate users suffer when resources serve attackers

**Security Context:**
- Modern attacks are sophisticated (APTs, slow HTTP attacks, application-layer DDoS)
- Traditional firewalls can't catch all malicious patterns
- ML can detect subtle anomalies humans miss

---

## 2. Literature Review & Theoretical Foundation

### 2.1 Anomaly Detection Theory

**Why Anomaly Detection vs Classification?**

**Classification Approach (Traditional):**
```
Training Data Required:
- Normal traffic samples ✓
- Attack traffic samples ✓ (SQL injection, XSS, DDoS, etc.)

Problems:
1. Attack patterns evolve constantly (zero-day attacks)
2. Impossible to collect all attack types
3. Imbalanced datasets (attacks are rare)
4. Model becomes outdated quickly
```

**Anomaly Detection Approach (Your Research):**
```
Training Data Required:
- Normal traffic samples ✓
- Attack traffic samples ✗ (NOT needed)

Advantages:
1. Detects unknown attacks (zero-day)
2. Only needs legitimate traffic for training
3. Adapts to concept drift (normal patterns change)
4. More robust in production
```

**Literature Support:**
- **Chandola et al. (2009)**: "Anomaly Detection: A Survey" - Establishes theoretical foundation
- **Goldstein & Uchida (2016)**: "A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms"
- **Pang et al. (2021)**: "Deep Learning for Anomaly Detection: A Review" - Modern approaches

### 2.2 Kubernetes Autoscaling Research

**HPA (Horizontal Pod Autoscaler) Fundamentals:**
```
Desired Replicas = ceil[Current Replicas × (Current Metric / Target Metric)]

Example:
- Current pods: 5
- Current CPU: 80%
- Target CPU: 50%
- New pods: ceil[5 × (80/50)] = 8 pods
```

**Research Evolution:**
1. **Static Thresholds** (Kubernetes default) - Reactive, slow
2. **Predictive Autoscaling** (AWS, GCP) - Uses historical patterns
3. **Reinforcement Learning** (Recent research) - Learns optimal policies
4. **Your Approach**: Adds security context to autoscaling decisions

**Literature Support:**
- **Lorido-Botran et al. (2014)**: "A Review of Auto-scaling Techniques for Elastic Applications in Cloud Environments"
- **Al-Dhuraibi et al. (2018)**: "Autonomic Vertical Elasticity of Docker Containers with ELASTICDOCKER"
- **Rossi et al. (2019)**: "Proactive Autoscaling for Cloud-Native Applications"

### 2.3 Security-Aware Resource Management

**Emerging Research Area:**
- Traditional: Security and performance treated separately
- Modern: Integrated security-performance tradeoffs

**Key Concepts:**
1. **Security Tax**: Resources spent on security vs performance
2. **Attack Amplification**: How attacks exploit autoscaling
3. **Economic Denial of Sustainability (EDoS)**: Attacks that drain cloud budgets

**Literature Support:**
- **Idziorek et al. (2012)**: "Exploiting Cloud Utility Models for Profit and Ruin" - Introduced EDoS
- **Kholidy & Baiardi (2012)**: "CIDS: A Framework for Intrusion Detection in Cloud Systems"
- **Gao et al. (2020)**: "Machine Learning for Security in Cloud Computing: A Survey"

---

## 3. System Architecture Design Rationale

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          KUBERNETES CLUSTER                              │
│                                                                          │
│  ┌────────────────┐                                                     │
│  │  Ingress       │ ← External Traffic (Legitimate + Malicious)         │
│  │  Controller    │                                                     │
│  └────────┬───────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌────────────────┐       ┌──────────────┐       ┌─────────────────┐  │
│  │  Application   │──────▶│   FluentD    │──────▶│   Elasticsearch │  │
│  │     Pods       │       │ (Log Collect)│       │   (Log Storage) │  │
│  │                │       └──────────────┘       └────────┬────────┘  │
│  │ • Web Server   │                                       │            │
│  │ • API Service  │                                       │            │
│  │ • Microservice │                                       ▼            │
│  └────────┬───────┘                              ┌─────────────────┐  │
│           │                                      │    Logstash     │  │
│           │                                      │ (Log Processing)│  │
│           │                                      └────────┬────────┘  │
│           │                                               │            │
│           │                                               ▼            │
│           │                                      ┌─────────────────┐  │
│           │                                      │ Feature Extract │  │
│           │                                      │ Pipeline        │  │
│           │                                      │                 │  │
│           │                                      │ Extract:        │  │
│           │                                      │ • Request rate  │  │
│           │                                      │ • Geo-location  │  │
│           │                                      │ • HTTP method   │  │
│           │                                      │ • Status codes  │  │
│           │                                      │ • User-Agent    │  │
│           │                                      │ • Payload size  │  │
│           │                                      │ • Time patterns │  │
│           │                                      └────────┬────────┘  │
│           │                                               │            │
│           │                                               ▼            │
│           │                                      ┌─────────────────┐  │
│           │                                      │   ML Models     │  │
│           │                                      │   (5 Models)    │  │
│           │                                      │                 │  │
│           │                                      │ 1. Autoencoder  │  │
│           │                                      │ 2. Isolation    │  │
│           │                                      │    Forest       │  │
│           │                                      │ 3. One-Class    │  │
│           │                                      │    SVM          │  │
│           │                                      │ 4. KNN          │  │
│           │                                      │ 5. GMM          │  │
│           │                                      └────────┬────────┘  │
│           │                                               │            │
│           │                                               ▼            │
│           │                                      ┌─────────────────┐  │
│           │                                      │ Anomaly Score   │  │
│           │                                      │ Aggregator      │  │
│           │                                      │                 │  │
│           │                                      │ Output:         │  │
│           │                                      │ Score: 0.0-1.0  │  │
│           │                                      │ (0=normal,      │  │
│           │                                      │  1=anomaly)     │  │
│           │                                      └────────┬────────┘  │
│           │                                               │            │
│           │                                               ▼            │
│           │                                      ┌─────────────────┐  │
│           └─────────────────────────────────────▶│  Threat-Aware   │  │
│                                                  │  Autoscaler     │  │
│                                                  │                 │  │
│                                                  │ IF anomaly_score│  │
│                                                  │    > threshold: │  │
│                                                  │   THEN adjust   │  │
│                                                  │   scaling       │  │
│                                                  └────────┬────────┘  │
│                                                           │            │
│                                                           ▼            │
│                                                  ┌─────────────────┐  │
│                                                  │  Kubernetes HPA │  │
│                                                  │  (Modified)     │  │
│                                                  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Design Decisions & Rationale

#### Decision 1: Why Kubernetes as Platform?

**Technical Reasons:**
1. **Industry Standard**: Used by 88% of enterprises (CNCF Survey 2023)
2. **Native Autoscaling**: Built-in HPA/VPA mechanisms to extend
3. **Observability**: Rich metrics and logging ecosystem
4. **Cloud Agnostic**: Works on any cloud or on-premise

**Research Reasons:**
1. **Reproducibility**: Other researchers can replicate on any K8s cluster
2. **Real-world Relevance**: Industry adoption means practical impact
3. **Extensibility**: Custom controllers are well-documented pattern

#### Decision 2: Why ELK Stack for Logging?

**Alternatives Considered:**
| Solution | Pros | Cons | Decision |
|----------|------|------|----------|
| **ELK Stack** | Industry standard, powerful search, Kibana visualization, scalable | Resource-heavy, complex setup | ✓ **CHOSEN** |
| Loki + Grafana | Lightweight, fast, cloud-native | Less powerful search, newer/less mature | Good alternative |
| CloudWatch/Stackdriver | Managed, easy setup | Vendor lock-in, expensive, limited ML integration | Not suitable |
| Plain files | Simple, no overhead | Not scalable, hard to query | Not suitable |

**Why ELK Specifically:**
1. **Elasticsearch**:
   - Full-text search for log analysis
   - Aggregations for feature extraction (e.g., "count requests per IP per minute")
   - Time-series optimized

2. **Logstash**:
   - Rich filtering and parsing plugins
   - Grok patterns for log parsing
   - Easy to extract features from logs

3. **Kibana**:
   - Visualization for debugging
   - Real-time dashboards for monitoring
   - Important for research presentation

**Literature Support:**
- **Barika et al. (2016)**: "Scalable Real-Time Log Analytics Using Elasticsearch"
- Used in production by Netflix, LinkedIn, Uber for log analysis

#### Decision 3: Why FluentD for Log Collection?

**Alternatives:**
| Solution | Pros | Cons | Decision |
|----------|------|------|----------|
| **FluentD** | CNCF project, Kubernetes-native, plugin ecosystem | Ruby-based (slower) | ✓ **CHOSEN** |
| Fluent Bit | Lightweight (C-based), faster | Fewer plugins, less mature | Good for resource-constrained |
| Filebeat | Part of ELK, lightweight | Less flexible, Elastic-specific | Tight coupling |
| Logstash | Powerful processing | Too heavy for log shipping | Use for processing only |

**Why FluentD:**
1. **DaemonSet Pattern**: Runs on every Kubernetes node, collects all pod logs
2. **Kubernetes Integration**: Auto-discovers pods, adds metadata (pod name, namespace, labels)
3. **Buffering**: Handles log bursts without data loss
4. **Routing**: Can send different logs to different destinations

**Architecture Pattern:**
```
Each K8s Node:
  FluentD DaemonSet → Reads all pod logs → Buffers → Ships to Elasticsearch
```

#### Decision 4: Why Multiple ML Models?

**Single Model Approach (Alternative):**
- Choose one "best" model (e.g., Isolation Forest)
- Simpler implementation
- Faster deployment

**Multi-Model Approach (Your Choice):**
- Compare 5 different algorithms
- Select best performer empirically
- Research contribution: comparative analysis

**Rationale:**
1. **No Free Lunch Theorem**: No single model is best for all datasets
2. **Research Rigor**: Comparative evaluation is more credible
3. **Practical Value**: Helps practitioners choose right model
4. **Ensemble Potential**: Can combine models for better accuracy

**Literature Support:**
- **Goldstein & Uchida (2016)**: Compared 19 anomaly detection algorithms - showed high variance in performance
- **Liu et al. (2018)**: "Isolation Forest" - showed superiority in some cases but not all
- **Domingos (2012)**: "A Few Useful Things to Know About Machine Learning" - discusses algorithm selection

---

## 4. Technology Stack Selection & Justification

### 4.1 Infrastructure Layer

#### Kubernetes Distribution Options

**For Development (Windows PC):**

| Option | Complexity | Features | Resource Usage | Decision |
|--------|-----------|----------|----------------|----------|
| **Docker Desktop K8s** | Low | Basic K8s, good for dev | 4GB RAM | ✓ **CHOSEN** |
| Minikube | Medium | Full K8s, add-ons | 2-4GB RAM | Alternative |
| Kind | Medium | Multi-node clusters | 2GB RAM | Good for CI/CD |
| K3d | Medium | Lightweight | 1GB RAM | Too minimal |

**Decision: Docker Desktop Kubernetes**
- **Reasoning**: Easiest setup on Windows, tight Docker integration, sufficient for development
- **Tradeoff**: Limited to single node, but adequate for testing components

**For Production (Ubuntu Server):**

| Option | Complexity | Features | Resource Usage | Decision |
|--------|-----------|----------|----------------|----------|
| **k3s** | Low | Lightweight, single-binary | 512MB RAM | ✓ **CHOSEN** |
| kubeadm | High | Full K8s, production-grade | 2GB+ RAM | Overkill for research |
| MicroK8s | Low | Snap-based, batteries included | 1GB RAM | Ubuntu-specific |
| Managed (EKS/GKE/AKS) | Low | Fully managed | Varies | Too expensive |

**Decision: k3s**
- **Reasoning**:
  - Lightweight (40MB binary vs 1GB+ for full K8s)
  - Production-ready (used by Rancher, SUSE)
  - Perfect for single-server deployments
  - Fast installation (< 30 seconds)
  - Includes Traefik ingress, local storage, metrics server

- **Tradeoff**: Some features removed (legacy/alpha features), but all core functionality present

**Literature Support:**
- **Burns et al. (2016)**: "Borg, Omega, and Kubernetes" - Establishes K8s patterns
- **Rancher Labs (2019)**: k3s whitepaper - Demonstrates production viability

### 4.2 Logging & Monitoring Stack

#### Complete ELK Stack Configuration

```yaml
# Why each component is necessary:

Elasticsearch:
  Purpose: "Search and analytics engine"
  Why: "Stores logs with indexing for fast queries"
  Alternatives: Splunk (expensive), Loki (less powerful search)

Logstash:
  Purpose: "Log processing pipeline"
  Why: "Parses unstructured logs into structured data"
  Example:
    Input: "192.168.1.1 - - [01/Jan/2024:12:00:00] GET /api/users 200"
    Output: {ip: "192.168.1.1", timestamp: "...", method: "GET",
             endpoint: "/api/users", status: 200}

Kibana:
  Purpose: "Visualization and exploration"
  Why: "Debug anomaly detection, visualize patterns"

FluentD:
  Purpose: "Log collection agent"
  Why: "Kubernetes-native log shipping"
```

**Resource Planning:**
```
Minimum Resources for ELK:
- Elasticsearch: 2GB RAM, 2 CPU cores
- Logstash: 1GB RAM, 1 CPU core
- Kibana: 1GB RAM, 1 CPU core
- FluentD: 128MB RAM per node

Recommended for Production:
- Elasticsearch: 8GB RAM (JVM heap: 4GB), 4 CPU cores
- Logstash: 2GB RAM, 2 CPU cores
- Kibana: 2GB RAM, 1 CPU core
```

**Scaling Considerations:**
```
Log Volume Estimation:
- 10 pods × 100 log lines/sec × 500 bytes/line = 500 KB/sec
- Daily volume: 500 KB/sec × 86400 sec = 43 GB/day
- Elasticsearch storage: 43 GB/day × 7 days retention = 301 GB

Index Strategy:
- Daily indices: logs-2024.01.01, logs-2024.01.02, ...
- Automatic rotation and deletion
- Hot-warm-cold architecture for cost optimization
```

### 4.3 ML/Data Science Stack

#### Programming Language: Python

**Why Python?**
1. **ML Ecosystem**: scikit-learn, TensorFlow, PyTorch
2. **Data Processing**: pandas, numpy for feature engineering
3. **Integration**: Kubernetes Python client, Elasticsearch Python client
4. **Research**: Jupyter notebooks for experimentation

**Alternatives Considered:**
- **R**: Strong for statistics, but poor production deployment
- **Java/Scala**: Good performance, but slower development
- **Go**: Excellent for K8s controllers, but limited ML libraries

#### ML Libraries Selection

```python
# Core ML Stack:

import numpy as np                    # Numerical computing
import pandas as pd                   # Data manipulation
import scikit-learn as sklearn        # Classical ML algorithms
import tensorflow as tf               # Deep learning (Autoencoder)

# Specific Libraries by Model:

# 1. Autoencoder
from tensorflow.keras import layers, models
# Why TensorFlow: Industry standard, Keras API is easy, TF Serving for deployment

# 2. Isolation Forest
from sklearn.ensemble import IsolationForest
# Why scikit-learn: Well-tested implementation, good documentation

# 3. One-Class SVM
from sklearn.svm import OneClassSVM
# Why scikit-learn: Mature implementation, proven in research

# 4. K-Nearest Neighbors
from sklearn.neighbors import NearestNeighbors
# Why scikit-learn: Efficient k-d tree implementation

# 5. Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
# Why scikit-learn: EM algorithm implementation, handles multiple modes
```

**Model Serving Options:**

| Option | Pros | Cons | Use Case |
|--------|------|------|----------|
| **TensorFlow Serving** | High performance, gRPC/REST API, production-grade | Only for TF models | Autoencoder ✓ |
| **FastAPI + scikit-learn** | Simple, flexible, Python-native | Slower than TF Serving | sklearn models ✓ |
| **Seldon Core** | K8s-native, multi-framework, A/B testing | Complex setup | Production scaling |
| **KServe** | Serverless ML, autoscaling | Steep learning curve | Advanced use |

**Decision: Hybrid Approach**
- TensorFlow Serving for Autoencoder (requires low latency)
- FastAPI for sklearn models (easier to containerize together)

### 4.4 Integration & Orchestration

#### Custom Controller (Threat-Aware Autoscaler)

**Language: Python**
```python
# Why Python for K8s controller:
from kubernetes import client, config, watch
# - Official Kubernetes Python client
# - Easy integration with ML models
# - Rapid development

# Alternative: Go
# Pros: Native K8s language, better performance
# Cons: Longer development time, harder to integrate with Python ML models
```

**Controller Pattern:**
```python
"""
Kubernetes Controller Pattern:
1. Watch K8s resources (Deployments, Pods)
2. Reconcile desired state vs current state
3. Take action to converge to desired state

Our Controller:
1. Watch: Pod metrics, Anomaly scores
2. Reconcile: Normal autoscaling vs Threat-aware autoscaling
3. Action: Adjust HPA target metrics
"""

class ThreatAwareAutoscaler:
    def __init__(self):
        self.ml_models = load_models()
        self.k8s_client = kubernetes.client.AutoscalingV2Api()

    def reconcile_loop(self):
        while True:
            # Get current pod metrics
            metrics = get_pod_metrics()

            # Get anomaly score from ML models
            anomaly_score = self.ml_models.predict(metrics)

            # Adjust autoscaling based on threat level
            if anomaly_score > 0.8:  # High anomaly
                # Don't scale up, possibly scale down
                adjust_hpa(scale_up=False)
            else:  # Normal traffic
                # Normal autoscaling
                adjust_hpa(scale_up=True)

            time.sleep(10)  # Reconcile every 10 seconds
```

**Deployment Strategy:**
```yaml
# The controller itself runs as a Deployment in K8s:

apiVersion: apps/v1
kind: Deployment
metadata:
  name: threat-aware-autoscaler
spec:
  replicas: 1  # Single instance (leader election for HA)
  template:
    spec:
      containers:
      - name: controller
        image: your-registry/threat-aware-autoscaler:v1
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
      serviceAccountName: autoscaler-sa  # Needs permissions to modify HPA
```

---

## 5. Component-by-Component Deep Dive

### 5.1 Log Collection Pipeline

#### FluentD Configuration Deep Dive

**Why FluentD DaemonSet Pattern?**
```
Traditional Logging:
  Application → STDOUT → Lost when pod dies ❌

FluentD DaemonSet:
  Application → STDOUT → Node filesystem → FluentD → Elasticsearch ✓

Benefits:
1. Automatic log collection (no app changes needed)
2. Survives pod restarts
3. Adds Kubernetes metadata automatically
```

**FluentD Configuration Example:**
```xml
<source>
  @type tail
  path /var/log/containers/*.log
  pos_file /var/log/fluentd-containers.log.pos
  tag kubernetes.*
  read_from_head true
  <parse>
    @type json
    time_format %Y-%m-%dT%H:%M:%S.%NZ
  </parse>
</source>

# Filter: Add Kubernetes metadata
<filter kubernetes.**>
  @type kubernetes_metadata
  @id filter_kube_metadata
</filter>

# Filter: Parse application logs
<filter kubernetes.var.log.containers.app-*.log>
  @type parser
  key_name log
  <parse>
    @type regexp
    expression /^(?<remote_addr>[^ ]*) - - \[(?<time>[^\]]*)\] "(?<method>\S+) (?<path>\S+) HTTP\/\S+" (?<status>[^ ]*) (?<body_bytes_sent>[^ ]*)/
    time_format %d/%b/%Y:%H:%M:%S %z
  </parse>
</filter>

# Output: Send to Elasticsearch
<match kubernetes.**>
  @type elasticsearch
  host elasticsearch.logging.svc.cluster.local
  port 9200
  logstash_format true
  logstash_prefix k8s-logs
  include_tag_key true
  type_name _doc

  <buffer>
    @type file
    path /var/log/fluentd-buffers/kubernetes.system.buffer
    flush_mode interval
    retry_type exponential_backoff
    flush_interval 5s
    retry_max_interval 30
    chunk_limit_size 2M
    queue_limit_length 8
    overflow_action block
  </buffer>
</match>
```

**Why This Configuration:**
1. **tail input**: Reads from container log files (K8s writes here)
2. **kubernetes_metadata filter**: Adds pod name, namespace, labels
3. **parser filter**: Extracts fields from log lines (IP, method, path, status)
4. **elasticsearch output**: Sends to ELK stack with buffering for reliability

**Buffer Strategy Explanation:**
```
Why buffering is critical:
1. Elasticsearch might be down → don't lose logs
2. Log bursts (traffic spike) → smooth out writes
3. Network issues → retry with exponential backoff

Buffer settings explained:
- flush_interval: 5s → send logs every 5 seconds
- chunk_limit_size: 2M → batch size
- queue_limit_length: 8 → max 16M in buffer (8 × 2M)
- overflow_action: block → slow down log generation if buffer full
```

### 5.2 Feature Extraction Pipeline

#### From Logs to ML Features

**Raw Log Example:**
```
192.168.1.100 - - [01/Jan/2024:12:34:56 +0000] "GET /api/users?page=1 HTTP/1.1" 200 1234 "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
```

**Feature Extraction Process:**

```python
# Step 1: Parse log line (done by FluentD)
log_entry = {
    "remote_addr": "192.168.1.100",
    "timestamp": "2024-01-01T12:34:56Z",
    "method": "GET",
    "path": "/api/users",
    "query": "page=1",
    "status": 200,
    "body_bytes_sent": 1234,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Step 2: Enrich with geo-location (using IP)
import geoip2.database
reader = geoip2.database.Reader('GeoLite2-City.mmdb')
response = reader.city(log_entry["remote_addr"])
log_entry["country"] = response.country.name
log_entry["city"] = response.city.name

# Step 3: Time-window aggregations (Elasticsearch aggregation)
# "How many requests from this IP in the last 1 minute?"
GET /k8s-logs-*/_search
{
  "query": {
    "bool": {
      "must": [
        {"term": {"remote_addr": "192.168.1.100"}},
        {"range": {"@timestamp": {"gte": "now-1m"}}}
      ]
    }
  },
  "aggs": {
    "request_count": {"value_count": {"field": "remote_addr"}}
  }
}

# Step 4: Feature vector creation
features = {
    # Categorical features (one-hot encoded)
    "method_GET": 1,
    "method_POST": 0,
    "status_2xx": 1,
    "status_4xx": 0,
    "status_5xx": 0,

    # Numerical features
    "request_rate": 45.2,  # requests/sec from this IP
    "avg_payload_size": 1234,
    "hour_of_day": 12,
    "day_of_week": 1,  # Monday

    # Geo features (one-hot or label encoded)
    "country_US": 1,
    "country_CN": 0,

    # User-Agent features
    "is_bot": 0,  # Detected from User-Agent string
    "browser_chrome": 1,

    # Path features
    "path_length": 10,  # len("/api/users")
    "query_params_count": 1,

    # Derived features
    "entropy_of_path": 2.54,  # Shannon entropy (high = random/suspicious)
    "unique_paths_from_ip": 5  # Diversity of requests
}
```

**Feature Engineering Rationale:**

| Feature | Why It Matters | Normal Pattern | Anomaly Pattern |
|---------|----------------|----------------|-----------------|
| **request_rate** | Attack detection | 1-10 req/sec | 1000+ req/sec (DDoS) |
| **status_code_distribution** | Error pattern | Mostly 2xx | Many 4xx (scanning) or 5xx (exploit attempt) |
| **geo_location** | Geographic anomaly | Expected regions | Unusual countries |
| **user_agent** | Bot detection | Real browsers | Scripted/missing UA |
| **path_entropy** | Random path detection | Low (0-2) | High (>4) indicates random paths |
| **time_of_day** | Temporal anomaly | Business hours | 3 AM traffic spike |
| **payload_size** | Exfiltration detection | Small (<10KB) | Large (>1MB uploads) |

**Literature Support:**
- **Sommer & Paxson (2010)**: "Outside the Closed World: On Using Machine Learning for Network Intrusion Detection" - Discusses feature engineering importance
- **Garcia et al. (2014)**: "An Empirical Comparison of Botnet Detection Methods" - Shows request rate and temporal features are critical

#### Logstash Pipeline for Feature Engineering

```ruby
# Logstash configuration for real-time feature extraction

filter {
  # 1. GeoIP enrichment
  geoip {
    source => "remote_addr"
    target => "geoip"
    database => "/usr/share/logstash/GeoLite2-City.mmdb"
  }

  # 2. User-Agent parsing
  useragent {
    source => "user_agent"
    target => "ua"
  }

  # 3. Time-based features
  ruby {
    code => "
      event.set('[time_features][hour]', event.get('@timestamp').hour)
      event.set('[time_features][day_of_week]', event.get('@timestamp').wday)
      event.set('[time_features][is_weekend]', [0, 6].include?(event.get('@timestamp').wday))
    "
  }

  # 4. Path analysis
  ruby {
    code => "
      path = event.get('path')
      event.set('[path_features][length]', path.length)
      event.set('[path_features][depth]', path.split('/').length)

      # Shannon entropy calculation
      require 'set'
      chars = path.chars
      entropy = 0.0
      chars.to_set.each do |c|
        p = chars.count(c).to_f / chars.length
        entropy -= p * Math.log2(p)
      end
      event.set('[path_features][entropy]', entropy)
    "
  }

  # 5. Elasticsearch lookup for request rate
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    query => 'remote_addr:%{remote_addr} AND @timestamp:[now-1m TO now]'
    index => "k8s-logs-*"
    fields => {
      "@timestamp" => "recent_requests"
    }
  }

  # 6. Calculate request rate
  ruby {
    code => "
      recent = event.get('recent_requests')
      event.set('[aggregated_features][request_rate_per_min]', recent ? recent.length : 0)
    "
  }
}

output {
  # Send enriched logs to Elasticsearch
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "k8s-features-%{+YYYY.MM.dd}"
  }

  # Also send to ML model inference service
  http {
    url => "http://ml-inference-service:8080/predict"
    http_method => "post"
    format => "json"
  }
}
```

**Why This Pipeline:**
1. **Real-time**: Features extracted as logs arrive (stream processing)
2. **Enrichment**: Adds external data (GeoIP, User-Agent DB)
3. **Aggregation**: Combines multiple logs for window-based features
4. **Dual Output**: Storage (Elasticsearch) + ML inference

### 5.3 ML Model Implementation Details

#### Model 1: Autoencoder (Deep Learning Approach)

**Architecture Design:**

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_dim=50, encoding_dim=10):
    """
    Autoencoder for anomaly detection.

    Concept: Learn compressed representation of normal traffic.
    If reconstruction error is high → anomaly.

    Args:
        input_dim: Number of features (50 in our case)
        encoding_dim: Compressed representation size (10)

    Architecture:
        Encoder: 50 → 30 → 20 → 10 (compress)
        Decoder: 10 → 20 → 30 → 50 (reconstruct)
    """

    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(30, activation='relu')(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Dense(20, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)  # Bottleneck

    # Decoder
    decoded = layers.Dense(20, activation='relu')(encoded)
    decoded = layers.Dense(30, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)  # Reconstruction

    # Full model
    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

# Training (only on normal traffic)
autoencoder = build_autoencoder()
history = autoencoder.fit(
    X_train_normal,  # Only legitimate traffic
    X_train_normal,  # Trying to reconstruct itself
    epochs=100,
    batch_size=256,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint('autoencoder_best.h5')
    ]
)

# Inference (detecting anomalies)
def detect_anomaly(features):
    reconstruction = autoencoder.predict(features)
    mse = np.mean(np.power(features - reconstruction, 2), axis=1)

    # Threshold based on training data (e.g., 95th percentile)
    threshold = np.percentile(mse_train, 95)

    anomaly_score = mse / threshold  # Normalized score
    is_anomaly = mse > threshold

    return anomaly_score, is_anomaly
```

**Why This Architecture:**
1. **Bottleneck (50→10)**: Forces model to learn essential patterns, ignores noise
2. **Batch Normalization**: Stabilizes training, faster convergence
3. **Dropout**: Prevents overfitting to training data
4. **Sigmoid output**: Maps to [0,1] range like input (after normalization)

**Training Strategy:**
```python
# Hyperparameter tuning
param_grid = {
    'encoding_dim': [5, 10, 15],
    'epochs': [50, 100, 150],
    'learning_rate': [0.001, 0.0001]
}

# Use validation loss to select best configuration
```

**Literature Support:**
- **Sakurada & Yairi (2014)**: "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction"
- **Zhou & Paffenroth (2017)**: "Anomaly Detection with Robust Deep Autoencoders"

#### Model 2: Isolation Forest

**Concept:**
```
Normal points: Dense, clustered together
Anomalies: Sparse, isolated, easy to separate

Algorithm:
1. Randomly select feature
2. Randomly select split value between min/max
3. Recursively partition data
4. Anomalies require fewer splits to isolate

Anomaly Score = 2^(-average_path_length / c(n))
where c(n) is average path length for normal data
```

**Implementation:**

```python
from sklearn.ensemble import IsolationForest

# Model configuration
iso_forest = IsolationForest(
    n_estimators=100,        # Number of trees
    max_samples='auto',      # Subsample size (256 or n if n<256)
    contamination=0.1,       # Expected proportion of anomalies
    max_features=1.0,        # Features to consider per split
    bootstrap=False,
    n_jobs=-1,               # Parallel processing
    random_state=42
)

# Training (only normal traffic)
iso_forest.fit(X_train_normal)

# Inference
anomaly_scores = iso_forest.decision_function(X_test)  # Negative scores = anomaly
predictions = iso_forest.predict(X_test)  # -1 = anomaly, 1 = normal

# Convert to probability-like score [0,1]
anomaly_prob = (1 - (anomaly_scores - anomaly_scores.min()) /
                (anomaly_scores.max() - anomaly_scores.min()))
```

**Why Isolation Forest for Traffic Anomaly:**
1. **No distance metric needed**: Works well with mixed feature types
2. **Fast**: O(n log n) training, O(log n) prediction
3. **Scalable**: Can handle millions of samples
4. **Robust**: Doesn't assume data distribution

**Parameter Tuning:**
```python
# n_estimators: More trees = better but slower
# - Start with 100, increase if underfitting
# contamination: Expected anomaly rate
# - Network traffic: 0.05-0.1 (5-10% anomalies)
# max_features: Feature subsampling
# - Use 1.0 (all features) if features are well-engineered
```

**Literature Support:**
- **Liu et al. (2008)**: "Isolation Forest" (Original paper)
- **Liu et al. (2012)**: "Isolation-Based Anomaly Detection" - Theoretical analysis

#### Model 3: One-Class SVM

**Concept:**
```
Learn a hypersphere (or hyperplane) that encloses normal data.
Anything outside this boundary = anomaly.

Kernel trick: Map data to higher dimension where separation is easier.
```

**Implementation:**

```python
from sklearn.svm import OneClassSVM

# Model configuration
oc_svm = OneClassSVM(
    kernel='rbf',           # Radial Basis Function (Gaussian kernel)
    gamma='scale',          # Kernel coefficient (1 / (n_features * X.var()))
    nu=0.1,                 # Upper bound on fraction of outliers
    max_iter=-1,            # No iteration limit
)

# Training
oc_svm.fit(X_train_normal)

# Inference
predictions = oc_svm.predict(X_test)  # -1 = anomaly, 1 = normal
decision_scores = oc_svm.decision_function(X_test)  # Distance from boundary

# Convert to [0,1] score
anomaly_prob = 1 / (1 + np.exp(decision_scores))  # Sigmoid transformation
```

**Kernel Selection Rationale:**

| Kernel | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **RBF (Gaussian)** | Default choice, non-linear boundaries | Flexible, handles complex patterns | Sensitive to gamma |
| Linear | High-dimensional data (>10k features) | Fast, interpretable | Only linear boundaries |
| Polynomial | Structured data with interactions | Captures feature interactions | Expensive for large data |

**Parameter Tuning:**
```python
# gamma: Controls decision boundary smoothness
# - Small gamma: Smooth boundary (may underfit)
# - Large gamma: Tight boundary (may overfit)
# - 'scale': 1 / (n_features * X.var()) - good default

# nu: Upper bound on training errors and lower bound on support vectors
# - Set to expected anomaly rate (e.g., 0.1 = 10%)
```

**Challenges with One-Class SVM:**
1. **Scalability**: O(n²) to O(n³) time complexity
2. **Memory**: Stores support vectors (can be large)
3. **Hyperparameter sensitivity**: gamma and nu require tuning

**Why Still Use It:**
- Strong theoretical foundation (statistical learning theory)
- Often outperforms other methods with proper tuning
- Robust to outliers in training data (nu parameter)

**Literature Support:**
- **Schölkopf et al. (2001)**: "Estimating the Support of a High-Dimensional Distribution"
- **Tax & Duin (2004)**: "Support Vector Data Description"

#### Model 4: K-Nearest Neighbors (KNN)

**Concept:**
```
Normal points: Close to many other normal points
Anomalies: Far from normal points

Anomaly score = Average distance to k nearest neighbors
```

**Implementation:**

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

class KNNAnomalyDetector:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm='auto',  # Let sklearn choose (ball_tree, kd_tree, brute)
            metric=metric
        )
        self.threshold = None

    def fit(self, X_train_normal):
        """Train on normal data only."""
        self.knn.fit(X_train_normal)

        # Calculate distances for training data to set threshold
        distances, _ = self.knn.kneighbors(X_train_normal)
        avg_distances = np.mean(distances, axis=1)

        # Threshold = 95th percentile of training distances
        self.threshold = np.percentile(avg_distances, 95)

        return self

    def predict(self, X_test):
        """Predict anomalies."""
        distances, _ = self.knn.kneighbors(X_test)
        avg_distances = np.mean(distances, axis=1)

        # Anomaly score (0-1 range)
        anomaly_scores = avg_distances / (self.threshold + 1e-10)

        # Binary prediction
        predictions = (avg_distances > self.threshold).astype(int)
        predictions = np.where(predictions == 1, -1, 1)  # -1 = anomaly, 1 = normal

        return predictions, anomaly_scores

# Usage
knn_detector = KNNAnomalyDetector(n_neighbors=5)
knn_detector.fit(X_train_normal)
predictions, scores = knn_detector.predict(X_test)
```

**Distance Metric Selection:**

| Metric | Use Case | Calculation |
|--------|----------|-------------|
| **Euclidean** | Continuous features | sqrt(Σ(xi - yi)²) |
| **Manhattan** | Sparse data, high dimensions | Σ\|xi - yi\| |
| **Cosine** | Text/high-dim data | 1 - (X·Y)/(\\|X\\|\\|Y\\|) |
| **Mahalanobis** | Correlated features | Accounts for covariance |

**Algorithm Selection (Auto):**
```python
# sklearn automatically chooses based on data:

if n_samples < 30:
    algorithm = 'brute'  # Check all points O(n²)
elif n_features > 20:
    algorithm = 'ball_tree'  # Works well in high dimensions
else:
    algorithm = 'kd_tree'  # Fastest for low dimensions
```

**Parameter Tuning:**
```python
# k (n_neighbors): Number of neighbors to consider
# - Too small (k=1): Sensitive to noise
# - Too large (k=50): May include anomalies in neighborhood
# - Rule of thumb: k = sqrt(n_samples) or 5-10 for small datasets

# Distance metric: Based on feature types
# - All numerical, same scale: Euclidean
# - Mixed scales: Normalize first, then Euclidean
# - Categorical: Hamming distance
```

**Advantages for Traffic Anomaly:**
1. **Interpretable**: Easy to explain (distance-based)
2. **No training**: Just stores data (lazy learning)
3. **Non-parametric**: No assumptions about data distribution

**Challenges:**
1. **Curse of dimensionality**: All points equidistant in high dimensions
2. **Computational cost**: O(n) for each prediction
3. **Memory**: Stores all training data

**Optimization Strategies:**
```python
# 1. Dimensionality reduction before KNN
from sklearn.decomposition import PCA
pca = PCA(n_components=20)  # Reduce from 50 to 20 features
X_reduced = pca.fit_transform(X_train_normal)

# 2. Approximate nearest neighbors (for large datasets)
from annoy import AnnoyIndex  # Spotify's library
index = AnnoyIndex(n_features, metric='euclidean')
# Much faster for millions of samples
```

**Literature Support:**
- **Ramaswamy et al. (2000)**: "Efficient Algorithms for Mining Outliers from Large Data Sets"
- **Angiulli & Pizzuti (2002)**: "Fast Outlier Detection in High Dimensional Spaces"

#### Model 5: Gaussian Mixture Model (GMM)

**Concept:**
```
Assume normal data comes from mixture of Gaussian distributions.
Points with low probability under this mixture = anomalies.

P(x) = Σ πk * N(x | μk, Σk)
where:
- πk: Weight of k-th Gaussian
- μk: Mean of k-th Gaussian
- Σk: Covariance of k-th Gaussian
```

**Implementation:**

```python
from sklearn.mixture import GaussianMixture
import numpy as np

class GMMAnomaly Detector:
    def __init__(self, n_components=3, covariance_type='full'):
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=100,
            n_init=10,  # Run EM algorithm 10 times, keep best
            random_state=42
        )
        self.threshold = None

    def fit(self, X_train_normal):
        """Train on normal data."""
        self.gmm.fit(X_train_normal)

        # Calculate log-likelihood for training data
        log_probs = self.gmm.score_samples(X_train_normal)

        # Threshold = 5th percentile (bottom 5% of training data)
        self.threshold = np.percentile(log_probs, 5)

        return self

    def predict(self, X_test):
        """Predict anomalies."""
        log_probs = self.gmm.score_samples(X_test)

        # Anomaly score (lower probability = higher anomaly score)
        anomaly_scores = -log_probs  # Negate so high score = anomaly
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())

        # Binary prediction
        predictions = (log_probs < self.threshold).astype(int)
        predictions = np.where(predictions == 1, -1, 1)

        return predictions, anomaly_scores

# Usage
gmm_detector = GMMAnomaly Detector(n_components=3)
gmm_detector.fit(X_train_normal)
predictions, scores = gmm_detector.predict(X_test)
```

**Covariance Type Selection:**

| Type | Assumption | Parameters | Use Case |
|------|------------|------------|----------|
| **full** | Each component has own full covariance | Most flexible | Small datasets, complex patterns |
| **tied** | All components share covariance | Medium | Clusters have same shape |
| **diag** | Diagonal covariance (features independent) | Fewer params | High dimensions |
| **spherical** | Single variance per component | Fewest params | Very high dimensions |

**Number of Components Selection:**

```python
# Use BIC (Bayesian Information Criterion) or AIC
from sklearn.mixture import GaussianMixture

bic_scores = []
aic_scores = []
n_components_range = range(1, 10)

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type='full')
    gmm.fit(X_train_normal)
    bic_scores.append(gmm.bic(X_train_normal))
    aic_scores.append(gmm.aic(X_train_normal))

# Choose n_components with lowest BIC
optimal_n = n_components_range[np.argmin(bic_scores)]
```

**Why GMM for Traffic Anomaly:**
1. **Multi-modal**: Traffic has different patterns (mobile, desktop, API, bot)
2. **Probabilistic**: Provides uncertainty estimates
3. **Density-based**: Good for detecting subtle anomalies

**Challenges:**
1. **EM algorithm**: Can get stuck in local optima (hence n_init=10)
2. **Dimensionality**: Struggles with >20 features (use 'diag' or 'spherical')
3. **Gaussian assumption**: May not fit heavy-tailed distributions

**Literature Support:**
- **Reynolds (2009)**: "Gaussian Mixture Models" - Encyclopedia chapter
- **Chandola et al. (2009)**: "Anomaly Detection: A Survey" - Discusses GMM for anomaly detection

---

### 5.4 Model Ensemble & Selection Strategy

#### Why Compare 5 Models?

**Research Perspective:**
```
Goal: Find the BEST model for this specific problem
Method: Empirical comparison on real data

Publishing:
- "We tested 5 models and Model X achieved 95% accuracy" is stronger than
- "We used Model X and achieved 85% accuracy" (but maybe Model Y could do 95%?)
```

**Practical Perspective:**
```
Different models excel at different anomaly types:

Autoencoder: Good for complex, non-linear patterns
Isolation Forest: Good for global outliers
One-Class SVM: Good for boundary-based anomalies
KNN: Good for local density anomalies
GMM: Good for multi-modal data with cluster-based anomalies
```

#### Ensemble Methods

**Approach 1: Majority Voting**
```python
def ensemble_predict(X_test, models):
    predictions = []

    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)

    # Stack predictions (n_samples, n_models)
    predictions = np.array(predictions).T

    # Majority vote
    final_pred = np.apply_along_axis(
        lambda x: 1 if np.sum(x == 1) > np.sum(x == -1) else -1,
        axis=1,
        arr=predictions
    )

    return final_pred
```

**Approach 2: Weighted Average of Scores**
```python
def ensemble_score(X_test, models, weights):
    """
    weights: Based on validation performance
    Example: [0.3, 0.25, 0.2, 0.15, 0.1] if Autoencoder is best
    """
    scores = []

    for model in models:
        score = model.anomaly_score(X_test)
        scores.append(score)

    # Weighted average
    ensemble_score = np.average(scores, axis=0, weights=weights)

    return ensemble_score

# Determine weights from validation set
weights = calculate_weights_from_validation(models, X_val, y_val)
```

**Approach 3: Stacking (Meta-Learner)**
```python
from sklearn.linear_model import LogisticRegression

# Step 1: Get predictions from all base models
base_predictions = np.column_stack([
    model.anomaly_score(X_train) for model in models
])

# Step 2: Train meta-learner
meta_learner = LogisticRegression()
meta_learner.fit(base_predictions, y_train)

# Step 3: Final prediction
def stacking_predict(X_test):
    base_preds = np.column_stack([
        model.anomaly_score(X_test) for model in models
    ])
    final_pred = meta_learner.predict(base_preds)
    return final_pred
```

**Recommendation for Your Research:**
```
Phase 1 (Initial): Compare models individually
- Train each model separately
- Evaluate on test set
- Report comparative results

Phase 2 (Advanced): Ensemble methods
- If no single model dominates, try ensemble
- Weighted average is good balance (simple but effective)
- Stacking for best performance (but harder to interpret)
```

---

## 6. Challenge-Solution Mapping

### 6.1 Technical Challenges

#### Challenge 1: Real-Time Processing Latency

**Problem:**
```
Requirement: Detect anomalies in <100ms (real-time autoscaling)
Reality:
- Log collection: 1-5s delay
- Feature extraction: 50-200ms
- ML inference: 10-100ms
- Total: 1-5 seconds (too slow!)
```

**Solutions:**

**Solution 1: Streaming Architecture**
```
Traditional (Batch):
FluentD → Elasticsearch → Batch job (every 5 min) → ML inference
Latency: 5+ minutes ❌

Streaming:
FluentD → Logstash → Kafka → Flink → ML inference → Redis
Latency: <1 second ✓

Components:
- Kafka: Message queue for log buffering
- Flink: Stream processing for feature extraction
- Redis: In-memory cache for recent anomaly scores
```

**Solution 2: Model Optimization**
```python
# Technique 1: Model quantization (for Autoencoder)
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('autoencoder')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Result: 4x faster inference, 4x smaller model

# Technique 2: Feature selection (reduce from 50 to 20 features)
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_train, y_train)

# Result: 2-3x faster inference

# Technique 3: Model caching
# Cache predictions for identical feature vectors
from functools import lru_cache

@lru_cache(maxsize=10000)
def predict_cached(features_tuple):
    features = np.array(features_tuple).reshape(1, -1)
    return model.predict(features)
```

**Solution 3: Approximate Detection**
```
Instead of scoring every request:
1. Sample 10% of requests for detailed ML analysis
2. Use simple rules for 90% (e.g., request rate > 100/s = likely DDoS)
3. Combine results

Trade-off: Miss some subtle anomalies, but 10x faster
```

**Evaluation Metric:**
```python
import time

def measure_latency(model, X_test, n_runs=100):
    latencies = []
    for _ in range(n_runs):
        start = time.time()
        model.predict(X_test[:100])  # Batch of 100
        end = time.time()
        latencies.append((end - start) / 100)  # Per-sample latency

    return {
        'mean': np.mean(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99)
    }

# Target: p95 < 100ms
```

#### Challenge 2: Concept Drift (Traffic Patterns Change)

**Problem:**
```
Today: Normal traffic = 1000 req/s
Next Month: Normal traffic = 5000 req/s (business growth)

Issue: Model trained on 1000 req/s flags 5000 req/s as anomaly ❌
```

**Solutions:**

**Solution 1: Periodic Retraining**
```python
class AdaptiveAnomalyDetector:
    def __init__(self, model, retrain_interval_days=7):
        self.model = model
        self.retrain_interval = retrain_interval_days
        self.last_training_date = datetime.now()

    def check_and_retrain(self):
        """Retrain model weekly with recent normal data."""
        if (datetime.now() - self.last_training_date).days >= self.retrain_interval:
            # Fetch last 7 days of traffic (excluding anomalies)
            recent_data = fetch_normal_traffic(days=7)

            # Retrain model
            self.model.fit(recent_data)
            self.last_training_date = datetime.now()

            print(f"Model retrained on {len(recent_data)} samples")
```

**Solution 2: Online Learning**
```python
# For models that support incremental learning
from sklearn.linear_model import SGDOneClassSVM

# Update model with each batch
for batch in data_stream:
    if is_verified_normal(batch):  # Only update with confirmed normal data
        model.partial_fit(batch)
```

**Solution 3: Adaptive Thresholds**
```python
from collections import deque

class AdaptiveThreshold:
    def __init__(self, window_size=1000):
        self.scores = deque(maxlen=window_size)

    def update(self, score, is_anomaly):
        if not is_anomaly:  # Only track normal scores
            self.scores.append(score)

    def get_threshold(self, percentile=95):
        if len(self.scores) < 100:
            return self.default_threshold
        return np.percentile(list(self.scores), percentile)

# Usage
adaptive_threshold = AdaptiveThreshold()

for features in stream:
    score = model.anomaly_score(features)
    threshold = adaptive_threshold.get_threshold()
    is_anomaly = score > threshold

    # Update threshold with feedback
    adaptive_threshold.update(score, is_anomaly)
```

**Monitoring Drift:**
```python
# Track distribution shift using KL divergence
from scipy.stats import entropy

def detect_drift(X_train, X_recent):
    """Detect if recent data differs significantly from training."""

    # Compare feature distributions
    for i in range(X_train.shape[1]):
        # Histogram of training data
        hist_train, bins = np.histogram(X_train[:, i], bins=50)
        hist_train = hist_train / hist_train.sum()

        # Histogram of recent data
        hist_recent, _ = np.histogram(X_recent[:, i], bins=bins)
        hist_recent = hist_recent / hist_recent.sum()

        # KL divergence
        kl_div = entropy(hist_train + 1e-10, hist_recent + 1e-10)

        if kl_div > 0.5:  # Significant drift
            print(f"Feature {i} has drifted (KL={kl_div:.2f})")
            return True

    return False

# Run weekly
if detect_drift(X_training, X_last_week):
    trigger_retraining()
```

**Literature Support:**
- **Gama et al. (2014)**: "A Survey on Concept Drift Adaptation"
- **Losing et al. (2018)**: "Incremental On-line Learning: A Review"

#### Challenge 3: Label Scarcity (No Ground Truth)

**Problem:**
```
Anomaly detection = unsupervised (no labels)
But evaluation requires labels!

How do you know if your model is good?
```

**Solutions:**

**Solution 1: Synthetic Anomaly Injection**
```python
def inject_anomalies(X_normal, anomaly_ratio=0.1):
    """
    Create labeled dataset by injecting synthetic anomalies.

    Types of synthetic anomalies:
    1. Point anomalies: Random noise
    2. Contextual anomalies: Swap feature values
    3. Collective anomalies: Sequences of unusual patterns
    """
    n_anomalies = int(len(X_normal) * anomaly_ratio)
    X_synthetic = X_normal.copy()
    y_labels = np.zeros(len(X_normal))  # 0 = normal

    # Strategy 1: Add Gaussian noise
    anomaly_indices = np.random.choice(len(X_normal), n_anomalies // 3, replace=False)
    X_synthetic[anomaly_indices] += np.random.normal(0, 3, X_synthetic[anomaly_indices].shape)
    y_labels[anomaly_indices] = 1

    # Strategy 2: Feature swapping
    anomaly_indices = np.random.choice(len(X_normal), n_anomalies // 3, replace=False)
    for idx in anomaly_indices:
        i, j = np.random.choice(X_normal.shape[1], 2, replace=False)
        X_synthetic[idx, i], X_synthetic[idx, j] = X_synthetic[idx, j], X_synthetic[idx, i]
    y_labels[anomaly_indices] = 1

    # Strategy 3: Extreme values
    anomaly_indices = np.random.choice(len(X_normal), n_anomalies // 3, replace=False)
    for idx in anomaly_indices:
        feature = np.random.randint(0, X_normal.shape[1])
        X_synthetic[idx, feature] = X_normal[:, feature].max() * 10  # 10x max value
    y_labels[anomaly_indices] = 1

    return X_synthetic, y_labels

# Evaluate model on synthetic data
X_test, y_test = inject_anomalies(X_normal_test)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

**Solution 2: Expert Labeling (Small Sample)**
```
1. Run model on production data
2. Sample 1000 requests (500 predicted anomalies, 500 predicted normal)
3. Have security expert manually label them
4. Use as evaluation set

Cost: ~2 hours of expert time
Benefit: Real labels for evaluation
```

**Solution 3: Proxy Labels**
```python
# Use indirect indicators as pseudo-labels

def create_proxy_labels(logs):
    """
    Heuristic-based labels (not perfect, but useful):
    - HTTP 4xx/5xx status codes → likely anomaly
    - Known malicious IPs (from threat intel) → anomaly
    - Rate limiting triggered → anomaly
    """
    labels = []

    for log in logs:
        is_anomaly = (
            log['status'] >= 400 or                          # Error status
            log['ip'] in blacklisted_ips or                  # Known bad IP
            log['request_rate'] > 100 or                     # Rate limit
            'bot' in log['user_agent'].lower()               # Bot traffic
        )
        labels.append(1 if is_anomaly else 0)

    return np.array(labels)

# Use for quick evaluation (not perfect, but indicative)
```

**Solution 4: Comparison with Baseline**
```python
# Even without labels, compare against simple baseline

class BaselineDetector:
    """Simple rule-based detector."""
    def predict(self, X):
        # Rule: Flag if request rate > 50/sec OR status = 5xx
        anomalies = (X[:, feature_index['request_rate']] > 50) | \
                    (X[:, feature_index['status_5xx']] == 1)
        return np.where(anomalies, -1, 1)

# If your ML model doesn't beat baseline, it's not useful
baseline = BaselineDetector()
ml_model = IsolationForest()

# Compare on same data
consistency = np.mean(baseline.predict(X) == ml_model.predict(X))
print(f"Agreement with baseline: {consistency:.2%}")

# ML should catch things baseline misses
ml_only_anomalies = (ml_model.predict(X) == -1) & (baseline.predict(X) == 1)
print(f"ML found {ml_only_anomalies.sum()} additional anomalies")
```

#### Challenge 4: Class Imbalance (Anomalies are Rare)

**Problem:**
```
Typical traffic:
- 99% normal requests
- 1% anomalies (attacks, errors, etc.)

Issue: Model can achieve 99% accuracy by predicting everything as normal!
```

**Solutions:**

**Solution 1: Use Appropriate Metrics**
```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def comprehensive_evaluation(y_true, y_pred, y_scores):
    """
    Don't use accuracy! Use these metrics instead:
    """

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        # Precision: Of predicted anomalies, how many are correct?
        'precision': precision_score(y_true, y_pred),

        # Recall: Of actual anomalies, how many did we catch?
        'recall': recall_score(y_true, y_pred),

        # F1: Harmonic mean of precision and recall
        'f1': f1_score(y_true, y_pred),

        # ROC-AUC: Area under ROC curve
        'roc_auc': roc_auc_score(y_true, y_scores),

        # PR-AUC: Better for imbalanced data
        'pr_auc': average_precision_score(y_true, y_scores),

        # False Positive Rate: Important for production (alert fatigue)
        'fpr': fp / (fp + tn),

        # True Negative Rate (Specificity)
        'tnr': tn / (tn + fp)
    }

    return metrics

# For imbalanced data, prioritize: Recall, PR-AUC, F1
```

**Solution 2: Cost-Sensitive Learning**
```python
# Assign higher cost to missing anomalies

# For models that support sample weights
class_weights = {
    0: 1.0,   # Normal - standard weight
    1: 10.0   # Anomaly - 10x weight (missing anomaly is 10x worse)
}

sample_weights = np.array([class_weights[y] for y in y_train])

# Train with weights
model.fit(X_train, y_train, sample_weight=sample_weights)
```

**Solution 3: Threshold Tuning**
```python
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(y_true, y_scores, target_recall=0.95):
    """
    Find threshold that achieves target recall.

    For security: High recall is critical (catch 95% of attacks)
    Accept lower precision (some false positives okay)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # Find threshold where recall >= target
    idx = np.argmax(recall >= target_recall)
    optimal_threshold = thresholds[idx]

    print(f"Threshold: {optimal_threshold:.3f}")
    print(f"Recall: {recall[idx]:.3f}")
    print(f"Precision: {precision[idx]:.3f}")

    return optimal_threshold

# Usage
threshold = find_optimal_threshold(y_val, anomaly_scores, target_recall=0.95)
final_predictions = (anomaly_scores > threshold).astype(int)
```

**Literature Support:**
- **He & Garcia (2009)**: "Learning from Imbalanced Data"
- **Branco et al. (2016)**: "A Survey of Predictive Modeling on Imbalanced Domains"

---

### 6.2 Integration Challenges

#### Challenge 5: Kubernetes Autoscaler Integration

**Problem:**
```
Standard HPA Controller:
if cpu_usage > 80%:
    scale_up()

Threat-Aware HPA:
if cpu_usage > 80% AND traffic_is_legitimate:
    scale_up()
else:
    scale_down()  # Don't waste resources on attackers
```

**Solutions:**

**Solution 1: Custom Metrics API**
```python
"""
Kubernetes allows custom metrics for HPA.
We'll expose anomaly_score as a custom metric.

Architecture:
ML Service → Prometheus → Metrics API → HPA
"""

# 1. ML service exposes metrics
from prometheus_client import Gauge, start_http_server

anomaly_score_metric = Gauge(
    'traffic_anomaly_score',
    'Current anomaly score (0-1)',
    ['pod', 'namespace']
)

def update_metrics():
    while True:
        score = calculate_current_anomaly_score()
        anomaly_score_metric.labels(
            pod='my-app',
            namespace='default'
        ).set(score)
        time.sleep(10)

# Start metrics server
start_http_server(8000)
threading.Thread(target=update_metrics, daemon=True).start()

# 2. Prometheus scrapes metrics
# prometheus.yml
scrape_configs:
  - job_name: 'ml-service'
    static_configs:
      - targets: ['ml-service:8000']

# 3. Register custom metric
apiVersion: v1
kind: ConfigMap
metadata:
  name: adapter-config
data:
  config.yaml: |
    rules:
    - seriesQuery: 'traffic_anomaly_score'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        as: "anomaly_score"
      metricsQuery: 'avg(traffic_anomaly_score{<<.LabelMatchers>>})'

# 4. HPA uses custom metric
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: threat-aware-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: anomaly_score
      target:
        type: AverageValue
        averageValue: "0.3"  # Scale up only if anomaly_score < 0.3
```

**Solution 2: Custom Controller (More Control)**
```python
"""
Implement custom controller that directly modifies HPA.
More flexible than custom metrics.
"""

from kubernetes import client, config
import time

class ThreatAwareAutoscaler:
    def __init__(self):
        config.load_incluster_config()  # Load K8s config
        self.apps_v1 = client.AppsV1Api()
        self.autoscaling_v1 = client.AutoscalingV1Api()
        self.ml_models = load_models()

    def get_current_metrics(self, namespace, deployment):
        """Fetch current pod metrics from Metrics Server."""
        # Simplified - in reality, use metrics-server API
        return {
            'cpu_percent': 75,
            'memory_percent': 60,
            'request_rate': 1000
        }

    def get_anomaly_score(self):
        """Get current anomaly score from ML model."""
        # Fetch recent traffic features from Elasticsearch
        features = fetch_recent_features()

        # Get predictions from all models
        scores = [model.predict(features) for model in self.ml_models]

        # Ensemble average
        anomaly_score = np.mean(scores)

        return anomaly_score

    def calculate_desired_replicas(self, current_replicas, metrics, anomaly_score):
        """
        Calculate desired replicas based on metrics and threat level.

        Logic:
        - If anomaly_score < 0.3: Normal autoscaling
        - If 0.3 <= anomaly_score < 0.7: Cautious scaling (slower)
        - If anomaly_score >= 0.7: No scaling up, consider scaling down
        """
        cpu_usage = metrics['cpu_percent']

        if anomaly_score >= 0.7:
            # High anomaly - likely attack
            # Scale down to minimum to save resources
            return max(2, current_replicas - 1)

        elif anomaly_score >= 0.3:
            # Medium anomaly - be cautious
            # Only scale if CPU very high
            if cpu_usage > 90:
                return current_replicas + 1
            elif cpu_usage < 30:
                return max(2, current_replicas - 1)
            else:
                return current_replicas

        else:
            # Low anomaly - normal traffic
            # Standard autoscaling
            target_cpu = 70
            desired = int(current_replicas * (cpu_usage / target_cpu))
            return max(2, min(10, desired))

    def reconcile(self):
        """Main reconciliation loop."""
        namespace = 'default'
        deployment_name = 'my-app'

        while True:
            try:
                # Get current state
                deployment = self.apps_v1.read_namespaced_deployment(
                    deployment_name, namespace
                )
                current_replicas = deployment.spec.replicas

                # Get metrics
                metrics = self.get_current_metrics(namespace, deployment_name)
                anomaly_score = self.get_anomaly_score()

                # Calculate desired state
                desired_replicas = self.calculate_desired_replicas(
                    current_replicas, metrics, anomaly_score
                )

                # Update if needed
                if desired_replicas != current_replicas:
                    deployment.spec.replicas = desired_replicas
                    self.apps_v1.patch_namespaced_deployment(
                        deployment_name, namespace, deployment
                    )
                    print(f"Scaled from {current_replicas} to {desired_replicas} "
                          f"(anomaly_score={anomaly_score:.2f})")

                time.sleep(30)  # Reconcile every 30 seconds

            except Exception as e:
                print(f"Error in reconcile loop: {e}")
                time.sleep(30)

    def run(self):
        """Start the controller."""
        print("Starting Threat-Aware Autoscaler...")
        self.reconcile()

# Deploy as pod in Kubernetes
if __name__ == '__main__':
    controller = ThreatAwareAutoscaler()
    controller.run()
```

**Deployment YAML:**
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: threat-aware-autoscaler
  namespace: default

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: threat-aware-autoscaler
rules:
- apiGroups: ["apps"]
  resources: ["deployments", "deployments/scale"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["pods", "nodes"]
  verbs: ["get", "list"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: threat-aware-autoscaler
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: threat-aware-autoscaler
subjects:
- kind: ServiceAccount
  name: threat-aware-autoscaler
  namespace: default

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: threat-aware-autoscaler
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: threat-aware-autoscaler
  template:
    metadata:
      labels:
        app: threat-aware-autoscaler
    spec:
      serviceAccountName: threat-aware-autoscaler
      containers:
      - name: controller
        image: your-registry/threat-aware-autoscaler:v1.0
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

**Literature Support:**
- **Kubernetes Documentation**: "Custom Metrics API"
- **Harchol-Balter (2013)**: "Performance Modeling and Design of Computer Systems" - Autoscaling theory

---

## 7. ML Model Selection Rationale

### 7.1 Theoretical Comparison

| Model | Type | Assumptions | Strengths | Weaknesses | Best For |
|-------|------|-------------|-----------|------------|----------|
| **Autoencoder** | Neural Network | Non-linear manifold | Complex patterns, feature learning | Slow training, requires tuning | High-dimensional, non-linear data |
| **Isolation Forest** | Ensemble | Anomalies are sparse | Fast, scalable, no assumptions | May miss local anomalies | Global outliers, large datasets |
| **One-Class SVM** | Support Vector | Decision boundary exists | Robust, kernel trick | Slow, memory-intensive | Small-medium datasets, clear boundary |
| **KNN** | Distance-based | Proximity = similarity | Simple, interpretable | Curse of dimensionality, slow prediction | Low-dimensional, small datasets |
| **GMM** | Probabilistic | Gaussian mixture | Probabilistic scores, multi-modal | EM algorithm instability | Data with clusters |

### 7.2 Computational Complexity

| Model | Training | Prediction | Memory |
|-------|----------|------------|---------|
| **Autoencoder** | O(n × epochs) | O(features) | O(parameters) ≈ MB |
| **Isolation Forest** | O(n log n) | O(log n) | O(trees × n) ≈ MB |
| **One-Class SVM** | O(n² to n³) | O(n_support_vectors) | O(n) ≈ GB for large n |
| **KNN** | O(1) (just store) | O(n) | O(n × features) ≈ GB |
| **GMM** | O(n × iterations) | O(components) | O(components × features²) ≈ MB |

**For Real-Time System (Target: <100ms prediction):**
- ✓ Autoencoder: ~10ms
- ✓ Isolation Forest: ~5ms
- ✓ GMM: ~3ms
- ⚠ One-Class SVM: ~50ms (borderline)
- ❌ KNN: ~200ms (too slow for n>100k)

### 7.3 Expected Performance on Traffic Data

**Hypothesis (to be validated):**

```
Predicted Ranking:
1. Isolation Forest - Fast, handles mixed features well
2. Autoencoder - Captures complex temporal patterns
3. GMM - Good for multi-modal (mobile, desktop, API traffic)
4. One-Class SVM - Solid baseline
5. KNN - Will struggle with high dimensions

Research Contribution:
Empirically validate this ranking on real Kubernetes traffic
```

**Evaluation Plan:**
```python
# Compare models on multiple metrics

results = {
    'model': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'roc_auc': [],
    'pr_auc': [],
    'training_time': [],
    'prediction_latency': [],
    'memory_usage': []
}

for model_name, model in models.items():
    # Training
    start = time.time()
    model.fit(X_train)
    training_time = time.time() - start

    # Prediction
    start = time.time()
    predictions = model.predict(X_test)
    prediction_latency = (time.time() - start) / len(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, model.anomaly_score(X_test))
    pr_auc = average_precision_score(y_test, model.anomaly_score(X_test))

    # Memory
    memory = get_model_size(model)

    # Store results
    results['model'].append(model_name)
    results['accuracy'].append(accuracy)
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1'].append(f1)
    results['roc_auc'].append(roc_auc)
    results['pr_auc'].append(pr_auc)
    results['training_time'].append(training_time)
    results['prediction_latency'].append(prediction_latency)
    results['memory_usage'].append(memory)

# Create comparison table
df = pd.DataFrame(results)
print(df.sort_values('f1', ascending=False))

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# F1 Score
axes[0,0].bar(df['model'], df['f1'])
axes[0,0].set_title('F1 Score')
axes[0,0].set_ylabel('Score')

# Latency
axes[0,1].bar(df['model'], df['prediction_latency'] * 1000)
axes[0,1].set_title('Prediction Latency')
axes[0,1].set_ylabel('Milliseconds')
axes[0,1].axhline(y=100, color='r', linestyle='--', label='Target: 100ms')

# PR-AUC (best for imbalanced data)
axes[1,0].bar(df['model'], df['pr_auc'])
axes[1,0].set_title('Precision-Recall AUC')
axes[1,0].set_ylabel('Score')

# Memory Usage
axes[1,1].bar(df['model'], df['memory_usage'])
axes[1,1].set_title('Memory Usage')
axes[1,1].set_ylabel('MB')

plt.tight_layout()
plt.savefig('model_comparison.png')
```

---

## 8. Integration Strategy & Data Flow

### 8.1 End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Traffic Generation & Logging                               │
└─────────────────────────────────────────────────────────────────────┘

User Request → Ingress → Service → Pod
                                     │
                                     ├─ Handle Request
                                     ├─ Generate Log: "GET /api 200 OK"
                                     └─ Write to STDOUT

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Log Collection & Storage                                   │
└─────────────────────────────────────────────────────────────────────┘

Pod STDOUT → Container Runtime → /var/log/containers/app-xxx.log
                                           │
                                           ▼
                                      FluentD (DaemonSet)
                                           │
                                           ├─ Add K8s metadata
                                           ├─ Buffer (5s window)
                                           ├─ Parse log format
                                           └─ Ship to Elasticsearch
                                                      │
                                                      ▼
                                                Elasticsearch
                                                (Index: k8s-logs-2024.01.01)

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Feature Extraction & Enrichment                            │
└─────────────────────────────────────────────────────────────────────┘

Elasticsearch ← Logstash (Query every 10s)
     │               │
     │               ├─ Fetch logs: [now-10s TO now]
     │               ├─ GeoIP lookup (IP → Country/City)
     │               ├─ User-Agent parsing
     │               ├─ Calculate window aggregations:
     │               │  - Request rate per IP
     │               │  - Unique paths per IP
     │               │  - Status code distribution
     │               └─ Create feature vector:
     │                  {
     │                    request_rate: 45.2,
     │                    geo_country: "US",
     │                    method: "GET",
     │                    status_2xx: 1,
     │                    ...
     │                  }
     │
     ├─────────────▶ Elasticsearch
     │              (Index: k8s-features-2024.01.01)
     │
     └─────────────▶ Kafka Topic: "traffic-features"
                    (Real-time stream)

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4: ML Inference                                               │
└─────────────────────────────────────────────────────────────────────┘

Kafka: "traffic-features"
     │
     ▼
ML Inference Service (Deployment)
     │
     ├─ Consume feature batch (100 samples)
     ├─ Normalize features
     ├─ Predict with all 5 models:
     │  - Autoencoder: score = 0.15
     │  - Isolation Forest: score = 0.12
     │  - One-Class SVM: score = 0.18
     │  - KNN: score = 0.20
     │  - GMM: score = 0.14
     ├─ Ensemble (weighted average): 0.158
     └─ Publish results
          │
          ├───────────▶ Redis (Key: "anomaly:score:current")
          │            Value: 0.158
          │            TTL: 60s
          │
          ├───────────▶ Prometheus Metric
          │            anomaly_score{pod="my-app"} = 0.158
          │
          └───────────▶ Elasticsearch
                       (Index: k8s-anomalies-2024.01.01)
                       {timestamp, score, features, prediction}

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 5: Autoscaling Decision                                       │
└─────────────────────────────────────────────────────────────────────┘

Threat-Aware Autoscaler (Custom Controller)
     │
     ├─ Every 30 seconds:
     │  │
     │  ├─ Read anomaly score from Redis: 0.158
     │  ├─ Read pod metrics from Metrics Server:
     │  │  - CPU: 75%
     │  │  - Memory: 60%
     │  │  - Current Replicas: 5
     │  │
     │  ├─ Decision Logic:
     │  │  IF anomaly_score < 0.3:  # Normal traffic
     │  │      IF cpu > 70%:
     │  │          desired_replicas = 5 * (75/70) = 6
     │  │      ELSE:
     │  │          desired_replicas = 5
     │  │  ELIF anomaly_score < 0.7:  # Medium anomaly
     │  │      IF cpu > 90%:
     │  │          desired_replicas = 6  # Cautious scaling
     │  │      ELSE:
     │  │          desired_replicas = 5  # Hold
     │  │  ELSE:  # High anomaly (likely attack)
     │  │      desired_replicas = max(2, 5-1) = 4  # Scale down
     │  │
     │  └─ Calculate: desired_replicas = 6
     │
     └─ Update Deployment:
        kubectl scale deployment/my-app --replicas=6

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 6: Monitoring & Feedback                                      │
└─────────────────────────────────────────────────────────────────────┘

Grafana Dashboard:
     │
     ├─ Panel 1: Request Rate (from Prometheus)
     ├─ Panel 2: Anomaly Score (from Prometheus)
     ├─ Panel 3: Pod Replicas (from K8s API)
     ├─ Panel 4: CPU/Memory Usage (from Metrics Server)
     └─ Panel 5: Detected Anomalies (from Elasticsearch)

Kibana Dashboard:
     │
     ├─ Raw Logs (k8s-logs-*)
     ├─ Extracted Features (k8s-features-*)
     ├─ Anomaly Events (k8s-anomalies-*)
     └─ Filtering by: Time, Pod, Anomaly Score, etc.

Alerts:
     │
     ├─ Prometheus AlertManager:
     │  - Alert if anomaly_score > 0.8 for 5 minutes
     │  - Alert if pods scaled down during high CPU (defensive scaling)
     │
     └─ Action: Send to Slack, PagerDuty, etc.
```

### 8.2 Component Dependencies

```
Dependency Graph:

Kubernetes Cluster (Base Layer)
     ├─ Container Runtime
     │    └─ Pods (generate logs)
     │
     ├─ Metrics Server
     │    └─ Provides CPU/Memory metrics
     │
     └─ FluentD DaemonSet
          └─ Requires: Pod logs via hostPath mount

Elasticsearch Cluster
     ├─ Depends on: Persistent storage (PVC)
     └─ Required by: Logstash, Kibana, Feature Extraction

Logstash
     ├─ Depends on: Elasticsearch
     └─ Required by: Feature extraction pipeline

Kibana
     ├─ Depends on: Elasticsearch
     └─ Used by: Human operators (debugging, visualization)

Kafka (Optional but recommended for production)
     ├─ Depends on: Zookeeper
     └─ Required by: Real-time feature streaming

ML Inference Service
     ├─ Depends on:
     │    - Trained ML models (loaded from storage)
     │    - Kafka (for input features)
     │    - Redis (for output cache)
     └─ Required by: Autoscaler

Redis
     ├─ Depends on: None (standalone)
     └─ Required by: Autoscaler (fast anomaly score lookup)

Prometheus
     ├─ Depends on: Metrics Server
     └─ Required by: Grafana, Alerting

Threat-Aware Autoscaler
     ├─ Depends on:
     │    - Kubernetes API
     │    - Redis (anomaly scores)
     │    - Metrics Server (CPU/Memory)
     └─ Controls: Deployment replicas

Grafana
     ├─ Depends on: Prometheus, Elasticsearch
     └─ Used by: Monitoring, visualization
```

### 8.3 Deployment Order

```
Step 1: Base Infrastructure
  ✓ Kubernetes cluster (k3s on Ubuntu)
  ✓ Persistent storage (local-path-provisioner with k3s)

Step 2: Monitoring Stack
  ✓ Prometheus (metrics collection)
  ✓ Grafana (visualization)

Step 3: Logging Stack
  ✓ Elasticsearch (3-node cluster for production, 1-node for dev)
  ✓ Kibana (UI for Elasticsearch)
  ✓ FluentD (DaemonSet on all nodes)

Step 4: Data Pipeline
  ✓ Logstash (feature extraction)
  ✓ Kafka + Zookeeper (optional, for real-time streaming)
  ✓ Redis (cache for anomaly scores)

Step 5: ML Infrastructure
  ✓ Train models (offline, on development machine)
  ✓ Deploy ML Inference Service
  ✓ Validate inference latency (<100ms)

Step 6: Custom Controller
  ✓ Deploy Threat-Aware Autoscaler
  ✓ Configure RBAC permissions
  ✓ Test with synthetic traffic

Step 7: Application
  ✓ Deploy sample application
  ✓ Configure HPA (standard, for comparison)
  ✓ Enable autoscaling

Step 8: Testing & Validation
  ✓ Generate normal traffic
  ✓ Generate attack traffic (DDoS simulation)
  ✓ Measure autoscaling behavior
  ✓ Compare with baseline (standard HPA)
```

---

## 9. Evaluation Methodology

### 9.1 Research Questions

**RQ1: Model Performance**
> Which anomaly detection model achieves the best performance on Kubernetes traffic data?

**Metrics:**
- Precision, Recall, F1-Score
- PR-AUC (primary metric for imbalanced data)
- ROC-AUC
- Confusion Matrix

**Hypothesis:**
Isolation Forest will outperform other models due to its robustness to high-dimensional data and mixed feature types.

**RQ2: System Latency**
> Can the system detect anomalies in real-time (<100ms) to support responsive autoscaling?

**Metrics:**
- End-to-end latency (log generation → anomaly detection)
- ML inference latency (per model)
- Feature extraction latency

**Target:**
- p95 latency < 100ms for ML inference
- p99 latency < 200ms

**RQ3: Autoscaling Effectiveness**
> Does threat-aware autoscaling reduce resource waste compared to standard HPA during attacks?

**Metrics:**
- Resource consumption (pod-hours) during DDoS attack
- Cost savings (estimated cloud cost)
- Legitimate request success rate during attack

**Hypothesis:**
Threat-aware autoscaling will reduce resource consumption by 50-70% during attacks while maintaining 95%+ success rate for legitimate requests.

**RQ4: Concept Drift Handling**
> How does model performance degrade over time without retraining?

**Metrics:**
- F1-Score over time (week 1, week 2, ..., week 8)
- KL divergence between training and recent data

**Hypothesis:**
Performance will degrade <5% over 4 weeks, but periodic retraining will restore performance.

### 9.2 Experimental Setup

#### Dataset Creation

**Phase 1: Normal Traffic Collection**
```
Duration: 2 weeks
Source: Real application traffic (if available) OR Synthetic
Volume: 10M requests
Distribution:
  - 70% GET requests (browsing)
  - 20% POST requests (API calls)
  - 10% Other (PUT, DELETE, OPTIONS)
Time: Business hours (9 AM - 6 PM) weighted higher
```

**Phase 2: Attack Traffic Generation**
```
Attack Types:
1. DDoS (Distributed Denial of Service)
   - High request rate from multiple IPs
   - Random paths, legitimate User-Agents
   - Duration: Bursts of 5-10 minutes

2. Slow HTTP (Slowloris)
   - Low request rate, but holds connections open
   - Incomplete headers
   - Duration: Continuous

3. Web Scraping / Scanning
   - Sequential path traversal (/admin, /config, /.env)
   - Single IP, automated User-Agent
   - Duration: Hours

4. API Abuse
   - Valid endpoints, but excessive rate
   - Single API key, legitimate-looking
   - Duration: Minutes to hours

Tools:
- Locust (for realistic load testing)
- hping3 (for DDoS simulation)
- Custom scripts (for targeted attacks)
```

**Dataset Split:**
```
Training: 60% (normal traffic only)
Validation: 20% (normal + 10% attacks)
Test: 20% (normal + 10% attacks)

Temporal Split:
- Training: Week 1-2 data
- Validation: Week 3 data
- Test: Week 4 data
  (Ensures models generalize to future, unseen traffic)
```

#### Baseline Comparisons

**Baseline 1: Standard HPA (No Anomaly Detection)**
```
Configuration:
- Target CPU: 70%
- Min Replicas: 2
- Max Replicas: 10

Behavior:
- Scales based only on CPU/Memory
- Blind to attack traffic
```

**Baseline 2: Rule-Based Anomaly Detection**
```
Rules:
- IF request_rate > 100/sec → Anomaly
- IF status_5xx > 10% → Anomaly
- IF geo_location in [CN, RU, KP] → Anomaly
- IF user_agent contains "bot" → Anomaly

Autoscaling:
- If anomaly_count > 50% → Don't scale up
```

**Baseline 3: Single Best Model (No Ensemble)**
```
Use only Isolation Forest (expected best performer)
Compare against ensemble of all 5 models
```

### 9.3 Metrics & KPIs

**ML Performance Metrics:**
```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    y_scores = model.anomaly_score(X_test)

    # Classification Report
    print(classification_report(y_test, y_pred,
                                 target_names=['Normal', 'Anomaly']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TP: {cm[1,1]}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC: {roc_auc:.3f}")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)
    print(f"PR-AUC: {pr_auc:.3f}")

    # Custom: Detection Rate at 1% FPR
    detection_rate_at_1pct_fpr = tpr[np.argmax(fpr >= 0.01)]
    print(f"Detection Rate @ 1% FPR: {detection_rate_at_1pct_fpr:.3f}")

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'detection_rate_1pct_fpr': detection_rate_at_1pct_fpr
    }
```

**System Performance Metrics:**
```python
def measure_system_performance():
    metrics = {}

    # Latency
    metrics['ml_inference_p50'] = measure_percentile_latency(50)
    metrics['ml_inference_p95'] = measure_percentile_latency(95)
    metrics['ml_inference_p99'] = measure_percentile_latency(99)

    # Throughput
    metrics['requests_per_second'] = measure_throughput()

    # Resource Usage
    metrics['cpu_usage_avg'] = measure_cpu_usage()
    metrics['memory_usage_avg'] = measure_memory_usage()
    metrics['network_bandwidth'] = measure_network()

    return metrics
```

**Autoscaling Metrics:**
```python
def evaluate_autoscaling(start_time, end_time):
    """
    Compare threat-aware vs standard HPA during attack.
    """
    # Pod scaling events
    threat_aware_pods = get_pod_count('threat-aware-hpa', start_time, end_time)
    standard_hpa_pods = get_pod_count('standard-hpa', start_time, end_time)

    # Resource consumption (pod-seconds)
    threat_aware_pod_seconds = np.sum(threat_aware_pods)
    standard_hpa_pod_seconds = np.sum(standard_hpa_pods)

    # Cost savings
    cost_per_pod_hour = 0.05  # Example: $0.05/pod/hour
    threat_aware_cost = (threat_aware_pod_seconds / 3600) * cost_per_pod_hour
    standard_hpa_cost = (standard_hpa_pod_seconds / 3600) * cost_per_pod_hour
    savings = standard_hpa_cost - threat_aware_cost
    savings_percent = (savings / standard_hpa_cost) * 100

    # Legitimate request success rate
    total_legit_requests = count_legitimate_requests(start_time, end_time)
    successful_legit_requests = count_successful_requests(start_time, end_time)
    success_rate = (successful_legit_requests / total_legit_requests) * 100

    return {
        'threat_aware_cost': threat_aware_cost,
        'standard_hpa_cost': standard_hpa_cost,
        'savings_usd': savings,
        'savings_percent': savings_percent,
        'legit_request_success_rate': success_rate
    }
```

### 9.4 Publication Strategy

**Target Venues:**
- **Tier 1**:
  - USENIX Security (Systems + Security)
  - IEEE S&P (Oakland)
  - ACM CCS (Computer and Communications Security)

- **Tier 2**:
  - ACSAC (Annual Computer Security Applications Conference)
  - RAID (International Symposium on Research in Attacks, Intrusions and Defenses)
  - IEEE Cloud (Cloud Computing)

**Paper Structure:**
```
1. Introduction
   - Problem: Blind autoscaling amplifies attacks
   - Solution: ML-based threat-aware autoscaling
   - Contributions:
     * Novel integration of anomaly detection + autoscaling
     * Comparative evaluation of 5 ML models
     * Open-source implementation for Kubernetes

2. Related Work
   - Autoscaling in cloud (predictive, reactive, RL-based)
   - Anomaly detection in networks (signature-based, ML-based)
   - Security in Kubernetes (admission control, RBAC, network policies)

3. System Design
   - Architecture
   - Components (ELK, ML models, Custom controller)
   - Integration with Kubernetes

4. ML Model Selection
   - 5 models (Autoencoder, IF, One-Class SVM, KNN, GMM)
   - Feature engineering (50 features from logs)
   - Training strategy (only normal traffic)

5. Implementation
   - Technology stack
   - Deployment on k3s
   - Code availability (GitHub)

6. Evaluation
   - RQ1: Model comparison (PR-AUC, latency)
   - RQ2: System performance (throughput, latency)
   - RQ3: Autoscaling effectiveness (cost savings, success rate)
   - RQ4: Concept drift (performance over time)

7. Discussion
   - Limitations (adversarial attacks, cold start)
   - Deployment considerations (resource overhead, false positives)
   - Future work (federated learning, explainability)

8. Conclusion
   - Summary of contributions
   - Impact on cloud security
```

**Reproducibility:**
```
Provide:
1. Code: GitHub repository with all components
2. Data: Synthetic traffic generator + sample dataset
3. Docker images: Pre-built containers for quick deployment
4. Documentation: Step-by-step setup guide
5. Experimental scripts: Automate all experiments
```

---

## 10. Implementation Roadmap

### 10.1 Phase 1: Infrastructure Setup (Week 1-2)

**Tasks:**
1. ✓ Set up Kubernetes on Windows (Docker Desktop)
2. ✓ Set up Kubernetes on Ubuntu (k3s)
3. ✓ Configure kubectl for multi-cluster management
4. ✓ Deploy Prometheus + Grafana
5. ✓ Deploy ELK Stack
6. ✓ Deploy FluentD DaemonSet
7. ✓ Verify log collection pipeline

**Deliverables:**
- Working Kubernetes clusters (Windows + Ubuntu)
- Logs flowing from pods → FluentD → Elasticsearch
- Kibana dashboard showing logs

**Testing:**
```bash
# Deploy test application
kubectl apply -f test-app.yaml

# Generate test traffic
for i in {1..1000}; do
  curl http://test-app.local/api/users
done

# Verify logs in Kibana
# Query: kubernetes.labels.app: "test-app"
```

### 10.2 Phase 2: Data Collection & Feature Engineering (Week 3-4)

**Tasks:**
1. ✓ Deploy sample web application
2. ✓ Generate synthetic normal traffic (Locust)
3. ✓ Implement feature extraction in Logstash
4. ✓ Validate features in Elasticsearch
5. ✓ Export features to CSV for ML training

**Deliverables:**
- 10M normal traffic samples
- Feature extraction pipeline
- CSV dataset with 50 features per request

**Code Example:**
```python
# scripts/generate_traffic.py
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def view_users(self):
        self.client.get("/api/users")

    @task(2)
    def view_posts(self):
        self.client.get("/api/posts")

    @task(1)
    def create_post(self):
        self.client.post("/api/posts", json={
            "title": "Test Post",
            "content": "Lorem ipsum..."
        })

# Run: locust -f generate_traffic.py --host=http://test-app.local
```

### 10.3 Phase 3: ML Model Training (Week 5-6)

**Tasks:**
1. ✓ Prepare training dataset (normal traffic only)
2. ✓ Implement 5 anomaly detection models
3. ✓ Train models on normal traffic
4. ✓ Hyperparameter tuning (GridSearch/RandomSearch)
5. ✓ Save trained models

**Deliverables:**
- 5 trained models (saved as .pkl or .h5 files)
- Training notebooks (Jupyter)
- Model comparison report

**Code Structure:**
```
ml/
├── data/
│   ├── train.csv         # Normal traffic
│   ├── val.csv           # Normal + 10% attacks
│   └── test.csv          # Normal + 10% attacks
├── models/
│   ├── autoencoder.py    # Autoencoder implementation
│   ├── isolation_forest.py
│   ├── one_class_svm.py
│   ├── knn.py
│   └── gmm.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── saved_models/
│   ├── autoencoder.h5
│   ├── isolation_forest.pkl
│   ├── one_class_svm.pkl
│   ├── knn.pkl
│   └── gmm.pkl
└── requirements.txt
```

### 10.4 Phase 4: ML Inference Service (Week 7-8)

**Tasks:**
1. ✓ Implement inference service (FastAPI)
2. ✓ Load all 5 models
3. ✓ Implement ensemble prediction
4. ✓ Containerize service (Docker)
5. ✓ Deploy to Kubernetes
6. ✓ Expose Prometheus metrics

**Deliverables:**
- ML inference service (REST API)
- Docker image
- Kubernetes deployment YAML

**Code Example:**
```python
# inference_service/app.py
from fastapi import FastAPI
import joblib
import tensorflow as tf
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()

# Load models
models = {
    'autoencoder': tf.keras.models.load_model('models/autoencoder.h5'),
    'isolation_forest': joblib.load('models/isolation_forest.pkl'),
    'one_class_svm': joblib.load('models/one_class_svm.pkl'),
    'knn': joblib.load('models/knn.pkl'),
    'gmm': joblib.load('models/gmm.pkl')
}

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
anomaly_score_metric = Gauge('current_anomaly_score', 'Current anomaly score')

@app.post("/predict")
@prediction_latency.time()
def predict(features: dict):
    # Convert to numpy array
    X = np.array([list(features.values())])

    # Predict with all models
    scores = []
    for name, model in models.items():
        if name == 'autoencoder':
            reconstruction = model.predict(X)
            mse = np.mean(np.power(X - reconstruction, 2))
            score = mse / threshold_autoencoder
        else:
            score = model.decision_function(X)[0]

        scores.append(score)

    # Ensemble (weighted average)
    weights = [0.25, 0.25, 0.20, 0.15, 0.15]  # Based on validation
    ensemble_score = np.average(scores, weights=weights)

    # Update metrics
    prediction_counter.inc()
    anomaly_score_metric.set(ensemble_score)

    return {
        'anomaly_score': float(ensemble_score),
        'is_anomaly': bool(ensemble_score > 0.5),
        'model_scores': {name: float(score) for name, score in zip(models.keys(), scores)}
    }

@app.get("/metrics")
def metrics():
    return generate_latest()
```

### 10.5 Phase 5: Custom Autoscaler (Week 9-10)

**Tasks:**
1. ✓ Implement custom controller (Python)
2. ✓ Integrate with ML inference service
3. ✓ Implement autoscaling logic
4. ✓ Configure RBAC permissions
5. ✓ Deploy and test

**Deliverables:**
- Threat-aware autoscaler (custom controller)
- Kubernetes deployment + RBAC
- Integration tests

**Testing Scenarios:**
```python
# tests/test_autoscaler.py

def test_normal_traffic():
    """During normal traffic, behaves like standard HPA."""
    # Simulate CPU at 80%, anomaly_score = 0.1
    assert autoscaler.calculate_replicas(current=5, cpu=80, anomaly=0.1) == 6

def test_high_anomaly():
    """During attack, doesn't scale up."""
    # Simulate CPU at 90%, anomaly_score = 0.9
    assert autoscaler.calculate_replicas(current=5, cpu=90, anomaly=0.9) <= 5

def test_medium_anomaly():
    """During medium anomaly, cautious scaling."""
    # Simulate CPU at 95%, anomaly_score = 0.5
    replicas = autoscaler.calculate_replicas(current=5, cpu=95, anomaly=0.5)
    assert 5 <= replicas <= 6  # At most 1 replica increase
```

### 10.6 Phase 6: Evaluation & Experiments (Week 11-12)

**Tasks:**
1. ✓ Generate attack traffic (DDoS, scanning, etc.)
2. ✓ Run experiments (threat-aware vs baselines)
3. ✓ Collect metrics (cost, success rate, latency)
4. ✓ Analyze results
5. ✓ Create visualizations

**Experiments:**
```
Experiment 1: Model Comparison
- Train all 5 models on same data
- Evaluate on test set
- Metrics: PR-AUC, ROC-AUC, F1, Latency
- Result: Determine best model

Experiment 2: System Latency
- Measure end-to-end latency (log → anomaly detection)
- Vary load (100, 1000, 10000 req/sec)
- Metrics: p50, p95, p99 latency
- Result: Verify <100ms p95 target

Experiment 3: DDoS Attack
- Baseline: Standard HPA
- Treatment: Threat-aware autoscaler
- Attack: 50,000 req/sec for 10 minutes
- Metrics: Pod count, cost, legit request success rate
- Result: Quantify cost savings

Experiment 4: Concept Drift
- Train model on Week 1 data
- Evaluate on Week 2, 3, 4, 8 data
- Metrics: F1-Score over time
- Result: Measure degradation, validate retraining

Experiment 5: Ablation Study
- Test individual models vs ensemble
- Test with/without feature engineering
- Test with different threshold values
- Result: Understand component contributions
```

**Data Collection:**
```python
# scripts/run_experiments.py

import pandas as pd
import time

def run_experiment(name, duration_minutes, attack_type=None):
    """Run a single experiment."""
    print(f"Starting experiment: {name}")

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    # Start traffic generator
    if attack_type:
        start_attack_traffic(attack_type)
    else:
        start_normal_traffic()

    # Collect metrics
    metrics = []
    while time.time() < end_time:
        m = collect_metrics()
        metrics.append(m)
        time.sleep(10)

    # Stop traffic
    stop_traffic()

    # Save results
    df = pd.DataFrame(metrics)
    df.to_csv(f"results/{name}.csv")

    print(f"Experiment {name} completed")
    return df

# Run all experiments
run_experiment("baseline_normal", duration_minutes=30)
run_experiment("baseline_ddos", duration_minutes=30, attack_type="ddos")
run_experiment("threat_aware_normal", duration_minutes=30)
run_experiment("threat_aware_ddos", duration_minutes=30, attack_type="ddos")
```

### 10.7 Phase 7: Paper Writing & Submission (Week 13-16)

**Tasks:**
1. ✓ Analyze experimental results
2. ✓ Create figures and tables
3. ✓ Write paper (8-12 pages)
4. ✓ Internal review
5. ✓ Submit to conference

**Timeline:**
- Week 13: Results analysis + figure creation
- Week 14: Write Sections 1-4 (Intro, Related Work, Design, Implementation)
- Week 15: Write Sections 5-7 (Evaluation, Discussion, Conclusion)
- Week 16: Polish, proofread, submit

---

## Summary: Why This Design?

### Technical Decisions Summary

| Decision | Alternative | Why Chosen |
|----------|-------------|------------|
| **Kubernetes** | Docker Swarm, Nomad | Industry standard, rich ecosystem |
| **k3s** | kubeadm, managed K8s | Lightweight, perfect for single-server |
| **ELK Stack** | Loki, Splunk, CloudWatch | Powerful search, industry standard |
| **FluentD** | Filebeat, Fluent Bit | K8s-native, plugin ecosystem |
| **5 ML Models** | Single model | Comparative analysis for research |
| **Python** | Go, Java | ML ecosystem, rapid development |
| **Custom Controller** | Modify HPA | More control, research flexibility |

### Research Contributions

1. **Novel Integration**: First work to combine ML anomaly detection with K8s autoscaling
2. **Comparative Study**: Empirical comparison of 5 models on real K8s traffic
3. **Practical System**: Full implementation, not just simulation
4. **Open Source**: Reproducible research, community benefit

### Next Steps

1. **Immediate**: Set up infrastructure (Windows + Ubuntu K8s clusters)
2. **Short-term**: Implement logging pipeline and feature extraction
3. **Medium-term**: Train ML models and build inference service
4. **Long-term**: Evaluate, write paper, publish

---

**This document provides the theoretical foundation and practical guidance for your research. Refer back to it as you implement each component. Good luck with your research!**
