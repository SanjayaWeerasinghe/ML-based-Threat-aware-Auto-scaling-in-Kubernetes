"""
Feature Extraction for HTTP Traffic Anomaly Detection
Extracts meaningful features from CSIC 2010 dataset
"""

import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs
import re
from scipy.stats import entropy

def extract_url_features(url):
    """Extract features from URL"""
    if pd.isna(url) or url == '':
        return {
            'url_length': 0,
            'num_parameters': 0,
            'num_special_chars': 0,
            'url_entropy': 0,
            'has_suspicious_keywords': 0
        }

    # URL length
    url_length = len(url)

    # Number of parameters
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        num_parameters = len(params)
    except:
        num_parameters = 0

    # Count special characters
    special_chars = r'[!@#$%^&*()+=\[\]{};:\'",<>?/\\|`~]'
    num_special_chars = len(re.findall(special_chars, url))

    # Calculate URL entropy (randomness)
    if len(url) > 0:
        char_freq = {}
        for char in url:
            char_freq[char] = char_freq.get(char, 0) + 1
        probabilities = [freq/len(url) for freq in char_freq.values()]
        url_entropy = entropy(probabilities, base=2)
    else:
        url_entropy = 0

    # Check for suspicious keywords (SQL injection, XSS patterns)
    suspicious_keywords = [
        'select', 'union', 'insert', 'delete', 'drop', 'update',
        'script', 'alert', 'onerror', 'onload', '../', '..',
        'exec', 'execute', 'cmd', 'system'
    ]
    url_lower = url.lower()
    has_suspicious = 1 if any(keyword in url_lower for keyword in suspicious_keywords) else 0

    return {
        'url_length': url_length,
        'num_parameters': num_parameters,
        'num_special_chars': num_special_chars,
        'url_entropy': url_entropy,
        'has_suspicious_keywords': has_suspicious
    }

def extract_features(df):
    """Extract all features from dataframe"""
    print("\n" + "=" * 60)
    print("Feature Extraction Started")
    print("=" * 60)

    features_list = []

    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            print(f"Processing record {idx}/{len(df)}...")

        # Extract URL features
        url_features = extract_url_features(row.get('URL', ''))

        # Basic features
        features = {
            # URL features
            'url_length': url_features['url_length'],
            'num_parameters': url_features['num_parameters'],
            'num_special_chars': url_features['num_special_chars'],
            'url_entropy': url_features['url_entropy'],
            'has_suspicious_keywords': url_features['has_suspicious_keywords'],

            # Method (encode as binary: GET=0, POST=1, others=2)
            'method_get': 1 if row.get('Method', '') == 'GET' else 0,
            'method_post': 1 if row.get('Method', '') == 'POST' else 0,

            # User-Agent length
            'user_agent_length': len(str(row.get('User-Agent', ''))),

            # Content length
            'content_length': len(str(row.get('content', ''))) if pd.notna(row.get('content', '')) else 0,

            # Has cookie
            'has_cookie': 0 if pd.isna(row.get('cookie', '')) or row.get('cookie', '') == '' else 1,

            # Has content-type
            'has_content_type': 0 if pd.isna(row.get('content-type', '')) or row.get('content-type', '') == '' else 1,
        }

        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    print(f"\n✓ Extracted {len(features_df.columns)} features from {len(df)} records")
    print(f"\nFeature names:")
    for col in features_df.columns:
        print(f"  - {col}")

    return features_df

if __name__ == "__main__":
    # Load training data
    print("\n1. Loading training data (normal traffic only)...")
    train_df = pd.read_csv("../../datasets/csic-2010/training_normal.csv")
    print(f"   Training samples: {len(train_df)}")

    # Load test data
    print("\n2. Loading test data (mixed traffic)...")
    test_df = pd.read_csv("../../datasets/csic-2010/testing_mixed.csv")
    print(f"   Test samples: {len(test_df)}")

    # Extract features from training data
    print("\n3. Extracting features from training data...")
    train_features = extract_features(train_df)

    # Extract features from test data
    print("\n4. Extracting features from test data...")
    test_features = extract_features(test_df)

    # Save test labels separately
    test_labels = test_df.iloc[:, 0]  # First column contains Normal/Anomalous

    # Save processed features
    print("\n5. Saving processed features...")
    train_features.to_csv("../../datasets/csic-2010/train_features.csv", index=False)
    test_features.to_csv("../../datasets/csic-2010/test_features.csv", index=False)
    test_labels.to_csv("../../datasets/csic-2010/test_labels.csv", index=False)

    print("\n" + "=" * 60)
    print("✅ Feature extraction complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  ✓ train_features.csv ({len(train_features)} samples, {len(train_features.columns)} features)")
    print(f"  ✓ test_features.csv ({len(test_features)} samples, {len(test_features.columns)} features)")
    print(f"  ✓ test_labels.csv ({len(test_labels)} samples)")

    print(f"\nFeature statistics (training data):")
    print(train_features.describe())
