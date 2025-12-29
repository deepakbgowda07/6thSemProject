# 🔐 Ensemble-Based Insider Threat Detection System

## Overview

This directory contains a **multi-variant ensemble detector** that combines diverse anomaly detection models for robust insider threat identification.

### Key Innovation: Model Diversity & Consensus

Instead of relying on a single model, this system trains **multiple variants** of each algorithm with different hyperparameters:

- **2 Isolation Forest variants** (different contamination rates & estimators)
- **2 One-Class SVM variants** (different nu values & gamma kernels)  
- **2 Autoencoder variants** (different network architectures)

**Total: 6 independent models** voting on each user's behavior.

---

## Files

### Training
- **`train_variants.py`** - Trains all 6 model variants and saves them to `models/saved/`
  ```bash
  python models/train_variants.py
  ```

### Detection
- **`strict_detector.py`** - Ensemble voting system that evaluates users
  - `StrictDetector` class: Load models, configure thresholds, evaluate feature vectors
  - `detect_threats_in_csv()`: Batch process entire CSV files
  - Example usage at bottom of file

### Saved Models
- **`saved/scaler.pkl`** - StandardScaler (fitted on training data)
- **`saved/iso_1.pkl`** - Isolation Forest (100 trees, 5% contamination)
- **`saved/iso_2.pkl`** - Isolation Forest (200 trees, 10% contamination)
- **`saved/ocsvm_1.pkl`** - One-Class SVM (nu=0.05, gamma='scale')
- **`saved/ocsvm_2.pkl`** - One-Class SVM (nu=0.10, gamma='auto')
- **`saved/ae_1.pkl`** - Autoencoder (8→4→8 architecture)
- **`saved/ae_2.pkl`** - Autoencoder (16→8→16 architecture)

---

## How It Works

### 1️⃣ Feature Scaling
Input features are scaled using the fitted `StandardScaler` to match training distribution.

### 2️⃣ Model Evaluation
Each of the 6 models scores the user independently:
- **Isolation Forest**: Negative anomaly score (higher = more anomalous)
- **One-Class SVM**: Negative decision function (higher = farther from normal boundary)
- **Autoencoder**: Reconstruction MSE (higher = worse reconstruction = anomaly)

### 3️⃣ Consensus Voting
**Triggers occur when:**
- Isolation Forest variant exceeds **0.60**
- One-Class SVM variant exceeds **0.60**
- Autoencoder variant exceeds **0.02**

### 4️⃣ Multi-Family Agreement (Strict Mode)
**Verdict = "MALICIOUS"** when anomalies are detected in **2+ model families**:
- If 2+ model families trigger → **High confidence threat**
- If 1 model family triggers → **Normal behavior**

This consensus requirement dramatically **reduces false positives** while maintaining detection capability.

---

## Usage Examples

### Basic Detection
```python
from models.strict_detector import StrictDetector
import numpy as np

detector = StrictDetector()

# Single user evaluation (11 features)
features = [8.0, 16.0, 5.0, 0.5, 3.0, 100, 0.7, 0.05, 0.5, 10, 0.2]
result = detector.evaluate(features)

print(f"Verdict: {result['verdict']}")
print(f"Triggered Models: {result['triggered_models']}")
print(f"Explanation: {result['explanation']}")
```

### Batch Processing
```python
from models.strict_detector import detect_threats_in_csv

results = detect_threats_in_csv('data/merged_features.csv')
malicious = results[results['verdict'] == 'MALICIOUS']
print(f"Found {len(malicious)} suspicious users")
```

---

## Performance Characteristics

### Advantages
✅ **Robustness**: Multiple models reduce overfitting from any single algorithm  
✅ **Diversity**: Different algorithms catch different types of anomalies  
✅ **Interpretability**: Can see which models triggered and why  
✅ **Security**: Consensus voting reduces false alarms  

### Trade-offs
- More computational cost (6 models vs 1)
- Requires more training data for meaningful diversity
- Slightly lower recall (intentionally conservative)

---

## Model Configurations

### Isolation Forest
```
iso_1: n_estimators=100, contamination=0.05
iso_2: n_estimators=200, contamination=0.10
```

### One-Class SVM
```
ocsvm_1: nu=0.05, kernel='rbf', gamma='scale'
ocsvm_2: nu=0.10, kernel='rbf', gamma='auto'
```

### Autoencoder
```
ae_1: hidden_layers=(8, 4, 8), max_iter=1500
ae_2: hidden_layers=(16, 8, 16), max_iter=1500
```

---

## Thresholds

These are **strict (security-first) thresholds**, tuned to prioritize precision over recall:

| Model Family | Threshold | Rationale |
|---|---|---|
| Isolation Forest | 0.60 | High anomaly score indicates deviation |
| One-Class SVM | 0.60 | Distance from normal boundary |
| Autoencoder | 0.02 | Reconstruction MSE (MSE scale is smaller) |

Adjust thresholds in `StrictDetector.__init__()` if you want to change detection sensitivity.

---

## Integration with Dashboard

The dashboard (`dashboard/combined_dashboard.py`) can optionally integrate StrictDetector for real-time threat scoring:

```python
detector = StrictDetector()
for user_features in user_data:
    result = detector.evaluate(user_features)
    st.write(f"{user}: {result['verdict']}")
```

---

## Training & Updating

To retrain all models after collecting new data:

```bash
python models/train_variants.py
```

This will:
1. Load fresh data from `data/merged_features.csv`
2. Retrain all 6 model variants
3. Update the scaler
4. Save everything to `models/saved/`

---

## Security Considerations

🔐 **Threat Model**: Insider threats with behavioral anomalies  
🔐 **False Positive Cost**: High (analyst fatigue, alert fatigue)  
🔐 **False Negative Cost**: Very High (missed threats)  

**Design Choice**: Use **conservative thresholds** + **consensus voting** to minimize false positives while catching real threats.

---

## Next Steps

1. ✅ Train models: `python models/train_variants.py`
2. ✅ Test detector: `python models/strict_detector.py`
3. 🔄 Integrate with dashboard for live detection
4. 📊 Monitor false positive / negative rates in production
5. 🔧 Adjust thresholds based on operational feedback
