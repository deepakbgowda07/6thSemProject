import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

DATA_DIR = "data"
MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, "merged_features.csv"))
X = df.drop(["user"], axis=1, errors='ignore')
# Drop is_red_team if it exists
if "is_red_team" in X.columns:
    X = X.drop("is_red_team", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print(f"Data shape: {X_scaled.shape}")

# -----------------------------
# Isolation Forest Variants
# -----------------------------
print("\nTraining Isolation Forest variants...")
iso_configs = [
    {"n_estimators": 100, "contamination": 0.05},
    {"n_estimators": 200, "contamination": 0.10}
]

for i, cfg in enumerate(iso_configs, start=1):
    print(f"  Training Isolation Forest variant {i}: {cfg}")
    iso = IsolationForest(
        n_estimators=cfg["n_estimators"],
        contamination=cfg["contamination"],
        random_state=42
    )
    iso.fit(X_scaled)
    joblib.dump(iso, f"{MODEL_DIR}/iso_{i}.pkl")
    print(f"    ✓ Saved: iso_{i}.pkl")

# -----------------------------
# One-Class SVM Variants
# -----------------------------
print("\nTraining One-Class SVM variants...")
svm_configs = [
    {"nu": 0.05, "gamma": "scale"},
    {"nu": 0.10, "gamma": "auto"}
]

for i, cfg in enumerate(svm_configs, start=1):
    print(f"  Training One-Class SVM variant {i}: {cfg}")
    svm = OneClassSVM(
        nu=cfg["nu"],
        kernel="rbf",
        gamma=cfg["gamma"]
    )
    svm.fit(X_scaled)
    joblib.dump(svm, f"{MODEL_DIR}/ocsvm_{i}.pkl")
    print(f"    ✓ Saved: ocsvm_{i}.pkl")

# -----------------------------
# Autoencoder Variants
# -----------------------------
print("\nTraining Autoencoder variants...")
ae_configs = [
    (8, 4, 8),
    (16, 8, 16)
]

for i, layers in enumerate(ae_configs, start=1):
    print(f"  Training Autoencoder variant {i}: hidden_layers={layers}")
    ae = MLPRegressor(
        hidden_layer_sizes=layers,
        max_iter=1500,
        random_state=42,
        early_stopping=False
    )
    ae.fit(X_scaled, X_scaled)
    joblib.dump(ae, f"{MODEL_DIR}/ae_{i}.pkl")
    print(f"    ✓ Saved: ae_{i}.pkl")

print("\n✅ Multiple model variants trained and saved successfully!")
print(f"Saved models in: {MODEL_DIR}")
