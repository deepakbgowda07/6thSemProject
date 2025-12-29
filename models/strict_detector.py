import joblib
import numpy as np
import os

MODEL_DIR = "models/saved"

class StrictDetector:
    """
    Ensemble detector that combines multiple model variants for robust insider threat detection.
    Uses a consensus-based approach: stricter thresholds + multiple model agreement = high confidence flags.
    """
    
    def __init__(self):
        """Load all model variants and configure thresholds."""
        try:
            self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
            
            self.iso_models = [
                joblib.load(os.path.join(MODEL_DIR, "iso_1.pkl")),
                joblib.load(os.path.join(MODEL_DIR, "iso_2.pkl"))
            ]
            
            self.svm_models = [
                joblib.load(os.path.join(MODEL_DIR, "ocsvm_1.pkl")),
                joblib.load(os.path.join(MODEL_DIR, "ocsvm_2.pkl"))
            ]
            
            self.ae_models = [
                joblib.load(os.path.join(MODEL_DIR, "ae_1.pkl")),
                joblib.load(os.path.join(MODEL_DIR, "ae_2.pkl"))
            ]
            
            # STRICT thresholds (security-first approach)
            self.thresholds = {
                "iso": 0.60,      # High isolation forest score = anomaly
                "svm": 0.60,      # High SVM decision value = anomaly
                "ae": 0.02        # High reconstruction error = anomaly
            }
            
            self.model_loaded = True
            print("✓ StrictDetector initialized successfully")
            
        except FileNotFoundError as e:
            print(f"⚠ Warning: Model files not found at {MODEL_DIR}")
            print(f"  Error: {e}")
            print(f"  Run 'python models/train_variants.py' first to generate model variants")
            self.model_loaded = False
    
    def evaluate(self, feature_vector):
        """
        Evaluate a user's feature vector against all model variants.
        
        Args:
            feature_vector: numpy array or list of features
            
        Returns:
            dict with verdict, triggered_models, scores, and explanation
        """
        
        if not self.model_loaded:
            return {
                "verdict": "UNKNOWN",
                "triggered_models": [],
                "scores": {},
                "explanation": "Models not loaded. Run training first.",
                "confidence": 0.0
            }
        
        x = self.scaler.transform([feature_vector])
        
        triggers = []
        scores = {}
        triggered_count = 0
        total_models = 6  # 2 iso + 2 svm + 2 ae
        
        # --- Isolation Forest Evaluation ---
        iso_triggers = 0
        for i, model in enumerate(self.iso_models):
            score = -model.score_samples(x)[0]
            scores[f"iso_{i+1}"] = score
            if score > self.thresholds["iso"]:
                triggers.append(f"IsolationForest_{i+1}")
                iso_triggers += 1
                triggered_count += 1
        
        # --- One-Class SVM Evaluation ---
        svm_triggers = 0
        for i, model in enumerate(self.svm_models):
            score = -model.decision_function(x)[0]
            scores[f"svm_{i+1}"] = score
            if score > self.thresholds["svm"]:
                triggers.append(f"OneClassSVM_{i+1}")
                svm_triggers += 1
                triggered_count += 1
        
        # --- Autoencoder Evaluation ---
        ae_triggers = 0
        for i, model in enumerate(self.ae_models):
            recon = model.predict(x)
            mse = np.mean((x - recon) ** 2)
            scores[f"ae_{i+1}"] = mse
            if mse > self.thresholds["ae"]:
                triggers.append(f"Autoencoder_{i+1}")
                ae_triggers += 1
                triggered_count += 1
        
        # --- Consensus-based Verdict ---
        # Require agreement from multiple model families for high confidence
        family_triggers = sum([
            iso_triggers > 0,
            svm_triggers > 0,
            ae_triggers > 0
        ])
        
        if family_triggers >= 2:
            verdict = "MALICIOUS"
            confidence = min(triggered_count / total_models, 1.0)
        else:
            verdict = "SAFE"
            confidence = 1.0 - (triggered_count / total_models)
        
        # --- Generate Explanation ---
        if verdict == "MALICIOUS":
            explanation = (
                f"🚨 Strict security alert: {triggered_count} out of {total_models} models triggered. "
                f"Behavior anomalies detected across {family_triggers} model families. "
                f"Triggered models: {', '.join(triggers)}"
            )
        else:
            explanation = (
                "✓ Behavior within normal limits. "
                f"Only {triggered_count} model(s) slightly elevated; no consensus anomaly detected."
            )
        
        return {
            "verdict": verdict,
            "triggered_models": triggers,
            "triggered_count": triggered_count,
            "total_models": total_models,
            "scores": scores,
            "explanation": explanation,
            "confidence": confidence,
            "model_families": {
                "isolation_forest": iso_triggers,
                "oneclass_svm": svm_triggers,
                "autoencoder": ae_triggers
            }
        }

    def batch_evaluate(self, feature_matrix):
        """
        Evaluate multiple feature vectors.
        
        Args:
            feature_matrix: 2D numpy array, shape (n_samples, n_features)
            
        Returns:
            list of evaluation results
        """
        results = []
        for i, feature_vector in enumerate(feature_matrix):
            result = self.evaluate(feature_vector)
            result['user_index'] = i
            results.append(result)
        return results


# ============================================
# Utility: Run detection on a CSV file
# ============================================
def detect_threats_in_csv(csv_path, feature_columns=None):
    """
    Load a CSV and run StrictDetector on all rows.
    
    Args:
        csv_path: Path to CSV file
        feature_columns: List of column names to use as features. 
                        If None, uses all numeric columns except 'user'.
    
    Returns:
        DataFrame with original data + detection results
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in ['user', 'is_red_team']]
    
    detector = StrictDetector()
    results = []
    
    for idx, row in df.iterrows():
        features = row[feature_columns].values.astype(float)
        result = detector.evaluate(features)
        results.append(result)
    
    results_df = pd.DataFrame(results)
    output_df = pd.concat([df[['user']], results_df], axis=1)
    
    return output_df


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*60)
    print("StrictDetector - Ensemble-based Insider Threat Detection")
    print("="*60 + "\n")
    
    detector = StrictDetector()
    
    if detector.model_loaded:
        print("Testing with sample feature vector...\n")
        
        # Create a random sample feature vector (11 features, matching training data)
        # Features: mean_login_hour, mean_logout_hour, files_per_day, usb_per_day, emails_per_day,
        #           out_of_session_access, degree_centrality, betweenness_centrality, keyword_flag, subject_len, sentiment
        sample = np.random.randn(11)
        result = detector.evaluate(sample)
        
        print(f"Verdict: {result['verdict']}")
        print(f"Triggered Models: {result['triggered_models']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nDetailed Scores:")
        for model, score in result['scores'].items():
            print(f"  {model}: {score:.4f}")
        print(f"\nExplanation: {result['explanation']}")
