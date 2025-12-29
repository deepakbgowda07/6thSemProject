import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def calculate_strict_score(df, scores_df):
    """
    Calculates a continuous, consensus-based strict risk score.
    The score is a weighted average of three components:
    1. Model Agreement: How many base models flag the user.
    2. Graph Risk: The user's highest network centrality score.
    3. Behavioral Risk: The user's most extreme behavioral deviation.
    """
    
    # --- Create a combined dataframe for easier processing ---
    combined_df = pd.merge(df, scores_df, on='user', suffixes=('_features', '_scores'))

    # --- Normalize all features and scores that will be used ---
    scaler = MinMaxScaler()
    features_to_normalize = [
        'isolation_forest', 'oneclass_svm', 'autoencoder',
        'degree_centrality', 'betweenness_centrality',
        'out_of_session_access', 'usb_per_day', 'files_per_day'
    ]
    
    normalized_features = pd.DataFrame(
        scaler.fit_transform(combined_df[features_to_normalize]),
        columns=[f'norm_{col}' for col in features_to_normalize],
        index=combined_df.index
    )

    # --- 1. Calculate Model Agreement Score ---
    # Score is 1 if 2+ models flag, 0.5 if 1 model flags, 0 otherwise
    model_flags = (normalized_features['norm_isolation_forest'] > 0.95).astype(int) + \
                  (normalized_features['norm_oneclass_svm'] > 0.95).astype(int) + \
                  (normalized_features['norm_autoencoder'] > 0.95).astype(int)
    
    score1 = np.where(model_flags >= 2, 1.0, np.where(model_flags == 1, 0.5, 0.0))

    # --- 2. Calculate Graph Risk Score ---
    # Score is the max of the normalized centrality metrics
    score2 = normalized_features[['norm_degree_centrality', 'norm_betweenness_centrality']].max(axis=1)

    # --- 3. Calculate Behavioral Risk Score ---
    # Score is the max of the normalized behavioral deviation metrics
    score3 = normalized_features[['norm_out_of_session_access', 'norm_usb_per_day', 'norm_files_per_day']].max(axis=1)

    # --- 4. Combine into Final Strict Risk Score ---
    # Using equal weights for simplicity, as requested by the user's high-level design.
    # The final score is a blend of model consensus, graph risk, and behavioral anomalies.
    strict_risk_score = (score1 + score2 + score3) / 3.0
    
    # Add the new score to the original scores_df
    scores_df['strict_risk_score'] = strict_risk_score

    # The user also wanted a binary 'strict_prediction' column in the CSV.
    # We'll use a high, fixed percentile here for the data export, but the dashboard will use a slider.
    # This meets the requirement "prediction must be based on a percentile"
    prediction_threshold = np.percentile(scores_df['strict_risk_score'], 98) # Default 98th percentile
    scores_df['strict_prediction'] = (scores_df['strict_risk_score'] >= prediction_threshold).astype(int)

    return scores_df
