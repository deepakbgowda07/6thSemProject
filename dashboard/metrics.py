import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import plotly.figure_factory as ff
import numpy as np
import streamlit as st

DATA_DIR = 'new/data'

MODEL_CONFIG = {
    "isolation_forest": {
        "name": "Isolation Forest",
        "overview": "Isolation Forest detects anomalies by randomly partitioning the feature space. Anomalies are isolated faster because they differ significantly from the majority of data points.",
        "criterion": "A shorter average path length in the random trees indicates a higher likelihood of being an anomaly.",
        "interpretation": "Isolation Forest often provides a good balance of precision and recall. It is computationally efficient and works well on high-dimensional data."
    },
    "oneclass_svm": {
        "name": "One-Class SVM",
        "overview": "One-Class SVM learns a boundary that encompasses the majority of the data. Points lying outside this learned boundary are considered anomalies.",
        "criterion": "A data point is flagged as an anomaly if it falls on the 'outer' side of the learned hyperplane in the feature space.",
        "interpretation": "One-Class SVM can be sensitive to parameter tuning (like the 'nu' parameter). It may have lower recall if the boundary is too tight but can achieve high precision."
    },
    "autoencoder": {
        "name": "Autoencoder",
        "overview": "An Autoencoder is a neural network trained to reconstruct its input. Anomalies are data points that the model fails to reconstruct accurately, resulting in a high reconstruction error.",
        "criterion": "A high Mean Squared Error (MSE) between the original input and its reconstruction is the primary indicator of an anomaly.",
        "interpretation": "Autoencoders are powerful for capturing complex, non-linear patterns. A high reconstruction error suggests the user's behavior is unlike anything seen in the training data."
    },
     "unified_risk_score": {
        "name": "Unified Risk Score",
        "overview": "This score is a composite metric created by normalizing and averaging the scores from all individual models.",
        "criterion": "A high unified score indicates that a user is considered anomalous by multiple models, increasing confidence in the detection.",
        "interpretation": "The unified score provides a more robust and stable measure of risk by leveraging the consensus of different anomaly detection techniques."
    },
    "strict_risk_score": {
        "name": "Strict Detector (High Precision)",
        "overview": "The Strict Insider Detection Model is a consensus-based detector that flags users only when multiple independent anomaly signals and structural risk indicators agree. This approach reduces false positives and is intended for high-confidence insider threat alerts.",
        "criterion": "A user is flagged if their composite 'strict risk score' (based on model agreement, graph risk, and behavioral flags) is in the top percentile (e.g., 98th or 99th) of the population.",
        "interpretation": "This model is conservative and prioritizes alert confidence (high precision) over coverage (recall). Lower recall is an expected and acceptable trade-off for reducing false positives."
    }
}


@st.cache_data
def load_evaluation_data():
    """
    Loads anomaly scores and red team labels, merging them for evaluation.
    Now uses a LEFT merge to include all users, not just red team members.
    """
    try:
        scores_df = pd.read_csv(os.path.join(DATA_DIR, 'anomaly_scores.csv'))
        red_team_df = pd.read_csv(os.path.join(DATA_DIR, 'red_team_users.csv'))
        
        # Perform a left merge to keep all users from scores_df
        eval_df = pd.merge(scores_df, red_team_df, on='user', how='left')
        
        # Fill NaN in 'is_red_team' with 0 (benign)
        eval_df['is_red_team'] = eval_df['is_red_team'].fillna(0).astype(int)
        
        # Add unified score for evaluation
        scaler = MinMaxScaler()
        score_cols = ['isolation_forest', 'oneclass_svm', 'autoencoder']
        for col in score_cols:
            eval_df[f'norm_{col}'] = scaler.fit_transform(eval_df[[col]])
        eval_df['unified_risk_score'] = eval_df[[f'norm_{col}' for col in score_cols]].mean(axis=1)

        return eval_df
    except FileNotFoundError as e:
        st.error(f"Error loading evaluation data: {e}. Please ensure 'anomaly_scores.csv' and 'red_team_users.csv' are in the '{DATA_DIR}' directory.")
        return None

def get_predictions(df, model_name, percentile_threshold=95):
    """
    Generates binary predictions based on a percentile threshold on a specific model's score.
    """
    score_col = model_name
    
    threshold = np.percentile(df[score_col], percentile_threshold)
    
    y_true = df['is_red_team']
    y_pred = (df[score_col] >= threshold).astype(int)
    
    return y_true, y_pred, df[score_col], threshold

def compute_metrics(y_true, y_pred):
    """
    Computes a dictionary of evaluation metrics.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def compute_roc_auc(y_true, scores):
    """
    Computes ROC-AUC score if applicable.
    """
    try:
        return roc_auc_score(y_true, scores)
    except ValueError:
        return None

def plot_confusion_matrix(y_true, y_pred):
    """
    Generates a Plotly heatmap for the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Benign', 'Insider Threat']
    
    # Pad with 0 if cm is not 2x2
    if cm.shape != (2,2):
        cm_padded = np.zeros((2,2))
        cm_padded[:cm.shape[0], :cm.shape[1]] = cm
        cm = cm_padded


    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    )
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis=dict(title='Predicted'),
        yaxis=dict(title='Actual'),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font=dict(color="white"),
    )
    
    return fig
