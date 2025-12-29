import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import os
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.strict_detector import StrictDetector

DATA_DIR = 'data'

st.set_page_config(layout="wide")
st.title('AI-Powered Insider Threat Detection: Combined Dashboard')

# Load data
@st.cache_data
def load_all_data():
    features = pd.read_csv(os.path.join(DATA_DIR, 'merged_features.csv'))
    scores = pd.read_csv(os.path.join(DATA_DIR, 'anomaly_scores.csv'))
    file_access = pd.read_csv(os.path.join(DATA_DIR, 'file_access.csv'), parse_dates=['access_time'])
    usb_usage = pd.read_csv(os.path.join(DATA_DIR, 'usb_usage.csv'), parse_dates=['plug_time', 'unplug_time'])
    
    # Try to load red team users, fallback to column if exists
    try:
        red_team = pd.read_csv(os.path.join(DATA_DIR, 'red_team_users.csv'))
        scores = pd.merge(scores, red_team, on='user', how='left')
    except FileNotFoundError:
        pass
    
    return features, scores, file_access, usb_usage

features, scores, file_access, usb_usage = load_all_data()
df = pd.merge(features, scores, on='user')

# Add unified risk score
@st.cache_data
def compute_unified_score(df):
    scaler = MinMaxScaler()
    score_cols = ['isolation_forest', 'oneclass_svm', 'autoencoder']
    df_copy = df.copy()
    for col in score_cols:
        df_copy[f'norm_{col}'] = scaler.fit_transform(df_copy[[col]])
    df_copy['unified_risk_score'] = df_copy[[f'norm_{col}' for col in score_cols]].mean(axis=1)
    return df_copy

df = compute_unified_score(df)

# Load StrictDetector for ensemble detection
@st.cache_resource
def load_strict_detector():
    """Load the ensemble detector (cached to avoid reloading)"""
    detector = StrictDetector()
    return detector

detector = load_strict_detector()

# Prepare node attributes for graph
def get_node_attrs():
    attrs = {}
    for _, row in scores.iterrows():
        anomaly = max(row['isolation_forest'], row['oneclass_svm'], row['autoencoder'])
        red_team = row.get('is_red_team', 0) if pd.notna(row.get('is_red_team', 0)) else 0
        attrs[row['user']] = {
            'anomaly': anomaly,
            'red_team': red_team,
            'high_risk': (anomaly > 1.0) or (red_team == 1)
        }
    return attrs
attrs = get_node_attrs()

# === UTILITY FUNCTIONS FROM user_detail.py ===

@st.cache_data
def get_population_quantiles(features_df):
    """Calculates population quantiles for feature comparison."""
    feature_cols = [
        'mean_login_hour', 'files_per_day', 'usb_per_day', 
        'out_of_session_access', 'degree_centrality', 'sentiment'
    ]
    return features_df[feature_cols].quantile([0.25, 0.75]).to_dict()

def get_risk_summary(user_row, risk_df, percentile_threshold=95):
    """Determines a user's risk status and provides a summary."""
    user_score = user_row.get('unified_risk_score', 0)
    threshold_high = np.percentile(risk_df['unified_risk_score'], percentile_threshold)
    threshold_medium = np.percentile(risk_df['unified_risk_score'], 85)

    if user_score >= threshold_high:
        status = "HIGH RISK"
        color = "red"
        reason = f"User's unified risk score ({user_score:.2f}) exceeds the {percentile_threshold}th percentile threshold ({threshold_high:.2f})."
    elif user_score >= threshold_medium:
        status = "MEDIUM RISK"
        color = "orange"
        reason = f"User's unified risk score ({user_score:.2f}) is elevated, falling between the 85th and {percentile_threshold}th percentiles."
    else:
        status = "LOW RISK"
        color = "green"
        reason = "User's behavior is within normal parameters."
        
    is_red_team = (user_row.get("is_red_team_x") == 1) or (user_row.get("is_red_team") == 1)

    st.markdown(f"### 🔴 User Risk Summary")
    st.markdown(f"**Risk Status:** <span style='color:{color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)
    if is_red_team:
        st.markdown("**Ground Truth:** <span style='color:red; font-weight:bold;'>🚩 Known Red Team User</span>", unsafe_allow_html=True)
    st.write(f"**Reason:** {reason}")

def generate_behavior_table(user_row, quantiles):
    """Creates a readable table of key behaviors."""
    st.markdown("### 📊 Key Behavior Indicators")
    
    data = []
    
    # Login Time
    login_h = user_row['mean_login_hour']
    interp = "Normal"
    if not (8 <= login_h <= 10): interp = "⚠️ Outside core hours"
    data.append(["Login Time", f"{login_h:.2f}", "8 AM–10 AM", interp])

    # File Access
    files_p_day = user_row['files_per_day']
    q75_files = quantiles['files_per_day'][0.75]
    interp = "Normal"
    if files_p_day > q75_files * 2: interp = "🚨 Very High"
    elif files_p_day > q75_files: interp = "⚠️ Higher than average"
    data.append(["File Access / Day", f"{files_p_day:.2f}", f"≤ {q75_files:.2f}", interp])

    # USB Usage
    usb_p_day = user_row['usb_per_day']
    q75_usb = quantiles['usb_per_day'][0.75]
    interp = "Normal"
    if usb_p_day > 0:
        if q75_usb == 0: interp = "🚨 Unusual (Peers have none)"
        elif usb_p_day > q75_usb: interp = "⚠️ Higher than average"
    data.append(["USB Usage / Day", f"{usb_p_day:.2f}", f"≤ {q75_usb:.2f}", interp])
    
    # Out-of-Session Access
    oos_access = user_row['out_of_session_access']
    q75_oos = quantiles['out_of_session_access'][0.75]
    interp = "Normal"
    if oos_access > q75_oos * 2: interp = "🚨 Very High"
    elif oos_access > q75_oos: interp = "⚠️ Higher than average"
    data.append(["Out-of-Session Access", f"{oos_access}", f"≤ {q75_oos:.0f}", interp])

    # Network Centrality
    centrality = user_row['degree_centrality']
    q75_centrality = quantiles['degree_centrality'][0.75]
    interp = "Low–Medium"
    if centrality > q75_centrality: interp = "⚠️ High Impact User"
    data.append(["Network Centrality", f"{centrality:.2f}", f"≤ {q75_centrality:.2f}", interp])

    df_table = pd.DataFrame(data, columns=["Behavior", "User Value", "Normal Range (75th %)", "Interpretation"])
    st.dataframe(df_table, hide_index=True, width='stretch')

def display_base_model_decisions(user_row, df):
    """Shows how each base model evaluated the user."""
    st.markdown("### 🧠 Base Model Decisions")
    
    models = ["isolation_forest", "oneclass_svm", "autoencoder"]
    flag_count = 0
    
    for model in models:
        threshold = df[model].quantile(0.95) # Top 5%
        user_score = user_row[model]
        
        if user_score >= threshold:
            decision = f"✔ **FLAGGED** (Score: {user_score:.3f} ≥ Threshold: {threshold:.3f})"
            flag_count += 1
        else:
            decision = f"✖ **Normal** (Score: {user_score:.3f} < Threshold: {threshold:.3f})"
        
        st.markdown(f"**{model.replace('_', ' ').title()}:** {decision}")
        
    st.write(f"**Conclusion:** User flagged by **{flag_count} out of {len(models)}** base models.")

def generate_investigation_hints(user_row, df):
    """Generates plain-English reasons for the flag."""
    st.markdown("### 🔍 Why This User Was Flagged")
    
    reasons = []
    quantiles = get_population_quantiles(df)
    
    if user_row['out_of_session_access'] > quantiles['out_of_session_access'][0.75] * 2:
        reasons.append("Extremely high number of file accesses outside of login hours.")
        
    if user_row['usb_per_day'] > 0 and quantiles['usb_per_day'][0.75] == 0:
        reasons.append("Frequent USB device usage, which is rare among peers.")
        
    if user_row['degree_centrality'] > quantiles['degree_centrality'][0.75]:
        reasons.append("High network centrality (many connections), increasing their potential blast radius.")
        
    if user_row['files_per_day'] > quantiles['files_per_day'][0.75] * 2:
        reasons.append("Accessed a significantly higher number of files per day than peers.")

    if not reasons:
        reasons.append("This user's risk score is driven by a combination of several smaller, subtle anomalies rather than a single large deviation.")

    for reason in reasons:
        st.markdown(f"- {reason}")
        
    st.markdown("""
    ---
    ### 🕵️ Recommended Analyst Actions
    1.  **Review file access logs** for the user, focusing on activity outside of standard business hours.
    2.  **Inspect USB activity logs** to understand the nature of the devices and data transfers.
    3.  **Cross-reference with email communications** if `keyword_flag` was triggered to understand the context.
    """)

# === VISUALIZATION FUNCTIONS FROM detection_logic.py ===

def plot_risk_distribution(df, threshold_value, selected_user_score=None, selected_user=None):
    """Plots a histogram of the unified risk score distribution."""
    fig = go.Figure()

    # Histogram of all user scores
    fig.add_trace(go.Histogram(
        x=df['unified_risk_score'],
        name='Risk Score Distribution',
        marker_color='#636EFA'
    ))

    # Vertical line for the threshold
    fig.add_vline(
        x=threshold_value,
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold_value:.2f})",
        annotation_position="top left"
    )

    # Marker for the selected user
    if selected_user_score is not None and selected_user is not None:
        fig.add_trace(go.Scatter(
            x=[selected_user_score],
            y=[0],
            mode='markers',
            marker=dict(color='orange', size=15, symbol='star'),
            name=f'User: {selected_user}'
        ))

    fig.update_layout(
        title='Risk Score Distribution and Detection Threshold',
        xaxis_title='Unified Risk Score',
        yaxis_title='Number of Users',
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font=dict(color="white"),
        showlegend=True
    )

    return fig

# === EVALUATION METRICS FROM metrics.py ===

def get_predictions(df, model_name, percentile_threshold=95):
    """Generates binary predictions based on a percentile threshold."""
    score_col = model_name
    threshold = np.percentile(df[score_col], percentile_threshold)
    
    y_true = df.get('is_red_team', pd.Series([0] * len(df))).fillna(0).astype(int)
    y_pred = (df[score_col] >= threshold).astype(int)
    
    return y_true, y_pred, df[score_col], threshold

def compute_metrics(y_true, y_pred):
    """Computes evaluation metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def compute_roc_auc(y_true, scores):
    """Computes ROC-AUC score."""
    try:
        return roc_auc_score(y_true, scores)
    except ValueError:
        return None

def plot_confusion_matrix(y_true, y_pred):
    """Generates a Plotly heatmap for the confusion matrix."""
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

# Build full graph
def build_graph():
    G = nx.Graph()
    for _, row in file_access.iterrows():
        G.add_edge(row['user'], row['file'], type='access')
    for _, row in usb_usage.iterrows():
        G.add_edge(row['user'], row['device'], type='usb')
    return G
G = build_graph()

# At-risk subgraph
def get_at_risk_subgraph(G, attrs):
    high_risk_nodes = {n for n, v in attrs.items() if v['high_risk']}
    connected_nodes = set()
    for node in high_risk_nodes:
        connected_nodes.add(node)
        connected_nodes.update(G.neighbors(node))
    return G.subgraph(connected_nodes).copy()

# Tabs
anomaly_tab, user_tab, ensemble_tab, graph_tab, metrics_tab, how_tab = st.tabs(["Anomaly Table", "User Detail", "🔐 Ensemble Detector", "At-Risk Graph", "Model Evaluation", "How Does It Work?"])

with anomaly_tab:
    st.header('User Anomaly Scores')
    score_method = st.selectbox('Select Model', ['isolation_forest', 'oneclass_svm', 'autoencoder', 'unified_risk_score'], key='score_method')
    df['Red Team'] = df.apply(lambda row: '🚩' if (row.get('is_red_team_x') == 1 or row.get('is_red_team') == 1) else '', axis=1)
    df['rank'] = df[score_method].rank(ascending=False)
    df_sorted = df.sort_values(score_method, ascending=False)
    cols = ['user', 'Red Team', score_method, 'rank'] + [c for c in df.columns if c not in ['user', score_method, 'rank', 'Red Team', 'is_red_team', 'is_red_team_x']]
    st.dataframe(df_sorted[cols], height=500, width='stretch')
    
    st.subheader('Top 5 Anomalous Users')
    top5 = df_sorted.head(5)
    st.bar_chart(top5.set_index('user')[score_method])
    
    # Risk distribution chart
    if score_method == 'unified_risk_score':
        threshold = np.percentile(df[score_method], 95)
        st.subheader('Risk Score Distribution')
        st.plotly_chart(plot_risk_distribution(df, threshold), use_container_width=True)

with user_tab:
    st.header('User Detail & Risk Analysis')
    selected_user = st.selectbox('Select User', df_sorted['user'], key='user_detail')
    user_row = df_sorted[df_sorted['user'] == selected_user].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Summary
        get_risk_summary(user_row, df)
        
        # Behavior Indicators
        quantiles = get_population_quantiles(df)
        generate_behavior_table(user_row, quantiles)
        
    with col2:
        # Base Model Decisions
        display_base_model_decisions(user_row, df)
        
        # Investigation Hints
        generate_investigation_hints(user_row, df)
    
    # Feature Details
    st.subheader('📋 Feature Details')
    feature_cols = ['mean_login_hour', 'mean_logout_hour', 'files_per_day', 'usb_per_day', 'emails_per_day', 'out_of_session_access', 'degree_centrality', 'betweenness_centrality', 'keyword_flag', 'subject_len', 'sentiment']
    features_dict = {k: user_row[k] for k in feature_cols if k in user_row}
    col_left, col_right = st.columns(2)
    with col_left:
        st.json({k: v for i, (k, v) in enumerate(features_dict.items()) if i < len(features_dict)//2})
    with col_right:
        st.json({k: v for i, (k, v) in enumerate(features_dict.items()) if i >= len(features_dict)//2})
    
    st.subheader('🎯 Anomaly Scores')
    scores_dict = {k: user_row[k] for k in ['isolation_forest', 'oneclass_svm', 'autoencoder', 'unified_risk_score'] if k in user_row}
    st.json(scores_dict)

with ensemble_tab:
    st.header('🔐 Ensemble Detector: Multi-Variant Consensus-Based Detection')
    st.markdown("""
    This tab uses a **consensus-based ensemble** of 6 different model variants:
    - 2 Isolation Forest models (different hyperparameters)
    - 2 One-Class SVM models (different kernels)
    - 2 Autoencoder variants (different architectures)
    
    **Verdict**: "MALICIOUS" only when 2+ model families agree (high confidence)
    """)
    
    # Two detection modes
    mode = st.radio("Select Detection Mode", ["Single User Analysis", "Batch Detection (All Users)"], horizontal=True)
    
    if mode == "Single User Analysis":
        st.subheader("Individual User Threat Assessment")
        selected_user_ensemble = st.selectbox('Select User', df['user'].unique(), key='ensemble_user')
        user_data = df[df['user'] == selected_user_ensemble].iloc[0]
        
        # Get features (use exactly the 11 features trained on)
        feature_cols = ['mean_login_hour', 'mean_logout_hour', 'files_per_day', 'usb_per_day', 
                       'emails_per_day', 'out_of_session_access', 'degree_centrality', 
                       'betweenness_centrality', 'keyword_flag', 'subject_len', 'sentiment']
        features = user_data[feature_cols].values.astype(float)
        
        try:
            # Evaluate with ensemble
            ensemble_result = detector.evaluate(features)
            
            # Display verdict with color coding
            col1, col2, col3 = st.columns(3)
            with col1:
                verdict_color = "🚨 red" if ensemble_result['verdict'] == "MALICIOUS" else "✓ green"
                st.metric("Ensemble Verdict", ensemble_result['verdict'], 
                         help="Consensus from 6-model ensemble")
            with col2:
                st.metric("Confidence Level", f"{ensemble_result['confidence']:.1%}",
                         help="0-100% confidence in verdict")
            with col3:
                st.metric("Models Triggered", ensemble_result['triggered_count'],
                         help=f"Out of {ensemble_result['total_models']} total models")
            
            # Triggered models
            st.subheader("🎯 Triggered Models")
            if ensemble_result['triggered_models']:
                cols = st.columns(len(ensemble_result['triggered_models']))
                for i, model in enumerate(ensemble_result['triggered_models']):
                    with cols[i]:
                        st.warning(f"⚠️ {model}")
            else:
                st.success("✓ No models triggered")
            
            # Model family breakdown
            st.subheader("📊 Model Family Breakdown")
            families = ensemble_result['model_families']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                iso_count = families['isolation_forest']
                st.metric("Isolation Forest", f"{iso_count}/2", 
                         delta="TRIGGERED" if iso_count > 0 else "Normal",
                         delta_color="inverse")
            with col2:
                svm_count = families['oneclass_svm']
                st.metric("One-Class SVM", f"{svm_count}/2",
                         delta="TRIGGERED" if svm_count > 0 else "Normal",
                         delta_color="inverse")
            with col3:
                ae_count = families['autoencoder']
                st.metric("Autoencoder", f"{ae_count}/2",
                         delta="TRIGGERED" if ae_count > 0 else "Normal",
                         delta_color="inverse")
            
            # Individual model scores
            st.subheader("🔬 Individual Model Scores")
            scores_data = []
            for model_name, score in ensemble_result['scores'].items():
                model_type = 'IF' if 'iso' in model_name else 'SVM' if 'svm' in model_name else 'AE'
                
                if 'iso' in model_name or 'svm' in model_name:
                    threshold = 0.60
                    triggered = "✓" if score > threshold else "✗"
                else:
                    threshold = 0.02
                    triggered = "✓" if score > threshold else "✗"
                
                scores_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Type': model_type,
                    'Score': f"{score:.4f}",
                    'Threshold': f"{threshold:.4f}",
                    'Triggered': triggered
                })
            
            scores_df = pd.DataFrame(scores_data)
            st.dataframe(scores_df, use_container_width=True, hide_index=True)
            
            # Explanation
            st.subheader("💡 Explanation")
            st.info(ensemble_result['explanation'])
            
            # Ground truth comparison
            if 'is_red_team' in user_data and pd.notna(user_data['is_red_team']):
                st.subheader("📋 Ground Truth Comparison")
                is_red_team = user_data['is_red_team']
                if is_red_team:
                    st.error("🚩 This user IS in the red team (ground truth)")
                else:
                    st.success("✓ This user is NOT in the red team (ground truth)")
                
                # Accuracy
                detection_correct = (ensemble_result['verdict'] == 'MALICIOUS') == is_red_team
                if detection_correct:
                    st.success("✓ Ensemble detection is CORRECT")
                else:
                    st.warning("⚠️ Ensemble detection MISMATCH with ground truth")
        
        except Exception as e:
            st.error(f"Error in ensemble detection: {e}")
            st.info("Make sure the StrictDetector models are trained. Run: `python models/train_variants.py`")
    
    else:  # Batch Detection
        st.subheader("Batch Threat Detection - All Users")
        st.info("Evaluating all users with the ensemble detector...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        try:
            for idx, (_, user_data) in enumerate(df.iterrows()):
                status_text.text(f"Processing: {user_data['user']} ({idx+1}/{len(df)})")
                
                # Get features (use exactly the 11 features trained on)
                feature_cols = ['mean_login_hour', 'mean_logout_hour', 'files_per_day', 'usb_per_day', 
                               'emails_per_day', 'out_of_session_access', 'degree_centrality', 
                               'betweenness_centrality', 'keyword_flag', 'subject_len', 'sentiment']
                features = user_data[feature_cols].values.astype(float)
                
                # Evaluate
                result = detector.evaluate(features)
                result['user'] = user_data['user']
                result['is_red_team'] = user_data.get('is_red_team', 0)
                
                all_results.append(result)
                progress_bar.progress((idx + 1) / len(df))
            
            status_text.empty()
            
            # Create results dataframe
            results_df = pd.DataFrame([
                {
                    'User': r['user'],
                    'Verdict': r['verdict'],
                    'Confidence': f"{r['confidence']:.1%}",
                    'Models Triggered': r['triggered_count'],
                    'Family: IF': r['model_families']['isolation_forest'],
                    'Family: SVM': r['model_families']['oneclass_svm'],
                    'Family: AE': r['model_families']['autoencoder'],
                    'Is Red Team': '🚩 Yes' if r['is_red_team'] else 'No'
                }
                for r in all_results
            ])
            
            # Display stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                malicious_count = sum(1 for r in all_results if r['verdict'] == 'MALICIOUS')
                st.metric("🚨 Malicious Verdicts", malicious_count)
            with col2:
                safe_count = sum(1 for r in all_results if r['verdict'] == 'SAFE')
                st.metric("✓ Safe Verdicts", safe_count)
            with col3:
                red_team_count = sum(1 for r in all_results if r['is_red_team'])
                st.metric("🚩 Known Red Team", red_team_count)
            with col4:
                # Accuracy
                correct = sum(1 for r in all_results if (r['verdict'] == 'MALICIOUS') == r['is_red_team'])
                accuracy = correct / len(all_results) if all_results else 0
                st.metric("✓ Accuracy", f"{accuracy:.1%}")
            
            # Display full results
            st.subheader("Detection Results Table")
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Threats summary
            st.subheader("🚨 Detected Threats")
            threats = results_df[results_df['Verdict'] == 'MALICIOUS']
            if len(threats) > 0:
                st.dataframe(threats, use_container_width=True, hide_index=True)
            else:
                st.success("No threats detected!")
            
            # Performance metrics
            st.subheader("📊 Performance Metrics")
            correct_malicious = sum(1 for r in all_results if r['verdict'] == 'MALICIOUS' and r['is_red_team'])
            false_positives = sum(1 for r in all_results if r['verdict'] == 'MALICIOUS' and not r['is_red_team'])
            false_negatives = sum(1 for r in all_results if r['verdict'] == 'SAFE' and r['is_red_team'])
            
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("True Positives (TP)", correct_malicious)
            with metrics_cols[1]:
                st.metric("False Positives (FP)", false_positives)
            with metrics_cols[2]:
                st.metric("False Negatives (FN)", false_negatives)
            with metrics_cols[3]:
                precision = correct_malicious / (correct_malicious + false_positives) if (correct_malicious + false_positives) > 0 else 0
                st.metric("Precision", f"{precision:.2%}")
        
        except Exception as e:
            st.error(f"Error in batch detection: {e}")
            st.info("Make sure the StrictDetector models are trained. Run: `python models/train_variants.py`")

with graph_tab:
    st.header('At-Risk Nodes and Their Connections')
    subG = get_at_risk_subgraph(G, attrs)
    net = Network(height='900px', width='100%', notebook=False, bgcolor='#222222', font_color='white')
    net.barnes_hut(gravity=-2000, central_gravity=0.1, spring_length=200, spring_strength=0.01, damping=0.85, overlap=1)
    net.set_options('''
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "fit": true, "iterations": 2500, "updateInterval": 50},
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.01,
          "damping": 0.85,
          "avoidOverlap": 1
        }
      }
    }
    ''')
    for node in subG.nodes():
        if node in attrs:
            score = attrs[node]['anomaly']
            red = attrs[node]['red_team']
            color = 'red' if red else ('orange' if score > 1.5 else 'yellow' if score > 1.0 else 'lightblue')
            size = 30 if red else (20 if score > 1.5 else 15 if score > 1.0 else 10)
            title = f"User: {node}<br>Anomaly Score: {score:.2f}<br>Red Team: {'Yes' if red else 'No'}"
        elif str(node).startswith('file'):
            color = 'green'
            size = 8
            title = f"File: {node}"
        elif str(node).startswith('usb'):
            color = 'purple'
            size = 8
            title = f"Device: {node}"
        else:
            color = 'gray'
            size = 8
            title = str(node)
        net.add_node(node, label=str(node), color=color, size=size, title=title)
    for edge in subG.edges(data=True):
        net.add_edge(edge[0], edge[1], color='gray' if edge[2]['type']=='access' else 'purple')
    net.save_graph('dashboard/graph.html')
    st.components.v1.html(open('dashboard/graph.html', 'r', encoding='utf-8').read(), height=900, scrolling=False)

with metrics_tab:
    st.header('Model Evaluation & Performance Metrics')
    
    # Model selection
    eval_model = st.selectbox('Select Model to Evaluate', ['isolation_forest', 'oneclass_svm', 'autoencoder', 'unified_risk_score'], key='eval_model')
    percentile_threshold = st.slider('Detection Threshold (Percentile)', 80, 99, 95, key='percentile_threshold')
    
    # Get predictions
    y_true, y_pred, scores, threshold = get_predictions(df, eval_model, percentile_threshold)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['Precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['Recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{metrics['F1-Score']:.3f}")
    
    # ROC-AUC
    roc_auc = compute_roc_auc(y_true, scores)
    if roc_auc is not None:
        st.metric("ROC-AUC Score", f"{roc_auc:.3f}")
    
    # Confusion Matrix
    st.subheader('Confusion Matrix')
    cm_fig = plot_confusion_matrix(y_true, y_pred)
    st.plotly_chart(cm_fig, use_container_width=True)
    
    # Detailed Results
    st.subheader('Detection Results')
    results_df = df.copy()
    results_df['Predicted'] = y_pred
    results_df['Actual'] = y_true
    results_df['Detection_Score'] = scores
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.write(f"**Total Users:** {len(results_df)}")
        st.write(f"**Predicted as Threats:** {y_pred.sum()}")
        st.write(f"**Actual Red Team Users:** {y_true.sum()}")
    with col_right:
        st.write(f"**True Positives (TP):** {((y_true == 1) & (y_pred == 1)).sum()}")
        st.write(f"**False Positives (FP):** {((y_true == 0) & (y_pred == 1)).sum()}")
        st.write(f"**False Negatives (FN):** {((y_true == 1) & (y_pred == 0)).sum()}")

with how_tab:
    st.header('How Does It Work?')
    st.markdown('''
## System Overview
This system detects insider threats by analyzing user behavior, system access, and relationships using advanced machine learning and graph analysis techniques.

---

### 1. **Data Simulation & Feature Engineering**
- **Simulated Logs:** The system generates synthetic logs for user logins, file access, USB usage, and emails, mimicking real organizational activity.
- **Feature Engineering:** Extracts features such as:
    - Login/logout patterns (mean hours, frequency)
    - File/USB/email activity rates
    - Out-of-session file access
    - Graph centrality (degree, betweenness)
    - NLP features from email subjects (keyword flags, length)

---

### 2. **Anomaly Detection Algorithms**
- **Isolation Forest**
    - *Mathematics:* Randomly partitions data to isolate points. Anomalies are isolated faster (shorter average path length in trees).
    - *Computer Science:* Ensemble of binary trees; each tree splits on random features/values. The anomaly score is based on the average path length to isolate a sample.
- **One-Class SVM**
    - *Mathematics:* Finds a boundary in feature space that encloses most data (support vectors). Points outside are anomalies.
    - *Computer Science:* Uses kernel methods (e.g., RBF) to map data to high-dimensional space and find a maximal margin hyperplane.
- **Autoencoder**
    - *Mathematics:* Neural network learns to compress and reconstruct input. High reconstruction error indicates anomaly.
    - *Computer Science:* Trains a feedforward neural network (MLP) to minimize reconstruction loss (MSE) between input and output.

---

### 3. **Graph Analysis**
- **Entity Graph:** Users, files, and devices are nodes; edges represent access or usage.
- **Centrality Measures:**
    - *Degree Centrality:* Number of connections (activity level).
    - *Betweenness Centrality:* Frequency a node lies on shortest paths (potential for information flow/control).
- **At-Risk Subgraph:** Focuses on high-risk users and their direct connections for visualization and investigation.

---

### 4. **Explainability**
- **SHAP (SHapley Additive exPlanations):**
    - *Mathematics:* Based on cooperative game theory; attributes model output to each feature by averaging over all possible feature orderings.
    - *Computer Science:* Computes feature importances for each prediction, helping analysts understand why a user is flagged.
- **LIME (Local Interpretable Model-agnostic Explanations):**
    - *Mathematics:* Fits a simple, interpretable model locally around a prediction to approximate the complex model.
    - *Computer Science:* Perturbs input data and observes output changes to estimate feature influence (not supported for Isolation Forest, but available for other models).

---

### 5. **Dashboard & Visualization**
- **Streamlit:** Interactive web app for data exploration, anomaly review, and graph visualization.
- **PyVis/NetworkX:** Renders interactive network graphs for at-risk nodes and their relationships.

---

### 6. **Red Team Simulation**
- Injects malicious behaviors (after-hours access, mass downloads, suspicious USB usage) to test detection capability.

---

## Summary
This system combines unsupervised machine learning, graph theory, and explainable AI to provide a robust, interpretable approach to insider threat detection.
''')