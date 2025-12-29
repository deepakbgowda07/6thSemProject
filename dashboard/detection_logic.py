import plotly.graph_objects as go
import pandas as pd

def plot_risk_distribution(df, threshold_value, selected_user_score=None, selected_user=None):
    """
    Plots a histogram of the unified risk score distribution, a threshold line,
    and an optional marker for the selected user.
    """
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

def get_feature_reasoning(user_row, features_df):
    """
    Provides a simple feature-level explanation for a user's risk score
    by comparing their features to the population median.
    """
    # These are the features that contribute to the anomaly scores
    feature_cols = [
        'mean_login_hour', 'mean_logout_hour', 'files_per_day', 
        'usb_per_day', 'emails_per_day', 'out_of_session_access', 
        'degree_centrality', 'betweenness_centrality', 'keyword_flag', 
        'subject_len', 'sentiment'
    ]
    
    # Calculate population medians
    population_medians = features_df[feature_cols].median()
    
    reasons = []
    for col in feature_cols:
        if col in user_row.index:
            user_value = user_row[col]
            median_value = population_medians[col]
            
            # Simple heuristic: if user's value is > 1.5x the median, it's a contributor
            # (avoid division by zero)
            if median_value > 0 and user_value > 1.5 * median_value:
                reasons.append(f"**{col.replace('_', ' ').title()}** ({user_value:.2f}) is significantly higher than the typical value ({median_value:.2f}).")
            elif user_value > 0 and median_value == 0:
                 reasons.append(f"**{col.replace('_', ' ').title()}** ({user_value:.2f}) is present where it's typically not.")


    if not reasons:
        return ["The user's behavior does not show significant deviations from the norm in the monitored features, but the combination of several smaller anomalies may have contributed to their score."]
        
    return reasons

