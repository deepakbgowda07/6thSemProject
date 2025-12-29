import streamlit as st
import pandas as pd
import numpy as np

# --- 1. User Risk Summary ---
def get_risk_summary(user_row, risk_df, percentile_threshold=95):
    """
    Determines a user's risk status and provides a summary.
    """
    user_score = risk_df.loc[risk_df['user'] == user_row['user'], 'unified_risk_score'].iloc[0]
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


# --- 2. Key Behavior Indicators ---
@st.cache_data
def get_population_quantiles(df):
    """Calculates population quantiles for feature comparison."""
    feature_cols = [
        'mean_login_hour', 'files_per_day', 'usb_per_day', 
        'out_of_session_access', 'degree_centrality', 'sentiment'
    ]
    return df[feature_cols].quantile([0.25, 0.75]).to_dict()

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


# --- 3. Anomaly Model Decisions ---
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

def display_strict_model_status(user_row, risk_df):
    """Shows the status according to the strict model."""
    st.markdown("### 🛡️ Strict Model Verdict")
    
    strict_score = user_row.get('strict_risk_score', 0)
    
    # Use a high percentile for the strict model's threshold
    threshold = np.percentile(risk_df['strict_risk_score'], 98)
    
    if strict_score >= threshold:
        st.error(f"**🚨 FLAGGED by Strict Model** (Score: {strict_score:.2f} ≥ Threshold: {threshold:.2f})")
        st.write("This user meets multiple high-risk criteria, indicating a high-confidence threat that requires immediate attention.")
    else:
        st.success(f"**✔ PASSED Strict Verification** (Score: {strict_score:.2f} < Threshold: {threshold:.2f})")
        st.write("This user does not meet the criteria for a high-confidence alert at this time.")


# --- 4. Investigation Hints ---
def generate_investigation_hints(user_row, df):
    """Generates plain-English reasons for the flag."""
    st.markdown("### 🔍 Why This User Was Flagged")
    
    reasons = []
    
    # Use the same logic as the behavior table for consistency
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