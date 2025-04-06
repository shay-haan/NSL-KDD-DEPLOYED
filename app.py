# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
    with open('models/all_models.pkl', 'rb') as f:
        return pickle.load(f)

model_data = load_models()
xgboost_models = model_data['xgboost_models']
logistic_models = model_data['logistic_models']
ensemble_models = model_data['ensemble_models']
scalers = model_data['scalers']
feature_names = model_data['feature_names']

# Sidebar
st.sidebar.title("🛡️ NIDS Control Panel")
st.sidebar.info(
    """
    Current Session Info:
    - Date: 2025-04-06 02:59:08 UTC
    - User: shay-haan
    """
)

# Main detection function
def detect_attacks(data):
    results = {}
    attack_types = ['DoS', 'Probe', 'R2L', 'U2R']
    
    for attack_type in attack_types:
        # Scale data
        X_scaled = scalers[attack_type].transform(data)
        
        # Get predictions
        ensemble_pred = ensemble_models[attack_type].predict_proba(X_scaled)
        xgb_pred = xgboost_models[attack_type].predict_proba(X_scaled)
        lr_pred = logistic_models[attack_type].predict_proba(X_scaled)
        
        results[attack_type] = {
            'ensemble': ensemble_pred[0][1],
            'xgboost': xgb_pred[0][1],
            'logistic': lr_pred[0][1]
        }
    
    return results

# Main app layout
st.title("Network Intrusion Detection System")
st.markdown("""
This system uses ensemble machine learning to detect four types of network attacks:
- **DoS** (Denial of Service)
- **Probe** (Surveillance/Scanning)
- **R2L** (Remote to Local)
- **U2R** (User to Root)
""")

# Input method selection
input_method = st.radio(
    "Choose input method:",
    ["Upload CSV", "Single Connection Analysis"]
)

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload network traffic data (CSV)", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} connections!")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Process button
            if st.button("Analyze Traffic"):
                with st.spinner("Analyzing network traffic..."):
                    # Ensure data has correct features
                    missing_features = [f for f in feature_names if f not in df.columns]
                    if missing_features:
                        st.error(f"Missing features: {', '.join(missing_features)}")
                    else:
                        # Process each row
                        results = []
                        for idx, row in df.iterrows():
                            row_results = detect_attacks(row[feature_names].to_frame().T)
                            results.append(row_results)
                        
                        # Create results visualization
                        st.subheader("Analysis Results")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        for attack_type, col in zip(['DoS', 'Probe', 'R2L', 'U2R'], 
                                                  [col1, col2, col3, col4]):
                            detected = sum(1 for r in results if r[attack_type]['ensemble'] > 0.5)
                            with col:
                                st.metric(
                                    f"{attack_type} Attacks",
                                    f"{detected}",
                                    f"{(detected/len(df)*100):.1f}%"
                                )
                        
                        # Detailed results
                        st.subheader("Detailed Analysis")
                        for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
                            with st.expander(f"{attack_type} Analysis"):
                                # Create probability distribution plot
                                fig = go.Figure()
                                probs = [r[attack_type]['ensemble'] for r in results]
                                fig.add_trace(go.Histogram(
                                    x=probs,
                                    name='Probability Distribution',
                                    nbinsx=50
                                ))
                                fig.update_layout(
                                    title=f"{attack_type} Attack Probability Distribution",
                                    xaxis_title="Probability",
                                    yaxis_title="Count"
                                )
                                st.plotly_chart(fig)
                                
                                # Show high-risk connections
                                high_risk = [
                                    (i, p) for i, p in enumerate(probs) 
                                    if p > 0.8
                                ]
                                if high_risk:
                                    st.warning(
                                        f"Found {len(high_risk)} high-risk connections "
                                        f"(probability > 0.8)"
                                    )
                                    risk_df = pd.DataFrame(
                                        high_risk,
                                        columns=['Connection ID', 'Probability']
                                    )
                                    st.dataframe(risk_df)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

else:
    st.subheader("Single Connection Analysis")
    
    # Create input form
    with st.form("connection_form"):
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        # Basic features
        with col1:
            duration = st.number_input("Duration", min_value=0)
            protocol = st.selectbox("Protocol Type", ['tcp', 'udp', 'icmp'])
            service = st.selectbox("Service", ['http', 'private', 'domain_u', 'smtp'])
            
        with col2:
            src_bytes = st.number_input("Source Bytes", min_value=0)
            dst_bytes = st.number_input("Destination Bytes", min_value=0)
            flag = st.selectbox("Flag", ['SF', 'S0', 'REJ', 'RSTR'])
            
        with col3:
            land = st.selectbox("Land", [0, 1])
            wrong_fragment = st.number_input("Wrong Fragment", min_value=0)
            urgent = st.number_input("Urgent", min_value=0)
            
        # Add more features as needed...
        
        submitted = st.form_submit_button("Analyze Connection")
        
        if submitted:
            # Create connection data
            connection_data = pd.DataFrame({
                'duration': [duration],
                'protocol_type': [protocol],
                'service': [service],
                'flag': [flag],
                'src_bytes': [src_bytes],
                'dst_bytes': [dst_bytes],
                'land': [land],
                'wrong_fragment': [wrong_fragment],
                'urgent': [urgent],
                # Add other features...
            })
            
            # Make predictions
            with st.spinner("Analyzing connection..."):
                results = detect_attacks(connection_data)
                
                # Show results
                st.subheader("Analysis Results")
                
                # Create gauge charts for each attack type
                for attack_type, probabilities in results.items():
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probabilities['ensemble'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"{attack_type} Attack Probability"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "orange"},
                                {'range': [80, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    st.plotly_chart(fig)
                    
                    # Show model comparison
                    st.write("Model Predictions:")
                    for model, prob in probabilities.items():
                        st.progress(prob)
                        st.write(f"{model.title()}: {prob*100:.2f}%")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### About
    This NIDS uses ensemble machine learning combining:
    - XGBoost
    - Logistic Regression
    - Stacking Ensemble
    
    Performance:
    - DoS: 99.998%
    - Probe: 100%
    - R2L: 99.999%
    - U2R: 100%
    """
)