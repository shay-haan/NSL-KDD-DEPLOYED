# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Network IDS Research Project",
    page_icon="🔒",
    layout="wide"
)

@st.cache_resource
def load_models():
    try:
        with open('all_models.pkl', 'rb') as f:
            models = pickle.load(f)
            st.success("Models loaded successfully!")
            return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def preprocess_data(df, feature_names):
    """Preprocess the input data to match training format"""
    try:
        # Create dummy variables for categorical features
        categorical_features = ['protocol_type', 'service', 'flag']
        df_processed = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
        
        # Ensure all required columns exist
        for col in feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # Select only the required features in the correct order
        df_processed = df_processed[feature_names]
        
        return df_processed
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

# Main title and info
st.title("🔒 Network Intrusion Detection System")
st.markdown("""
### Research Project - Final Semester
This system uses ensemble machine learning to detect network attacks:
- Denial of Service (DoS)
- Probe Attacks
- Remote to Local (R2L)
- User to Root (U2R)
""")

# Load models
models = load_models()

if models:
    # Show feature requirements
    with st.expander("📋 Required Features Information"):
        st.write("Your CSV file must contain the following features:")
        required_features = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
            'num_failed_logins', 'logged_in', 'num_compromised', 
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
        st.write(", ".join(required_features))

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file with network traffic data",
        type="csv"
    )

    if uploaded_file:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} records!")
            
            # Show data preview
            with st.expander("Preview Raw Data"):
                st.dataframe(df.head())
                st.info(f"Dataset Shape: {df.shape}")
            
            # Verify required features
            missing_features = [f for f in required_features if f not in df.columns]
            if missing_features:
                st.error(f"Missing required features: {', '.join(missing_features)}")
                st.stop()
            
            # Process data button
            if st.button("Analyze Network Traffic"):
                with st.spinner("Processing data..."):
                    # Preprocess data
                    df_processed = preprocess_data(df, models['feature_names'])
                    
                    if df_processed is not None:
                        # Get predictions for each attack type
                        results = {}
                        for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
                            # Scale data
                            scaler = models['scalers'][attack_type]
                            X_scaled = scaler.transform(df_processed)
                            
                            # Get predictions
                            ensemble_pred = models['ensemble_models'][attack_type].predict_proba(X_scaled)
                            xgb_pred = models['xgboost_models'][attack_type].predict_proba(X_scaled)
                            lr_pred = models['logistic_models'][attack_type].predict_proba(X_scaled)
                            
                            results[attack_type] = {
                                'Ensemble': ensemble_pred[:, 1],
                                'XGBoost': xgb_pred[:, 1],
                                'Logistic': lr_pred[:, 1]
                            }
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Summary metrics
                        cols = st.columns(4)
                        for attack_type, col in zip(results.keys(), cols):
                            detected = (results[attack_type]['Ensemble'] > 0.5).sum()
                            with col:
                                st.metric(
                                    f"{attack_type} Attacks",
                                    f"{detected}",
                                    f"{(detected/len(df)*100):.2f}%"
                                )
                        
                        # Detailed analysis
                        for attack_type in results:
                            with st.expander(f"{attack_type} Analysis"):
                                # Distribution plot
                                fig = px.histogram(
                                    pd.DataFrame(results[attack_type]),
                                    title=f"{attack_type} Attack Probability Distribution"
                                )
                                st.plotly_chart(fig)
                                
                                # High risk connections
                                high_risk = np.where(results[attack_type]['Ensemble'] > 0.8)[0]
                                if len(high_risk) > 0:
                                    st.warning(f"Found {len(high_risk)} high-risk connections!")
                                    st.dataframe(df.iloc[high_risk])
                        
                        # Download results
                        if st.button("Download Results"):
                            # Add predictions to original dataframe
                            for attack_type in results:
                                df[f'{attack_type}_probability'] = results[attack_type]['Ensemble']
                            
                            # Convert to CSV
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "Download Complete Analysis",
                                csv,
                                f"nids_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                key='download-csv'
                            )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please check your CSV file format and required features.")

# Sidebar info
st.sidebar.info(f"""
    Session Info:
    - Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
    - User: shay-haan
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Model Performance
- DoS: 99.998%
- Probe: 100%
- R2L: 99.999%
- U2R: 100%

*Based on test dataset evaluation*
""")