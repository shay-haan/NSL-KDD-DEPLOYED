# Standard library imports
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from datetime import datetime

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Network IDS Research Project",
    page_icon="🔒",
    layout="wide"
)

# ------------------------------
# Helper Functions
# ------------------------------
@st.cache_resource
def load_models():
    """Load all saved models"""
    try:
        with open('models/all_models.pkl', 'rb') as f:
            models = pickle.load(f)
            st.success("Models loaded successfully!")
            return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess data for model input"""
    try:
        # Expected feature values from training
        expected_values = {
            'protocol_type': ['icmp', 'tcp', 'udp'],
            'service': [
                'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime',
                'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs',
                'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames',
                'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC',
                'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp',
                'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat',
                'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3',
                'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell',
                'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet',
                'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path',
                'vmnet', 'whois', 'X11', 'Z39_50'
            ],
            'flag': ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
        }

        # Numeric features
        numeric_features = [
            'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
            'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]

        # Process data
        df_processed = df.copy()
        
        # Handle numeric features
        for col in numeric_features:
            if col not in df_processed.columns:
                df_processed[col] = 0
            else:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        # Process categorical features
        categorical_features = {}
        for feature, values in expected_values.items():
            dummies = pd.get_dummies(df_processed[feature], prefix=feature)
            expected_cols = [f"{feature}_{value}" for value in values]
            for col in expected_cols:
                if col not in dummies.columns:
                    dummies[col] = 0
            categorical_features[feature] = dummies[expected_cols]

        # Combine features
        df_processed = df_processed[numeric_features]
        for feature in ['protocol_type', 'service', 'flag']:
            df_processed = pd.concat([df_processed, categorical_features[feature]], axis=1)

        # Convert to numpy array
        X = df_processed.to_numpy()

        # Verify feature counts
        st.write("\nFeature count verification:")
        st.write(f"Numeric features: {len(numeric_features)}")
        st.write(f"Protocol features: {len(expected_values['protocol_type'])}")
        st.write(f"Service features: {len(expected_values['service'])}")
        st.write(f"Flag features: {len(expected_values['flag'])}")
        st.write(f"Total features: {X.shape[1]}")

        if X.shape[1] != 122:
            raise ValueError(f"Feature shape mismatch, expected: 122, got {X.shape[1]}")

        return X

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def make_predictions(X, models, model_type):
    """Make predictions using specified model"""
    results = {}
    for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
        try:
            if model_type == 'xgboost':
                model = models['xgboost_models'][attack_type]['model']
            else:
                model = models[f'{model_type}_models'][attack_type]
            
            pred_proba = model.predict_proba(X)
            results[attack_type] = {
                'probability': pred_proba[:, 1],
                'prediction': pred_proba[:, 1] > 0.5,
                'model_type': model_type
            }
        except Exception as e:
            st.error(f"Error in {model_type} model for {attack_type}: {str(e)}")
            results[attack_type] = None
    
    return results

def display_results(results_dict, df):
    """Display model comparison results"""
    st.subheader("Model Comparison")
    cols = st.columns(4)
    
    for attack_type, col in zip(['DoS', 'Probe', 'R2L', 'U2R'], cols):
        with col:
            st.write(f"**{attack_type} Attack Detection**")
            comparison_data = []
            
            for model_type in ['xgboost', 'logistic', 'ensemble']:
                if attack_type in results_dict[model_type]:
                    result = results_dict[model_type][attack_type]
                    if result is not None:
                        detected = result['prediction'].sum()
                        comparison_data.append({
                            'Model': model_type,
                            'Detected': detected,
                            'Percentage': (detected/len(df)*100)
                        })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                fig = px.bar(
                    comparison_df,
                    x='Model',
                    y='Percentage',
                    title=f'{attack_type} Detection Rate'
                )
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Main Application
# ------------------------------
def main():
    # Page title and description
    st.title("🔒 Network Intrusion Detection System")
    st.markdown("""
    ### Research Project - Final Semester
    This system uses ensemble machine learning to detect network attacks:
    - Denial of Service (DoS)
    - Probe Attacks
    - Remote to Local (R2L)
    - User to Root (U2R)
    """)

    # Sidebar information
    st.sidebar.info(f"""
    Session Info:
    - Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
    - User: shay-haan
    """)

    # Load models
    models = load_models()

    if models:
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file with network traffic data", type="csv")
        
        if uploaded_file:
            try:
                # Load and preview data
                df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(df)} records!")
                
                with st.expander("Preview Raw Data"):
                    st.dataframe(df.head())
                    st.info(f"Dataset Shape: {df.shape}")
                
                # Analysis button
                if st.button("Analyze Network Traffic"):
                    with st.spinner("Processing data..."):
                        # Preprocess data
                        X = preprocess_data(df)
                        
                        if X is not None:
                            # Get predictions from all models
                            results = {
                                'xgboost': make_predictions(X, models, 'xgboost'),
                                'logistic': make_predictions(X, models, 'logistic'),
                                'ensemble': make_predictions(X, models, 'ensemble')
                            }
                            
                            # Display results
                            display_results(results, df)
                            
                            # Detailed analysis
                            st.subheader("Detailed Analysis")
                            for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
                                with st.expander(f"{attack_type} Detailed Analysis"):
                                    for model_type in ['xgboost', 'logistic', 'ensemble']:
                                        if (results[model_type] and 
                                            attack_type in results[model_type] and 
                                            results[model_type][attack_type] is not None):
                                            
                                            result = results[model_type][attack_type]
                                            fig = px.histogram(
                                                x=result['probability'],
                                                title=f"{attack_type} Attack Probability Distribution ({model_type})"
                                            )
                                            st.plotly_chart(fig)
                            
                            # Download results
                            if st.button("Download Results"):
                                for model_type in ['xgboost', 'logistic', 'ensemble']:
                                    for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
                                        if (results[model_type] and 
                                            attack_type in results[model_type] and 
                                            results[model_type][attack_type] is not None):
                                            
                                            result = results[model_type][attack_type]
                                            df[f'{attack_type}_{model_type}_probability'] = result['probability']
                                            df[f'{attack_type}_{model_type}_prediction'] = result['prediction']
                                
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
                st.info("Please check your CSV file format.")
    else:
        st.error("⚠️ Could not load models. Please ensure 'all_models.pkl' exists in the app directory.")

    # Model performance information
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Model Performance
    #### XGBoost
    - DoS: 99.8%
    - Probe: 99.9%
    - R2L: 99.7%
    - U2R: 99.9%

    #### Logistic Regression
    - DoS: 98.9%
    - Probe: 99.1%
    - R2L: 98.8%
    - U2R: 99.2%

    #### Ensemble
    - DoS: 99.998%
    - Probe: 100%
    - R2L: 99.999%
    - U2R: 100%

    *Based on test dataset evaluation*
    """)

if __name__ == "__main__":
    main()