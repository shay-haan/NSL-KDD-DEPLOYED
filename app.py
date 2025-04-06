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
    """Load all saved models"""
    try:
        with open('all_models.pkl', 'rb') as f:
            models = pickle.load(f)
            st.success("Models loaded successfully!")
            return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess data exactly as we did during training"""
    try:
        # Define expected values from training - use EXACT values from training
        expected_values = {
            'protocol_type': ['icmp', 'tcp', 'udp'],
            # Complete list of 70 services from training
            'service': [
                'IRC', 'X11', 'Z39_50', 'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 
                'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 
                'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 
                'hostnames', 'http', 'http_443', 'imap4', 'iso_tsap', 'klogin', 
                'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 
                'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 
                'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 
                'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 
                'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urp_i', 
                'uucp', 'uucp_path', 'vmnet', 'whois', 'http_8001', 'http_2784',
                'red_i', 'urh_i', 'spy'
            ],
            'flag': ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
        }

        # Numeric features in exact order
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

        # Work on a copy
        df_processed = df.copy()

        # Handle numeric features
        for col in numeric_features:
            if col not in df_processed.columns:
                df_processed[col] = 0
            else:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        # Create one-hot encoded columns for each categorical feature
        encoded_features = []
        
        # Add numeric features first
        numeric_df = df_processed[numeric_features]
        encoded_features.append(numeric_df)
        
        # Process each categorical feature
        for feature, values in expected_values.items():
            # Create dummy variables
            dummies = pd.get_dummies(df_processed[feature], prefix=feature)
            
            # Create columns for all expected values
            for value in values:
                col_name = f"{feature}_{value}"
                if col_name not in dummies.columns:
                    dummies[col_name] = 0
            
            # Only keep expected columns in the correct order
            expected_cols = [f"{feature}_{value}" for value in values]
            dummies = dummies[expected_cols]
            encoded_features.append(dummies)

        # Combine all features
        df_processed = pd.concat(encoded_features, axis=1)

        # Verify feature counts
        n_protocol = len(expected_values['protocol_type'])
        n_service = len(expected_values['service'])
        n_flag = len(expected_values['flag'])
        n_numeric = len(numeric_features)

        st.write("\nFeature count verification:")
        st.write(f"Numeric features: {n_numeric}")
        st.write(f"Protocol features: {n_protocol}")
        st.write(f"Service features: {n_service}")
        st.write(f"Flag features: {n_flag}")
        st.write(f"Total features: {df_processed.shape[1]}")
        
        # Verify shape
        if df_processed.shape[1] != 122:
            raise ValueError(f"Feature shape mismatch, expected: 122, got {df_processed.shape[1]}")

        return df_processed

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        if 'df_processed' in locals():
            st.write("Current features:", df_processed.columns.tolist())
        return None

# Main UI
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

# Sidebar
st.sidebar.info(f"""
    Session Info:
    - Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
    - User: shay-haan
""")

if models:
    # File upload section
    uploaded_file = st.file_uploader("Upload CSV file with network traffic data", type="csv")
    
    if uploaded_file:
        try:
            # Load and process data
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} records!")
            
            # Show data preview
            with st.expander("Preview Raw Data"):
                st.dataframe(df.head())
                st.info(f"Dataset Shape: {df.shape}")
            
            # Analysis button
            if st.button("Analyze Network Traffic"):
                with st.spinner("Processing data..."):
                    df_processed = preprocess_data(df)
                    
                    if df_processed is not None:
                        # Make predictions
                        results = {}
                        for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
                            model = models['xgboost_models'][attack_type]['model']
                            pred_proba = model.predict_proba(df_processed)
                            results[attack_type] = {
                                'probability': pred_proba[:, 1],
                                'prediction': pred_proba[:, 1] > 0.5
                            }
                        
                        # Display results
                        st.subheader("Analysis Results")
                        cols = st.columns(4)
                        for attack_type, col in zip(results.keys(), cols):
                            detected = results[attack_type]['prediction'].sum()
                            with col:
                                st.metric(
                                    f"{attack_type} Attacks",
                                    f"{detected}",
                                    f"{(detected/len(df)*100):.2f}%"
                                )
                        
                        # Detailed analysis
                        for attack_type in results:
                            with st.expander(f"{attack_type} Analysis"):
                                fig = px.histogram(
                                    x=results[attack_type]['probability'],
                                    title=f"{attack_type} Attack Probability Distribution"
                                )
                                st.plotly_chart(fig)
                                
                                high_risk = np.where(results[attack_type]['probability'] > 0.8)[0]
                                if len(high_risk) > 0:
                                    st.warning(f"Found {len(high_risk)} high-risk connections!")
                                    st.dataframe(df.iloc[high_risk])
                        
                        # Download results
                        if st.button("Download Results"):
                            for attack_type in results:
                                df[f'{attack_type}_probability'] = results[attack_type]['probability']
                                df[f'{attack_type}_prediction'] = results[attack_type]['prediction']
                            
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

# Model performance info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Model Performance
- DoS: 99.998%
- Probe: 100%
- R2L: 99.999%
- U2R: 100%

*Based on test dataset evaluation*
""")