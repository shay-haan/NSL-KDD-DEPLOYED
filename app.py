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

# Main title
st.title("🔒 Network Intrusion Detection System")
st.markdown("""
### Research Project - Final Semester
This system uses ensemble machine learning to detect network attacks:
- Denial of Service (DoS)
- Probe Attacks
- Remote to Local (R2L)
- User to Root (U2R)
""")

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

def preprocess_data(df, models):
    """Preprocess data exactly as we did during training"""
    try:
        # Get the exact feature names from our models
        feature_names = models['feature_names']
        st.write("Number of features expected:", len(feature_names))
        
        # Work on a copy of the dataframe
        df_processed = df.copy()
        
        # Define base numeric features (these are the original numeric columns)
        base_numeric_features = [
            "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", 
            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
            "root_shell", "su_attempted", "num_root", "num_file_creations",
            "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
            "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
            "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]
        
        # Keep only numeric features initially
        numeric_df = df_processed[base_numeric_features].copy()
        
        # Add binary target columns (initialized to 0)
        for target in ['DoS', 'Probe', 'R2L', 'U2R']:
            numeric_df[target] = 0
            
        # Now handle categorical features
        # IMPORTANT: We only use the categories that were in the training data
        categorical_features = {
            'protocol_type': ['icmp', 'tcp', 'udp'],
            'service': [
                'IRC', 'X11', 'Z39_50', 'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 
                'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 
                'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest',
                'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 
                'ISO_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name',
                'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp',
                'ntp_u', 'other', 'pop_2', 'pop_3', 'printer', 'private', 'remote_job',
                'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat',
                'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 
                'uucp_path', 'vmnet', 'whois'
            ],
            'flag': [
                'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 
                'SF', 'SH'
            ]
        }
        
        # Create dummy variables for each categorical feature
        for feature, categories in categorical_features.items():
            # Create dummy columns for each known category
            for category in categories:
                col_name = f"{feature}_{category}"
                numeric_df[col_name] = (df_processed[feature] == category).astype(int)
        
        # Ensure we have exactly the features the model expects
        final_features = numeric_df.reindex(columns=feature_names, fill_value=0)
        
        st.write("Number of features after preprocessing:", len(final_features.columns))
        st.write("Features match expected:", set(final_features.columns) == set(feature_names))
        
        return final_features
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Columns in input data:", df.columns.tolist())
        st.write("Number of columns in input:", len(df.columns))
        return None

# Load models first!
models = load_models()

# Sidebar: Session info
st.sidebar.info(f"""
    Session Info:
    - Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
    - User: shay-haan
""")

if models:
    # Display expected feature info in an expander
    with st.expander("📋 Feature Information"):
        st.write("Required numeric features:")
        numeric_features = [f for f in models['feature_names'] 
                          if not any(x in f for x in ['protocol_type_', 'service_', 'flag_'])]
        st.write(", ".join(numeric_features))
        
        st.write("\nAccepted categorical values:")
        st.write("protocol_type:", [f.replace('protocol_type_', '') 
                                  for f in models['feature_names'] if 'protocol_type_' in f])
        st.write("service:", [f.replace('service_', '') 
                            for f in models['feature_names'] if 'service_' in f])
        st.write("flag:", [f.replace('flag_', '') 
                         for f in models['feature_names'] if 'flag_' in f])

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file with network traffic data",
        type="csv"
    )

    if uploaded_file:
        try:
            # Load data from CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} records!")
            
            # Data Preview
            with st.expander("Preview Raw Data"):
                st.dataframe(df.head())
                st.info(f"Dataset Shape: {df.shape}")
            
            # Analyze data when button is clicked
            if st.button("Analyze Network Traffic"):
                with st.spinner("Processing data..."):
                    df_processed = preprocess_data(df, models)
                    
                    if df_processed is not None:
                        # Get predictions for each attack type
                        results = {}
                        for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
                            scaler = models['scalers'][attack_type]
                            X_scaled = scaler.transform(df_processed)
                            
                            ensemble_pred = models['ensemble_models'][attack_type].predict_proba(X_scaled)
                            results[attack_type] = {
                                'probability': ensemble_pred[:, 1],
                                'prediction': ensemble_pred[:, 1] > 0.5
                            }
                        
                        # Display overall analysis results
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
                        
                        # Detailed analysis with histograms and high-risk connections
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
                        
                        # Option to download analysis results
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

# Sidebar Footer: Model performance
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Model Performance
- DoS: 99.998%
- Probe: 100%
- R2L: 99.999%
- U2R: 100%

*Based on test dataset evaluation*
""")