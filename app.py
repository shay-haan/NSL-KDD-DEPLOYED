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
        # Load the exact feature order from training
        feature_order = [
            # Numeric features first
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
            'dst_host_srv_rerror_rate',
            # Protocol features
            'protocol_type_icmp', 'protocol_type_tcp', 'protocol_type_udp',
            # Service features
            'service_IRC', 'service_X11', 'service_Z39_50', 'service_auth', 'service_bgp',
            'service_courier', 'service_csnet_ns', 'service_ctf', 'service_daytime',
            'service_discard', 'service_domain', 'service_domain_u', 'service_echo',
            'service_eco_i', 'service_ecr_i', 'service_efs', 'service_exec',
            'service_finger', 'service_ftp', 'service_ftp_data', 'service_gopher',
            'service_hostnames', 'service_http', 'service_http_443', 'service_http_8001',
            'service_http_2784', 'service_imap4', 'service_iso_tsap', 'service_klogin',
            'service_kshell', 'service_ldap', 'service_link', 'service_login',
            'service_mtp', 'service_name', 'service_netbios_dgm', 'service_netbios_ns',
            'service_netbios_ssn', 'service_netstat', 'service_nnsp', 'service_nntp',
            'service_ntp_u', 'service_other', 'service_pm_dump', 'service_pop_2',
            'service_pop_3', 'service_printer', 'service_private', 'service_red_i',
            'service_remote_job', 'service_rje', 'service_shell', 'service_smtp',
            'service_sql_net', 'service_ssh', 'service_sunrpc', 'service_supdup',
            'service_systat', 'service_telnet', 'service_tftp_u', 'service_tim_i',
            'service_time', 'service_urh_i', 'service_urp_i', 'service_uucp',
            'service_uucp_path', 'service_vmnet', 'service_whois',
            # Flag features
            'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR',
            'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH'
        ]

        # Create a copy of the input dataframe
        df_processed = df.copy()

        # Convert numeric features
        numeric_features = feature_order[:38]  # First 38 features are numeric
        for col in numeric_features:
            if col not in df_processed.columns:
                df_processed[col] = 0
            else:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        # Create categorical features
        # Protocol type
        protocol_dummies = pd.get_dummies(df_processed['protocol_type'], prefix='protocol_type')
        protocol_features = feature_order[38:41]  # Protocol features
        for col in protocol_features:
            if col not in protocol_dummies.columns:
                protocol_dummies[col] = 0

        # Service
        service_dummies = pd.get_dummies(df_processed['service'], prefix='service')
        service_features = feature_order[41:111]  # Service features
        for col in service_features:
            if col not in service_dummies.columns:
                service_dummies[col] = 0

        # Flag
        flag_dummies = pd.get_dummies(df_processed['flag'], prefix='flag')
        flag_features = feature_order[111:]  # Flag features
        for col in flag_features:
            if col not in flag_dummies.columns:
                flag_dummies[col] = 0

        # Combine all features in exact order
        df_processed = pd.concat([
            df_processed[numeric_features],
            protocol_dummies[protocol_features],
            service_dummies[service_features],
            flag_dummies[flag_features]
        ], axis=1)

        # Verify we have all features in correct order
        missing_features = set(feature_order) - set(df_processed.columns)
        extra_features = set(df_processed.columns) - set(feature_order)

        if missing_features:
            st.warning(f"Missing features: {missing_features}")
        if extra_features:
            st.warning(f"Extra features: {extra_features}")

        # Add debug information
        st.write("\nFeature Statistics:")
        st.write("Sample values for first 5 rows:")
        st.write(df_processed.head())
        
        st.write("\nFeature value ranges:")
        st.write(df_processed.describe())

        # Verify final shape
        if df_processed.shape[1] != 122:
            raise ValueError(f"Feature shape mismatch, expected: 122, got {df_processed.shape[1]}")

        return df_processed[feature_order]  # Ensure exact feature order

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        if 'df_processed' in locals():
            st.write("Current features:", df_processed.columns.tolist())
        return None

# In the main app, add more debugging for predictions
if df_processed is not None:
    # Make predictions
    results = {}
    for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
        model = models['xgboost_models'][attack_type]['model']
        
        # Add debugging info
        st.write(f"\nPredicting {attack_type}:")
        st.write(f"Model feature count: {model.n_features_in_}")
        st.write(f"Input feature count: {df_processed.shape[1]}")
        
        # Make prediction
        pred_proba = model.predict_proba(df_processed)
        
        # Show probability distribution
        st.write(f"{attack_type} probability distribution:")
        st.write(pd.DataFrame(pred_proba).describe())
        
        results[attack_type] = {
            'probability': pred_proba[:, 1],
            'prediction': pred_proba[:, 1] > 0.5
        }

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