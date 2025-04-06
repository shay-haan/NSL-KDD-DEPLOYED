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
        # Define expected values from training - EXACT list of 70 services
        expected_values = {
            'protocol_type': ['icmp', 'tcp', 'udp'],
            'service': [
                'IRC', 'X11', 'Z39_50', 'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 
                'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 
                'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 
                'hostnames', 'http', 'http_443', 'http_2784', 'http_8001', 'imap4', 
                'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 
                'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat',
                'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 
                'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp',
                'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u',
                'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet',
                'whois', 'X11', 'Z39_50', 'aol', 'harvest'  # Added missing service
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

        # Create copy of input data
        df_processed = df.copy()

        # Handle numeric features
        for col in numeric_features:
            if col not in df_processed.columns:
                df_processed[col] = 0
            else:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        # Process categorical features
        for feature, values in expected_values.items():
            # Create dummy variables
            dummies = pd.get_dummies(df_processed[feature], prefix=feature)
            
            # Add missing columns with zeros
            for value in values:
                col_name = f"{feature}_{value}"
                if col_name not in dummies.columns:
                    dummies[col_name] = 0
            
            # Keep only expected columns in correct order
            expected_cols = [f"{feature}_{value}" for value in values]
            dummies = dummies[expected_cols]
            
            # Add to processed dataframe
            df_processed = pd.concat([df_processed, dummies], axis=1)

        # Drop original categorical columns
        df_processed = df_processed.drop(['protocol_type', 'service', 'flag'], axis=1)

        # Create final feature order
        final_features = (
            numeric_features +
            [f'protocol_type_{p}' for p in expected_values['protocol_type']] +
            [f'service_{s}' for s in expected_values['service']] +
            [f'flag_{f}' for f in expected_values['flag']]
        )

        # Select features in correct order
        df_processed = df_processed[final_features]

        # Print verification info
        st.write("\nFeature count verification:")
        st.write(f"Numeric features: {len(numeric_features)}")
        st.write(f"Protocol features: {len(expected_values['protocol_type'])}")
        st.write(f"Service features: {len(expected_values['service'])}")
        st.write(f"Flag features: {len(expected_values['flag'])}")
        st.write(f"Total features: {df_processed.shape[1]}")

        return df_processed

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def make_predictions(df_processed, models, model_type):
    """Make predictions using specified model type"""
    results = {}
    for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
        try:
            # Get model and make predictions
            if model_type == 'xgboost':
                model = models['xgboost_models'][attack_type]['model']
            else:
                model = models[f'{model_type}_models'][attack_type]
            
            # Make predictions
            pred_proba = model.predict_proba(df_processed)
            
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
    """Display comparison of model predictions"""
    st.subheader("Model Comparison")
    
    # Create columns for each attack type
    cols = st.columns(4)
    
    for attack_type, col in zip(['DoS', 'Probe', 'R2L', 'U2R'], cols):
        with col:
            st.write(f"**{attack_type} Attack Detection**")
            
            # Compare predictions from each model
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
            
            # Display comparison table
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                # Create bar chart
                fig = px.bar(
                    comparison_df,
                    x='Model',
                    y='Percentage',
                    title=f'{attack_type} Detection Rate'
                )
                st.plotly_chart(fig, use_container_width=True)

# Main app code
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

# Sidebar info
st.sidebar.info(f"""
Session Info:
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
- User: shay-haan
""")

if models:
    uploaded_file = st.file_uploader("Upload CSV file with network traffic data", type="csv")
# After loading models
if models:
    st.write("Model Structure Verification:")
    for model_type in ['xgboost_models', 'logistic_models', 'ensemble_models']:
        st.write(f"\n{model_type} structure:")
        if model_type in models:
            for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
                if attack_type in models[model_type]:
                    if model_type == 'xgboost_models':
                        st.write(f"  {attack_type}: {type(models[model_type][attack_type]['model'])}")
                    else:
                        st.write(f"  {attack_type}: {type(models[model_type][attack_type])}")
    
    if uploaded_file:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} records!")
            
            # Show data preview
            with st.expander("Preview Raw Data"):
                st.dataframe(df.head())
                st.info(f"Dataset Shape: {df.shape}")
            
            # Analyze button
            if st.button("Analyze Network Traffic"):
                with st.spinner("Processing data..."):
                    # Preprocess data
                    df_processed = preprocess_data(df)
                    
                    if df_processed is not None:
                        # Get predictions from all models
                        results = {
                            'xgboost': make_predictions(df_processed, models, 'xgboost'),
                            'logistic': make_predictions(df_processed, models, 'logistic'),
                            'ensemble': make_predictions(df_processed, models, 'ensemble')
                        }
                        
                        # Display results comparison
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
                                
                                # Show high-risk connections
                                ensemble_result = results['ensemble'][attack_type]
                                if ensemble_result is not None:
                                    high_risk = np.where(ensemble_result['probability'] > 0.8)[0]
                                    if len(high_risk) > 0:
                                        st.warning(f"Found {len(high_risk)} high-risk connections!")
                                        st.dataframe(df.iloc[high_risk])
                        
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

# Model performance info
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