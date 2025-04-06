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
        
        # Drop label column from features if it exists (it's not used for prediction)
        if 'label' in df_processed.columns:
            df_processed.drop('label', axis=1, inplace=True)
            
        # Create binary target columns from label (these will be dropped later)
        attack_types = {
            'DoS': ['neptune', 'smurf', 'pod', 'teardrop', 'land', 'back', 'apache2', 'udpstorm', 'processtable', 'mailbomb'],
            'Probe': ['ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint'],
            'R2L': ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named'],
            'U2R': ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'sqlattack', 'xterm', 'ps']
        }
        
        # Extract categorical columns
        categorical_columns = ['protocol_type', 'service', 'flag']
        
        # Get the mapping dictionaries for categorical features
        protocol_types = sorted(list(set(x.replace('protocol_type_', '') 
                              for x in feature_names if x.startswith('protocol_type_'))))
        services = sorted(list(set(x.replace('service_', '') 
                        for x in feature_names if x.startswith('service_'))))
        flags = sorted(list(set(x.replace('flag_', '') 
                     for x in feature_names if x.startswith('flag_'))))
        
        # Create dummy variables with fixed categories
        all_dummies = pd.DataFrame()
        for col, categories, prefix in [
            ('protocol_type', protocol_types, 'protocol_type_'),
            ('service', services, 'service_'),
            ('flag', flags, 'flag_')
        ]:
            # Create dummies with a fixed set of columns
            dummies = pd.get_dummies(df_processed[col], prefix=prefix)
            
            # Ensure all expected categories exist
            for cat in categories:
                col_name = f"{prefix}{cat}"
                if col_name not in dummies.columns:
                    dummies[col_name] = 0
            
            # Only keep the expected categories and in the right order
            expected_cols = [f"{prefix}{cat}" for cat in categories]
            dummies = dummies.reindex(columns=expected_cols, fill_value=0)
            
            all_dummies = pd.concat([all_dummies, dummies], axis=1)
        
        # Drop original categorical columns
        df_processed = df_processed.drop(categorical_columns, axis=1)
        
        # Add the dummy variables
        df_processed = pd.concat([df_processed, all_dummies], axis=1)
        
        # Keep only numeric features that are in the feature_names list
        numeric_features = [col for col in feature_names 
                          if not any(x in col for x in ['protocol_type_', 'service_', 'flag_'])]
        
        # Create final dataframe with exact features in exact order
        final_features = df_processed[feature_names]
        
        st.write("Number of features after preprocessing:", len(final_features.columns))
        
        return final_features
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please check your CSV file format.")
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