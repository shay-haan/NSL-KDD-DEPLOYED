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
        
        # Extract categories from the training feature names
        protocol_types = [col.replace('protocol_type_', '') for col in feature_names if 'protocol_type_' in col]
        services = [col.replace('service_', '') for col in feature_names if 'service_' in col]
        flags = [col.replace('flag_', '') for col in feature_names if 'flag_' in col]
        
        # Create dummy variables only for the categories seen during training
        protocol_dummies = pd.get_dummies(df_processed['protocol_type'], prefix='protocol_type')
        service_dummies = pd.get_dummies(df_processed['service'], prefix='service')
        flag_dummies = pd.get_dummies(df_processed['flag'], prefix='flag')
        
        # Retain only the desired columns (if a column is missing, add it later with zeros)
        protocol_cols = ['protocol_type_' + p for p in protocol_types]
        service_cols = ['service_' + s for s in services]
        flag_cols = ['flag_' + f for f in flags]
        
        protocol_dummies = protocol_dummies.reindex(columns=protocol_cols, fill_value=0)
        service_dummies = service_dummies.reindex(columns=service_cols, fill_value=0)
        flag_dummies = flag_dummies.reindex(columns=flag_cols, fill_value=0)
        
        # Drop original categorical columns
        df_processed = df_processed.drop(['protocol_type', 'service', 'flag'], axis=1)
        
        # Concatenate dummy variables with the dataset
        df_processed = pd.concat([df_processed, protocol_dummies, service_dummies, flag_dummies], axis=1)
        
        # Ensure all required columns exist; if missing, add with zeros
        for col in feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # Select only the required features in the correct order
        df_processed = df_processed[feature_names]
        
        st.write("Expected feature names:", feature_names)
        st.write("Processed feature names:", df_processed.columns.tolist())
        st.write("Number of features after preprocessing:", df_processed.shape[1])
        
        return df_processed
        
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        st.write("Columns in input data:", df.columns.tolist())
        st.write("Expected features:", feature_names)
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