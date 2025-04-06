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
        
        # Separate numeric and categorical features from feature_names
        numeric_features = []
        categorical_features = {'protocol_type': [], 'service': [], 'flag': []}
        
        for feature in feature_names:
            if any(feature.startswith(f"{cat}_") for cat in categorical_features.keys()):
                for cat in categorical_features:
                    if feature.startswith(f"{cat}_"):
                        categorical_features[cat].append(feature.replace(f"{cat}_", ""))
                        break
            else:
                numeric_features.append(feature)
        
        # Create dummy variables for categorical features
        for cat_feature, values in categorical_features.items():
            # Create dummy variables
            dummies = pd.get_dummies(df_processed[cat_feature], prefix=cat_feature)
            
            # Add missing columns with zeros
            for value in values:
                col_name = f"{cat_feature}_{value}"
                if col_name not in dummies.columns:
                    dummies[col_name] = 0
            
            # Keep only the columns we expect
            expected_cols = [f"{cat_feature}_{value}" for value in values]
            dummies = dummies[expected_cols]
            
            # Add to processed dataframe
            for col in dummies.columns:
                df_processed[col] = dummies[col]
        
        # Select only the features we need in the correct order
        df_processed = df_processed[feature_names]
        
        st.write("Number of features after preprocessing:", df_processed.shape[1])
        return df_processed
        
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
    # Add a demo data option
    data_option = st.radio(
        "Choose data source:",
        ["Upload CSV", "Use NSL-KDD Test Data"]
    )

    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV file with network traffic data",
            type="csv"
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
    else:
        # Load NSL-KDD test data
        test_url = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Test.csv'
        col_names = ["duration","protocol_type","service","flag","src_bytes",
                    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
                    "logged_in","num_compromised","root_shell","su_attempted","num_root",
                    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
                    "is_host_login","is_guest_login","count","srv_count","serror_rate",
                    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
                    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
                    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
                    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
        df = pd.read_csv(test_url, header=None, names=col_names)
        st.success("Loaded NSL-KDD test dataset!")

    if 'df' in locals():
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
else:
    st.error("⚠️ Could not load models. Please ensure 'all_models.pkl' exists in the app directory.")

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Model Performance
- DoS: 99.998%
- Probe: 100%
- R2L: 99.999%
- U2R: 100%

*Based on test dataset evaluation*
""")