import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import pickle
import os
import io

# Set page config
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Define required columns
REQUIRED_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot", 
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

@st.cache_resource
def load_models():
    try:
        models = {
            'xgboost': {
                'DoS': pickle.load(open('models/xgb_DoS.pkl', 'rb')),
                'Probe': pickle.load(open('models/xgb_Probe.pkl', 'rb')),
                'R2L': pickle.load(open('models/xgb_R2L.pkl', 'rb')),
                'U2R': pickle.load(open('models/xgb_U2R.pkl', 'rb'))
            },
            'logistic': {
                'DoS': pickle.load(open('models/lr_DoS.pkl', 'rb')),
                'Probe': pickle.load(open('models/lr_Probe.pkl', 'rb')),
                'R2L': pickle.load(open('models/lr_R2L.pkl', 'rb')),
                'U2R': pickle.load(open('models/lr_U2R.pkl', 'rb'))
            },
            'ensemble': {
                'DoS': pickle.load(open('models/ensemble_DoS.pkl', 'rb')),
                'Probe': pickle.load(open('models/ensemble_Probe.pkl', 'rb')),
                'R2L': pickle.load(open('models/ensemble_R2L.pkl', 'rb')),
                'U2R': pickle.load(open('models/ensemble_U2R.pkl', 'rb'))
            }
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def preprocess_data(df):
    # Check if all required columns are present
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {', '.join(missing_cols)}")
        return None

    # Categorical columns
    categorical_columns = ['protocol_type', 'service', 'flag']
    
    # Get categorical values
    df_categorical_values = df[categorical_columns]
    
    # Label Encoding
    df_categorical_values_enc = df_categorical_values.apply(LabelEncoder().fit_transform)
    
    # One-Hot Encoding
    enc = OneHotEncoder(categories='auto', sparse=False)
    df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
    
    # Get numeric columns
    numeric_columns = [col for col in REQUIRED_COLUMNS if col not in categorical_columns]
    df_numeric_values = df[numeric_columns].astype(float)
    
    # Combine numeric and categorical features
    X = np.hstack((df_numeric_values, df_categorical_values_encenc))
    
    return X

def make_predictions(X, models):
    results = {}
    
    for model_type in ['xgboost', 'logistic', 'ensemble']:
        results[model_type] = {}
        for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
            try:
                predictions = models[model_type][attack_type].predict(X)
                attack_count = np.sum(predictions == 1)
                attack_percentage = (attack_count / len(predictions)) * 100
                results[model_type][attack_type] = {
                    'count': int(attack_count),
                    'percentage': float(attack_percentage),
                    'predictions': predictions
                }
            except Exception as e:
                st.error(f"Error in {model_type} model for {attack_type}: {str(e)}")
                return None
    
    return results

def display_results(results, df):
    if results is None:
        return

    st.write("## Detection Results")
    
    # Create tabs for different models
    tabs = st.tabs(['XGBoost', 'Logistic Regression', 'Ensemble'])
    
    for tab, model_type in zip(tabs, ['xgboost', 'logistic', 'ensemble']):
        with tab:
            st.write(f"### {model_type.upper()} Model Results")
            
            # Create columns for different attack types
            cols = st.columns(4)
            
            for col, attack_type in zip(cols, ['DoS', 'Probe', 'R2L', 'U2R']):
                with col:
                    result = results[model_type][attack_type]
                    st.metric(
                        label=f"{attack_type} Attacks",
                        value=f"{result['count']}",
                        delta=f"{result['percentage']:.2f}%"
                    )
            
            # Create detailed results DataFrame
            results_df = pd.DataFrame({
                'DoS': results[model_type]['DoS']['predictions'],
                'Probe': results[model_type]['Probe']['predictions'],
                'R2L': results[model_type]['R2L']['predictions'],
                'U2R': results[model_type]['U2R']['predictions']
            })
            
            # Combine with original features
            detailed_results = pd.concat([df, results_df], axis=1)
            
            st.write("### Detailed Results")
            st.dataframe(detailed_results)
            
            # Download button for results
            csv = detailed_results.to_csv(index=False)
            st.download_button(
                label=f"Download {model_type.upper()} results as CSV",
                data=csv,
                file_name=f'{model_type}_detection_results.csv',
                mime='text/csv'
            )

def main():
    st.title('Network Intrusion Detection System')
    st.write('Upload your test dataset to detect DoS, Probe, R2L, and U2R attacks')
    
    # Load models
    models = load_models()
    if models is None:
        st.error("Failed to load models. Please check if model files exist in the 'models' directory.")
        return
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview")
            st.write(df.head())
            
            X = preprocess_data(df)
            if X is not None:
                with st.spinner('Making predictions...'):
                    results = make_predictions(X, models)
                    display_results(results, df)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please make sure your file is in the correct format.")

if __name__ == "__main__":
    main()