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
st.write('Upload your test dataset to detect DoS, Probe, R2L, and U2R attacks')

# Load the trained models
@st.cache_resource
def load_models():
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

def preprocess_data(df):
    # Categorical columns
    categorical_columns = ['protocol_type', 'service', 'flag']
    
    # Get categorical values
    df_categorical_values = df[categorical_columns]
    
    # Label Encoding
    df_categorical_values_enc = df_categorical_values.apply(LabelEncoder().fit_transform)
    
    # One-Hot Encoding
    enc = OneHotEncoder(categories='auto')
    df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
    
    # Get numeric columns (excluding 'label' if it exists)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = numeric_columns[numeric_columns != 'label']
    df_numeric_values = df[numeric_columns]
    
    # Combine numeric and categorical features
    X = np.hstack((df_numeric_values, df_categorical_values_encenc.toarray()))
    
    return X

def main():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Preprocess the data
        X = preprocess_data(df)
        
        # Load models
        models = load_models()
        
        # Make predictions
        predictions = {}
        
        for model_type in ['xgboost', 'logistic', 'ensemble']:
            st.subheader(f"{model_type.upper()} Predictions")
            
            for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
                predictions[f"{model_type}_{attack_type}"] = models[model_type][attack_type].predict(X)
                
                # Calculate statistics
                attack_count = np.sum(predictions[f"{model_type}_{attack_type}"] == 1)
                total_records = len(predictions[f"{model_type}_{attack_type}"])
                
                # Display results
                st.write(f"{attack_type} Attacks Detected: {attack_count} ({(attack_count/total_records)*100:.2f}%)")
            
            st.write("---")
        
        # Download predictions
        for model_type in ['xgboost', 'logistic', 'ensemble']:
            combined_predictions = pd.DataFrame({
                'DoS': predictions[f"{model_type}_DoS"],
                'Probe': predictions[f"{model_type}_Probe"],
                'R2L': predictions[f"{model_type}_R2L"],
                'U2R': predictions[f"{model_type}_U2R"]
            })
            
            # Convert predictions to CSV
            csv = combined_predictions.to_csv(index=False)
            
            # Create download button
            st.download_button(
                label=f"Download {model_type.upper()} predictions as CSV",
                data=csv,
                file_name=f'{model_type}_predictions.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()