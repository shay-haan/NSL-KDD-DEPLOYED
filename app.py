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

def preprocess_data(df):
    # Check if all required columns are present
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.error(f"The following required columns are missing from your dataset: {', '.join(missing_cols)}")
        st.write("Your dataset columns:", ", ".join(df.columns))
        st.write("""
        Please ensure your dataset has all required columns. 
        The expected format is the NSL-KDD dataset format with the following columns:
        """)
        st.write(", ".join(REQUIRED_COLUMNS))
        return None

    # Categorical columns
    categorical_columns = ['protocol_type', 'service', 'flag']
    
    # Get categorical values
    df_categorical_values = df[categorical_columns]
    
    # Label Encoding
    df_categorical_values_enc = df_categorical_values.apply(LabelEncoder().fit_transform)
    
    # One-Hot Encoding
    enc = OneHotEncoder(categories='auto')
    df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
    
    # Get numeric columns
    numeric_columns = [col for col in REQUIRED_COLUMNS if col not in categorical_columns]
    df_numeric_values = df[numeric_columns]
    
    # Combine numeric and categorical features
    X = np.hstack((df_numeric_values, df_categorical_values_encenc.toarray()))
    
    return X

def main():
    st.title('Network Intrusion Detection System')
    st.write('Upload your test dataset to detect DoS, Probe, R2L, and U2R attacks')
    
    # Add file format instructions
    st.write("""
    ### Required File Format
    Your CSV file should contain the following columns:
    """)
    st.write(", ".join(REQUIRED_COLUMNS))
    
    # Example dataset download
    st.write("""
    ### Need an example?
    Download a sample dataset in the correct format:
    """)
    
    # Create a small sample dataset
    sample_data = pd.DataFrame(columns=REQUIRED_COLUMNS)
    sample_data.loc[0] = [0, 'tcp', 'http', 'SF', 181, 5450, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 1, 0, 0, 9, 9, 1, 0, 0.11, 0, 0, 0, 0, 0]
    
    # Convert sample data to CSV
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="Download Sample Dataset",
        data=csv,
        file_name="sample_dataset.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.write(df.head())
            
            X = preprocess_data(df)
            if X is None:
                return
            
            # Load models and continue with predictions...
            # [Rest of your prediction code here]
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please make sure your file is in the correct format.")

if __name__ == "__main__":
    main()