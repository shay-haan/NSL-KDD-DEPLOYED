import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import pickle
import os

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
    try:
        # Check if all required columns are present
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
            return None

        # Categorical columns
        categorical_columns = ['protocol_type', 'service', 'flag']
        
        # Get numeric columns
        numeric_columns = [col for col in REQUIRED_COLUMNS if col not in categorical_columns]
        
        # Convert numeric columns to float
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Create separate DataFrame for categorical data
        df_categorical = df[categorical_columns].copy()
        
        # Initialize label encoders dictionary
        label_encoders = {}
        
        # Apply Label Encoding to each categorical column
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            df_categorical[column] = label_encoders[column].fit_transform(df_categorical[column])
        
        # Convert to numpy array for OneHotEncoder
        categorical_values = df_categorical.values
        
        # Apply OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse_output=False)
        categorical_encoded = onehot_encoder.fit_transform(categorical_values)
        
        # Combine numeric and encoded categorical features
        numeric_values = df[numeric_columns].values
        X = np.hstack((numeric_values, categorical_encoded))
        
        st.write(f"Feature vector shape: {X.shape}")
        
        return X

    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def load_models():
    try:
        models = {
            'xgboost': {
                'DoS': pickle.load(open('models/xgb_DoS.pkl', 'rb')),
                'Probe': pickle.load(open('models/xgb_Probe.pkl', 'rb')),
                'R2L': pickle.load(open('models/xgb_R2L.pkl', 'rb')),
                'U2R': pickle.load(open('models/xgb_U2R.pkl', 'rb'))
            }
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def make_predictions(X, models):
    results = {}
    
    for model_type in ['xgboost']:
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

def main():
    st.title('Network Intrusion Detection System')
    st.write('Upload your test dataset to detect network intrusions')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Show data preview
            st.write("### Dataset Preview")
            st.write(df.head())
            
            # Preprocess data
            X = preprocess_data(df)
            
            if X is not None:
                # Load models
                models = load_models()
                
                if models is not None:
                    # Make predictions
                    with st.spinner('Making predictions...'):
                        results = make_predictions(X, models)
                        
                        if results is not None:
                            st.write("### Detection Results")
                            
                            for model_type in results:
                                st.write(f"#### {model_type.upper()} Model Results")
                                
                                cols = st.columns(4)
                                for col, (attack_type, result) in zip(cols, results[model_type].items()):
                                    with col:
                                        st.metric(
                                            label=f"{attack_type} Attacks",
                                            value=f"{result['count']}",
                                            delta=f"{result['percentage']:.2f}%"
                                        )
                                
                                # Create detailed results DataFrame
                                detailed_results = df.copy()
                                for attack_type, result in results[model_type].items():
                                    detailed_results[f'{attack_type}_prediction'] = result['predictions']
                                
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
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()