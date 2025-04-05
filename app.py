import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle

# Set page config
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Define the exact categories used in NSL-KDD dataset
PROTOCOL_TYPES = ['tcp', 'udp', 'icmp']
SERVICES = [
    'http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet', 'ftp', 'eco_i', 'ntp_u',
    'ecr_i', 'other', 'private', 'pop_3', 'ftp_data', 'rje', 'time', 'mtp', 'link',
    'remote_job', 'gopher', 'ssh', 'name', 'whois', 'domain', 'login', 'imap4',
    'daytime', 'ctf', 'nntp', 'shell', 'IRC', 'nnsp', 'http_443', 'exec', 'printer',
    'efs', 'courier', 'uucp', 'klogin', 'kshell', 'echo', 'discard', 'systat',
    'supdup', 'iso_tsap', 'hostnames', 'csnet_ns', 'pop_2', 'sunrpc', 'uucp_path',
    'netbios_ns', 'netbios_ssn', 'netbios_dgm', 'sql_net', 'vmnet', 'bgp', 'Z39_50',
    'ldap', 'netstat', 'urh_i', 'X11', 'urp_i', 'pm_dump', 'tftp_u', 'tim_i', 
    'red_i'
]
FLAGS = ['SF', 'S1', 'REJ', 'S2', 'S0', 'S3', 'RSTO', 'RSTR', 'RSTOS0', 'OTH', 'SH']

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

        # Get numeric columns
        numeric_columns = [col for col in REQUIRED_COLUMNS if col not in ['protocol_type', 'service', 'flag']]
        
        # Create one-hot encodings for categorical columns
        protocol_dummies = pd.get_dummies(df['protocol_type'], prefix='protocol_type')
        service_dummies = pd.get_dummies(df['service'], prefix='service')
        flag_dummies = pd.get_dummies(df['flag'], prefix='flag')
        
        # Ensure all expected categories are present
        for protocol in PROTOCOL_TYPES:
            if f'protocol_type_{protocol}' not in protocol_dummies.columns:
                protocol_dummies[f'protocol_type_{protocol}'] = 0
                
        for service in SERVICES:
            if f'service_{service}' not in service_dummies.columns:
                service_dummies[f'service_{service}'] = 0
                
        for flag in FLAGS:
            if f'flag_{flag}' not in flag_dummies.columns:
                flag_dummies[f'flag_{flag}'] = 0
        
        # Convert numeric columns to float
        numeric_data = df[numeric_columns].astype(float)
        
        # Combine all features
        final_df = pd.concat([
            numeric_data,
            protocol_dummies[sorted([f'protocol_type_{p}' for p in PROTOCOL_TYPES])],
            service_dummies[sorted([f'service_{s}' for s in SERVICES])],
            flag_dummies[sorted([f'flag_{f}' for f in FLAGS])]
        ], axis=1)
        
        # Convert to numpy array
        X = final_df.values
        
        st.write(f"Feature vector shape: {X.shape}")
        
        # Verify we have 122 features
        if X.shape[1] != 122:
            st.error(f"Expected 122 features, but got {X.shape[1]}")
            return None
            
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

# ... rest of your code remains the same ...

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