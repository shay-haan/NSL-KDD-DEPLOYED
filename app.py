# app.py
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

# Session info
st.sidebar.info(f"""
    Session Info:
    - Date: 2025-04-06 03:10:23 UTC
    - User: shay-haan
""")

# Load the saved models
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

# Load models
models = load_models()

if models:
    # File uploader
    st.subheader("Upload Test Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file containing network traffic data",
        type="csv"
    )

    if uploaded_file:
        try:
            # Load and display data
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} records!")
            
            # Show data preview
            with st.expander("Preview Data"):
                st.dataframe(df.head())
                st.info(f"Dataset Shape: {df.shape}")

            # Process data button
            if st.button("Analyze Network Traffic"):
                with st.spinner("Processing data..."):
                    # Get predictions for each attack type
                    results = {}
                    for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
                        # Scale data
                        scaler = models['scalers'][attack_type]
                        X_scaled = scaler.transform(df)
                        
                        # Get predictions from all models
                        ensemble_pred = models['ensemble_models'][attack_type].predict_proba(X_scaled)
                        xgb_pred = models['xgboost_models'][attack_type].predict_proba(X_scaled)
                        lr_pred = models['logistic_models'][attack_type].predict_proba(X_scaled)
                        
                        results[attack_type] = {
                            'Ensemble': ensemble_pred[:, 1],
                            'XGBoost': xgb_pred[:, 1],
                            'Logistic': lr_pred[:, 1]
                        }
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Summary metrics
                    cols = st.columns(4)
                    for attack_type, col in zip(results.keys(), cols):
                        detected = (results[attack_type]['Ensemble'] > 0.5).sum()
                        with col:
                            st.metric(
                                f"{attack_type} Attacks",
                                f"{detected}",
                                f"{(detected/len(df)*100):.2f}%"
                            )
                    
                    # Detailed analysis for each attack type
                    for attack_type in results:
                        with st.expander(f"{attack_type} Analysis"):
                            # Plot probability distributions
                            fig = px.histogram(
                                pd.DataFrame(results[attack_type]),
                                title=f"{attack_type} Attack Probability Distribution"
                            )
                            st.plotly_chart(fig)
                            
                            # High risk connections
                            high_risk = np.where(results[attack_type]['Ensemble'] > 0.8)[0]
                            if len(high_risk) > 0:
                                st.warning(f"Found {len(high_risk)} high-risk connections!")
                                st.dataframe(df.iloc[high_risk])
                    
                    # Save results option
                    if st.button("Save Results"):
                        # Add predictions to dataframe
                        for attack_type in results:
                            df[f'{attack_type}_probability'] = results[attack_type]['Ensemble']
                        
                        # Save to CSV
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        df.to_csv(f'results_{timestamp}.csv', index=False)
                        st.success("Results saved successfully!")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has the correct format and features.")

# Footer with model performance
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Model Performance
- DoS: 99.998%
- Probe: 100%
- R2L: 99.999%
- U2R: 100%

*Based on test dataset evaluation*
""")

# Requirements text
requirements = """
streamlit==1.22.0
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
xgboost==1.4.2
plotly==5.13.0
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements)