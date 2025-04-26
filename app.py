# Standard library imports
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from datetime import datetime
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Network IDS Research Project",
    page_icon="🔒",
    layout="wide"
)

# ------------------------------
# Load Models and Preprocessing Objects
# ------------------------------
@st.cache_resource
def load_resources():
    """Load all saved models and preprocessing objects"""
    resources = {}
    try:
        # Load Preprocessing Objects
        if os.path.exists('label_encoders.pkl'):
            with open('label_encoders.pkl', 'rb') as f:
                resources['label_encoders'] = pickle.load(f)
            st.sidebar.success("Label Encoders loaded.")
        else:
            st.sidebar.error("label_encoders.pkl not found!")
            return None

        if os.path.exists('onehot_encoder.pkl'):
            with open('onehot_encoder.pkl', 'rb') as f:
                # Expecting a dictionary {'encoder': enc, 'feature_names': feature_names}
                ohe_data = pickle.load(f)
                resources['onehot_encoder'] = ohe_data['encoder']
                resources['ohe_feature_names'] = ohe_data['feature_names']
            st.sidebar.success("OneHot Encoder loaded.")
        else:
            st.sidebar.error("onehot_encoder.pkl not found!")
            return None

        if os.path.exists('column_order.pkl'):
            with open('column_order.pkl', 'rb') as f:
                resources['column_order'] = pickle.load(f)
            st.sidebar.success("Column Order loaded.")
        else:
            st.sidebar.error("column_order.pkl not found!")
            return None

        # Load Models
        model_path = 'models/all_models.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                resources['models'] = pickle.load(f)
            st.sidebar.success("Models loaded successfully!")
        else:
            st.sidebar.error(f"{model_path} not found!")
            return None
            
        # --- Calculate expected feature count ---
        # Start with the full column order from training
        expected_cols = resources['column_order'].copy()
        # Remove the original label column and the derived attack type columns
        cols_to_remove = ['label', 'DoS', 'Probe', 'R2L', 'U2R']
        # Also remove original categorical columns that were replaced by OHE
        cols_to_remove.extend(['protocol_type', 'service', 'flag'])
        
        final_feature_columns = [col for col in expected_cols if col not in cols_to_remove]
        resources['final_feature_columns'] = final_feature_columns
        resources['expected_feature_count'] = len(final_feature_columns)
        st.sidebar.info(f"Expected model input features: {resources['expected_feature_count']}")
        # -----------------------------------------

        return resources

    except FileNotFoundError as e:
        st.error(f"Error loading resources: {str(e)}. Make sure all .pkl files are in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred loading resources: {str(e)}")
        return None

# ------------------------------
# Preprocessing Function
# ------------------------------
def preprocess_data(df_raw, resources):
    """Preprocess raw data to match the training pipeline using loaded objects."""
    if not resources:
        st.error("Preprocessing resources not loaded.")
        return None

    try:
        df = df_raw.copy() # Use a copy of the raw data for processing

        label_encoders = resources['label_encoders']
        onehot_encoder = resources['onehot_encoder']
        ohe_feature_names = resources['ohe_feature_names'] # Feature names from OHE
        final_feature_columns = resources['final_feature_columns'] # Final expected columns for the model
        expected_feature_count = resources['expected_feature_count']

        categorical_columns = ['protocol_type', 'service', 'flag']
        # Recalculate numeric features based on the final expected columns minus the OHE columns
        numeric_features = [col for col in final_feature_columns if col not in ohe_feature_names]

        st.write("Starting preprocessing...")

        # 1. Handle Numeric Features (Fill missing, ensure numeric) - UPDATED SECTION
        st.write(f"Processing {len(numeric_features)} numeric features...")
        processed_numeric_cols = {} # Store processed columns temporarily
        for col in numeric_features:
            if col not in df.columns:
                st.warning(f"Expected numeric feature '{col}' not found in uploaded data. Adding it as 0.")
                # Create a Series of zeros with the same index as the original DataFrame
                processed_numeric_cols[col] = pd.Series(0, index=df.index, dtype=float)
            else:
                # Step 1: Attempt conversion to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                # Step 2: Fill NaN values (resulting from coercion or original NaNs) with 0
                filled_series = numeric_series.fillna(0)
                # Step 3: Explicitly cast to float to prevent potential type issues
                processed_numeric_cols[col] = filled_series.astype(float)

        # Create a DataFrame from the processed numeric columns, ensuring index alignment
        df_numeric_processed = pd.DataFrame(processed_numeric_cols, index=df.index)
        # --- END OF UPDATED SECTION ---


        # 2. Apply Label Encoding (using the original df for categorical columns)
        st.write("Applying Label Encoding...")
        df_label_encoded_cats = pd.DataFrame(index=df.index) # Ensure index alignment
        found_cats = False
        for col in categorical_columns:
            if col in df.columns:
                try:
                    le = label_encoders[col]
                    # Handle potential unseen values by checking if they exist in encoder classes
                    unseen = set(df[col]) - set(le.classes_)
                    if unseen:
                         # Option: Map unseen to a default value (e.g., -1 or a specific category)
                         # For now, we'll rely on OHE's handle_unknown='ignore'
                         st.warning(f"Column '{col}' contains unseen values: {unseen}. OHE will ignore them.")
                         # Transform known values, leave others as is (will become NaN-like for OHE or error)
                         df_label_encoded_cats[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1) # Use -1 for unseen
                    else:
                         df_label_encoded_cats[col] = le.transform(df[col])
                    found_cats = True
                except ValueError as ve:
                     st.error(f"Error applying LabelEncoder to '{col}': {ve}. Check data format.")
                     # Handle error, maybe skip column or return None
                     return None # Stop preprocessing on error
            else:
                 st.warning(f"Categorical column '{col}' not found in uploaded data. Skipping Label Encoding for it.")

        # 3. Apply One-Hot Encoding
        st.write("Applying One-Hot Encoding...")
        if found_cats and not df_label_encoded_cats.empty:
            # Ensure columns match the order expected by the fitted OHE
            df_label_encoded_cats = df_label_encoded_cats.reindex(columns=onehot_encoder.feature_names_in_, fill_value=-1) # Use -1 for missing cols
            
            encoded_cats_array = onehot_encoder.transform(df_label_encoded_cats)
            df_one_hot_encoded = pd.DataFrame(encoded_cats_array, columns=ohe_feature_names, index=df.index)
        else:
            # If no categorical columns were found/processed, create empty OHE df with correct columns
            st.warning("No categorical data found or processed for One-Hot Encoding. Creating placeholder OHE columns.")
            df_one_hot_encoded = pd.DataFrame(0, columns=ohe_feature_names, index=df.index) # Fill with 0


        # 4. Combine Processed Numeric and Encoded Categorical Features
        st.write("Combining features...")
        # df_numeric_processed and df_one_hot_encoded should have the same index now
        df_combined = pd.concat([df_numeric_processed, df_one_hot_encoded], axis=1)

        # 5. Ensure Correct Feature Order and Set
        st.write(f"Reindexing to match {len(final_feature_columns)} expected features...")

        # Add any missing columns expected by the model, fill with 0
        missing_cols = set(final_feature_columns) - set(df_combined.columns)
        if missing_cols:
            st.warning(f"Adding missing expected feature columns as 0: {missing_cols}")
            for col in missing_cols:
                df_combined[col] = 0

        # Keep only the final expected features in the correct order
        # Handle potential duplicates if column names overlap (shouldn't happen with correct derivation)
        df_final = df_combined.loc[:, ~df_combined.columns.duplicated()]
        df_final = df_final.reindex(columns=final_feature_columns, fill_value=0)


        # 6. Verify Final Shape
        st.write(f"Final preprocessed shape: {df_final.shape}")
        if df_final.shape[1] != expected_feature_count:
            raise ValueError(f"Feature shape mismatch after preprocessing. Expected {expected_feature_count} features, but got {df_final.shape[1]}. Check preprocessing steps and loaded objects.")

        # 7. Convert to NumPy array
        X = df_final.to_numpy()
        st.success("Preprocessing complete.")
        return X

    except KeyError as e:
        st.error(f"Preprocessing Error: Missing key '{e}'. This might be due to inconsistencies between saved objects and expected data structure, or missing columns in uploaded file.")
        st.exception(e) # Show full traceback in Streamlit logs
        return None
    except ValueError as e:
        st.error(f"Preprocessing Value Error: {str(e)}. Ensure uploaded data format matches training data.")
        st.exception(e)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during preprocessing: {str(e)}")
        st.exception(e) # Show full traceback
        return None

# ------------------------------
# Prediction Function
# ------------------------------
def make_predictions(X, resources, model_type):
    """Make predictions using specified model type"""
    results = {}
    models_dict = resources['models']
    attack_types = ['DoS', 'Probe', 'R2L', 'U2R']

    for attack in attack_types:
        try:
            # Adjust access based on how 'all_models.pkl' is structured
            if model_type == 'xgboost':
                 # Example: Check if xgboost models are nested
                 if 'xgboost_models' in models_dict and attack in models_dict['xgboost_models']:
                     # Check if there's another level like 'model'
                     if isinstance(models_dict['xgboost_models'][attack], dict) and 'model' in models_dict['xgboost_models'][attack]:
                         model = models_dict['xgboost_models'][attack]['model']
                     else: # Assume the model is directly under the attack type
                         model = models_dict['xgboost_models'][attack]
                 else:
                     st.error(f"XGBoost model structure not found for {attack} in loaded resources.")
                     results[attack] = None
                     continue
            elif model_type == 'logistic':
                 # Example: Check structure for logistic models
                 if 'logistic_models' in models_dict and attack in models_dict['logistic_models']:
                     model = models_dict['logistic_models'][attack]
                 else: # Maybe they are stored directly?
                     st.warning(f"Trying direct access for logistic model {attack}")
                     if f'logistic_{attack}' in models_dict:
                         model = models_dict[f'logistic_{attack}']
                     else:
                         st.error(f"Logistic model structure not found for {attack} in loaded resources.")
                         results[attack] = None
                         continue
            elif model_type == 'ensemble':
                 # Example: Check structure for ensemble models
                 if 'ensemble_models' in models_dict and attack in models_dict['ensemble_models']:
                      model = models_dict['ensemble_models'][attack]
                 else:
                     st.error(f"Ensemble model structure not found for {attack} in loaded resources.")
                     results[attack] = None
                     continue
            else:
                st.error(f"Unknown model type: {model_type}")
                results[attack] = None
                continue

            # Make prediction
            pred_proba = model.predict_proba(X)
            # Use a threshold (0.5 is default, but could be tuned)
            threshold = 0.5
            predictions = (pred_proba[:, 1] >= threshold).astype(int)

            results[attack] = {
                'probability': pred_proba[:, 1],
                'prediction': predictions,
                'threshold': threshold,
                'model_type': model_type
            }
            st.write(f"Prediction successful for {attack} using {model_type}.")

        except AttributeError as e:
             st.error(f"Prediction Error (AttributeError) for {attack} using {model_type}: {str(e)}. Does the loaded object have a 'predict_proba' method?")
             results[attack] = None
        except ValueError as e:
             st.error(f"Prediction Error (ValueError) for {attack} using {model_type}: {str(e)}. Possible feature mismatch error during prediction.")
             results[attack] = None
        except Exception as e:
            st.error(f"Unexpected Error during prediction for {attack} using {model_type}: {str(e)}")
            results[attack] = None

    return results

# ------------------------------
# Results Display Function
# ------------------------------
def display_results(results_all_models, df_raw_len):
    """Display model comparison results"""
    st.subheader("Model Comparison: Detected Attacks")
    cols = st.columns(4)
    attack_types = ['DoS', 'Probe', 'R2L', 'U2R']
    model_types = ['xgboost', 'logistic', 'ensemble']

    for attack_type, col in zip(attack_types, cols):
        with col:
            st.write(f"**{attack_type}**")
            comparison_data = []

            for model_type in model_types:
                # Check if results exist for this model and attack type
                if (model_type in results_all_models and
                    results_all_models[model_type] is not None and
                    attack_type in results_all_models[model_type] and
                    results_all_models[model_type][attack_type] is not None):

                    result = results_all_models[model_type][attack_type]
                    detected_count = result['prediction'].sum()
                    percentage = (detected_count / df_raw_len * 100) if df_raw_len > 0 else 0

                    comparison_data.append({
                        'Model': model_type.capitalize(),
                        'Detected': detected_count,
                        'Percentage': percentage
                    })
                else:
                     # Add placeholder if results are missing
                     comparison_data.append({
                        'Model': model_type.capitalize(),
                        'Detected': 'N/A',
                        'Percentage': 'N/A'
                    })


            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                # Display table - use st.dataframe for better formatting
                st.dataframe(comparison_df.style.format({'Percentage': "{:.2f}%"}), hide_index=True)

                # Filter out N/A before plotting
                plot_df = comparison_df[comparison_df['Percentage'] != 'N/A'].copy()
                plot_df['Percentage'] = pd.to_numeric(plot_df['Percentage'])

                if not plot_df.empty:
                    fig = px.bar(
                        plot_df,
                        x='Model',
                        y='Percentage',
                        title=f'{attack_type} Detection Rate (%)',
                        text='Percentage' # Show percentage on bars
                    )
                    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                    fig.update_layout(yaxis_title="Detection Rate (%)", uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig, use_container_width=True)

    # Display Probability Distributions
    st.subheader("Attack Probability Distributions")
    for attack_type in attack_types:
         with st.expander(f"Show {attack_type} Probability Histograms"):
             dist_cols = st.columns(len(model_types))
             for model_type, dist_col in zip(model_types, dist_cols):
                  with dist_col:
                      st.write(f"**{model_type.capitalize()}**")
                      if (model_type in results_all_models and
                          results_all_models[model_type] is not None and
                          attack_type in results_all_models[model_type] and
                          results_all_models[model_type][attack_type] is not None):

                          result = results_all_models[model_type][attack_type]
                          fig_hist = px.histogram(
                              x=result['probability'],
                              nbins=50, # Adjust number of bins if needed
                              title=f"{attack_type} Probabilities"
                          )
                          fig_hist.update_layout(xaxis_title="Predicted Probability", yaxis_title="Count")
                          st.plotly_chart(fig_hist, use_container_width=True)
                          st.caption(f"Threshold: {result['threshold']:.2f}")
                      else:
                          st.write("No results available.")


# ------------------------------
# Main Application
# ------------------------------
def main():
    # Page title and description
    st.title("🔒 Network Intrusion Detection System (NSL-KDD)")
    st.markdown("""
    Upload a CSV file containing network traffic data (NSL-KDD format) to analyze potential intrusions using XGBoost, Logistic Regression, and an Ensemble model.
    """)

    # Sidebar information
    st.sidebar.header("Session Info")
    st.sidebar.info(f"""
    - Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
    - User: {os.getenv('USER', 'shay-haan')}
    """) # Use environment variable or default

    st.sidebar.header("Resource Loading Status")
    # Load models and preprocessing objects
    resources = load_resources()

    if resources:
        st.sidebar.header("Analysis Options")
        # File upload
        uploaded_file = st.sidebar.file_uploader("Upload NSL-KDD CSV", type="csv", key="file_uploader")

        if uploaded_file:
            st.subheader(f"Uploaded File: `{uploaded_file.name}`")
            try:
                # Load and preview data
                df_raw = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(df_raw):,} records!")

                with st.expander("Preview Raw Data (First 5 Rows)"):
                    st.dataframe(df_raw.head())
                    st.info(f"Raw Dataset Shape: {df_raw.shape}")

                # Analysis button
                if st.sidebar.button("Analyze Network Traffic", key="analyze_button"):
                    st.subheader("Analysis Results")
                    with st.spinner("Preprocessing data..."):
                        X = preprocess_data(df_raw, resources)

                    if X is not None:
                        st.success(f"Data preprocessed into {X.shape[0]} samples and {X.shape[1]} features.")

                        results_all = {}
                        model_types_to_run = ['xgboost', 'logistic', 'ensemble']

                        for model_type in model_types_to_run:
                             with st.spinner(f"Running {model_type.capitalize()} models..."):
                                 st.write(f"--- Running {model_type.capitalize()} ---")
                                 results_all[model_type] = make_predictions(X, resources, model_type)

                        # Display results
                        display_results(results_all, len(df_raw))

                        # Download results option
                        st.subheader("Download Full Results")
                        if st.button("Prepare Download File", key="prepare_download"):
                            df_results = df_raw.copy()
                            for model_type in model_types_to_run:
                                if model_type in results_all and results_all[model_type]:
                                    for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']:
                                        if attack_type in results_all[model_type] and results_all[model_type][attack_type]:
                                            result = results_all[model_type][attack_type]
                                            pred_col = f'{attack_type}_{model_type}_pred'
                                            prob_col = f'{attack_type}_{model_type}_prob'
                                            df_results[pred_col] = result['prediction']
                                            df_results[prob_col] = result['probability']

                            csv_data = df_results.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Click to Download Analysis CSV",
                                data=csv_data,
                                file_name=f"nids_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key='download_csv_button'
                            )

            except pd.errors.EmptyDataError:
                 st.error("The uploaded CSV file is empty.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure the uploaded file is a valid CSV in the expected NSL-KDD format.")
    else:
        st.error("⚠️ Could not load necessary models or preprocessing files. Application cannot proceed.")
        st.info("Please ensure `label_encoders.pkl`, `onehot_encoder.pkl`, `column_order.pkl`, and `models/all_models.pkl` are present in the application directory.")

    # Model performance information (Static - Update if necessary)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Model Performance (Test Set)
    _(As reported)_

    **XGBoost**
    - DoS: 99.8% | Probe: 99.9%
    - R2L: 99.7% | U2R: 99.9%

    **Logistic Regression**
    - DoS: 98.9% | Probe: 99.1%
    - R2L: 98.8% | U2R: 99.2%

    **Ensemble**
    - DoS: 99.998% | Probe: 100%
    - R2L: 99.999% | U2R: 100%
    """)

if __name__ == "__main__":
    main()