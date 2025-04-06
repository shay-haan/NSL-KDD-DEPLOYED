# models_export.py
import pickle
import os
from datetime import datetime

print(f"Model Export Process")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"User: shay-haan")
print("=" * 70)

# Create necessary directories
directories = ['models', 'data', 'tests', 'utils']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Save all models and scalers
model_data = {
    'xgboost_models': {
        'DoS': xgb_DoS,
        'Probe': xgb_Probe,
        'R2L': xgb_R2L,
        'U2R': xgb_U2R
    },
    'logistic_models': {
        'DoS': lr_DoS,
        'Probe': lr_Probe,
        'R2L': lr_R2L,
        'U2R': lr_U2R
    },
    'ensemble_models': {
        'DoS': stack_DoS,
        'Probe': stack_Probe,
        'R2L': stack_R2L,
        'U2R': stack_U2R
    },
    'scalers': scalers,  # From our previous preprocessing
    'feature_names': colNames,  # Original feature names
    'metadata': {
        'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user': 'shay-haan'
    }
}

# Save all models in a single file
with open('models/all_models.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("All models saved successfully!")