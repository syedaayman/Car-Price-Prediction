"""
preprocessing.py — Input preprocessing for the Car Price Prediction app.

Replicates the exact pipeline from Notebook 2 (Feature Engineering):
  1. Convert year → car_age  (car_age = REFERENCE_YEAR - year)
  2. One-hot encode company and fuel_type
  3. Reindex to match exact training column order
  4. Scale car_age and kms_driven using the saved StandardScaler
"""

import pandas as pd
import os

from src.utils import load_pickle_file, get_artifacts_directory

# Must match the REFERENCE_YEAR used in Notebook 2
REFERENCE_YEAR = 2024


def load_artifacts():
    """
    Load scaler, encoder_data, and feature_columns from artifacts/.

    Returns
    -------
    tuple : (scaler, encoder_data, feature_columns)
    """
    artifacts_dir = get_artifacts_directory()
    scaler          = load_pickle_file(os.path.join(artifacts_dir, 'scaler.pkl'))
    encoder_data    = load_pickle_file(os.path.join(artifacts_dir, 'encoder.pkl'))
    feature_columns = load_pickle_file(os.path.join(artifacts_dir, 'columns.pkl'))
    return scaler, encoder_data, feature_columns


def preprocess_user_input(company, year, kms_driven, fuel_type):
    """
    Transform raw user input into a model-ready feature vector.

    Parameters
    ----------
    company    : str  — e.g. 'Maruti'
    year       : int  — manufacturing year, e.g. 2015
    kms_driven : int  — kilometres driven, e.g. 45000
    fuel_type  : str  — 'Petrol', 'Diesel', or 'LPG'

    Returns
    -------
    pd.DataFrame — one row, columns matching training data exactly
    """
    scaler, encoder_data, feature_columns = load_artifacts()

    # Step 1: year → car_age  (mirrors Notebook 2)
    car_age = REFERENCE_YEAR - year

    # Step 2: Base numerical DataFrame
    input_df = pd.DataFrame({'car_age': [car_age], 'kms_driven': [kms_driven]})

    # Step 3: One-hot encode fuel_type
    for fuel in encoder_data['fuel_type_list']:
        input_df[f'fuel_type_{fuel}'] = 1 if fuel == fuel_type else 0

    # Step 4: One-hot encode company
    for comp in encoder_data['company_list']:
        input_df[f'company_{comp}'] = 1 if comp == company else 0

    # Step 5: Reindex to exact training column order (fills any missing with 0)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Step 6: Scale car_age and kms_driven with the saved scaler
    num_cols = ['car_age', 'kms_driven']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    return input_df


def get_valid_options():
    """
    Return (company_list, fuel_type_list) for Streamlit dropdown menus.
    """
    _, encoder_data, _ = load_artifacts()
    return encoder_data['company_list'], encoder_data['fuel_type_list']
