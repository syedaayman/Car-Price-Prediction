"""
prediction.py — Load the trained model and make price predictions.

The model is trained on log(Price).  This module reads model_meta.pkl
to detect this and applies np.exp() before returning results.
"""

import os
import numpy as np
import pandas as pd

from src.utils import load_pickle_file, get_artifacts_directory, format_price_inr


def load_model():
    """Load the best trained model and its metadata."""
    artifacts_dir = get_artifacts_directory()
    model = load_pickle_file(os.path.join(artifacts_dir, 'model.pkl'))

    # Load metadata to know whether log-transform was used
    meta_path = os.path.join(artifacts_dir, 'model_meta.pkl')
    if os.path.exists(meta_path):
        meta = load_pickle_file(meta_path)
    else:
        meta = {'log_transform': False}

    return model, meta


def predict_price(preprocessed_input: pd.DataFrame) -> dict:
    """
    Predict the car price from a preprocessed input DataFrame.

    Parameters
    ----------
    preprocessed_input : pd.DataFrame
        Single-row DataFrame from preprocessing.preprocess_user_input()

    Returns
    -------
    dict
        'price_raw'       : float — raw predicted price in INR
        'price_formatted' : str   — human-readable INR string (e.g., '₹3,37,157')
    """
    model, meta = load_model()

    # model.predict() returns an array — take the first element
    raw_output = model.predict(preprocessed_input)[0]

    # If the model was trained on log(Price), convert back to INR scale
    if meta.get('log_transform', False):
        price_inr = float(np.exp(raw_output))
    else:
        price_inr = float(raw_output)

    formatted = format_price_inr(price_inr)

    return {
        'price_raw': price_inr,
        'price_formatted': formatted
    }
