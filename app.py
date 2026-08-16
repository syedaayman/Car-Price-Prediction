"""
app.py
======

STEP 4 (final step) of the project: The Streamlit Web App.

WHAT THIS FILE DOES
--------------------
This is the ONLY file a normal user ever interacts with. It:
    1. Directly loads the pickle files saved earlier
       (model.pkl, scaler.pkl, encoder.pkl, columns.pkl, model_meta.pkl)
    2. Shows a simple form: company, year, kms driven, fuel type
    3. When the user clicks "Predict", converts their input into the
       same numeric format the model was trained on
    4. Asks the model for a price and shows it in Indian Rupees

WHY THIS FILE IS SIMPLE
------------------------
On purpose, this file does NOT import from any other custom Python
file (no src/preprocessing.py, no src/prediction.py). Everything is
written directly here, step by step, so a beginner can read this ONE
file top to bottom and see the entire prediction process without
jumping between files.

IMPORTANT IDEA TO REMEMBER
----------------------------
Whatever steps we did to the TRAINING data in feature_engineering.py
(car_age, one-hot encoding, scaling, column order) must be repeated
IN THE EXACT SAME WAY here for the new user input. That is the whole
point of saving scaler.pkl / encoder.pkl / columns.pkl — they are the
saved "instructions" for repeating those exact same steps later.

HOW TO RUN
----------
    streamlit run app.py

This requires artifacts/model.pkl, scaler.pkl, encoder.pkl, columns.pkl,
and model_meta.pkl to already exist. Run feature_engineering.py and
model_training.py first if they don't.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------------------------------------------------------------------
# STEP 1: Load all the pickle files directly
# ---------------------------------------------------------------------------
# We load everything ONCE, right at the top of the app, before showing
# anything to the user.

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

ARTIFACTS_FOLDER = "artifacts"

try:
    with open(os.path.join(ARTIFACTS_FOLDER, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(ARTIFACTS_FOLDER, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(ARTIFACTS_FOLDER, "encoder.pkl"), "rb") as f:
        encoder_data = pickle.load(f)

    with open(os.path.join(ARTIFACTS_FOLDER, "columns.pkl"), "rb") as f:
        feature_columns = pickle.load(f)

    with open(os.path.join(ARTIFACTS_FOLDER, "model_meta.pkl"), "rb") as f:
        model_meta = pickle.load(f)

except FileNotFoundError:
    st.error(
        "Artifact files not found in the 'artifacts' folder.\n\n"
        "Please run these files first, in order:\n"
        "1. eda.py\n"
        "2. feature_engineering.py\n"
        "3. model_training.py"
    )
    st.stop()

# These come straight out of encoder.pkl — the list of companies and
# fuel types the model was trained on.
company_list = encoder_data["company_list"]
fuel_type_list = encoder_data["fuel_type_list"]

# This tells us the fixed year we used to calculate car_age during
# training. We MUST use the same reference year here.
REFERENCE_YEAR = model_meta["reference_year"]

# ---------------------------------------------------------------------------
# STEP 2: Page header
# ---------------------------------------------------------------------------
st.title("🚗 Used Car Price Predictor")
st.write(
    "Fill in the car details below and click **Predict Price** to get "
    "an estimated price in Indian Rupees."
)
st.caption(f"Model in use: **{model_meta['model_name']}**")
st.divider()

# ---------------------------------------------------------------------------
# STEP 3: Collect user input with simple widgets
# ---------------------------------------------------------------------------
selected_company = st.selectbox("Car Company (Brand)", sorted(company_list))

selected_year = st.slider(
    "Manufacturing Year",
    min_value=1995,
    max_value=REFERENCE_YEAR,
    value=2015
)

selected_kms = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500000,
    value=40000,
    step=1000
)

selected_fuel = st.selectbox("Fuel Type", sorted(fuel_type_list))

predict_clicked = st.button("Predict Price", type="primary")

# ---------------------------------------------------------------------------
# STEP 4: When the button is clicked, repeat the training-time steps
# ---------------------------------------------------------------------------
if predict_clicked:

    # --- 4a. year -> car_age ---------------------------------------------
    # Same formula used in feature_engineering.py.
    car_age = REFERENCE_YEAR - selected_year

    # --- 4b. Start building one row of input data -------------------------
    input_row = pd.DataFrame({
        "car_age": [car_age],
        "kms_driven": [selected_kms]
    })

    # --- 4c. One-Hot Encode fuel_type by hand ------------------------------
    # For every fuel type seen during training, create a column that is
    # 1 if it matches the user's choice, else 0. This recreates exactly
    # what pd.get_dummies() did during feature engineering.
    for fuel in fuel_type_list:
        column_name = f"fuel_type_{fuel}"
        input_row[column_name] = 1 if fuel == selected_fuel else 0

    # --- 4d. One-Hot Encode company by hand --------------------------------
    for company in company_list:
        column_name = f"company_{company}"
        input_row[column_name] = 1 if company == selected_company else 0

    # --- 4e. Make sure column order EXACTLY matches training ---------------
    # Any column that doesn't exist yet gets filled with 0.
    # This step is critical — without it, the model would receive
    # columns in the wrong order or a different shape than it expects.
    input_row = input_row.reindex(columns=feature_columns, fill_value=0)

    # --- 4f. Scale car_age and kms_driven using the SAVED scaler -----------
    # We do NOT fit a new scaler here — we reuse the exact one fitted
    # on the training data, so the scaling is consistent.
    numeric_columns = ["car_age", "kms_driven"]
    input_row[numeric_columns] = scaler.transform(input_row[numeric_columns])

    # --- 4g. Ask the model for a prediction ---------------------------------
    predicted_log_price = model.predict(input_row)[0]

    # --- 4h. Undo the log-transform to get a real rupee value --------------
    # We trained on log(Price), so we must reverse it with np.exp().
    if model_meta.get("log_transform", False):
        predicted_price = np.exp(predicted_log_price)
    else:
        predicted_price = predicted_log_price

    # --- 4i. Apply a safety floor ---------------------------------------
    # Linear-style models can occasionally predict unrealistic values
    # (even negative!) for very old or very high-km cars, since they
    # are extrapolating outside the range of prices they were trained on.
    MIN_PRICE = 30000
    predicted_price = max(MIN_PRICE, predicted_price)

    # --- 4j. Display the result ------------------------------------------
    st.divider()
    st.subheader("Estimated Price")
    st.success(f"₹ {predicted_price:,.0f}")
    st.write(f"That is approximately ₹ {predicted_price / 100000:.2f} Lakh")

else:
    st.info("Fill in the details above and click **Predict Price**.")
