"""
app.py
------
Streamlit app for smartphone price prediction.

Run locally:
    streamlit run app.py

Requires model_pipeline.joblib and metadata.json (produced by train.py)
to be present in the same folder.
"""

import json
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model_pipeline.joblib"
META_PATH = "metadata.json"

st.set_page_config(page_title="Smartphone Price Predictor", page_icon="📱", layout="centered")


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)
    return pipeline, metadata


try:
    pipeline, meta = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train.py` first "
        "(with Smartphones.csv in this folder) to generate "
        "`model_pipeline.joblib` and `metadata.json`."
    )
    st.stop()

st.title("📱 Smartphone Price Predictor")
st.caption("Enter the specs below to estimate the price (INR) — model: Gradient Boosting Regressor")

with st.expander("Model performance on test data"):
    m = meta["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"₹{m['mae']:.0f}")
    c2.metric("RMSE", f"₹{m['rmse']:.0f}")
    c3.metric("R²", f"{m['r2']:.3f}")

st.divider()
st.subheader("Specifications")

inputs = {}

# Categorical inputs
col1, col2 = st.columns(2)
cat_cols = meta["categorical_cols"]
for i, c in enumerate(cat_cols):
    target_col = col1 if i % 2 == 0 else col2
    options = meta["categorical_options"][c]
    inputs[c] = target_col.selectbox(c.replace("_", " ").title(), options)

st.divider()

# Numeric inputs (exclude derived features, computed automatically below)
derived = {"battery_per_inch", "camera_total_mp"}
numeric_cols = [c for c in meta["numeric_cols"] if c not in derived]

col1, col2 = st.columns(2)
for i, c in enumerate(numeric_cols):
    r = meta["numeric_ranges"][c]
    target_col = col1 if i % 2 == 0 else col2
    step = 1.0 if r["max"] - r["min"] > 20 else 0.1
    inputs[c] = target_col.number_input(
        c.replace("_", " ").title(),
        min_value=float(r["min"]),
        max_value=float(r["max"]) * 1.2,
        value=float(r["median"]),
        step=step,
    )

st.divider()
st.subheader("Features")

col1, col2, col3 = st.columns(3)
bool_cols = meta["boolean_cols"]
bool_labels = {c: c.replace("has_", "").replace("_", " ").title() for c in bool_cols}
cols_cycle = [col1, col2, col3]
for i, c in enumerate(bool_cols):
    inputs[c] = 1 if cols_cycle[i % 3].checkbox(bool_labels[c]) else 0

st.divider()

if st.button("Predict Price", type="primary", use_container_width=True):
    row = dict(inputs)
    # Derived features
    row["battery_per_inch"] = row["battery_mah"] / row["display_inches"]
    row["camera_total_mp"] = row["rear_camera_main_mp"] + row["front_camera_main_mp"]

    X_new = pd.DataFrame([row])
    prediction = pipeline.predict(X_new)[0]

    st.success(f"### Estimated Price: ₹{prediction:,.0f}")
