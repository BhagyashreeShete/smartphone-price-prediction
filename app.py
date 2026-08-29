import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("smartphone_price_model.pkl")
scaler = joblib.load("smartphone_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Smartphone Price Prediction",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Smartphone Price Prediction")
st.write("Enter smartphone specifications to predict the price.")

# -----------------------------
# Input Features
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    rating_score = st.number_input("Rating Score", 0.0, 10.0, 8.0)
    core_count = st.number_input("Core Count", 1, 20, 8)
    clock_speed_ghz = st.number_input("Clock Speed (GHz)", 0.1, 5.0, 2.5)
    ram_gb = st.number_input("RAM (GB)", 1, 32, 8)
    storage_gb = st.number_input("Storage (GB)", 8, 2048, 128)
    display_inches = st.number_input("Display (Inches)", 3.0, 10.0, 6.5)
    res_width_px = st.number_input("Resolution Width", 100, 5000, 1080)

with col2:
    res_height_px = st.number_input("Resolution Height", 100, 5000, 2400)
    refresh_rate_hz = st.number_input("Refresh Rate (Hz)", 30, 240, 120)
    battery_mah = st.number_input("Battery (mAh)", 500, 10000, 5000)
    charging_watt = st.number_input("Charging Watt", 1, 300, 33)
    rear_camera_count = st.number_input("Rear Camera Count", 1, 10, 3)
    front_camera_count = st.number_input("Front Camera Count", 1, 5, 1)
    rear_camera_main_mp = st.number_input("Rear Camera MP", 1.0, 300.0, 50.0)

with col3:
    front_camera_main_mp = st.number_input("Front Camera MP", 1.0, 100.0, 16.0)
    has_5g = st.selectbox("5G", [0, 1])
    has_nfc = st.selectbox("NFC", [0, 1])
    has_ir_blaster = st.selectbox("IR Blaster", [0, 1])
    fast_charging = st.selectbox("Fast Charging", [0, 1])

# -----------------------------
# Derived Features
# -----------------------------

battery_per_inch = battery_mah / display_inches
camera_total_mp = rear_camera_main_mp + front_camera_main_mp

# -----------------------------
# Brand
# -----------------------------

brand = st.selectbox(
    "Smartphone Brand",
    [
        "ai+",
        "alcatel",
        "apple",
        "blackzone"
    ]
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Price"):

    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

    values = {
        "rating_score": rating_score,
        "core_count": core_count,
        "clock_speed_ghz": clock_speed_ghz,
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "has_5g": has_5g,
        "has_nfc": has_nfc,
        "has_ir_blaster": has_ir_blaster,
        "display_inches": display_inches,
        "res_width_px": res_width_px,
        "res_height_px": res_height_px,
        "refresh_rate_hz": refresh_rate_hz,
        "battery_mah": battery_mah,
        "fast_charging": fast_charging,
        "charging_watt": charging_watt,
        "rear_camera_count": rear_camera_count,
        "front_camera_count": front_camera_count,
        "rear_camera_main_mp": rear_camera_main_mp,
        "front_camera_main_mp": front_camera_main_mp,
        "battery_per_inch": battery_per_inch,
        "camera_total_mp": camera_total_mp
    }

    for col, value in values.items():
        if col in input_data.columns:
            input_data.loc[0, col] = value

    # Brand one-hot encoding
    brand_col = "smartphone_brand_" + brand

    if brand_col in input_data.columns:
        input_data.loc[0, brand_col] = 1

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    st.success(
        f"💰 Predicted Smartphone Price: ₹{prediction[0]:,.2f}"
    )