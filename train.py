"""
train.py
--------
Smartphone price prediction - training script.

Steps (same logic as the original notebook):
1. Load Smartphones.csv
2. Clean data (drop duplicates, fill missing numeric values with median)
3. Feature engineering (battery_per_inch, camera_total_mp)
4. Build a preprocessing + GradientBoostingRegressor pipeline
5. Train, evaluate, and save the pipeline + metadata for the Streamlit app

Run:
    python train.py
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "Smartphones.csv"
MODEL_PATH = "model_pipeline.joblib"
META_PATH = "metadata.json"

TARGET = "price_inr"

# Columns dropped because they are free-text / too high-cardinality
# to be useful as categorical inputs in a simple form (model name, exact
# processor string). Brand-level info is kept instead.
DROP_COLS = ["model"]

NUMERIC_COLS = [
    "rating_score", "core_count", "clock_speed_ghz", "ram_gb", "storage_gb",
    "display_inches", "res_width_px", "res_height_px", "refresh_rate_hz",
    "battery_mah", "charging_watt", "rear_camera_count", "front_camera_count",
    "rear_camera_main_mp", "front_camera_main_mp",
]

CATEGORICAL_COLS = [
    "smartphone_brand", "processor_brand", "processor_name", "os_name",
    "memory_card_type",
]

BOOLEAN_COLS = ["has_5g", "has_nfc", "has_ir_blaster", "fast_charging",
                 "memory_card_supported"]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates()

    # Fill numeric NaNs with median (same as notebook)
    df = df.fillna(df.median(numeric_only=True))

    # memory_card_type is categorical text; fill missing with "not_supported"
    if "memory_card_type" in df.columns:
        df["memory_card_type"] = df["memory_card_type"].fillna("not_supported")

    # memory_card_supported has NaN where no card slot exists -> treat as False
    if "memory_card_supported" in df.columns:
        df["memory_card_supported"] = df["memory_card_supported"].fillna(False).astype(bool)

    # Feature engineering
    df["battery_per_inch"] = df["battery_mah"] / df["display_inches"]
    df["camera_total_mp"] = df["rear_camera_main_mp"] + df["front_camera_main_mp"]

    return df


def main():
    df = load_and_clean(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    numeric_cols = NUMERIC_COLS + ["battery_per_inch", "camera_total_mp"]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    boolean_cols = [c for c in BOOLEAN_COLS if c in df.columns]

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Treat booleans as numeric 0/1 so they scale fine
    for c in boolean_cols:
        X[c] = X[c].astype(int)

    feature_numeric = numeric_cols + boolean_cols

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = GradientBoostingRegressor(random_state=42)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.4f}")

    # Save the trained pipeline
    joblib.dump(pipeline, MODEL_PATH)

    # Save metadata so the Streamlit app can build the input form
    metadata = {
        "numeric_cols": numeric_cols,
        "boolean_cols": boolean_cols,
        "categorical_cols": categorical_cols,
        "categorical_options": {
            c: sorted(df[c].dropna().unique().tolist()) for c in categorical_cols
        },
        "numeric_ranges": {
            c: {
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "median": float(df[c].median()),
            }
            for c in numeric_cols
        },
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved pipeline -> {MODEL_PATH}")
    print(f"Saved metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
