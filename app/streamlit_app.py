"""Streamlit app for AI Digital Payment Failure Predictor.

This app collects transaction inputs and predicts failure probability and a risk label
using the project's trained model when available. If no trained model is found, a
clear demo-mode heuristic is used for quick local demos (with a disclaimer).
"""

from pathlib import Path
from typing import Optional, Tuple, Dict
import json

import streamlit as st
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "synthetic" / "upi_transactions_2020_2025.csv"
MODEL_PATHS = [ROOT_DIR / "models" / "model.joblib", ROOT_DIR / "models" / "model.pkl"]
FEATURE_COL_JSON = ROOT_DIR / "models" / "feature_columns.json"

st.set_page_config(page_title="AI Digital Payment Failure Predictor", layout="centered")

st.title("AI Digital Payment Failure Predictor")
st.write("Predict the risk of a payment failure before executing a transaction.")


@st.cache_data
def load_options() -> Tuple[list, list]:
    # Try to read some example data to populate bank and device options
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH, nrows=10000)
            banks = sorted(df["bank_name"].dropna().unique().tolist())[:50]
            devices = sorted(df["device_type"].dropna().unique().tolist())
            if not banks:
                banks = ["Bank A", "Bank B", "Bank C"]
            if not devices:
                devices = ["mobile", "web"]
            return banks, devices
        except Exception:
            pass
    # Fallback options (Indian banks)
    return ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "YesBank"], [
        "mobile",
        "web",
    ]


@st.cache_data
def load_model_and_features() -> Tuple[Optional[object], Optional[list], Optional[str]]:
    """Attempt to load a serialized model and feature columns.

    Returns (model, feature_columns, message)
    message is None when a usable model was loaded, else contains a user-facing note.
    """
    # Try to import joblib to load models
    try:
        import joblib
    except Exception:
        joblib = None

    for p in MODEL_PATHS:
        if p.exists() and joblib is not None:
            try:
                model = joblib.load(p)
                # Try loading feature columns
                if FEATURE_COL_JSON.exists():
                    with open(FEATURE_COL_JSON) as f:
                        feat_cols = json.load(f)
                elif hasattr(model, "feature_names_in_"):
                    feat_cols = list(model.feature_names_in_)
                else:
                    feat_cols = None
                return model, feat_cols, None
            except Exception:
                continue
    return (
        None,
        None,
        "No serialized model found. The app can run in demo mode for quick previews.",
    )


def demo_predict(input_df: pd.DataFrame) -> float:
    """A conservative demo heuristic for showing probability in absence of a trained model.

    This is for demonstration only and is not a substitute for a trained model.
    """
    # Basic weighted heuristic (normalized)
    latency = input_df["network_latency_ms"].astype(float).iloc[0]
    retry = float(input_df["retry_count"].iloc[0])
    past_rate = float(input_df["past_user_failure_rate"].iloc[0])
    bank_load = float(input_df["bank_load_score"].iloc[0])

    prob = 0.0
    prob += min(1.0, latency / 1000.0) * 0.25
    prob += (1 - 1 / (1 + retry)) * 0.25
    prob += past_rate * 0.35
    prob += bank_load * 0.15
    # clamp
    prob = max(0.0, min(1.0, prob))
    return prob


def align_and_predict(
    model, feat_cols: Optional[list], input_df: pd.DataFrame
) -> float:
    # One-hot encode categorical features same as training
    df_in = pd.get_dummies(
        input_df.copy(), columns=["bank_name", "device_type"], dummy_na=False
    )

    if feat_cols is not None:
        for c in feat_cols:
            if c not in df_in.columns:
                df_in[c] = 0
        # drop unexpected cols
        extra = [c for c in df_in.columns if c not in feat_cols]
        if extra:
            df_in = df_in.drop(columns=extra)
        df_in = df_in[feat_cols]

    # Ensure numeric
    df_in = df_in.astype(float)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df_in)[:, 1][0])
    else:
        # fallback to predict output (0/1)
        pred = int(model.predict(df_in)[0])
        prob = 0.9 if pred == 1 else 0.1
    return prob


model, feat_cols, model_msg = load_model_and_features()

banks, devices = load_options()

# If feature columns are available, derive bank categories from them to
# ensure dropdown values align exactly with the trained model's features.
if feat_cols is not None:
    bank_cols = [c for c in feat_cols if c.startswith("bank_name_")]
    if bank_cols:
        # columns look like 'bank_name_SBI' -> extract suffix; preserve order
        banks = [c.replace("bank_name_", "") for c in bank_cols]
    else:
        # ensure fallback banks are Indian-only
        indian_banks = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "YesBank"]
        banks = [b for b in banks if b in indian_banks]
        if not banks:
            banks = indian_banks

with st.form("predict_form"):
    amount = st.number_input(
        "Amount (INR ₹)", min_value=0.0, value=100.0, step=1.0, format="%.2f"
    )
    hour = st.slider("Hour", 0, 23, 12)
    day_of_week = st.slider("Day of Week", 0, 6, 2)
    bank_name = st.selectbox("Bank", banks)
    network_latency_ms = st.number_input(
        "Network latency (ms)", min_value=0.0, value=100.0, step=1.0, format="%.1f"
    )
    device_type = st.selectbox("Device type", devices)
    retry_count = st.slider("Retry count", 0, 10, 0)
    bank_load_score = st.slider("Bank load score", 0.0, 1.0, 0.5, step=0.01)
    past_user_failure_rate = st.slider(
        "Past user failure rate", 0.0, 1.0, 0.02, step=0.01
    )

    submitted = st.form_submit_button("Predict Risk")

if model is None and submitted:
    # Run demo heuristic
    st.warning(
        "Model not available — running in demo mode. This is a heuristic preview, not a trained model."
    )

# Build input row
input_data = {
    "amount": amount,
    "hour": hour,
    "day_of_week": day_of_week,
    "bank_name": bank_name,
    "network_latency_ms": network_latency_ms,
    "device_type": device_type,
    "retry_count": retry_count,
    "bank_load_score": bank_load_score,
    "past_user_failure_rate": past_user_failure_rate,
}
input_df = pd.DataFrame([input_data])

if submitted:
    if model is not None:
        try:
            prob = align_and_predict(model, feat_cols, input_df)
            source = "Model"
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            prob = demo_predict(input_df)
            source = "Demo fallback"
    else:
        prob = demo_predict(input_df)
        source = "Demo"

    # Risk labeling
    if prob < 0.3:
        label = "Low"
    elif prob <= 0.6:
        label = "Medium"
    else:
        label = "High"

    st.markdown("---")
    st.subheader("Prediction")
    # Display amount with Indian Rupee symbol; no conversion applied
    st.write(f"**Amount (INR ₹):** ₹{amount:,.2f}")
    st.write(f"**Failure probability:** {prob:.2%}  \n**Risk label:** **{label}**")
    st.caption(f"Source: {source}")

    if model_msg:
        st.info(model_msg)

# Footer
st.markdown("---")
st.write(
    "Tip: To use a trained model in the app, place a serialized model in `models/model.joblib` and optionally `models/feature_columns.json` for column alignment."
)
