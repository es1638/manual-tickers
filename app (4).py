import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import traceback

# Page config
st.set_page_config(layout="wide")
st.title("📈 Intraday Breakout Dashboard")

# Load model
try:
    model = joblib.load("lightgbm_model_converted.pkl")
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# Session ticker list
if "tickers" not in st.session_state:
    st.session_state.tickers = []

# Upload ticker list
uploaded_file = st.file_uploader("Upload ticker list (.txt)", type=["txt"])
if uploaded_file:
    tickers = [line.strip() for line in uploaded_file.getvalue().decode().splitlines() if line.strip()]
    st.session_state.tickers = tickers
    st.success(f"✅ Loaded {len(tickers)} tickers from text.")

# Threshold slider
threshold = st.slider("Buy Signal Threshold", 0.90, 1.00, 0.98, step=0.01)

# Feature engineering
def get_live_features(ticker):
    data = yf.download(ticker, period="2d", interval="1m")
    if data.empty:
        raise ValueError("No intraday data")
    data.index = pd.to_datetime(data.index)
    volume_series = data["Volume"].squeeze()
    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = volume_series.rolling(window=5).mean()
    data["rolling_volume_ratio"] = volume_series / data["rolling_volume"]
    features = data[["momentum_10min", "price_change_5min", "rolling_volume", "rolling_volume_ratio"]]
    return features.dropna().iloc[-1:]

# Predict
if st.session_state.tickers:
    results = []
    for ticker in st.session_state.tickers:
        with st.expander(f"🔍 Debug for {ticker}", expanded=False):
            try:
                X = get_live_features(ticker)
                pred = model.predict(X)
                prob = pred[0] if hasattr(pred, '__getitem__') else float(pred)
                results.append({
                    "Ticker": ticker,
                    "Buy Signal": "✅ Buy" if prob >= threshold else "❌ No",
                    "Probability": f"{prob:.2f}"
                })
            except Exception as e:
                results.append({
                    "Ticker": ticker,
                    "Buy Signal": "⚠️ Error",
                    "Probability": str(e)
                })

    df = pd.DataFrame(results)
    st.dataframe(df)
else:
    st.info("Please upload a ticker list to begin.")
