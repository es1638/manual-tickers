
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import lightgbm as lgb
import traceback

# Set Streamlit page config
st.set_page_config(layout="wide")
st.title("📈 Intraday Breakout Dashboard")

# Load model
try:
    model = joblib.load("lightgbm_model_converted.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Ticker input section
st.subheader("Upload ticker list (.txt) or paste tickers below")
ticker_list = []

uploaded_file = st.file_uploader("Upload ticker list (.txt)", type="txt")
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    ticker_list = [line.strip().upper() for line in content.splitlines() if line.strip()]
    st.success(f"Loaded {len(ticker_list)} tickers from file.")

pasted = st.text_area("Paste comma-separated tickers", "AAPL, MSFT, TSLA")
if pasted:
    pasted_tickers = [t.strip().upper() for t in pasted.split(",") if t.strip()]
    ticker_list.extend([t for t in pasted_tickers if t not in ticker_list])
    st.success(f"Loaded {len(pasted_tickers)} tickers from text.")

# Buy threshold
threshold = st.slider("Buy Signal Threshold", 0.90, 1.00, 0.98, step=0.01)

# Feature engineering function
def get_live_features(ticker):
    data = yf.download(ticker, period="2d", interval="1m")
    if data.empty:
        raise ValueError("No intraday data available")
    data.index = pd.to_datetime(data.index)
    volume_series = data["Volume"]
    data["momentum_10min"] = data["Close"].pct_change(periods=10)
    data["price_change_5min"] = data["Close"].pct_change(periods=5)
    data["rolling_volume"] = volume_series.rolling(window=5).mean()
    data["rolling_volume_ratio"] = volume_series / data["rolling_volume"]
    features = data[["momentum_10min", "price_change_5min", "rolling_volume", "rolling_volume_ratio"]]
    return features.dropna().iloc[-1:]

# Run evaluation
if ticker_list:
    results = []
    for ticker in ticker_list:
        with st.expander(f"🔍 Debug for {ticker}", expanded=False):
            try:
                X = get_live_features(ticker)
                if X.empty:
                    raise ValueError("No intraday data available")

                try:
                    pred = model.predict(X)
                    prob = pred[0] if hasattr(pred, '__getitem__') else float(pred)
                except Exception as e:
                    raise Exception(f"Model prediction error: {e}")

                results.append({
                    "Ticker": ticker,
                    "Buy Signal": "✅ Buy" if prob >= threshold else "❌ No",
                    "Probability": round(prob, 4)
                })
            except Exception as e:
                error_details = traceback.format_exc()
                results.append({
                    "Ticker": ticker,
                    "Buy Signal": "⚠️ Error",
                    "Probability": str(e)
                })

    df_results = pd.DataFrame(results)
    st.dataframe(df_results)
else:
    st.info("Please upload a ticker list or paste tickers to begin.")

