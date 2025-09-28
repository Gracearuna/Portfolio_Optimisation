# portfolio_streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
from sklearn.covariance import LedoitWolf

# Import your model functions
from models import (
    predict_returns_lstm,
    optimize_portfolio,
    equal_weight_portfolio,
    black_litterman,
    backtest_annual,
    performance_metrics_full,
    plot_efficient_frontier
)

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")
st.title("📈 Portfolio Optimization & Backtest Dashboard")

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("Settings")
frequency = st.sidebar.selectbox("Data Frequency", ["daily", "weekly", "monthly"], index=1)
n_lags = st.sidebar.slider("LSTM lookback (lags)", 1, 12, 3)
lookback = st.sidebar.slider("Backtest Lookback Period (periods)", 1, 24, 12)

tickers = st.sidebar.text_area(
    "Tickers (comma-separated)", 
    "JPM,GS,AAPL,MSFT,NVDA,GOOGL,META,AMZN,HD,KO,XOM,CVX,UNH,PFE,CAT,UNP,NFLX,DIS,NEE,PLD"
).replace(" ", "").split(",")

start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-06-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-06-01"))

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return df

stock_data = load_data(tickers, start, end)
st.subheader("Stock Data Preview")
st.dataframe(stock_data.tail())

# ===============================
# Resample Returns
# ===============================
FREQ_MAP = {'daily': None, 'weekly': 'W-FRI', 'monthly': 'M'}
rf_divisor_map = {'daily': 252, 'weekly': 52, 'monthly': 12}

def resample_returns(stock_data, freq):
    rule = FREQ_MAP[freq]
    if rule:
        stock_data = stock_data.resample(rule).last()
    returns = np.log(stock_data / stock_data.shift(1)).dropna()
    return returns

returns = resample_returns(stock_data, frequency)
st.subheader(f"{frequency.capitalize()} Returns Preview")
st.dataframe(returns.tail())

# ===============================
# Run Backtest
# ===============================
with st.spinner("Running annual backtest..."):
    cum_returns_df = backtest_annual(returns, n_lags=n_lags, lookback=lookback)

# ===============================
# Plot Cumulative Returns with Drawdown
# ===============================
st.subheader("Cumulative Returns with Drawdown")
plt.figure(figsize=(14,8))

for col in cum_returns_df.columns:
    cum = cum_returns_df[col]
    rolling_max = cum.cummax()
    drawdown = rolling_max - cum

    # Plot cumulative return line
    plt.plot(cum_returns_df.index, cum, label=col, linewidth=2)
    # Drawdown shading
    plt.fill_between(cum_returns_df.index, cum, rolling_max, color='red', alpha=0.1)

plt.title("Portfolio Cumulative Returns with Drawdown", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Cumulative Return", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
st.pyplot(plt.gcf())

# ===============================
# Performance Metrics
# ===============================
metrics_df = performance_metrics_full(cum_returns_df)
st.subheader("Performance Metrics")
st.dataframe(metrics_df)

# ===============================
# Efficient Frontier
# ===============================
st.subheader("Efficient Frontier")
mu = predict_returns_lstm(returns, n_lags=n_lags)
Sigma = LedoitWolf().fit(returns).covariance_

# Risk-free rate
treasury = web.DataReader("DGS5", "fred", start, end)
rf_annual = treasury["DGS5"].mean()/100
rf = rf_annual / rf_divisor_map[frequency]

# Compute portfolios
weights_mvo, ret_mvo, vol_mvo, sharpe_mvo, weights_sharpe, ret_sharpe, vol_sharpe, sharpe_sharpe = optimize_portfolio(mu, Sigma, rf, tickers)
w_eq, ret_eq, vol_eq, sharpe_eq = equal_weight_portfolio(mu, Sigma, rf)
mu_bl, weights_bl, ret_bl, vol_bl, sharpe_bl = black_litterman(mu, Sigma, rf, tickers, returns)

portfolio_metrics = {
    "Equal Weight": (ret_eq, vol_eq, sharpe_eq),
    "Max Sharpe": (ret_sharpe, vol_sharpe, sharpe_sharpe),
    "MVO": (ret_mvo, vol_mvo, sharpe_mvo),
    "Black-Litterman": (ret_bl, vol_bl, sharpe_bl)
}

ef_summary = plot_efficient_frontier(mu, Sigma, rf, portfolio_metrics, title=f"{frequency.title()} Efficient Frontier")
st.pyplot(plt.gcf())

st.subheader("Portfolio Metrics")
st.dataframe(ef_summary)
