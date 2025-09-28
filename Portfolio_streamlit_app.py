# ===============================
# Streamlit Dashboard for Portfolio Optimization
# ===============================
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
    performance_metrics_full
)

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("Settings")
frequency = st.sidebar.selectbox("Data Frequency", ["daily", "weekly", "monthly"], index=1)
n_lags = st.sidebar.slider("LSTM lookback lags", 1, 12, 3)
lookback = st.sidebar.slider("Backtest lookback (periods)", 1, 24, 12)
n_random = st.sidebar.slider("Random Portfolios for Efficient Frontier", 100, 5000, 3000)

# ===============================
# Load or Download Data
# ===============================
tickers = [
    "JPM","GS","AAPL","MSFT","NVDA","GOOGL","META",
    "AMZN","HD","KO","XOM","CVX","UNH","PFE",
    "CAT","UNP","NFLX","DIS","NEE","PLD"
]

@st.cache_data
def load_stock_data(tickers):
    df = yf.download(tickers, start="2020-06-01", end="2025-06-01", auto_adjust=True)["Close"]
    return df

stock_data = load_stock_data(tickers)
st.subheader("Sample Stock Prices")
st.dataframe(stock_data.tail())

# ===============================
# Resample & Compute Returns
# ===============================
FREQUENCY_MAP = {'daily': {'resample': None, 'rf_divisor': 252},
                 'weekly': {'resample': 'W-FRI', 'rf_divisor': 52},
                 'monthly': {'resample': 'M', 'rf_divisor': 12}}

def resample_returns(stock_data, freq_key):
    rule = FREQUENCY_MAP[freq_key]['resample']
    if rule:
        stock_data = stock_data.resample(rule).last()
    returns = np.log(stock_data / stock_data.shift(1)).dropna()
    return returns

returns = resample_returns(stock_data, frequency)
st.subheader(f"{frequency.title()} Returns")
st.dataframe(returns.tail())

# ===============================
# Run Backtest
# ===============================
st.subheader("Backtest Portfolio Performance")
cum_returns_df = backtest_annual(returns, n_lags=n_lags, lookback=lookback)

# Plot cumulative returns
fig, ax = plt.subplots(figsize=(12,6))
for col in cum_returns_df.columns:
    ax.plot(cum_returns_df.index, cum_returns_df[col], label=col, linewidth=2)
    # Drawdown shading
    cum = cum_returns_df[col]
    rolling_max = cum.cummax()
    ax.fill_between(cum_returns_df.index, cum, rolling_max, color='red', alpha=0.1)

ax.set_title("Portfolio Cumulative Returns with Drawdowns")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
st.pyplot(fig)

# Performance Metrics
metrics_df = performance_metrics_full(cum_returns_df)
st.subheader(f"Portfolio Performance Metrics ({frequency.capitalize()})")
st.dataframe(metrics_df)

# ===============================
# Efficient Frontier
# ===============================
st.subheader("Efficient Frontier")

# Compute predicted returns and covariance
mu = predict_returns_lstm(returns, n_lags=n_lags)
Sigma = LedoitWolf().fit(returns).covariance_

treasury = web.DataReader("DGS5", "fred", returns.index[0], returns.index[-1])
rf_annual = treasury["DGS5"].mean()/100
rf = rf_annual / FREQUENCY_MAP[frequency]['rf_divisor']

w_mvo, ret_mvo, vol_mvo, sharpe_mvo, w_sharpe, ret_sharpe, vol_sharpe, sharpe_sharpe = optimize_portfolio(mu, Sigma, rf, returns.columns)
w_eq, ret_eq, vol_eq, sharpe_eq = equal_weight_portfolio(mu, Sigma, rf)
mu_bl, w_bl, ret_bl, vol_bl, sharpe_bl = black_litterman(mu, Sigma, rf, returns.columns, returns)

portfolio_metrics = {
    "Equal Weight": (ret_eq, vol_eq, sharpe_eq),
    "Max Sharpe": (ret_sharpe, vol_sharpe, sharpe_sharpe),
    "MVO": (ret_mvo, vol_mvo, sharpe_mvo),
    "Black-Litterman": (ret_bl, vol_bl, sharpe_bl)
}

# Plot Efficient Frontier
def plot_efficient_frontier(mu, Sigma, rf, portfolios, n_random=3000, max_weight=0.3, title="Efficient Frontier"):
    n = len(mu)
    np.random.seed(42)
    results = np.zeros((3, n_random))
    for i in range(n_random):
        weights = np.random.dirichlet(np.ones(n))
        ret = np.dot(weights, mu)
        vol = np.sqrt(weights.T @ Sigma @ weights)
        sharpe = (ret - rf)/vol
        results[:, i] = [ret, vol, sharpe]

    # Compute analytical efficient frontier
    n_points = 50
    target_returns = np.linspace(min(mu), max(mu), n_points)
    ef_returns, ef_vols = [], []
    for r_target in target_returns:
        w = cp.Variable(n)
        portfolio_var = cp.quad_form(w, Sigma)
        constraints = [cp.sum(w)==1, w>=0, w<=max_weight, mu @ w==r_target]
        prob = cp.Problem(cp.Minimize(portfolio_var), constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        if w.value is not None:
            ef_returns.append(r_target)
            ef_vols.append(np.sqrt(w.value.T @ Sigma @ w.value))

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    sc = ax.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', alpha=0.4)
    plt.colorbar(sc, label='Sharpe Ratio')
    ax.plot(ef_vols, ef_returns, 'r--', lw=2, label='Efficient Frontier')
    for label, metrics in portfolios.items():
        r, v, s = metrics
        ax.scatter(v, r, marker='X', s=200, label=f"{label} (Sharpe: {s:.3f})")
    ax.set_xlabel("Volatility (Std Dev)")
    ax.set_ylabel("Expected Return")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    st.pyplot(fig)

    # Summary
    summary_list = []
    for label, metrics in portfolios.items():
        r, v, s = metrics
        summary_list.append([label, r, v, s])
    return pd.DataFrame(summary_list, columns=['Portfolio','Return','Volatility','Sharpe Ratio'])

ef_summary = plot_efficient_frontier(mu, Sigma, rf, portfolio_metrics, n_random=n_random)
st.subheader("Efficient Frontier Portfolio Summary")
st.dataframe(ef_summary)
