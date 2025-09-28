# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import cvxpy as cp
from scipy.optimize import minimize

import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")
st.title("📊 Portfolio Optimization & Backtesting Dashboard")

# ===============================
# User Inputs
# ===============================
tickers_input = st.text_area(
    "Enter stock tickers separated by comma (default example):",
    value="JPM, GS, AAPL, MSFT, NVDA, GOOGL, META, AMZN, HD, KO, XOM, CVX, UNH, PFE, CAT, UNP, NFLX, DIS, NEE, PLD"
)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

frequency = st.selectbox("Select Return Frequency:", ['daily','weekly','monthly'])
n_lags = st.slider("LSTM Lag Period", 1, 12, 3)
lookback = st.slider("Backtest Lookback Period (periods):", 1, 24, 12)
start_date = st.date_input("Start Date", pd.to_datetime("2020-06-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-06-01"))

FREQ_SETTINGS = {
    'daily': {'resample': None, 'rf_divisor': 252},
    'weekly': {'resample': 'W-FRI', 'rf_divisor': 52},
    'monthly': {'resample': 'M', 'rf_divisor': 12}
}

# ===============================
# Download Data
# ===============================
@st.cache_data
def download_data(tickers, start, end):
    return yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]

stock_data = download_data(tickers, start_date, end_date)
st.write("### Stock Data Sample", stock_data.head())

# ===============================
# Compute Returns
# ===============================
def resample_returns(stock_data, freq_key):
    rule = FREQ_SETTINGS[freq_key]['resample']
    data = stock_data.copy()
    if rule:
        data = data.resample(rule).last()
    returns = np.log(data / data.shift(1)).dropna()
    return returns

returns = resample_returns(stock_data, frequency)

# ===============================
# LSTM Prediction
# ===============================
def predict_returns_lstm(returns, n_lags=3, epochs=50, batch_size=16):
    tickers = returns.columns
    n_tickers = len(tickers)
    X_all, y_all = [], []

    for i in range(n_lags, len(returns)-1):
        X_all.append(returns.iloc[i-n_lags:i].values)
        y_all.append(returns.iloc[i+1].values)

    X = np.array(X_all)
    Y = np.array(y_all)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, n_tickers)).reshape(X.shape)
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y)

    latest_input = returns.iloc[-n_lags:].values.reshape(1, n_lags, n_tickers)
    latest_input_scaled = scaler_X.transform(latest_input.reshape(-1, n_tickers)).reshape(1, n_lags, n_tickers)

    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=(n_lags, n_tickers)))
    model.add(Dense(n_tickers))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, Y_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

    pred_scaled = model.predict(latest_input_scaled, verbose=0)
    return scaler_Y.inverse_transform(pred_scaled)[0]

# ===============================
# Portfolio Optimization Functions
# ===============================
def optimize_portfolio(mu, Sigma, rf, tickers, max_variance=0.001):
    n = len(mu)
    shrinkage_factor = 0.5
    mu_shrunk = shrinkage_factor * mu + (1 - shrinkage_factor) * np.mean(mu)

    # MVO
    w_mvo = cp.Variable(n)
    lambda_reg = 0.01
    portfolio_return = mu_shrunk @ w_mvo - lambda_reg * cp.sum_squares(w_mvo)
    portfolio_variance = cp.quad_form(w_mvo, Sigma)
    max_weight = 0.3
    constraints = [cp.sum(w_mvo) == 1, w_mvo >= 0, w_mvo <= max_weight, portfolio_variance <= max_variance]
    cp.Problem(cp.Maximize(portfolio_return), constraints).solve()
    weights_mvo = w_mvo.value if w_mvo.value is not None else np.repeat(1/n, n)

    # Max Sharpe
    def neg_sharpe(w):
        ret = np.dot(w, mu_shrunk)
        vol = np.sqrt(w.T @ Sigma @ w)
        return -(ret - rf)/vol + lambda_reg*np.sum(w**2)
    bounds = [(0, max_weight)]*n
    constraints = [{'type':'eq','fun':lambda w: np.sum(w)-1}]
    result = minimize(neg_sharpe, np.repeat(1/n,n), method='SLSQP', bounds=bounds, constraints=constraints)
    weights_sharpe = result.x if result.success else np.repeat(1/n,n)

    return weights_mvo, weights_sharpe

def equal_weight_portfolio(n):
    return np.repeat(1/n, n)

def _nearest_psd(A, eps=1e-10):
    B = 0.5*(A + A.T)
    w, V = np.linalg.eigh(B)
    w_clipped = np.clip(w, eps, None)
    return (V * w_clipped) @ V.T

def market_implied_delta(returns, rf, market_weights):
    mu_mkt = returns.mean().values @ market_weights
    var_mkt = market_weights.T @ returns.cov().values @ market_weights
    delta = (mu_mkt - rf)/max(var_mkt,1e-12)
    return max(delta,0.0)

def black_litterman(mu_view, Sigma, rf, tickers, returns, tau=0.5, omega_scalar=0.01):
    n = len(mu_view)
    caps = []
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info
            caps.append(info.get("marketCap",0))
        except:
            caps.append(0)
    caps = np.array(caps, dtype=float)
    market_weights = caps/np.nansum(caps) if np.nansum(caps)>0 else np.full(n,1.0/n)

    mu_view = np.asarray(mu_view, dtype=float).reshape(-1)
    Sigma = _nearest_psd(np.asarray(Sigma))
    delta = market_implied_delta(returns, rf, market_weights)
    Pi = delta * (Sigma @ market_weights)
    P = np.eye(n)
    Omega = np.eye(n) * omega_scalar
    A = np.linalg.inv(tau*Sigma)
    post_prec = A + P.T @ np.linalg.inv(Omega) @ P
    post_mean = np.linalg.inv(post_prec) @ (A @ Pi + P.T @ np.linalg.inv(Omega) @ mu_view)

    w = cp.Variable(n)
    ret = post_mean @ w
    risk = cp.quad_form(w, Sigma)
    constraints = [cp.sum(w)==1, w>=0]
    cp.Problem(cp.Maximize(ret - delta*risk), constraints).solve()
    weights_bl = w.value if w.value is not None else np.repeat(1/n, n)
    return post_mean, weights_bl

# ===============================
# Backtest
# ===============================
def backtest_annual(returns, n_lags=3, lookback=12):
    portfolios = ["Equal-Weight", "MVO", "Max-Sharpe", "Black-Litterman"]
    cum_returns = {p:[1.0] for p in portfolios}
    backtest_index = [returns.index[0]]

    n_periods = len(returns)
    for start_idx in range(lookback, n_periods, lookback):
        end_idx = min(start_idx + lookback, n_periods)
        window_data = returns.iloc[start_idx-lookback:start_idx]

        mu = predict_returns_lstm(window_data, n_lags)
        Sigma = LedoitWolf().fit(window_data).covariance_

        treasury = web.DataReader("DGS5", "fred", window_data.index[0], window_data.index[-1])
        rf_annual = treasury["DGS5"].mean()/100
        rf = rf_annual / FREQ_SETTINGS[frequency]['rf_divisor']

        w_mvo, w_sharpe = optimize_portfolio(mu, Sigma, rf, returns.columns)
        w_eq = equal_weight_portfolio(len(returns.columns))
        _, w_bl = black_litterman(mu, Sigma, rf, returns.columns, window_data)

        actual_returns = returns.iloc[start_idx:end_idx].values
        cum_returns["Equal-Weight"].append(cum_returns["Equal-Weight"][-1]*np.prod(1 + actual_returns @ w_eq))
        cum_returns["MVO"].append(cum_returns["MVO"][-1]*np.prod(1 + actual_returns @ w_mvo))
        cum_returns["Max-Sharpe"].append(cum_returns["Max-Sharpe"][-1]*np.prod(1 + actual_returns @ w_sharpe))
        cum_returns["Black-Litterman"].append(cum_returns["Black-Litterman"][-1]*np.prod(1 + actual_returns @ w_bl))

        backtest_index.append(returns.index[end_idx-1])

    return pd.DataFrame(cum_returns, index=backtest_index)

cum_returns_df = backtest_annual(returns, n_lags, lookback)

# ===============================
# Performance Metrics
# ===============================
def max_drawdown(cum_returns):
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max)/rolling_max
    return drawdown.min()

def performance_metrics_full(cum_returns_df):
    n_days = (cum_returns_df.index[-1] - cum_returns_df.index[0]).days
    n_years = n_days/365.25 if n_days>0 else 0.0001
    metrics = {}
    for col in cum_returns_df.columns:
        total_return = cum_returns_df[col].iloc[-1]/cum_returns_df[col].iloc[0]-1
        CAGR = (1+total_return)**(1/n_years)-1
        period_returns = cum_returns_df[col].pct_change().dropna()
        vol = period_returns.std()*np.sqrt(FREQ_SETTINGS[frequency]['rf_divisor'])
        sharpe = CAGR/vol if vol else np.nan
        mdd = max_drawdown(cum_returns_df[col])
        metrics[col] = [CAGR, vol, sharpe, mdd]
    return pd.DataFrame(metrics, index=['CAGR','Volatility','Sharpe','Max Drawdown']).T

metrics_df = performance_metrics_full(cum_returns_df)

# ===============================
# Efficient Frontier
# ===============================
def compute_efficient_frontier(mu, Sigma, rf, n_points=50, max_weight=0.3):
    n = len(mu)
    target_returns = np.linspace(min(mu), max(mu), n_points)
    frontier_returns, frontier_vols, frontier_sharpes = [], [], []
    for r_target in target_returns:
        w = cp.Variable(n)
        portfolio_var = cp.quad_form(w, Sigma)
        constraints = [cp.sum(w)==1, w>=0, w<=max_weight, mu @ w==r_target]
        cp.Problem(cp.Minimize(portfolio_var), constraints).solve()
        if w.value is not None:
            frontier_returns.append(r_target)
            vol = np.sqrt(w.value.T @ Sigma @ w.value)
            frontier_vols.append(vol)
            frontier_sharpes.append((r_target - rf)/vol)
    return np.array(frontier_returns), np.array(frontier_vols), np.array(frontier_sharpes)

mu_latest = predict_returns_lstm(returns, n_lags)
Sigma_latest = LedoitWolf().fit(returns).covariance_
treasury = web.DataReader("DGS5", "fred", start_date, end_date)
rf_annual = treasury["DGS5"].mean()/100
rf = rf_annual / FREQ_SETTINGS[frequency]['rf_divisor']

ef_returns, ef_vols, ef_sharpes = compute_efficient_frontier(mu_latest, Sigma_latest, rf)

# ===============================
# Streamlit Outputs
# ===============================
st.write("### Cumulative Returns")
st.line_chart(cum_returns_df)

st.write("### Portfolio Performance Metrics")
st.dataframe(metrics_df.style.format("{:.2%}"))

st.write("### Portfolio Drawdowns")
for col in cum_returns_df.columns:
    drawdown = (cum_returns_df[col]/cum_returns_df[col].cummax() - 1)
    st.line_chart(drawdown.rename(f"{col} Drawdown"))

st.write("### Efficient Frontier")
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(ef_vols, ef_returns, 'r--', lw=2, label='Efficient Frontier')
for col in cum_returns_df.columns:
    ret = cum_returns_df[col].iloc[-1]/cum_returns_df[col].iloc[0]-1
    vol = cum_returns_df[col].pct_change().std()*np.sqrt(FREQ_SETTINGS[frequency]['rf_divisor'])
    ax.scatter(vol, ret, s=100, label=col)
ax.set_xlabel("Volatility (Std Dev)")
ax.set_ylabel("Expected Return")
ax.set_title("Efficient Frontier with Portfolios")
ax.legend()
st.pyplot(fig)
