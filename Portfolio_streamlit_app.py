import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.covariance import LedoitWolf
import cvxpy as cp
from scipy.optimize import minimize
import pandas_datareader.data as web
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Portfolio Optimization Dashboard")

# === SIDEBAR OPTIONS ===
frequency = st.sidebar.selectbox("Select Return Frequency", ['daily', 'weekly', 'monthly'])
n_lags = st.sidebar.slider("ML Lag Period", 1, 5, 2)

# Selected 20 tickers
tickers = [
    "JPM", "GS", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "HD", "KO", "XOM", "CVX", "UNH", "PFE",
    "CAT", "UNP", "NFLX", "DIS", "NEE", "PLD"
]

FREQUENCY_MAP = {
    'daily': {'resample': None, 'rf_divisor': 252},
    'weekly': {'resample': 'W-FRI', 'rf_divisor': 52},
    'monthly': {'resample': 'M', 'rf_divisor': 12}
}

start = "2020-06-01"
end = "2025-06-01"

# === DATA DOWNLOAD ===
@st.cache_data
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data

stock_data = load_data(tickers, start, end)

# === RESAMPLE RETURNS ===
def resample_returns(stock_data, freq_key):
    rule = FREQUENCY_MAP[freq_key]['resample']
    if rule:
        stock_data = stock_data.resample(rule).last()
    returns = np.log(stock_data / stock_data.shift(1)).dropna()
    return returns

returns = resample_returns(stock_data, frequency)

# === ML RETURN PREDICTION ===
def predict_returns(returns, n_lags):
    X_all, y_all_dict = [], {ticker: [] for ticker in returns.columns}
    for i in range(n_lags, len(returns)-1):
        lagged = returns.iloc[i-n_lags:i].values.flatten()
        X_all.append(lagged)
        for ticker in returns.columns:
            y_all_dict[ticker].append(returns.iloc[i+1][ticker])
    X = np.array(X_all)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    latest_input = returns.iloc[-n_lags:].values.flatten().reshape(1,-1)
    latest_input_scaled = scaler.transform(latest_input)
    predicted_returns = []
    for ticker in returns.columns:
        y = np.array(y_all_dict[ticker])
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_scaled[:len(y)], y)
        pred = model.predict(latest_input_scaled)[0]
        predicted_returns.append(pred)
    return np.array(predicted_returns)

mu = predict_returns(returns, n_lags)
Sigma = LedoitWolf().fit(returns).covariance_

# === RISK-FREE RATE ===
treasury = web.DataReader("DGS5", "fred", start, end)
rf_annual = treasury["DGS5"].mean() / 100
rf = rf_annual / FREQUENCY_MAP[frequency]['rf_divisor']

# === PORTFOLIO FUNCTIONS ===
def optimize_portfolio(mu, Sigma, rf, max_variance=None):
    n = len(mu)
    if max_variance is None:
        scale_factor = np.trace(Sigma)/n
        max_variance = scale_factor * 2
    w_mvo = cp.Variable(n)
    portfolio_return = mu @ w_mvo
    portfolio_variance = cp.quad_form(w_mvo, Sigma)
    constraints = [cp.sum(w_mvo)==1, w_mvo>=0, cp.max(w_mvo)<=0.3, portfolio_variance<=max_variance]
    prob = cp.Problem(cp.Maximize(portfolio_return), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    weights_mvo = np.nan_to_num(w_mvo.value)
    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
        return -(ret - rf)/vol
    bounds = [(0,0.2)]*n
    cons = [{'type':'eq','fun': lambda w: np.sum(w)-1}]
    res = minimize(neg_sharpe, np.repeat(1/n,n), method='SLSQP', bounds=bounds, constraints=cons)
    weights_sharpe = np.nan_to_num(res.x)
    return weights_mvo, weights_sharpe

def equal_weight_portfolio(mu, Sigma, rf):
    n = len(mu)
    w_eq = np.repeat(1/n, n)
    return w_eq

def _nearest_psd(A, eps=1e-10):
    B = 0.5*(A + A.T)
    w, V = np.linalg.eigh(B)
    w_clipped = np.clip(w, eps, None)
    return (V * w_clipped) @ V.T

def market_implied_delta(returns, rf, market_weights):
    mu_mkt = returns.mean().values @ market_weights
    var_mkt = market_weights.T @ returns.cov().values @ market_weights
    delta = (mu_mkt - rf)/max(var_mkt, 1e-12)
    return float(max(delta, 0.0))

def black_litterman(mu_view, Sigma, rf, tickers, returns, tau=0.2, omega_scalar=0.1):
    n = len(mu_view)
    caps = []
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info
            caps.append(info.get("marketCap",0))
        except Exception:
            caps.append(0)
    caps = np.array(caps, dtype=float)
    market_weights = caps/np.nansum(caps) if np.nansum(caps)>0 else np.full(n, 1.0/n)
    mu_view = np.asarray(mu_view, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma_psd = _nearest_psd(Sigma)
    delta = market_implied_delta(returns, rf, market_weights)
    Pi = delta * (Sigma_psd @ market_weights)
    P = np.eye(n)
    Omega = np.eye(n) * omega_scalar
    A = np.linalg.inv(tau*Sigma_psd)
    post_prec = A + P.T @ np.linalg.inv(Omega) @ P
    post_mean = np.linalg.inv(post_prec) @ (A @ Pi + P.T @ np.linalg.inv(Omega) @ mu_view)
    w = cp.Variable(n)
    ret = post_mean @ w
    risk = cp.quad_form(w, Sigma_psd)
    constraints = [cp.sum(w)==1, w>=0]
    prob = cp.Problem(cp.Maximize(ret - delta*risk), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    weights_bl = np.nan_to_num(w.value)
    return post_mean, weights_bl

# === CALCULATE PORTFOLIOS ===
w_eq = equal_weight_portfolio(mu, Sigma, rf)
w_mvo, w_sharpe = optimize_portfolio(mu, Sigma, rf)
mu_bl, w_bl = black_litterman(mu, Sigma, rf, tickers, returns)

portfolios = {
    "Equal": w_eq,
    "MVO": w_mvo,
    "Max Sharpe": w_sharpe,
    "BL": w_bl
}

# === SINGLE-PORTFOLIO ANALYSIS ===
st.subheader("Single Portfolio Analysis")
single_option = st.selectbox("Select Portfolio", ["Equal", "MVO", "Max Sharpe", "BL"])
weights_single = portfolios[single_option]

# Weights plot
fig, ax = plt.subplots(figsize=(12,6))
ax.bar(tickers, np.nan_to_num(weights_single))
ax.set_ylabel("Weights")
ax.set_title(f"{single_option} Portfolio Weights")
ax.set_xticklabels(tickers, rotation=45)
ax.grid(True)
st.pyplot(fig)

# Cumulative returns + drawdown
daily_log = pd.Series(returns.values @ np.nan_to_num(weights_single), index=returns.index)
cum_ret = np.exp(daily_log.cumsum())
running_max = cum_ret.cummax()
drawdown = cum_ret / running_max - 1

fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(cum_ret, label=f"{single_option} Cumulative Return", color="blue")
ax2.fill_between(drawdown.index, cum_ret, running_max, where=drawdown<0, color='red', alpha=0.2, label="Drawdown")
ax2.set_title(f"{single_option} Portfolio Cumulative Return and Drawdown")
ax2.set_xlabel("Date")
ax2.set_ylabel("Cumulative Return")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# Portfolio metrics
def max_drawdown(cum_ret): return (cum_ret/cum_ret.cummax()-1).min()
def cagr_log(daily_log_returns, trading_days=252): return np.exp(daily_log_returns.sum()/ (len(daily_log_returns)/trading_days))-1
def annualized_volatility(daily_log_returns, trading_days=252): return daily_log_returns.std()*np.sqrt(trading_days)
def sharpe_ratio_log(daily_log_returns, rf_annual=rf_annual, trading_days=252):
    rf_daily = rf_annual/trading_days
    excess = daily_log_returns-rf_daily
    return np.sqrt(trading_days)*excess.mean()/excess.std()

metrics = {
    "Max Drawdown": max_drawdown(cum_ret),
    "CAGR": cagr_log(daily_log),
    "Volatility": annualized_volatility(daily_log),
    "Sharpe Ratio": sharpe_ratio_log(daily_log)
}
st.subheader(f"{single_option} Portfolio Metrics")
st.dataframe(pd.DataFrame(metrics, index=[0]).T)

# === MULTI-PORTFOLIO COMPARISON ===
st.subheader("Multi-Portfolio Comparison")
multi_option = st.multiselect("Select Portfolios to Compare", ["Equal", "MVO", "Max Sharpe", "BL"], default=["Equal","MVO","Max Sharpe","BL"])
fig3, ax3 = plt.subplots(figsize=(14,7))
for name in multi_option:
    w = portfolios[name]
    daily_log = pd.Series(returns.values @ np.nan_to_num(w), index=returns.index)
    cum_ret = np.exp(daily_log.cumsum())
    running_max = cum_ret.cummax()
    drawdown = cum_ret/running_max -1
    ax3.plot(cum_ret, label=f"{name} Cumulative Return")
    ax3.fill_between(drawdown.index, cum_ret, running_max, where=drawdown<0, alpha=0.2)
ax3.set_title("Comparison of Selected Portfolio Cumulative Returns and Drawdowns")
ax3.set_xlabel("Date")
ax3.set_ylabel("Cumulative Return")
ax3.grid(True)
ax3.legend()
st.pyplot(fig3)
