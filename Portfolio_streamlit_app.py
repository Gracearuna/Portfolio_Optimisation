import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from datetime import date

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Portfolio Settings")
DEFAULT_TICKERS = [
    "JPM", "GS", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "HD", "KO", "XOM", "CVX", "UNH", "PFE",
    "CAT", "UNP", "NFLX", "DIS", "NEE", "PLD"
]

tickers = st.sidebar.multiselect("Select Tickers", DEFAULT_TICKERS, default=DEFAULT_TICKERS[:10])
start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-06-01").date())
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-06-01").date())
frequency = st.sidebar.selectbox("Frequency", ["daily", "weekly", "monthly"])
n_lags = st.sidebar.slider("ML Lag Period", 1, 5, 2)
max_variance = st.sidebar.number_input("Max Variance (MVO)", 0.0001, 0.01, 0.0002, format="%.6f")
tau = st.sidebar.number_input("BL τ", 0.01, 1.0, 0.2, format="%.4f")
omega_scalar = st.sidebar.number_input("BL Ω Scalar", 0.01, 1.0, 0.1, format="%.4f")

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data
def download_prices(tickers, start, end):
    if not tickers: return pd.DataFrame()
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series): df = df.to_frame()
    return df.dropna(how="all")

def resample_returns(df, freq):
    freq_map = {'daily': None, 'weekly': 'W-FRI', 'monthly': 'M'}
    rule = freq_map.get(freq)
    if rule: df = df.resample(rule).last()
    returns = np.log(df / df.shift(1)).dropna()
    return returns

def get_rf(start, end):
    try:
        rf_series = web.DataReader("DGS5", "fred", start, end)
        rf_annual = float(rf_series["DGS5"].mean()) / 100
    except:
        rf_annual = 0.02
    return rf_annual

def predict_mu(returns, n_lags):
    if returns.shape[0] <= n_lags + 2: return returns.mean().values
    X, y_dict = [], {t: [] for t in returns.columns}
    for i in range(n_lags, len(returns)-1):
        lag = returns.iloc[i-n_lags:i].values.flatten()
        X.append(lag)
        for t in returns.columns: y_dict[t].append(returns.iloc[i+1][t])
    X = np.array(X); scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    latest = scaler.transform(returns.iloc[-n_lags:].values.flatten().reshape(1,-1))
    mu_pred = []
    for t in returns.columns:
        y = np.array(y_dict[t])
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_scaled[:len(y)], y)
        mu_pred.append(model.predict(latest)[0])
    return np.array(mu_pred)

def _nearest_psd(A, eps=1e-10):
    B = 0.5*(A + A.T)
    w, V = np.linalg.eigh(B)
    return (V * np.clip(w, eps, None)) @ V.T

def optimize_portfolio(mu, Sigma, rf, max_var):
    n = len(mu); Sigma = _nearest_psd(Sigma)
    # MVO
    w = cp.Variable(n)
    ret = mu @ w; risk = cp.quad_form(w, Sigma)
    constraints = [cp.sum(w)==1, w>=0, risk<=max_var]
    cp.Problem(cp.Maximize(ret), constraints).solve(solver=cp.SCS, verbose=False)
    w_mvo = np.nan_to_num(w.value)
    # Max Sharpe
    def neg_sharpe(w): return -((w@mu - rf)/np.sqrt(w.T @ Sigma @ w + 1e-12))
    res = minimize(neg_sharpe, np.repeat(1/n, n), method='SLSQP', bounds=[(0,0.2)]*n,
                   constraints={'type':'eq','fun':lambda w: np.sum(w)-1})
    w_sharpe = np.nan_to_num(res.x)
    return w_mvo, w_sharpe

def equal_weights(n): return np.repeat(1/n, n)

def black_litterman(mu_view, Sigma, rf, tickers, returns, tau, omega):
    n = len(mu_view); Sigma = _nearest_psd(Sigma)
    caps = []
    for tk in tickers:
        try: caps.append(yf.Ticker(tk).info.get("marketCap",0))
        except: caps.append(0)
    caps = np.array(caps); w_mkt = caps/np.nansum(caps) if np.nansum(caps)>0 else np.repeat(1/n,n)
    delta = max((returns.mean().values @ w_mkt - rf)/(w_mkt.T @ returns.cov().values @ w_mkt),0)
    Pi = delta * Sigma @ w_mkt
    P, Omega = np.eye(n), np.eye(n)*omega
    post = np.linalg.inv(tau*Sigma)**1 + P.T @ np.linalg.inv(Omega) @ P
    mu_post = np.linalg.inv(post) @ (np.linalg.inv(tau*Sigma)@Pi + P.T @ np.linalg.inv(Omega) @ mu_view)
    w = cp.Variable(n)
    ret, risk = mu_post@w, cp.quad_form(w,Sigma)
    cp.Problem(cp.Maximize(ret - delta*risk), [cp.sum(w)==1,w>=0]).solve(solver=cp.SCS, verbose=False)
    return mu_post, np.nan_to_num(w.value)

# ---------------------------
# Data Pipeline
# ---------------------------
if not tickers: st.warning("Select at least one ticker."); st.stop()
prices = download_prices(tickers, start, end)
if prices.empty: st.error("No price data returned."); st.stop()
returns = resample_returns(prices, frequency)
rf_annual = get_rf(start, end); rf = rf_annual / {'daily':252,'weekly':52,'monthly':12}[frequency]

mu = predict_mu(returns, n_lags)
Sigma = LedoitWolf().fit(returns).covariance_
w_mvo, w_sharpe = optimize_portfolio(mu, Sigma, rf, max_variance)
w_eq = equal_weights(len(mu))
mu_bl, w_bl = black_litterman(mu, Sigma, rf, tickers, returns, tau, omega_scalar)

portfolios = {"Equal": w_eq, "MVO": w_mvo, "Max Sharpe": w_sharpe, "BL": w_bl}

# ---------------------------
# Portfolio Plots
# ---------------------------
st.subheader("Portfolio Weights Comparison")
fig, ax = plt.subplots(figsize=(12,5))
for name, w in portfolios.items():
    ax.bar(np.arange(len(tickers))+0.2*list(portfolios.keys()).index(name), w, width=0.2, label=name)
ax.set_xticks(np.arange(len(tickers))); ax.set_xticklabels(tickers, rotation=45)
ax.set_ylabel("Weights"); ax.set_title("Portfolio Weights"); ax.legend(); ax.grid(True)
st.pyplot(fig)

st.subheader("Cumulative Returns & Drawdowns")
daily_port = pd.DataFrame({name: returns.values @ w for name, w in portfolios.items()}, index=returns.index)
cumulative = np.exp(daily_port.cumsum())
fig2, ax2 = plt.subplots(figsize=(12,6))
for col in cumulative.columns:
    series = cumulative[col]; running_max = series.cummax(); drawdown = series/running_max-1
    ax2.plot(series.index, series, label=col)
    ax2.fill_between(series.index, series, running_max, where=drawdown<0, alpha=0.2)
ax2.set_title("Cumulative Returns with Drawdowns"); ax2.set_xlabel("Date"); ax2.set_ylabel("Cumulative Return")
ax2.grid(True); ax2.legend()
st.pyplot(fig2)
