# streamlit_portfolio_dashboard_fixed_v2.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import cvxpy as cp
from scipy.optimize import minimize
import pandas_datareader.data as web
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Advanced Portfolio Dashboard", layout="wide")

# === SIDEBAR INPUTS ===
st.sidebar.header("Portfolio Settings")
tickers_input = st.sidebar.text_area(
    "Enter tickers (comma-separated)", 
    value="AAPL,MSFT,GOOGL,META,NVDA,AMD,CRM,ADBE,CSCO"
)
tickers = [x.strip().upper() for x in tickers_input.split(",")]

start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-06-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-06-01"))
frequency = st.sidebar.selectbox("Resampling Frequency", ['daily', 'weekly', 'monthly', 'annual'])
n_lags = st.sidebar.slider("ML Lag Days", 1, 10, 2)
window = st.sidebar.slider("Rolling Window Size (Days)", 5, 60, 21)

st.sidebar.markdown("---")
st.sidebar.write("Developed by: Your Name")

FREQ_MAP = {
    'daily': {'resample': None, 'rf_divisor': 252},
    'weekly': {'resample': 'W-FRI', 'rf_divisor': 52},
    'monthly': {'resample': 'M', 'rf_divisor': 12},
    'annual': {'resample': 'Y', 'rf_divisor': 1}
}

# === DATA DOWNLOAD & RETURNS ===
@st.cache_data
def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data

@st.cache_data
def get_rf_rate(start, end, freq_key):
    try:
        treasury = web.DataReader("DGS5", "fred", start, end)
        rf_annual = treasury.mean()[0]/100
        return rf_annual / FREQ_MAP[freq_key]['rf_divisor']
    except:
        return 0.01 / FREQ_MAP[freq_key]['rf_divisor']  # fallback 1% annual

stock_data = download_data(tickers, start, end)
rf = get_rf_rate(start, end, frequency)

def resample_returns(stock_data, freq_key):
    rule = FREQ_MAP[freq_key]['resample']
    if rule:
        stock_data = stock_data.resample(rule).last()
    returns = np.log(stock_data / stock_data.shift(1)).dropna()
    return returns

returns = resample_returns(stock_data, frequency)

st.subheader("Stock Prices & Returns")
st.dataframe(stock_data.tail())
st.dataframe(returns.tail())

# === ML PREDICTED RETURNS ===
def predict_returns(returns, n_lags):
    X_all, y_all_dict = [], {ticker: [] for ticker in returns.columns}
    for i in range(n_lags, len(returns) - 1):
        lagged = returns.iloc[i - n_lags:i].values.flatten()
        X_all.append(lagged)
        for ticker in returns.columns:
            y_all_dict[ticker].append(returns.iloc[i + 1][ticker])
    X = np.array(X_all)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    latest_input = returns.iloc[-n_lags:].values.flatten().reshape(1, -1)
    latest_input_scaled = scaler.transform(latest_input)
    predicted_returns = []
    for ticker in returns.columns:
        y = np.array(y_all_dict[ticker])
        if len(y) < 2:
            predicted_returns.append(returns[ticker].iloc[-1])
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_scaled[:len(y)], y)
            predicted_returns.append(model.predict(latest_input_scaled)[0])
    return np.array(predicted_returns)

mu = predict_returns(returns, n_lags)
Sigma = LedoitWolf().fit(returns).covariance_

# === PORTFOLIO FUNCTIONS ===
def equal_weight_portfolio(n):
    return np.repeat(1/n, n)

def optimize_portfolio(mu, Sigma, rf, max_variance=0.0002):
    n = len(mu)
    # MVO
    w_mvo = cp.Variable(n)
    portfolio_return = mu @ w_mvo
    portfolio_variance = cp.quad_form(w_mvo, Sigma)
    constraints = [cp.sum(w_mvo)==1, w_mvo>=0, cp.max(w_mvo)<=0.3, portfolio_variance<=max_variance]
    prob = cp.Problem(cp.Maximize(portfolio_return), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    weights_mvo = np.nan_to_num(w_mvo.value)

    # Max Sharpe
    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
        return -(ret - rf)/vol
    bounds = [(0,0.2)]*n
    cons = [{'type':'eq','fun': lambda w: np.sum(w)-1}]
    res = minimize(neg_sharpe, np.repeat(1/n,n), method='SLSQP', bounds=bounds, constraints=cons)
    weights_sharpe = np.nan_to_num(res.x)
    return weights_mvo, weights_sharpe

def _nearest_psd(A, eps=1e-10):
    B = 0.5*(A + A.T)
    w, V = np.linalg.eigh(B)
    w_clipped = np.clip(w, eps, None)
    return (V*w_clipped) @ V.T

def market_implied_delta(returns, rf, market_weights):
    mu_mkt = returns.mean().values @ market_weights
    var_mkt = market_weights.T @ returns.cov().values @ market_weights
    delta = (mu_mkt - rf)/max(var_mkt,1e-12)
    return float(max(delta,0.0))

def black_litterman(mu_view, Sigma, rf, tickers, returns, tau=0.2, omega_scalar=0.1):
    n = len(mu_view)
    caps = []
    for tk in tickers:
        try: caps.append(yf.Ticker(tk).info.get("marketCap",0))
        except: caps.append(0)
    caps = np.array(caps,dtype=float)
    market_weights = caps/np.nansum(caps) if np.nansum(caps)>0 else np.full(n,1/n)
    Sigma_psd = _nearest_psd(Sigma)
    delta = market_implied_delta(returns, rf, market_weights)
    Pi = delta * Sigma_psd @ market_weights
    P = np.eye(n)
    Omega = np.eye(n)*omega_scalar
    A = np.linalg.inv(tau*Sigma_psd)
    post_prec = A + P.T @ np.linalg.inv(Omega) @ P
    post_mean = np.linalg.inv(post_prec) @ (A@Pi + P.T @ np.linalg.inv(Omega) @ mu_view)
    w = cp.Variable(n)
    ret = post_mean @ w
    risk = cp.quad_form(w, Sigma_psd)
    cons = [cp.sum(w)==1, w>=0]
    prob = cp.Problem(cp.Maximize(ret - delta*risk), cons)
    prob.solve(solver=cp.SCS, verbose=False)
    return post_mean, np.nan_to_num(w.value)

# --- CALCULATE PORTFOLIOS ---
w_eq = equal_weight_portfolio(len(tickers))
w_mvo, w_sharpe = optimize_portfolio(mu, Sigma, rf)
mu_bl, w_bl = black_litterman(mu, Sigma, rf, tickers, returns)

# --- DISPLAY PORTFOLIOS ---
st.subheader("Portfolio Weights")
st.dataframe(pd.DataFrame({
    'Ticker': tickers,
    'Equal Weight': w_eq,
    'MVO': w_mvo,
    'Max Sharpe': w_sharpe,
    'Black-Litterman': w_bl
}))

# === PLOT FUNCTION ===
def plot_weights(weights_dict, tickers, title):
    fig, ax = plt.subplots(figsize=(12,6))
    n_assets = len(tickers)
    n_portfolios = len(weights_dict)
    width = 0.8 / n_portfolios
    for i, (name, w) in enumerate(weights_dict.items()):
        w = np.array(w).flatten()
        w = np.nan_to_num(w, nan=0.0)
        if len(w) != n_assets:
            w = np.zeros(n_assets)
        ax.bar(np.arange(n_assets) + i*width, w, width=width, label=name)
    ax.set_xticks(np.arange(n_assets) + width*(n_portfolios-1)/2)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

plot_weights({'Equal Weight': w_eq, 'MVO': w_mvo, 'Max Sharpe': w_sharpe, 'BL': w_bl}, tickers, "Portfolio Weights Comparison")

# === CUMULATIVE RETURNS ===
daily_returns = returns
cumulative_dict = {}
for name, w in zip(['Equal','MVO','Sharpe','BL'], [w_eq,w_mvo,w_sharpe,w_bl]):
    port_ret = daily_returns.values @ w
    cumulative_dict[name] = np.exp(np.log1p(port_ret).cumsum())

plt.figure(figsize=(12,6))
for name, cum in cumulative_dict.items():
    plt.plot(daily_returns.index, cum, label=name)
plt.title("Cumulative Portfolio Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.legend()
st.pyplot(plt)

# === ROLLING METRICS ===
plt.figure(figsize=(12,6))
for name, w in zip(['Equal','MVO','Sharpe','BL'], [w_eq,w_mvo,w_sharpe,w_bl]):
    port_ret = pd.Series(daily_returns.values @ w, index=daily_returns.index)
    rolling_vol = port_ret.rolling(window).std() * np.sqrt(FREQ_MAP[frequency]['rf_divisor'])
    plt.plot(port_ret.index, rolling_vol, label=name)
plt.title(f"Rolling {window}-Period Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.grid(True)
st.pyplot(plt)

plt.figure(figsize=(12,6))
for name, w in zip(['Equal','MVO','Sharpe','BL'], [w_eq,w_mvo,w_sharpe,w_bl]):
    port_ret = pd.Series(daily_returns.values @ w, index=daily_returns.index)
    rolling_sharpe = port_ret.rolling(window).mean() / port_ret.rolling(window).std() * np.sqrt(FREQ_MAP[frequency]['rf_divisor'])
    plt.plot(port_ret.index, rolling_sharpe, label=name)
plt.title(f"Rolling {window}-Period Sharpe Ratio")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.grid(True)
st.pyplot(plt)

# === PORTFOLIO CORRELATION ===
daily_portfolio_returns = pd.DataFrame({name: daily_returns.values @ w for name, w in zip(['Equal','MVO','Sharpe','BL'], [w_eq,w_mvo,w_sharpe,w_bl])}, index=daily_returns.index)
st.subheader("Portfolio Daily Returns Correlation")
st.dataframe(daily_portfolio_returns.corr())
plt.figure(figsize=(8,6))
sns.heatmap(daily_portfolio_returns.corr(), annot=True, cmap='coolwarm')
plt.title("Portfolio Daily Returns Correlation")
st.pyplot(plt)
