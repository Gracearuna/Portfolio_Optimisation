import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas_datareader.data as web

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

# --------------------------------
# Sidebar Controls
# --------------------------------
st.sidebar.header("Portfolio Settings")
tickers = st.sidebar.multiselect(
    "Select Tickers",
    ["JPM", "GS", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
     "AMZN", "HD", "KO", "XOM", "CVX", "UNH", "PFE",
     "CAT", "UNP", "NFLX", "DIS", "NEE", "PLD"],
    default=["JPM", "AAPL", "MSFT", "NVDA", "GOOGL", "META"]
)

start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-06-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-06-01"))
n_lags = st.sidebar.slider("Number of Lags", 1, 5, 2)
frequency = st.sidebar.selectbox("Frequency", ["daily", "weekly", "monthly", "annual"])
max_variance = st.sidebar.number_input("Max Variance Constraint (MVO)", value=0.0002)
tau = st.sidebar.number_input("Black–Litterman τ", value=0.2)
omega_scalar = st.sidebar.number_input("Black–Litterman Ω Scalar", value=0.1)

FREQUENCY_MAP = {
    'daily': {'resample': None, 'rf_divisor': 252},
    'weekly': {'resample': 'W-FRI', 'rf_divisor': 52},
    'monthly': {'resample': 'M', 'rf_divisor': 12},
    'annual': {'resample': 'Y', 'rf_divisor': 1}
}

# --------------------------------
# Helper Functions
# --------------------------------
@st.cache_data
def resample_returns(stock_data, freq_key):
    rule = FREQUENCY_MAP[freq_key]['resample']
    if rule:
        stock_data = stock_data.resample(rule).last()
    returns = np.log(stock_data / stock_data.shift(1)).dropna()
    return returns

def predict_returns(returns, n_lags):
    X_all = []
    y_all_dict = {ticker: [] for ticker in returns.columns}
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
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_scaled[:len(y)], y)
        pred = model.predict(latest_input_scaled)[0]
        predicted_returns.append(pred)
    return np.array(predicted_returns)

def optimize_portfolio(mu, Sigma, rf, tickers, max_variance=0.0002):
    n = len(mu)
    w_mvo = cp.Variable(n)
    portfolio_return = mu @ w_mvo
    portfolio_variance = cp.quad_form(w_mvo, Sigma)
    constraints = [cp.sum(w_mvo) == 1, w_mvo >= 0, portfolio_variance <= max_variance]
    prob = cp.Problem(cp.Maximize(portfolio_return), constraints)
    prob.solve()
    weights_mvo = w_mvo.value

    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
        return -(ret - rf) / vol

    bounds = [(0, 0.2)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    init_guess = np.repeat(1/n, n)
    result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    weights_sharpe = result.x
    return weights_mvo, weights_sharpe

def equal_weight_portfolio(mu, Sigma, rf):
    n = len(mu)
    return np.repeat(1/n, n)

def _nearest_psd(A, eps=1e-10):
    B = 0.5*(A + A.T)
    w, V = np.linalg.eigh(B)
    w_clipped = np.clip(w, eps, None)
    return (V * w_clipped) @ V.T

def market_implied_delta(returns, rf, market_weights):
    mu_mkt = returns.mean().values @ market_weights
    var_mkt = market_weights.T @ returns.cov().values @ market_weights
    delta = (mu_mkt - rf) / max(var_mkt, 1e-12)
    return float(max(delta, 0.0))

def black_litterman(mu_view, Sigma, rf, tickers, returns, tau=0.2, omega_scalar=0.1):
    n = len(mu_view)
    caps = []
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info
            caps.append(info.get("marketCap", 0))
        except Exception:
            caps.append(0)
    caps = np.array(caps, dtype=float)
    if np.nansum(caps) <= 0:
        market_weights = np.full(n, 1.0/n)
    else:
        market_weights = caps/np.nansum(caps)
    mu_view = np.asarray(mu_view, dtype=float).reshape(-1)
    Sigma_psd = _nearest_psd(Sigma)
    delta = market_implied_delta(returns, rf, market_weights)
    Pi = delta * (Sigma_psd @ market_weights)

    P = np.eye(n)
    Omega = np.eye(n) * omega_scalar
    A = np.linalg.inv(tau * Sigma_psd)
    post_prec = A + P.T @ np.linalg.inv(Omega) @ P
    post_mean = np.linalg.inv(post_prec) @ (A @ Pi + P.T @ np.linalg.inv(Omega) @ mu_view)

    w = cp.Variable(n)
    ret = post_mean @ w
    risk = cp.quad_form(w, Sigma_psd)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Maximize(ret - delta * risk), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    return post_mean, w.value

# --------------------------------
# Data Download & Processing
# --------------------------------
stock_data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
returns = resample_returns(stock_data, frequency)
treasury = web.DataReader("DGS5", "fred", start, end)
rf_annual = treasury["DGS5"].mean() / 100
rf = rf_annual / FREQUENCY_MAP[frequency]['rf_divisor']
mu = predict_returns(returns, n_lags)
Sigma = LedoitWolf().fit(returns).covariance_
w_mvo, w_sharpe = optimize_portfolio(mu, Sigma, rf, tickers, max_variance)
w_eq = equal_weight_portfolio(mu, Sigma, rf)
mu_bl, w_bl = black_litterman(mu, Sigma, rf, tickers, returns, tau, omega_scalar)

# --------------------------------
# Tabs for Outputs
# --------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Weights", "Efficient Frontier", "Cumulative Returns", "Metrics", "Rolling", "Correlation", "VaR & CVaR"]
)

with tab1:
    st.subheader("Portfolio Weights")
    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Equal Weight": w_eq,
        "MVO": w_mvo,
        "Max Sharpe": w_sharpe,
        "Black–Litterman": w_bl
    })
    st.dataframe(weights_df.set_index("Ticker"))

with tab2:
    st.subheader("Efficient Frontier")
    fig, ax = plt.subplots()
    ax.scatter(mu, np.sqrt(np.diag(Sigma)), c='blue', label='Assets')
    ax.set_xlabel("Expected Return")
    ax.set_ylabel("Volatility")
    ax.legend()
    st.pyplot(fig)

# (Remaining tabs would contain your rolling backtest, cumulative return plots, metrics table, rolling volatility/sharpe, correlation heatmap, VaR/CVaR plots)
