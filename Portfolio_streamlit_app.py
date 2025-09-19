import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import cvxpy as cp
import pandas_datareader.data as web
import warnings

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")
warnings.filterwarnings("ignore", category=UserWarning)

# --- Sidebar Inputs ---
st.sidebar.title("Settings")
frequency = st.sidebar.selectbox("Return Frequency", ["daily", "weekly", "monthly", "annual"])
n_lags = st.sidebar.slider("Lag Days for ML Prediction", min_value=1, max_value=5, value=2)

tickers = [
   "JPM", "GS", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "HD", "KO", "XOM", "CVX", "UNH", "PFE",
    "CAT", "UNP", "NFLX", "DIS", "NEE", "PLD"
]

start = "2020-06-01"
end = "2025-06-01"

FREQUENCY_MAP = {
    'daily': {'resample': None, 'rf_divisor': 252},
    'weekly': {'resample': 'W-FRI', 'rf_divisor': 52},
    'monthly': {'resample': 'M', 'rf_divisor': 12},
    'annual': {'resample': 'Y', 'rf_divisor': 1}
}

@st.cache_data
def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data

stock_data = download_data(tickers, start, end)
st.subheader("Stock Data")
st.dataframe(stock_data.tail())

# --- Calculate Returns ---
def resample_returns(stock_data, freq_key):
    rule = FREQUENCY_MAP[freq_key]['resample']
    if rule:
        stock_data = stock_data.resample(rule).last()
    returns = np.log(stock_data / stock_data.shift(1)).dropna()
    return returns

returns = resample_returns(stock_data, frequency)
st.subheader(f"{frequency.title()} Log Returns")
st.dataframe(returns.tail())

# --- Risk-Return Summary ---
risk_return_summary = pd.DataFrame({
    'Mean Return': returns.mean(),
    'Volatility': returns.std()
}).sort_values(by='Mean Return', ascending=False)

st.subheader("Risk-Return Summary")
st.dataframe(risk_return_summary)

# --- Covariance & Correlation ---
cov_matrix = returns.cov()
corr_matrix = returns.corr()

st.subheader("Covariance Matrix")
fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(cov_matrix, cmap='coolwarm', center=0, ax=ax)
st.pyplot(fig)

st.subheader("Correlation Matrix")
fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax)
st.pyplot(fig)

# --- ML Predicted Returns ---
def predict_returns(returns, n_lags):
    X_all = []
    y_all_dict = {ticker: [] for ticker in returns.columns}
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
        predicted_returns.append(model.predict(latest_input_scaled)[0])
    return np.array(predicted_returns)

mu = predict_returns(returns, n_lags)
st.subheader("ML Predicted Returns")
st.dataframe(pd.DataFrame({"Ticker": tickers, "Predicted Return": mu}))

# --- Portfolio Optimization Functions ---
Sigma = LedoitWolf().fit(returns).covariance_
treasury = web.DataReader("DGS5", "fred", start, end)
rf_annual = treasury["DGS5"].mean() / 100
rf = rf_annual / FREQUENCY_MAP[frequency]['rf_divisor']

def equal_weight_portfolio(mu, Sigma, rf):
    n = len(mu)
    w_eq = np.repeat(1/n, n)
    ret = np.dot(w_eq, mu)
    vol = np.sqrt(np.dot(w_eq.T, np.dot(Sigma, w_eq)))
    sharpe = (ret - rf)/vol
    return w_eq, ret, vol, sharpe

def optimize_portfolio(mu, Sigma, rf, max_variance=0.0002):
    n = len(mu)
    w_mvo = cp.Variable(n)
    portfolio_return = mu @ w_mvo
    portfolio_variance = cp.quad_form(w_mvo, Sigma)
    max_weight = 0.3
    constraints = [cp.sum(w_mvo)==1, w_mvo>=0, w_mvo<=max_weight, portfolio_variance<=max_variance]
    prob = cp.Problem(cp.Maximize(portfolio_return), constraints)
    prob.solve()
    weights_mvo = w_mvo.value

    # Max Sharpe
    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
        return -(ret - rf)/vol

    bounds = [(0,0.2)]*n
    cons = [{'type':'eq', 'fun': lambda w: np.sum(w)-1}]
    init_guess = np.repeat(1/n, n)
    res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    weights_sharpe = res.x
    return weights_mvo, weights_sharpe

w_eq, _, _, _ = equal_weight_portfolio(mu, Sigma, rf)
weights_mvo, weights_sharpe = optimize_portfolio(mu, Sigma, rf)

# --- Portfolio Weights Plots ---
st.subheader("Portfolio Weights")
def plot_weights(weights, tickers, title):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(tickers, weights)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    st.pyplot(fig)

plot_weights(w_eq, tickers, "Equal Weight Portfolio")
plot_weights(weights_mvo, tickers, "MVO Portfolio")
plot_weights(weights_sharpe, tickers, "Max Sharpe Portfolio")

# --- Cumulative Returns ---
daily_portfolio_returns = np.dot(returns.values, np.vstack([w_eq, weights_mvo, weights_sharpe]).T)
cum_returns = pd.DataFrame(np.exp(np.log1p(daily_portfolio_returns).cumsum()),
                           columns=["Equal Weight","MVO","Max Sharpe"],
                           index=returns.index)
st.subheader("Cumulative Returns")
st.line_chart(cum_returns)

# --- Drawdowns ---
st.subheader("Portfolio Drawdowns")
fig, ax = plt.subplots(figsize=(12,6))
for col in cum_returns.columns:
    running_max = cum_returns[col].cummax()
    drawdown = cum_returns[col]/running_max - 1
    ax.fill_between(drawdown.index, cum_returns[col], running_max, alpha=0.1)
    ax.plot(cum_returns[col], label=col)
ax.set_title("Cumulative Returns with Drawdowns")
ax.legend()
st.pyplot(fig)

# --- Correlation of Portfolios ---
st.subheader("Portfolio Correlation")
port_corr = cum_returns.pct_change().corr()
st.dataframe(port_corr)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(port_corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
