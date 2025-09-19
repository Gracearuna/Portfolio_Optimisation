# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import cvxpy as cp
import pandas_datareader.data as web
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

st.title("📊 Portfolio Optimization Dashboard")

# --- Sidebar Settings ---
st.sidebar.header("Settings")
tickers = st.sidebar.multiselect(
    "Select tickers:",
    [
        "JPM", "GS", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "AMZN", "HD", "KO", "XOM", "CVX", "UNH", "PFE",
        "CAT", "UNP", "NFLX", "DIS", "NEE", "PLD"
    ],
    default=["JPM", "GS", "AAPL", "MSFT", "NVDA"]
)

start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-06-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-06-01"))
frequency = st.sidebar.selectbox("Return Frequency", ["daily", "weekly", "monthly", "annual"])
n_lags = st.sidebar.slider("ML Lookback Lags", 1, 10, 2)

FREQUENCY_MAP = {
    'daily': {'resample': None, 'rf_divisor': 252},
    'weekly': {'resample': 'W-FRI', 'rf_divisor': 52},
    'monthly': {'resample': 'M', 'rf_divisor': 12},
    'annual': {'resample': 'Y', 'rf_divisor': 1}
}

# --- Download Stock Data ---
@st.cache_data
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data

stock_data = load_data(tickers, start, end)
st.subheader("Stock Prices")
st.dataframe(stock_data.tail())

# --- Returns ---
def resample_returns(stock_data, freq_key):
    rule = FREQUENCY_MAP[freq_key]['resample']
    if rule:
        stock_data = stock_data.resample(rule).last()
    returns = np.log(stock_data / stock_data.shift(1)).dropna()
    return returns

returns = resample_returns(stock_data, frequency)

# --- ML Forecast ---
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

mu = predict_returns(returns, n_lags)
Sigma = LedoitWolf().fit(returns).covariance_

# --- Risk-Free Rate ---
treasury = web.DataReader("DGS5", "fred", start, end)
rf_annual = treasury["DGS5"].mean() / 100
rf = rf_annual / FREQUENCY_MAP[frequency]['rf_divisor']

# --- Portfolio Functions ---
def get_portfolio_perf(weights, mu, Sigma, rf):
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
    sharpe = (port_return - rf) / port_vol
    return port_return, port_vol, sharpe

def equal_weight_portfolio(mu, Sigma, rf):
    n = len(mu)
    w_eq = np.repeat(1/n, n)
    return w_eq, *get_portfolio_perf(w_eq, mu, Sigma, rf)

def optimize_portfolio(mu, Sigma, rf, tickers, max_variance=0.0002):
    n = len(mu)
    w_mvo = cp.Variable(n)
    portfolio_return = mu @ w_mvo
    portfolio_variance = cp.quad_form(w_mvo, Sigma)
    max_weight = 0.3
    constraints = [
        cp.sum(w_mvo) == 1,
        w_mvo >= 0,
        w_mvo <= max_weight,
        portfolio_variance <= max_variance
    ]
    prob = cp.Problem(cp.Maximize(portfolio_return), constraints)
    prob.solve()
    weights_mvo = w_mvo.value

    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
        return -(ret - rf)/vol
    bounds = [(0,0.2)]*n
    constraints_s = [{'type':'eq','fun':lambda w: np.sum(w)-1}]
    init_guess = np.repeat(1/n,n)
    result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints_s)
    weights_sharpe = result.x

    return weights_mvo, weights_sharpe

# --- Calculate Portfolios ---
w_eq, ret_eq, vol_eq, sharpe_eq = equal_weight_portfolio(mu, Sigma, rf)
weights_mvo, weights_sharpe = optimize_portfolio(mu, Sigma, rf, tickers)

# --- Display Portfolio Weights (formatted as %) ---
df_weights = pd.DataFrame({
    "Ticker": tickers,
    "Equal Weight": w_eq,
    "MVO": weights_mvo,
    "Max Sharpe": weights_sharpe
})
df_weights_display = df_weights.copy()
for col in ["Equal Weight", "MVO", "Max Sharpe"]:
    df_weights_display[col] = df_weights_display[col].apply(lambda x: f"{x:.2%}")

st.subheader("Portfolio Weights")
st.dataframe(df_weights_display)

# --- Cumulative Returns Simulation ---
daily_log_returns = returns
cum_eq = np.exp(np.dot(daily_log_returns, w_eq).cumsum())
cum_mvo = np.exp(np.dot(daily_log_returns, weights_mvo).cumsum())
cum_sharpe = np.exp(np.dot(daily_log_returns, weights_sharpe).cumsum())

st.subheader("Cumulative Returns")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(cum_eq, label="Equal Weight")
ax.plot(cum_mvo, label="MVO")
ax.plot(cum_sharpe, label="Max Sharpe")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.legend()
st.pyplot(fig)

# --- Correlation Heatmap ---
st.subheader("Correlation Heatmap of Daily Returns")
fig2, ax2 = plt.subplots(figsize=(10,8))
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# --- Risk/Return Scatter ---
st.subheader("Risk-Return Scatter")
risk_return = pd.DataFrame({
    "Mean Return": returns.mean(),
    "Volatility": returns.std()
})
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.scatterplot(x="Volatility", y="Mean Return", data=risk_return, s=100, ax=ax3)
for i, ticker in enumerate(risk_return.index):
    ax3.text(risk_return.Volatility[i]+0.0001, risk_return["Mean Return"][i], ticker)
ax3.set_xlabel("Volatility")
ax3.set_ylabel("Mean Return")
st.pyplot(fig3)
