#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import pandas_datareader.data as web
from datetime import datetime
import cvxpy as cp
from sklearn.covariance import LedoitWolf

# Removed deprecated Streamlit option
st.title("Portfolio Optimization Dashboard")

TICKERS = st.multiselect("Select Assets:", [
    "JPM", "GS", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "HD", "KO", "XOM", "CVX", "UNH", "PFE",
    "CAT", "UNP", "NFLX", "DIS", "NEE", "PLD"
], default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"])

START = "2020-06-01"
END = "2025-06-01"

@st.cache_data
def download_data(tickers):
    return yf.download(tickers, start=START, end=END, auto_adjust=True)['Close']

if not TICKERS:
    st.warning("Please select at least one ticker.")
    st.stop()

stock_data = download_data(TICKERS)
st.write("### Adjusted Close Prices Sample")
st.dataframe(stock_data.tail())

returns_df = np.log(stock_data / stock_data.shift(1)).dropna()

# Risk-return summary
risk_return_summary = pd.DataFrame({
    'Mean Daily Return': returns_df.mean(),
    'Daily Volatility (Std Dev)': returns_df.std()
}).sort_values(by='Mean Daily Return', ascending=False)

st.write("### Risk-Return Summary")
st.dataframe(risk_return_summary)

# Correlation heatmaps
st.write("### Covariance and Correlation Matrices")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(returns_df.cov(), cmap='coolwarm', center=0, ax=axes[0])
axes[0].set_title('Covariance Matrix')
sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title('Correlation Matrix')
st.pyplot(fig)

# Risk-return scatter plot
fig2 = plt.figure(figsize=(10, 6))
sns.scatterplot(x='Daily Volatility (Std Dev)', y='Mean Daily Return', data=risk_return_summary, s=100)
for ticker in risk_return_summary.index:
    plt.text(risk_return_summary.loc[ticker, 'Daily Volatility (Std Dev)'] + 0.0001,
             risk_return_summary.loc[ticker, 'Mean Daily Return'],
             ticker, fontsize=9)
plt.xlabel('Daily Volatility (Std Dev)')
plt.ylabel('Mean Daily Return')
plt.title('Risk-Return Scatter Plot')
plt.grid(True)
st.pyplot(fig2)

# Feature Engineering for Random Forest
n_lags = 5
X_all = []
y_all_dict = {ticker: [] for ticker in TICKERS}

for i in range(n_lags, len(returns_df) - 1):
    lag_features = returns_df.iloc[i - n_lags:i].values.flatten()
    X_all.append(lag_features)
    for ticker in TICKERS:
        y_all_dict[ticker].append(returns_df.iloc[i + 1][ticker])

X = np.array(X_all)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

latest_input = returns_df.iloc[-n_lags:].values.flatten().reshape(1, -1)
latest_input_scaled = scaler.transform(latest_input)

# Model Training & Prediction
predicted_returns = []
for ticker in TICKERS:
    y = np.array(y_all_dict[ticker])
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled[:len(y)], y)
    pred = model.predict(latest_input_scaled)[0]
    predicted_returns.append(pred)

Pred_returns = np.array(predicted_returns)
pred_df = pd.DataFrame({'Ticker': TICKERS, 'Predicted Return': Pred_returns})
st.write("### Predicted Next-Day Returns")
st.dataframe(pred_df)

# Risk-Free Rate
rf_data = web.DataReader("DGS5", "fred", START, END)
avg_rf = rf_data["DGS5"].mean() / 100
rf_daily = avg_rf / 252

# --- Mean-Variance Optimization ---
st.header("Mean-Variance Optimization")
n = len(Pred_returns)
mu = Pred_returns
Sigma = LedoitWolf().fit(returns_df).covariance_

max_variance = st.slider("Select maximum acceptable portfolio variance:", min_value=0.00001, max_value=0.001, value=0.0001, step=0.00001)
weights_mvo = cp.Variable(n)
portfolio_return = mu @ weights_mvo
portfolio_variance = cp.quad_form(weights_mvo, Sigma)
constraints = [cp.sum(weights_mvo) == 1, portfolio_variance <= max_variance, weights_mvo >= 0]
objective = cp.Maximize(portfolio_return)
problem = cp.Problem(objective, constraints)
problem.solve()

w_mvo = weights_mvo.value
st.write("### Optimal Weights (MVO)")
st.bar_chart(pd.Series(w_mvo, index=TICKERS))
st.write(f"Expected Portfolio Return: {portfolio_return.value:.6f}")
st.write(f"Portfolio Variance: {portfolio_variance.value:.8f}")

# --- Max Sharpe Ratio ---
st.header("Maximum Sharpe Ratio Optimization")

def neg_sharpe(weights, ret, cov, rf):
    port_return = np.dot(weights, ret)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return -(port_return - rf) / port_vol

bounds = [(0, 0.2)] * n
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
initial = np.repeat(1/n, n)
cov_matrix = returns_df[-252:].cov().values

res = minimize(neg_sharpe, initial, args=(mu, cov_matrix, rf_daily), method='SLSQP', bounds=bounds, constraints=constraints)
w_sharpe = res.x

st.write("### Optimal Weights (Max Sharpe Ratio)")
st.bar_chart(pd.Series(w_sharpe, index=TICKERS))
port_return = np.dot(w_sharpe, mu)
port_vol = np.sqrt(np.dot(w_sharpe.T, np.dot(cov_matrix, w_sharpe)))
sharpe = (port_return - rf_daily) / port_vol
st.write(f"Expected Return: {port_return:.6f}")
st.write(f"Volatility: {port_vol:.6f}")
st.write(f"Sharpe Ratio: {sharpe:.4f}")

# --- Equal Weight Portfolio ---
st.header("Equal Weight Portfolio")
weights_eq = np.repeat(1/n, n)
ret_eq = np.dot(weights_eq, mu)
vol_eq = np.sqrt(np.dot(weights_eq.T, np.dot(cov_matrix, weights_eq)))
sharpe_eq = (ret_eq - rf_daily) / vol_eq
st.bar_chart(pd.Series(weights_eq, index=TICKERS))
st.write(f"Expected Return: {ret_eq:.6f}")
st.write(f"Volatility: {vol_eq:.6f}")
st.write(f"Sharpe Ratio: {sharpe_eq:.4f}")

# --- Black-Litterman ---
st.header("Black-Litterman Optimization")
market_caps = []
for ticker in TICKERS:
    try:
        cap = yf.Ticker(ticker).info.get("marketCap", 0)
        market_caps.append(cap)
    except:
        market_caps.append(0)
market_weights = pd.Series(market_caps, index=TICKERS)
market_weights = market_weights / market_weights.sum()

spy = yf.download("SPY", start=START, end=END, auto_adjust=True)['Close']
spy_returns = np.log(spy / spy.shift(1)).dropna()
delta = ((spy_returns.mean() - rf_daily) / spy_returns.var()).mean()

Sigma = LedoitWolf().fit(returns_df).covariance_
Sigma_inv = np.linalg.inv(Sigma)
Pi = delta * Sigma @ market_weights.values
Omega = np.eye(n) * 0.1
Omega_inv = np.linalg.inv(Omega)
P = np.eye(n)
tau = 0.2
combined = np.linalg.inv(tau * Sigma_inv + P.T @ Omega_inv @ P)
combined_mu = tau * Sigma_inv @ Pi + P.T @ Omega_inv @ mu
posterior_mu = combined @ combined_mu

w_bl = cp.Variable(n)
portfolio_variance = cp.quad_form(w_bl, Sigma)
constraints = [cp.sum(w_bl) == 1, w_bl >= 0]
objective = cp.Minimize(portfolio_variance)
problem = cp.Problem(objective, constraints)
problem.solve()

st.write("### Optimal Weights (Black-Litterman)")
st.bar_chart(pd.Series(w_bl.value, index=TICKERS))
st.write(f"Portfolio Variance: {portfolio_variance.value:.8f}")


# In[ ]:




