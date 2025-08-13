import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# 1️⃣ Load Historical Data
# -----------------------------
st.title("Portfolio Optimization Dashboard")
adj_close = pd.read_csv('adj_close_prices.csv', index_col=0, parse_dates=True)
tickers = adj_close.columns.tolist()

st.subheader("Historical Price Data")
st.dataframe(adj_close.tail())

# -----------------------------
# 2️⃣ Compute Returns
# -----------------------------
returns = adj_close.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
risk_free_rate = 0.01  # 1% annual

st.subheader("Returns & Covariance Snapshot")
st.write("Mean Returns:")
st.write(mean_returns.head())
st.write("Covariance Matrix:")
st.write(cov_matrix.iloc[:5, :5])

# -----------------------------
# 3️⃣ Predicted Returns (Random Forest)
# -----------------------------
# Placeholder for simplicity; normally you'd train RF on features
predicted_returns = mean_returns.values  # using historical mean as predicted
st.subheader("Predicted Returns (Placeholder)")
st.write(pd.Series(predicted_returns, index=tickers).head())

# -----------------------------
# 4️⃣ Portfolio Optimization Functions
# -----------------------------
def portfolio_perf(weights, mean_ret, cov):
    ret = np.dot(weights, mean_ret)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return ret, vol

def max_sharpe_portfolio(mean_ret, cov, rf):
    num_assets = len(mean_ret)
    weights = np.array([1/num_assets]*num_assets)  # placeholder
    ret, vol = portfolio_perf(weights, mean_ret, cov)
    sharpe = (ret - rf)/vol
    return weights, ret, vol, sharpe

def min_vol_portfolio(mean_ret, cov):
    num_assets = len(mean_ret)
    weights = np.array([1/num_assets]*num_assets)  # placeholder
    ret, vol = portfolio_perf(weights, mean_ret, cov)
    sharpe = (ret - risk_free_rate)/vol
    return weights, ret, vol, sharpe

def equal_weight_portfolio(num_assets):
    weights = np.array([1/num_assets]*num_assets)
    return weights

# Placeholder for Black-Litterman
def black_litterman_portfolio(mean_ret, cov):
    num_assets = len(mean_ret)
    weights = np.array([1/num_assets]*num_assets)
    return weights

# -----------------------------
# 5️⃣ Compute Strategies
# -----------------------------
strategies = {}

w_max, r_max, v_max, s_max = max_sharpe_portfolio(predicted_returns, cov_matrix, risk_free_rate)
strategies['Max Sharpe'] = w_max
w_min, r_min, v_min, s_min = min_vol_portfolio(predicted_returns, cov_matrix)
strategies['Min Volatility'] = w_min
strategies['Equal Weight'] = equal_weight_portfolio(len(tickers))
strategies['Black-Litterman'] = black_litterman_portfolio(predicted_returns, cov_matrix)

# Display
st.subheader("Portfolio Strategy Allocations")
alloc_df = pd.DataFrame({k: v for k, v in strategies.items()}, index=tickers)
st.dataframe(alloc_df.style.format("{:.2%}"))

# -----------------------------
# 6️⃣ Interactive User Portfolio
# -----------------------------
st.sidebar.subheader("Adjust Portfolio Weights")
user_weights = []
total_weight = 0
for ticker in tickers:
    w = st.sidebar.slider(f"{ticker} weight", 0.0, 1.0, 1.0/len(tickers), 0.01)
    user_weights.append(w)
    total_weight += w

# Normalize
user_weights = np.array(user_weights) / total_weight

st.subheader("User-Defined Portfolio Allocation")
user_alloc_df = pd.DataFrame({'Ticker': tickers, 'Weight': user_weights})
st.dataframe(user_alloc_df.style.format({"Weight": "{:.2%}"}))

user_ret, user_vol = portfolio_perf(user_weights, predicted_returns, cov_matrix)
user_sharpe = (user_ret - risk_free_rate)/user_vol

st.subheader("User-Defined Portfolio Performance")
st.write(f"Expected Return: {user_ret:.2%}")
st.write(f"Volatility: {user_vol:.2%}")
st.write(f"Sharpe Ratio: {user_sharpe:.2f}")

# -----------------------------
# 7️⃣ Cumulative Returns Plot
# -----------------------------
st.subheader("Cumulative Returns of Portfolios")
cumulative_df = pd.DataFrame()
for name, w in strategies.items():
    port_returns = (returns @ w)
    cumulative_df[name] = (1 + port_returns).cumprod()

cumulative_df['User Portfolio'] = (1 + returns @ user_weights).cumprod()

st.line_chart(cumulative_df)
