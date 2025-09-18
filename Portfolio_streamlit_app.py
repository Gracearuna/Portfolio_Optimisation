# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import cvxpy as cp
import pandas_datareader.data as web
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

# --- SIDEBAR INPUTS ---
st.sidebar.title("Portfolio Inputs")
tickers_input = st.sidebar.text_area(
    "Tickers (comma-separated)", 
    "AAPL,MSFT,GOOGL,META,NVDA,AMD,CRM,ADBE,CSCO,JPM,BAC,GS,MS,C,AXP,JNJ,PFE,MRK,UNH,LLY,ABT,AMZN,TSLA,HD,MCD,NKE,PG,KO,PEP,WMT,XOM,CVX,COP,SLB,UNP,CAT,HON,GE,DIS,CMCSA,VZ,PLD,AMT,SPG,DOW,NEM,SHW,NEE,DUK,SO"
)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-06-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-06-01"))
frequency = st.sidebar.selectbox("Frequency", ["daily", "weekly", "monthly", "annual"])
n_lags = st.sidebar.number_input("ML Lag Window", min_value=1, max_value=10, value=2)
max_weight = st.sidebar.slider("Max Weight per Asset", 0.0, 1.0, 0.3)

FREQUENCY_MAP = {
    "daily": {"resample": None, "rf_divisor": 252},
    "weekly": {"resample": "W-FRI", "rf_divisor": 52},
    "monthly": {"resample": "M", "rf_divisor": 12},
    "annual": {"resample": "Y", "rf_divisor": 1},
}

# --- FUNCTIONS ---
def resample_returns(stock_data, freq_key):
    rule = FREQUENCY_MAP[freq_key]["resample"]
    if rule:
        stock_data = stock_data.resample(rule).last()
    returns = np.log(stock_data / stock_data.shift(1)).dropna()
    return returns

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

def optimize_portfolio(mu, Sigma, rf, max_weight=0.3):
    n = len(mu)
    # MVO
    w_mvo = cp.Variable(n)
    portfolio_return = mu @ w_mvo
    portfolio_variance = cp.quad_form(w_mvo, Sigma)
    constraints = [cp.sum(w_mvo)==1, w_mvo>=0, w_mvo<=max_weight, portfolio_variance<=0.0002]
    cp.Problem(cp.Maximize(portfolio_return), constraints).solve()
    weights_mvo = w_mvo.value
    # Max Sharpe
    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
        return -(ret - rf)/vol
    bounds = [(0, max_weight)]*n
    constraints = [{'type':'eq','fun': lambda w: np.sum(w)-1}]
    result = minimize(neg_sharpe, np.repeat(1/n,n), method='SLSQP', bounds=bounds, constraints=constraints)
    weights_sharpe = result.x
    ret_sharpe = np.dot(weights_sharpe, mu)
    vol_sharpe = np.sqrt(weights_sharpe.T @ Sigma @ weights_sharpe)
    sharpe = (ret_sharpe - rf)/vol_sharpe
    # Equal weight
    w_eq = np.repeat(1/n, n)
    ret_eq = np.dot(w_eq, mu)
    vol_eq = np.sqrt(w_eq.T @ Sigma @ w_eq)
    sharpe_eq = (ret_eq - rf)/vol_eq
    return weights_mvo, weights_sharpe, w_eq, ret_sharpe, vol_sharpe, sharpe, ret_eq, vol_eq, sharpe_eq

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
    market_weights = caps/np.nansum(caps) if np.nansum(caps)>0 else np.full(n,1.0/n)
    Sigma_psd = _nearest_psd(np.asarray(Sigma))
    delta = market_implied_delta(returns, rf, market_weights)
    Pi = delta * (Sigma_psd @ market_weights)
    P = np.eye(n)
    Omega = np.eye(n)*omega_scalar
    A = np.linalg.inv(tau*Sigma_psd)
    post_prec = A + P.T @ np.linalg.inv(Omega) @ P
    post_mean = np.linalg.inv(post_prec) @ (A@Pi + P.T @ np.linalg.inv(Omega) @ mu_view)
    w = cp.Variable(n)
    ret = post_mean @ w
    risk = cp.quad_form(w, Sigma_psd)
    constraints = [cp.sum(w)==1, w>=0]
    prob = cp.Problem(cp.Maximize(ret - delta*risk), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    return post_mean, w.value

def cumulative_returns(log_returns):
    return np.exp(log_returns.cumsum())

def max_drawdown(cum_returns):
    return (cum_returns / cum_returns.cummax() - 1).min()

def rolling_metrics(log_returns, window=21, trading_days=252):
    rolling_vol = log_returns.rolling(window).std()*np.sqrt(trading_days)
    rolling_sharpe = (log_returns.rolling(window).mean()/log_returns.rolling(window).std())*np.sqrt(trading_days)
    return rolling_vol, rolling_sharpe

def get_portfolio_perf(weights, mu, Sigma, rf):
    port_return = np.dot(weights, mu)
    port_vol = np.sqrt(weights.T @ Sigma @ weights)
    sharpe = (port_return - rf)/port_vol
    return port_return, port_vol, sharpe

def efficient_frontier(mu, Sigma, points=50):
    n = len(mu)
    frontier_vols = []
    target_returns = np.linspace(min(mu), max(mu), points)
    bounds = [(0,1)]*n
    base_cons = {'type':'eq','fun': lambda x: np.sum(x)-1}
    for target in target_returns:
        cons = [base_cons, {'type':'eq','fun': lambda x,target=target: np.dot(x,mu)-target}]
        res = minimize(lambda w: np.sqrt(w.T@Sigma@w), np.repeat(1/n,n), method='SLSQP', bounds=bounds, constraints=cons)
        frontier_vols.append(res.fun if res.success else np.nan)
    return target_returns, np.array(frontier_vols)

# --- MAIN EXECUTION ---
st.title("📈 Portfolio Optimization Dashboard with Black-Litterman")

with st.spinner("Downloading stock data..."):
    stock_data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]

returns = resample_returns(stock_data, frequency)

st.subheader("Returns Head")
st.dataframe(returns.head())

# --- ML Predictions & Covariance ---
with st.spinner("Predicting returns & covariance..."):
    mu = predict_returns(returns, n_lags)
    Sigma = LedoitWolf().fit(returns).covariance_
    treasury = web.DataReader("DGS5","fred", start, end)
    rf_annual = treasury["DGS5"].mean()/100
    rf = rf_annual / FREQUENCY_MAP[frequency]["rf_divisor"]
    w_mvo, w_sharpe, w_eq, r_s, v_s, s_s, r_eq, v_eq, s_eq = optimize_portfolio(mu, Sigma, rf, max_weight)
    mu_bl, w_bl = black_litterman(mu, Sigma, rf, tickers, returns)

# --- Portfolio Weights ---
st.subheader("Portfolio Weights & Predicted Returns")
df_portfolios = pd.DataFrame({
    "Ticker": tickers,
    "MVO": w_mvo,
    "Max Sharpe": w_sharpe,
    "Equal Weight": w_eq,
    "Black-Litterman": w_bl,
    "Predicted Return": mu,
    "BL Posterior Return": mu_bl
})
st.dataframe(df_portfolios)

# --- Portfolio Metrics ---
st.subheader("Portfolio Metrics")
metrics_df = pd.DataFrame({
    "Metric": ["Return","Volatility","Sharpe"],
    "Max Sharpe":[r_s, v_s, s_s],
    "Equal Weight":[r_eq, v_eq, s_eq],
    "Black-Litterman":[np.dot(mu_bl, w_bl), np.sqrt(w_bl.T @ Sigma @ w_bl), (np.dot(mu_bl,w_bl)-rf)/np.sqrt(w_bl.T @ Sigma @ w_bl)]
})
st.dataframe(metrics_df)

# --- Cumulative Returns & Drawdowns ---
st.subheader("Cumulative Returns & Drawdowns")
cum_mvo = cumulative_returns(returns @ w_mvo)
cum_sharpe = cumulative_returns(returns @ w_sharpe)
cum_eq = cumulative_returns(returns @ w_eq)
cum_bl = cumulative_returns(returns @ w_bl)

fig, ax = plt.subplots(figsize=(12,6))
for name, cum in zip(["MVO","Max Sharpe","Equal Weight","Black-Litterman"], [cum_mvo,cum_sharpe,cum_eq,cum_bl]):
    ax.plot(cum, label=name)
    drawdown = cum / cum.cummax() - 1
    ax.fill_between(drawdown.index, cum, cum.cummax(), where=drawdown<0, alpha=0.1)
ax.set_title("Cumulative Returns & Drawdowns")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Rolling Metrics ---
st.subheader("Rolling Volatility & Sharpe")
fig2, ax2 = plt.subplots(figsize=(12,5))
for name, w in zip(["MVO","Max Sharpe","Equal Weight","Black-Litterman"], [w_mvo,w_sharpe,w_eq,w_bl]):
    log_ret = returns @ w
    roll_vol, roll_sharpe = rolling_metrics(log_ret)
    ax2.plot(roll_vol, label=f"{name} Volatility")
ax2.set_title("Rolling Volatility")
ax2.set_xlabel("Date")
ax2.set_ylabel("Volatility")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(12,5))
for name, w in zip(["MVO","Max Sharpe","Equal Weight","Black-Litterman"], [w_mvo,w_sharpe,w_eq,w_bl]):
    log_ret = returns @ w
    roll_vol, roll_sharpe = rolling_metrics(log_ret)
    ax3.plot(roll_sharpe, label=f"{name} Sharpe")
ax3.set_title("Rolling Sharpe Ratio")
ax3.set_xlabel("Date")
ax3.set_ylabel("Sharpe Ratio")
ax3.grid(True)
ax3.legend()
st.pyplot(fig3)

# --- Efficient Frontier ---
st.subheader("Efficient Frontier")
target_ret, frontier_vols = efficient_frontier(mu, Sigma)
fig4, ax4 = plt.subplots(figsize=(12,6))
ax4.plot(frontier_vols, target_ret, 'r--', label="Efficient Frontier")
for name, w in zip(["MVO","Max Sharpe","Equal Weight","Black-Litterman"], [w_mvo,w_sharpe,w_eq,w_bl]):
    r, v, s = get_portfolio_perf(w, mu, Sigma, rf)
    ax4.scatter(v, r, marker='X', s=200, label=f"{name} (Sharpe: {s:.2f})")
ax4.set_title("Efficient Frontier with Portfolios")
ax4.set_xlabel("Volatility")
ax4.set_ylabel("Expected Return")
ax4.grid(True)
ax4.legend()
st.pyplot(fig4)
