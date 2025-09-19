import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import cvxpy as cp
import pandas_datareader.data as web
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# === PAGE CONFIG ===
st.set_page_config(layout="wide", page_title="Portfolio Optimization Dashboard")

# === SIDEBAR OPTIONS ===
frequency = st.sidebar.selectbox("Select Return Frequency", ['daily', 'weekly', 'monthly'])
n_lags = st.sidebar.slider("ML Lag Period", 1, 5, 2)
window_rolling = st.sidebar.slider("Rolling Window (days)", 5, 60, 21)

# === TICKERS & DATA ===
tickers = [
    "AAPL","MSFT","GOOGL","META","NVDA","AMD","CRM","ADBE","CSCO",
    "JPM","BAC","GS","MS","C","AXP",
    "JNJ","PFE","MRK","UNH","LLY","ABT",
    "AMZN","TSLA","HD","MCD","NKE",
    "PG","KO","PEP","WMT",
    "XOM","CVX","COP","SLB",
    "UNP","CAT","HON","GE",
    "DIS","CMCSA","VZ",
    "PLD","AMT","SPG",
    "DOW","NEM","SHW",
    "NEE","DUK","SO"
]

start = "2020-06-01"
end = "2025-06-01"

FREQUENCY_MAP = {
    'daily': {'resample': None, 'rf_divisor': 252},
    'weekly': {'resample': 'W-FRI', 'rf_divisor': 52},
    'monthly': {'resample': 'M', 'rf_divisor': 12}
}

# === FUNCTIONS ===

@st.cache_data
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data

def resample_returns(stock_data, freq_key):
    rule = FREQUENCY_MAP[freq_key]['resample']
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
        pred = model.predict(latest_input_scaled)[0]
        predicted_returns.append(pred)
    return np.array(predicted_returns)

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
    return float(max(delta,0.0))

def black_litterman(mu_view, Sigma, rf, tickers, returns, tau=0.2, omega_scalar=0.1):
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
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma_psd = _nearest_psd(Sigma)
    delta = market_implied_delta(returns, rf, market_weights)
    Pi = delta * (Sigma_psd @ market_weights)
    P = np.eye(n)
    Omega = np.eye(n) * omega_scalar
    A = np.linalg.inv(tau*Sigma_psd)
    post_prec = A + P.T @ np.linalg.inv(Omega) @ P
    post_mean = np.linalg.inv(post_prec) @ (A @ Pi + P.T @ np.linalg.inv(Omega) @ mu_view)
    # MVO on posterior mean
    w = cp.Variable(n)
    ret = post_mean @ w
    risk = cp.quad_form(w, Sigma_psd)
    constraints = [cp.sum(w)==1, w>=0]
    prob = cp.Problem(cp.Maximize(ret - delta*risk), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    weights_bl = np.nan_to_num(w.value)
    return post_mean, weights_bl

# === LOAD DATA & PREPARE RETURNS ===
stock_data = load_data(tickers, start, end)
returns = resample_returns(stock_data, frequency)
st.subheader("Stock Returns Head")
st.dataframe(returns.head())

# === RISK-FREE RATE ===
treasury = web.DataReader("DGS5","fred",start,end)
rf_annual = treasury["DGS5"].mean()/100
rf = rf_annual / FREQUENCY_MAP[frequency]['rf_divisor']

# === PREDICT RETURNS & OPTIMIZE ===
mu = predict_returns(returns, n_lags)
Sigma = LedoitWolf().fit(returns).covariance_

w_eq = equal_weight_portfolio(mu, Sigma, rf)
w_mvo, w_sharpe = optimize_portfolio(mu, Sigma, rf)
mu_bl, w_bl = black_litterman(mu, Sigma, rf, tickers, returns)

# === PORTFOLIO WEIGHTS PLOT ===
st.subheader("Portfolio Weights Comparison")
def plot_weights(weights_dict, tickers, title):
    fig, ax = plt.subplots(figsize=(14,6))
    width = 0.15
    for i,(name,w) in enumerate(weights_dict.items()):
        ax.bar(np.arange(len(tickers))+i*width, w, width=width, label=name)
    ax.set_xticks(np.arange(len(tickers))+width*(len(weights_dict)-1)/2)
    ax.set_xticklabels(tickers, rotation=45)
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

plot_weights({
    "Equal": w_eq,
    "MVO": w_mvo,
    "Max Sharpe": w_sharpe,
    "BL": w_bl
}, tickers, "Portfolio Weights Comparison")

# === REBALANCED DAILY RETURNS (1-year lookback, 1-year rebalance) ===
lookback_period = pd.DateOffset(years=1)
rebalance_frequency = pd.DateOffset(years=1)
start_date = pd.to_datetime(start)
end_date = pd.to_datetime(end)

dates_list = []
daily_mvo, daily_maxsharpe, daily_eq, daily_bl = [], [], [], []
current_start = start_date
while True:
    current_lookback_end = current_start + lookback_period
    current_out_sample_end = current_lookback_end + lookback_period
    if current_out_sample_end > end_date:
        break
    window_returns = returns.loc[current_start:current_lookback_end]
    mu_t = predict_returns(window_returns, n_lags)
    Sigma_t = LedoitWolf().fit(window_returns).covariance_
    mu_bl_t, w_bl_t = black_litterman(mu_t, Sigma_t, rf, tickers, window_returns)
    w_mvo_t, w_sharpe_t = optimize_portfolio(mu_t, Sigma_t, rf)
    w_eq_t = equal_weight_portfolio(mu_t, Sigma_t, rf)
    out_sample_returns = returns.loc[current_lookback_end + pd.Timedelta(days=1):current_out_sample_end]
    daily_mvo.extend(np.log1p(np.dot(out_sample_returns.values, w_mvo_t)))
    daily_maxsharpe.extend(np.log1p(np.dot(out_sample_returns.values, w_sharpe_t)))
    daily_eq.extend(np.log1p(np.dot(out_sample_returns.values, w_eq_t)))
    daily_bl.extend(np.log1p(np.dot(out_sample_returns.values, w_bl_t)))
    dates_list.extend(out_sample_returns.index)
    current_start += rebalance_frequency

daily_mvo = pd.Series(daily_mvo, index=dates_list)
daily_maxsharpe = pd.Series(daily_maxsharpe, index=dates_list)
daily_eq = pd.Series(daily_eq, index=dates_list)
daily_bl = pd.Series(daily_bl, index=dates_list)

portfolios = {"MVO": daily_mvo, "Max Sharpe": daily_maxsharpe, "Equal Weight": daily_eq, "BL": daily_bl}

# === CUMULATIVE RETURNS & DRAWDOWNS ===
st.subheader("Cumulative Returns & Drawdowns")
fig, ax = plt.subplots(figsize=(14,6))
for name, daily_log in portfolios.items():
    cum_ret = np.exp(daily_log.cumsum())
    ax.plot(cum_ret, label=name)
    running_max = cum_ret.cummax()
    drawdown = cum_ret/running_max - 1
    ax.fill_between(drawdown.index, cum_ret, running_max, where=drawdown<0, color='red', alpha=0.1)
ax.set_title("Cumulative Returns & Drawdowns")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.legend()
ax.grid(True)
st.pyplot(fig)

# === ROLLING METRICS ===
st.subheader("Rolling Volatility & Sharpe Ratio")
fig, ax = plt.subplots(2,1, figsize=(14,10))
for name, daily_log in portfolios.items():
    rolling_vol = daily_log.rolling(window_rolling).std() * np.sqrt(252)
    ax[0].plot(rolling_vol, label=name)
ax[0].set_title(f"Rolling {window_rolling}-Day Volatility")
ax[0].set_xlabel("Date"); ax[0].set_ylabel("Volatility")
ax[0].legend(); ax[0].grid(True)

for name, daily_log in portfolios.items():
    rolling_sharpe = daily_log.rolling(window_rolling).mean() / daily_log.rolling(window_rolling).std() * np.sqrt(252)
    ax[1].plot(rolling_sharpe, label=name)
ax[1].set_title(f"Rolling {window_rolling}-Day Sharpe Ratio")
ax[1].set_xlabel("Date"); ax[1].set_ylabel("Sharpe Ratio")
ax[1].legend(); ax[1].grid(True)
st.pyplot(fig)

# === CORRELATION HEATMAP ===
st.subheader("Portfolio Correlation Heatmap")
daily_df = pd.DataFrame(portfolios)
corr_matrix = daily_df.corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.write("Portfolio Correlation Matrix:")
st.dataframe(corr_matrix)
