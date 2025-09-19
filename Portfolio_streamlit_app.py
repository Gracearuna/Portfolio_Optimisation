import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
all_tickers = [
    "JPM", "GS", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "HD", "KO", "XOM", "CVX", "UNH", "PFE",
    "CAT", "UNP", "NFLX", "DIS", "NEE", "PLD"
]

selected_tickers = st.sidebar.multiselect(
    "Select stocks (default all 20)",
    all_tickers,
    default=all_tickers
)

if len(selected_tickers) == 0:
    st.warning("Please select at least 1 ticker.")
    st.stop()

tickers = selected_tickers

# === FREQUENCY MAP ===
FREQUENCY_MAP = {
    'daily': {'resample': None, 'rf_divisor': 252},
    'weekly': {'resample': 'W-FRI', 'rf_divisor': 52},
    'monthly': {'resample': 'M', 'rf_divisor': 12}
}

start = "2020-06-01"
end = "2025-06-01"

# === DOWNLOAD DATA ===
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

st.subheader("Returns Head")
st.dataframe(returns.head())

# === RISK-RETURN SUMMARY ===
risk_return_summary = pd.DataFrame({
    'Mean Return': returns.mean(),
    'Volatility': returns.std()
}).sort_values(by='Mean Return', ascending=False)

st.subheader("Risk-Return Summary")
st.dataframe(risk_return_summary)

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

# === ADAPTIVE MVO FUNCTION ===
def optimize_portfolio(mu, Sigma, rf, max_variance=None):
    n = len(mu)
    if max_variance is None:
        scale_factor = np.trace(Sigma)/n
        max_variance = scale_factor * 2

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

# === EQUAL WEIGHT PORTFOLIO ===
def equal_weight_portfolio(mu, Sigma, rf):
    n = len(mu)
    w_eq = np.repeat(1/n, n)
    return w_eq

w_eq = equal_weight_portfolio(mu, Sigma, rf)
w_mvo, w_sharpe = optimize_portfolio(mu, Sigma, rf)

# === PORTFOLIO WEIGHTS PLOT ===
def plot_weights(weights_dict, tickers, title):
    n_assets = len(tickers)
    width = 0.15
    fig, ax = plt.subplots(figsize=(12,6))
    for i, (name, w) in enumerate(weights_dict.items()):
        w = np.nan_to_num(w)
        ax.bar(np.arange(n_assets)+i*width, w, width=width, label=name)
    ax.set_xticks(np.arange(n_assets) + width*(len(weights_dict)-1)/2)
    ax.set_xticklabels(tickers, rotation=45)
    ax.set_ylabel("Weights")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

plot_weights({
    "Equal": w_eq,
    "MVO": w_mvo,
    "Max Sharpe": w_sharpe
}, tickers, "Portfolio Weights Comparison")

# === SAFE PORTFOLIO RETURN FUNCTION ===
def safe_portfolio_return(returns, weights):
    weights = np.nan_to_num(weights).flatten()
    return returns.values @ weights

# === PORTFOLIO METRICS ===
def max_drawdown(cum_ret):
    return (cum_ret / cum_ret.cummax() - 1).min()

def cagr_log(daily_log_returns, trading_days=252):
    total_log_return = daily_log_returns.sum()
    total_years = len(daily_log_returns)/trading_days
    return np.exp(total_log_return/total_years)-1

def annualized_volatility(daily_log_returns, trading_days=252):
    return daily_log_returns.std()*np.sqrt(trading_days)

def sharpe_ratio_log(daily_log_returns, rf_annual=rf_annual, trading_days=252):
    rf_daily = rf_annual/trading_days
    excess = daily_log_returns - rf_daily
    return np.sqrt(trading_days)*excess.mean()/excess.std()

portfolios = {
    "Equal": safe_portfolio_return(returns, w_eq),
    "MVO": safe_portfolio_return(returns, w_mvo),
    "Max Sharpe": safe_portfolio_return(returns, w_sharpe)
}

metrics = {}
for name, daily_ret in portfolios.items():
    cum_ret = np.exp(pd.Series(daily_ret).cumsum())
    metrics[name] = {
        "Max Drawdown": max_drawdown(cum_ret),
        "CAGR": cagr_log(pd.Series(daily_ret)),
        "Volatility": annualized_volatility(pd.Series(daily_ret)),
        "Sharpe Ratio": sharpe_ratio_log(pd.Series(daily_ret))
    }

st.subheader("Portfolio Metrics")
st.dataframe(pd.DataFrame(metrics).T)

# === CUMULATIVE RETURNS PLOT ===
plt.figure(figsize=(12,6))
for name, daily_ret in portfolios.items():
    cum_ret = np.exp(pd.Series(daily_ret).cumsum())
    plt.plot(cum_ret, label=name)
plt.title("Portfolio Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
st.pyplot(plt)

# === CORRELATION MATRIX ===
daily_df = pd.DataFrame({name: portfolios[name] for name in portfolios})
st.subheader("Portfolio Daily Returns Correlation")
st.dataframe(daily_df.corr())
