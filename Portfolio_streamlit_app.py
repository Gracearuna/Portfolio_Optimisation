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
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Portfolio Optimization Dashboard")

# === SIDEBAR OPTIONS ===
frequency = st.sidebar.selectbox("Select Return Frequency", ['daily', 'weekly', 'monthly'])
n_lags = st.sidebar.slider("ML Lag Period", 1, 5, 2)

# Selected 20 tickers
tickers = [
    "JPM", "GS", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "HD", "KO", "XOM", "CVX", "UNH", "PFE",
    "CAT", "UNP", "NFLX", "DIS", "NEE", "PLD"
]

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

# === PORTFOLIO OPTIMIZATION ===
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

def equal_weight_portfolio(mu, Sigma, rf):
    n = len(mu)
    w_eq = np.repeat(1/n, n)
    return w_eq

w_eq = equal_weight_portfolio(mu, Sigma, rf)
w_mvo, w_sharpe = optimize_portfolio(mu, Sigma, rf)

# === BLACK-LITTERMAN PORTFOLIO ===
def _nearest_psd(A, eps=1e-10):
    B = 0.5*(A + A.T)
    w, V = np.linalg.eigh(B)
    w_clipped = np.clip(w, eps, None)
    return (V * w_clipped) @ V.T

def market_implied_delta(returns, rf, market_weights):
    mu_mkt = returns.mean().values @ market_weights
    var_mkt = market_weights.T @ returns.cov().values @ market_weights
    delta = (mu_mkt - rf)/max(var_mkt, 1e-12)
    return float(max(delta, 0.0))

def black_litterman(mu_view, Sigma, rf, tickers, returns, tau=0.2, omega_scalar=0.1):
    n = len(mu_view)
    caps = []
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info
            caps.append(info.get("marketCap",0))
        except Exception:
            caps.append(0)
    caps = np.array(caps, dtype=float)
    if np.nansum(caps) <= 0:
        market_weights = np.full(n, 1.0/n)
    else:
        market_weights = caps/np.nansum(caps)
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

mu_bl, w_bl = black_litterman(mu, Sigma, rf, tickers, returns)

# === SAFE PORTFOLIO RETURNS ===
def safe_portfolio_return(returns, weights):
    weights = np.nan_to_num(weights).flatten()
    return returns.values @ weights

portfolios = {
    "Equal": safe_portfolio_return(returns, w_eq),
    "MVO": safe_portfolio_return(returns, w_mvo),
    "Max Sharpe": safe_portfolio_return(returns, w_sharpe),
    "BL": safe_portfolio_return(returns, w_bl)
}

# === PERFORMANCE METRICS ===
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

# === SIDEBAR DATE PICKER ===
default_start = datetime(2021, 7, 1)
plot_start_date = st.sidebar.date_input("Select Start Date for Plots", value=default_start, min_value=returns.index.min(), max_value=returns.index.max())
plot_start_date = pd.to_datetime(plot_start_date)

# === TABS ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Portfolio Weights", "Efficient Frontier", "Cumulative Returns & Drawdowns",
    "Rolling Volatility & Sharpe", "Correlation Heatmap"
])

# === TAB 1: Portfolio Weights ===
with tab1:
    st.subheader("Portfolio Weights Comparison")
    def plot_weights(weights_dict, tickers, title):
        n_assets = len(tickers)
        width = 0.15
        fig, ax = plt.subplots(figsize=(12,6))
        for i, (name, w) in enumerate(weights_dict.items()):
            w = np.nan_to_num(w)
            ax.bar(np.arange(n_assets)+i*width, w, width=width, label=name)
        ax.set_xticks(np.arange(n_assets)+width*(len(weights_dict)-1)/2)
        ax.set_xticklabels(tickers, rotation=45)
        ax.set_ylabel("Weights")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    plot_weights({
        "Equal": w_eq,
        "MVO": w_mvo,
        "Max Sharpe": w_sharpe,
        "BL": w_bl
    }, tickers, "Portfolio Weights Comparison")

# === TAB 2: Efficient Frontier ===
with tab2:
    st.subheader("Efficient Frontier & Random Portfolios")
    def efficient_frontier(mu, Sigma, rf, n_points=50):
        n = len(mu)
        w_list, rets, vols, sharpes = [], [], [], []
        for target_ret in np.linspace(mu.min(), mu.max(), n_points):
            w = cp.Variable(n)
            ret = mu @ w
            risk = cp.quad_form(w, Sigma)
            constraints = [cp.sum(w)==1, w>=0, ret==target_ret]
            prob = cp.Problem(cp.Minimize(risk), constraints)
            prob.solve(solver=cp.SCS, verbose=False)
            if w.value is not None:
                w_list.append(np.nan_to_num(w.value))
                rets.append(target_ret)
                vol = np.sqrt(risk.value)
                vols.append(vol)
                sharpes.append((target_ret-rf)/vol)
        return rets, vols, sharpes

    rets_ef, vols_ef, sharpes_ef = efficient_frontier(mu, Sigma, rf)
    plt.figure(figsize=(12,6))
    plt.plot(vols_ef, rets_ef, 'g--', label="Efficient Frontier")
    plt.scatter(np.sqrt(np.diag(Sigma)), mu, c='red', label='Individual Stocks')
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# === TAB 3: Cumulative Returns & Drawdowns ===
with tab3:
    st.subheader("Cumulative Returns & Drawdowns")
    # Slice returns starting from selected date
    dates = returns.index[returns.index >= plot_start_date]

    # Cumulative Returns
    plt.figure(figsize=(12,6))
    for name, daily_ret in portfolios.items():
        daily_series = pd.Series(daily_ret, index=returns.index)
        daily_series = daily_series[daily_series.index >= plot_start_date]
        cum_ret = np.exp(daily_series.cumsum())
        plt.plot(cum_ret, label=name)
    plt.title("Portfolio Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Drawdowns
    plt.figure(figsize=(12,6))
    for name, daily_ret in portfolios.items():
        daily_series = pd.Series(daily_ret, index=returns.index)
        daily_series = daily_series[daily_series.index >= plot_start_date]
        cum_ret = np.exp(daily_series.cumsum())
        drawdown = cum_ret / cum_ret.cummax() - 1
        plt.plot(drawdown, label=name)
    plt.title("Portfolio Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    st.pyplot(plt)

# === TAB 4: Rolling Volatility & Sharpe ===
with tab4:
    st.subheader("Rolling Volatility & Sharpe Ratio")
    rolling_window = 21

    # Rolling Volatility
    plt.figure(figsize=(12,6))
    for name, daily_ret in portfolios.items():
        daily_series = pd.Series(daily_ret, index=returns.index)
        daily_series = daily_series[daily_series.index >= plot_start_date]
        roll_vol = daily_series.rolling(rolling_window).std() * np.sqrt(FREQUENCY_MAP[frequency]['rf_divisor'])
        plt.plot(roll_vol, label=name)
    plt.title(f"Rolling {rolling_window}-day Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Rolling Sharpe
    plt.figure(figsize=(12,6))
    for name, daily_ret in portfolios.items():
        daily_series = pd.Series(daily_ret, index=returns.index)
        daily_series = daily_series[daily_series.index >= plot_start_date]
        roll_sharpe = (daily_series - rf).rolling(rolling_window).mean() / daily_series.rolling(rolling_window).std() * np.sqrt(FREQUENCY_MAP[frequency]['rf_divisor'])
        plt.plot(roll_sharpe, label=name)
    plt.title(f"Rolling {rolling_window}-day Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    st.pyplot(plt)

# === TAB 5: Correlation Heatmap ===
with tab5:
    st.subheader("Portfolio Daily Returns Correlation")
    daily_df = pd.DataFrame({name: portfolios[name] for name in portfolios})
    st.dataframe(daily_df.corr())
    plt.figure(figsize=(10,6))
    sns.heatmap(daily_df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)
