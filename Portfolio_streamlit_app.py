import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
from datetime import date

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Portfolio Settings")
DEFAULT_TICKERS = [
    "JPM", "GS", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "HD", "KO", "XOM", "CVX", "UNH", "PFE",
    "CAT", "UNP", "NFLX", "DIS", "NEE", "PLD"
]

tickers = st.sidebar.multiselect("Select Tickers", DEFAULT_TICKERS, default=DEFAULT_TICKERS[:10])
start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-06-01").date())
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-06-01").date())
frequency = st.sidebar.selectbox("Frequency", ["daily", "weekly", "monthly"])
n_lags = st.sidebar.slider("ML Lag Period", 1, 5, 2)
max_variance = st.sidebar.number_input("Max Variance (MVO)", 0.0001, 0.01, 0.0002, format="%.6f")
tau = st.sidebar.number_input("BL τ", 0.01, 1.0, 0.2, format="%.4f")
omega_scalar = st.sidebar.number_input("BL Ω Scalar", 0.01, 1.0, 0.1, format="%.4f")
rolling_window_days = st.sidebar.slider("Rolling Window (days)", 10, 126, 21)

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data
def download_prices(tickers, start, end):
    if not tickers: return pd.DataFrame()
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series): df = df.to_frame()
    return df.dropna(how="all")

def resample_returns(df, freq):
    freq_map = {'daily': None, 'weekly': 'W-FRI', 'monthly': 'M'}
    rule = freq_map.get(freq)
    if rule: df = df.resample(rule).last()
    returns = np.log(df / df.shift(1)).dropna()
    return returns

def get_rf(start, end):
    try:
        rf_series = web.DataReader("DGS5", "fred", start, end)
        rf_annual = float(rf_series["DGS5"].mean()) / 100
    except:
        rf_annual = 0.02
    return rf_annual

def predict_mu(returns, n_lags):
    if returns.shape[0] <= n_lags + 2: return returns.mean().values
    X, y_dict = [], {t: [] for t in returns.columns}
    for i in range(n_lags, len(returns)-1):
        lag = returns.iloc[i-n_lags:i].values.flatten()
        X.append(lag)
        for t in returns.columns: y_dict[t].append(returns.iloc[i+1][t])
    X = np.array(X); scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    latest = scaler.transform(returns.iloc[-n_lags:].values.flatten().reshape(1,-1))
    mu_pred = []
    for t in returns.columns:
        y = np.array(y_dict[t])
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_scaled[:len(y)], y)
        mu_pred.append(model.predict(latest)[0])
    return np.array(mu_pred)

def _nearest_psd(A, eps=1e-10):
    B = 0.5*(A + A.T)
    w, V = np.linalg.eigh(B)
    return (V * np.clip(w, eps, None)) @ V.T

def optimize_portfolio(mu, Sigma, rf, max_var):
    n = len(mu); Sigma = _nearest_psd(Sigma)
    # MVO
    w = cp.Variable(n)
    ret = mu @ w; risk = cp.quad_form(w, Sigma)
    constraints = [cp.sum(w)==1, w>=0, risk<=max_var]
    cp.Problem(cp.Maximize(ret), constraints).solve(solver=cp.SCS, verbose=False)
    w_mvo = np.nan_to_num(w.value)
    # Max Sharpe
    def neg_sharpe(w): return -((w@mu - rf)/np.sqrt(w.T @ Sigma @ w + 1e-12))
    res = minimize(neg_sharpe, np.repeat(1/n, n), method='SLSQP', bounds=[(0,0.2)]*n,
                   constraints={'type':'eq','fun':lambda w: np.sum(w)-1})
    w_sharpe = np.nan_to_num(res.x)
    return w_mvo, w_sharpe

def equal_weights(n): return np.repeat(1/n, n)

def black_litterman(mu_view, Sigma, rf, tickers, returns, tau, omega):
    n = len(mu_view); Sigma = _nearest_psd(Sigma)
    caps = []
    for tk in tickers:
        try: caps.append(yf.Ticker(tk).info.get("marketCap",0))
        except: caps.append(0)
    caps = np.array(caps); w_mkt = caps/np.nansum(caps) if np.nansum(caps)>0 else np.repeat(1/n,n)
    delta = max((returns.mean().values @ w_mkt - rf)/(w_mkt.T @ returns.cov().values @ w_mkt),0)
    Pi = delta * Sigma @ w_mkt
    P, Omega = np.eye(n), np.eye(n)*omega
    post = np.linalg.inv(tau*Sigma)**1 + P.T @ np.linalg.inv(Omega) @ P
    mu_post = np.linalg.inv(post) @ (np.linalg.inv(tau*Sigma)@Pi + P.T @ np.linalg.inv(Omega) @ mu_view)
    w = cp.Variable(n)
    ret, risk = mu_post@w, cp.quad_form(w,Sigma)
    cp.Problem(cp.Maximize(ret - delta*risk), [cp.sum(w)==1,w>=0]).solve(solver=cp.SCS, verbose=False)
    return mu_post, np.nan_to_num(w.value)

def get_portfolio_perf(weights, mu, Sigma, rf):
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights.T @ Sigma @ weights))
    sharpe = (ret - rf)/(vol + 1e-12)
    return ret, vol, sharpe

def efficient_frontier(mu, Sigma, points=100):
    n = len(mu)
    frontier_vols, frontier_rets = [], np.linspace(min(mu), max(mu), points)
    for target in frontier_rets:
        res = minimize(lambda w: np.sqrt(w.T@Sigma@w),
                       x0=np.repeat(1/n,n),
                       bounds=[(0,1)]*n,
                       constraints=[{'type':'eq','fun': lambda w: np.sum(w)-1},
                                    {'type':'eq','fun': lambda w, t=target: np.dot(w, mu)-t}])
        frontier_vols.append(res.fun if res.success else np.nan)
    return frontier_rets, np.array(frontier_vols)

# ---------------------------
# Data Pipeline
# ---------------------------
if not tickers: st.warning("Select at least one ticker."); st.stop()
prices = download_prices(tickers, start, end)
if prices.empty: st.error("No price data returned."); st.stop()
returns = resample_returns(prices, frequency)
rf_annual = get_rf(start, end)
rf = rf_annual / {'daily':252,'weekly':52,'monthly':12}[frequency]

mu = predict_mu(returns, n_lags)
Sigma = LedoitWolf().fit(returns).covariance_
w_mvo, w_sharpe = optimize_portfolio(mu, Sigma, rf, max_variance)
w_eq = equal_weights(len(mu))
mu_bl, w_bl = black_litterman(mu, Sigma, rf, tickers, returns, tau, omega_scalar)

portfolios = {"Equal": w_eq, "MVO": w_mvo, "Max Sharpe": w_sharpe, "BL": w_bl}

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Weights","Efficient Frontier","Cumulative Returns & Drawdowns","Performance Metrics","Rolling Volatility & Sharpe","Correlation Heatmap"]
)

# Tab 1: Portfolio Weights
with tab1:
    st.subheader("Optimized Portfolio Weights")
    df_w = pd.DataFrame({ "Ticker": tickers })
    for name, w in portfolios.items(): df_w[name] = w
    st.dataframe(df_w.set_index("Ticker"))
    st.download_button("Download Weights CSV", df_w.to_csv(index=False).encode(), "weights.csv")

    fig, ax = plt.subplots(figsize=(12,5))
    for i,(name,w) in enumerate(portfolios.items()):
        ax.bar(np.arange(len(tickers))+0.2*i, w, width=0.2, label=name)
    ax.set_xticks(np.arange(len(tickers))); ax.set_xticklabels(tickers, rotation=45)
    ax.set_ylabel("Weight"); ax.set_title("Portfolio Weights"); ax.legend(); ax.grid(True)
    st.pyplot(fig)

# Tab 2: Efficient Frontier
with tab2:
    st.subheader("Efficient Frontier & Random Portfolios")
    n = len(mu)
    Sigma_psd = _nearest_psd(Sigma)
    num_port = 2000
    results = np.zeros((3, num_port))
    for i in range(num_port):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = get_portfolio_perf(w, mu, Sigma_psd, rf)
        results[:, i] = [r, v, s]
    frontier_rets, frontier_vols = efficient_frontier(mu, Sigma_psd)
    fig, ax = plt.subplots(figsize=(10,6))
    sc = ax.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', alpha=0.5, label='Random Portfolios')
    ax.plot(frontier_vols, frontier_rets, 'r--', lw=2, label='Efficient Frontier')
    for name,w in portfolios.items():
        r,v,s = get_portfolio_perf(w, mu, Sigma_psd, rf)
        ax.scatter(v,r,marker='X',s=160,label=f"{name} (Sharpe: {s:.2f})")
    ax.set_xlabel("Volatility"); ax.set_ylabel("Expected Return"); ax.set_title(f"{frequency.title()} Efficient Frontier")
    ax.grid(True); ax.legend(); cbar=plt.colorbar(sc); cbar.set_label("Sharpe Ratio")
    st.pyplot(fig)

# Tab 3: Cumulative Returns & Drawdowns
with tab3:
    st.subheader("Cumulative Returns & Drawdowns")
    daily_port = pd.DataFrame({name: returns.values @ w for name,w in portfolios.items()}, index=returns.index)
    cumulative = np.exp(daily_port.cumsum())
    fig, ax = plt.subplots(figsize=(12,6))
    for col in cumulative.columns:
        series = cumulative[col]; running_max = series.cummax(); drawdown = series/running_max-1
        ax.plot(series.index, series, label=col)
        ax.fill_between(series.index, series, running_max, where=drawdown<0, alpha=0.2)
    ax.set_title("Cumulative Returns with Drawdowns"); ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return")
    ax.grid(True); ax.legend()
    st.pyplot(fig)
    st.session_state['daily_port'] = daily_port

# Tab 4: Performance Metrics
with tab4:
    st.subheader("Performance Metrics")
    if 'daily_port' not in st.session_state: st.info("Run backtest first."); st.stop()
    daily_df = st.session_state['daily_port']
    metrics = {}
    def max_dd(series): return float((series/series.cummax()-1).min())
    def cagr_log(logrets, td=252): return float(np.exp(logrets.sum()/max(len(logrets)/td,1e-12))-1)
    def ann_vol(logrets, td=252): return float(logrets.std()*np.sqrt(td))
    def sharpe_log(logrets, rf, td=252): return float(np.sqrt(td)*(logrets - rf/td).mean() / (logrets.std()+1e-12))
    for name in daily_df.columns:
        series = daily_df[name]; cum = np.exp(series.cumsum())
        metrics[name] = {"Max Drawdown": max_dd(cum),"CAGR": cagr_log(series),
                         "Volatility": ann_vol(series),"Sharpe": sharpe_log(series, rf_annual)}
    df_metrics = pd.DataFrame(metrics).T
    st.dataframe(df_metrics.style.format({"Max Drawdown":"{:.2%}","CAGR":"{:.2%}","Volatility":"{:.2%}","Sharpe":"{:.2f}"}))
    st.download_button("Download Metrics CSV", df_metrics.to_csv().encode(), "metrics.csv")

# Tab 5: Rolling Volatility & Sharpe
with tab5:
    st.subheader("Rolling Volatility & Sharpe")
    window = rolling_window_days
    fig, ax = plt.subplots(figsize=(12,5))
    for name in daily_df.columns:
        ax.plot(daily_df[name].rolling(window).std()*np.sqrt(252), label=name)
    ax.set_title(f"Rolling {window}-Day Annualized Volatility"); ax.set_xlabel("Date"); ax.set_ylabel("Volatility")
    ax.grid(True); ax.legend()
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(12,5))
    for name in daily_df.columns:
        rolling_mean = daily_df[name].rolling(window).mean()
        rolling_std = daily_df[name].rolling(window).std()
        rolling_sharpe = rolling_mean/(rolling_std+1e-12)*np.sqrt(252)
        ax.plot(rolling_sharpe, label=name)
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio"); ax.set_xlabel("Date"); ax.set_ylabel("Sharpe")
    ax.grid(True); ax.legend()
    st.pyplot(fig)

# Tab 6: Correlation Heatmap
with tab6:
    st.subheader("Correlation Heatmap (Daily Log Returns)")
    corr = daily_df.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Portfolio Daily Returns Correlation")
    st.pyplot(fig)
    st.dataframe(corr.style.format("{:.2f}"))
