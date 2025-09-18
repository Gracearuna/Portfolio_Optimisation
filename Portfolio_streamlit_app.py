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
from datetime import date

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

# --------------------------------
# Sidebar Controls
# --------------------------------
st.sidebar.header("Portfolio Settings")
DEFAULT_TICKERS = [
    "JPM", "GS", "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "HD", "KO", "XOM", "CVX", "UNH", "PFE",
    "CAT", "UNP", "NFLX", "DIS", "NEE", "PLD"
]

tickers = st.sidebar.multiselect(
    "Select Tickers",
    DEFAULT_TICKERS,
    default=["JPM", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "XOM", "UNH", "PLD"],
)

start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-06-01").date())
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-06-01").date())

n_lags = st.sidebar.slider("Number of Lags (ML)", 1, 5, 2)
frequency = st.sidebar.selectbox("Frequency", ["daily", "weekly", "monthly", "annual"], index=0)
max_variance = st.sidebar.number_input("Max Variance Constraint (MVO)", value=0.0002, format="%.6f")

tau = st.sidebar.number_input("Black–Litterman τ", value=0.2, format="%.4f")
omega_scalar = st.sidebar.number_input("Black–Litterman Ω Scalar", value=0.1, format="%.4f")

# Backtest controls
st.sidebar.markdown("---")
st.sidebar.header("Backtest Settings")
lookback_years = st.sidebar.slider("Lookback Window (years)", 1, 3, 1)
rebalance_months = st.sidebar.slider("Rebalance Frequency (months)", 1, 12, 12)
rolling_window_days = st.sidebar.slider("Rolling Window (days)", 10, 126, 21)

FREQUENCY_MAP = {
    'daily': {'resample': None, 'rf_divisor': 252},
    'weekly': {'resample': 'W-FRI', 'rf_divisor': 52},
    'monthly': {'resample': 'M', 'rf_divisor': 12},
    'annual': {'resample': 'Y', 'rf_divisor': 1}
}

# --------------------------------
# Helpers & Models
# --------------------------------
@st.cache_data(show_spinner=False)
def download_prices(tickers, start, end):
    if len(tickers) == 0:
        return pd.DataFrame()
    df = yf.download(tickers, start=pd.to_datetime(start), end=pd.to_datetime(end), auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how='all')

@st.cache_data(show_spinner=False)
def resample_returns(stock_data: pd.DataFrame, freq_key: str) -> pd.DataFrame:
    if stock_data.empty:
        return stock_data
    rule = FREQUENCY_MAP[freq_key]['resample']
    if rule:
        stock_data = stock_data.resample(rule).last()
    returns = np.log(stock_data / stock_data.shift(1)).dropna()
    return returns

@st.cache_data(show_spinner=False)
def get_rf_series(start, end):
    try:
        treasury = web.DataReader("DGS5", "fred", start, end)
        rf_annual = float(treasury["DGS5"].mean()) / 100.0
    except Exception:
        rf_annual = 0.02  # fallback 2%
    return rf_annual

def predict_returns(returns: pd.DataFrame, n_lags: int) -> np.ndarray:
    if returns.shape[0] <= n_lags + 2:
        return returns.mean().values  # fallback to historical mean
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

def _nearest_psd(A, eps=1e-10):
    B = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(B)
    w_clipped = np.clip(w, eps, None)
    return (V * w_clipped) @ V.T

def optimize_portfolio(mu, Sigma, rf, max_variance=0.0002):
    n = len(mu)
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma_psd = _nearest_psd(Sigma)

    # MVO
    w_mvo = cp.Variable(n)
    portfolio_return = mu @ w_mvo
    portfolio_variance = cp.quad_form(w_mvo, Sigma_psd)
    constraints = [cp.sum(w_mvo) == 1, w_mvo >= 0, portfolio_variance <= max_variance]
    prob = cp.Problem(cp.Maximize(portfolio_return), constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
        weights_mvo = np.array(w_mvo.value).reshape(-1)
    except Exception:
        weights_mvo = np.repeat(1/n, n)

    # Max Sharpe
    def neg_sharpe(w):
        ret = float(np.dot(w, mu))
        vol = float(np.sqrt(np.dot(w.T, np.dot(Sigma_psd, w))))
        return -((ret - rf) / (vol + 1e-12))

    bounds = [(0, 0.2)] * n
    constraints_slsqp = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    init_guess = np.repeat(1/n, n)
    try:
        result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints_slsqp)
        weights_sharpe = result.x if result.success else init_guess
    except Exception:
        weights_sharpe = init_guess

    return weights_mvo, weights_sharpe

def equal_weight_portfolio(n):
    return np.repeat(1/n, n)

def market_implied_delta(returns, rf, market_weights):
    mu_mkt = returns.mean().values @ market_weights
    var_mkt = market_weights.T @ returns.cov().values @ market_weights
    delta = (mu_mkt - rf) / max(var_mkt, 1e-12)
    return float(max(delta, 0.0))

def black_litterman(mu_view, Sigma, rf, tickers, returns, tau=0.2, omega_scalar=0.1):
    n = len(mu_view)
    Sigma_psd = _nearest_psd(Sigma)

    caps = []
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info
            caps.append(info.get("marketCap", 0))
        except Exception:
            caps.append(0)
    caps = np.array(caps, dtype=float)
    market_weights = caps / np.nansum(caps) if np.nansum(caps) > 0 else np.full(n, 1.0 / n)

    delta = market_implied_delta(returns, rf, market_weights)
    Pi = delta * (Sigma_psd @ market_weights)

    P = np.eye(n)
    Omega = np.eye(n) * omega_scalar
    Ainv = np.linalg.inv(tau * Sigma_psd)
    post_prec = Ainv + P.T @ np.linalg.inv(Omega) @ P
    post_mean = np.linalg.inv(post_prec) @ (Ainv @ Pi + P.T @ np.linalg.inv(Omega) @ mu_view)

    w = cp.Variable(n)
    ret = post_mean @ w
    risk = cp.quad_form(w, Sigma_psd)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Maximize(ret - delta * risk), constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
        w_bl = np.array(w.value).reshape(-1)
    except Exception:
        w_bl = market_weights

    return post_mean, w_bl

def get_portfolio_perf(weights, mu, Sigma, rf):
    port_return = float(np.dot(weights, mu))
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(Sigma, weights))))
    sharpe = (port_return - rf) / (port_vol + 1e-12)
    return port_return, port_vol, sharpe

def efficient_frontier_curve(mu, Sigma, points=100):
    n = len(mu)
    frontier_vols = []
    target_returns = np.linspace(min(mu), max(mu), points)
    bounds = tuple((0, 1) for _ in range(n))
    base_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    for target in target_returns:
        constraints = [base_constraint,
                       {'type': 'eq', 'fun': lambda x, t=target: np.dot(x, mu) - t}]
        result = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(Sigma, w))),
                          x0=np.ones(n)/n,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
        frontier_vols.append(result.fun if result.success else np.nan)
    return target_returns, np.array(frontier_vols)

def align_weights(weights, master, active):
    aligned = []
    for t in active:
        if t in master:
            aligned.append(weights[master.index(t)])
        else:
            # fallback: equally distribute for tickers missing in master
            aligned.append(1.0 / len(active))
    return np.array(aligned)


# --------------------------------
# Data Pipeline
# --------------------------------
if len(tickers) == 0:
    st.warning("Please select at least one ticker.")
    st.stop()

prices = download_prices(tickers, start, end)
if prices.empty:
    st.error("No price data returned. Try different dates or tickers.")
    st.stop()

# Only keep active tickers with price data
active_tickers = list(prices.columns)

returns = resample_returns(prices, frequency)
rf_annual = get_rf_series(start, end)
rf = rf_annual / FREQUENCY_MAP[frequency]['rf_divisor']

mu = predict_returns(returns, n_lags)
Sigma = LedoitWolf().fit(returns).covariance_

w_mvo, w_sharpe = optimize_portfolio(mu, Sigma, rf, max_variance)
w_eq = np.repeat(1/len(active_tickers), len(active_tickers))
mu_bl, w_bl = black_litterman(mu, Sigma, rf, active_tickers, returns, tau, omega_scalar)

# Align weights to active tickers
w_mvo = align_weights(w_mvo, tickers, active_tickers)
w_sharpe = align_weights(w_sharpe, tickers, active_tickers)
w_bl = align_weights(w_bl, tickers, active_tickers)

# --------------------------------
# Tabs
# --------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Weights", "Efficient Frontier", "Cumulative Returns & Drawdowns", 
     "Metrics", "Rolling", "Correlation", "VaR & CVaR"]
)

# --------------------------
# Tab 1: Weights
# --------------------------
with tab1:
    st.subheader("Optimized Portfolio Weights")
    weights_df = pd.DataFrame({
        "Ticker": active_tickers,
        "Equal Weight": w_eq,
        "MVO": w_mvo,
        "Max Sharpe": w_sharpe,
        "Black–Litterman": w_bl
    })
    st.dataframe(weights_df.set_index("Ticker"))

    csv = weights_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Weights CSV", data=csv, file_name="weights.csv", mime="text/csv")

# --------------------------
# Tab 2: Efficient Frontier
# --------------------------
with tab2:
    st.subheader("Efficient Frontier & Random Portfolios")
    n = len(mu)
    num_portfolios = 3000
    results = np.zeros((3, num_portfolios))
    Sigma_psd = _nearest_psd(Sigma)
    for i in range(num_portfolios):
        w = np.random.dirichlet(np.ones(n))
        r, v, s = get_portfolio_perf(w, mu, Sigma_psd, rf)
        results[:, i] = [r, v, s]

    frontier_returns, frontier_vols = efficient_frontier_curve(mu, Sigma_psd)
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', alpha=0.5, label='Random Portfolios')
    if np.isfinite(frontier_vols).any():
        ax.plot(frontier_vols, frontier_returns, 'r--', linewidth=2, label='Efficient Frontier')

    portfolios = {
        "Equal Weight": w_eq,
        "Max Sharpe": w_sharpe,
        "MVO": w_mvo,
        "Black-Litterman": w_bl,
    }
    for label, w in portfolios.items():
        r, v, s = get_portfolio_perf(w, mu, Sigma_psd, rf)
        ax.scatter(v, r, marker='X', s=160, label=f"{label} (Sharpe: {s:.2f})")

    ax.set_xlabel('Volatility (Std Dev)')
    ax.set_ylabel('Expected Return')
    ax.set_title(f"{frequency.title()} Efficient Frontier (ML Forecasted Returns)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    cbar = plt.colorbar(sc)
    cbar.set_label('Sharpe Ratio')
    st.pyplot(fig)

# --------------------------
# Remaining tabs (Tab3-7)
# --------------------------
# I can provide the remaining tabs fully implemented next, including backtest and metrics.
