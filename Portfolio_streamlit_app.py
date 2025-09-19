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

    # MVO: Maximize return under risk constraint
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

    # Max Sharpe (SLSQP with 20% cap)
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

    # Market cap weights (best-effort; yfinance may omit some caps)
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

    # Mean-variance optimal weights on posterior
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


# Performance utilities

def get_portfolio_perf(weights, mu, Sigma, rf):
    port_return = float(np.dot(weights, mu))
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(Sigma, weights))))
    sharpe = (port_return - rf) / (port_vol + 1e-12)
    return port_return, port_vol, sharpe


def min_variance(Sigma):
    n = Sigma.shape[0]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(lambda w: np.dot(w.T, np.dot(Sigma, w)),
                      x0=np.ones(n)/n,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    return result.x


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

returns = resample_returns(prices, frequency)
rf_annual = get_rf_series(start, end)
rf = rf_annual / FREQUENCY_MAP[frequency]['rf_divisor']

mu = predict_returns(returns, n_lags)
Sigma = LedoitWolf().fit(returns).covariance_

w_mvo, w_sharpe = optimize_portfolio(mu, Sigma, rf, max_variance)
w_eq = equal_weight_portfolio(len(mu))
mu_bl, w_bl = black_litterman(mu, Sigma, rf, tickers, returns, tau, omega_scalar)

# --------------------------------
# Tabs for Outputs
# --------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Weights", "Efficient Frontier", "Cumulative Returns & Drawdowns", "Metrics", "Rolling", "Correlation", "VaR & CVaR"]
)

with tab1:
    st.subheader("Optimized Portfolio Weights")
    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Equal Weight": w_eq,
        "MVO": w_mvo,
        "Max Sharpe": w_sharpe,
        "Black–Litterman": w_bl
    })
    st.dataframe(weights_df.set_index("Ticker"))

    csv = weights_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Weights CSV", data=csv, file_name="weights.csv", mime="text/csv")

with tab2:
    st.subheader("Efficient Frontier & Random Portfolios")
    n = len(mu)

    # Random portfolios scatter
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

    # Mark specific portfolios
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

with tab3:
    st.subheader("Cumulative Returns & Drawdowns")

    lookback_period = pd.DateOffset(years=lookback_years)
    rebalance_frequency = pd.DateOffset(months=rebalance_months)

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
        if window_returns.empty or window_returns.shape[0] <= n_lags + 2:
            current_start += rebalance_frequency
            continue
        mu_t = predict_returns(window_returns, n_lags)
        Sigma_t = LedoitWolf().fit(window_returns).covariance_
        mu_bl_t, w_bl_t = black_litterman(mu_t, Sigma_t, rf, tickers, window_returns, tau, omega_scalar)
        w_mvo_t, w_sharpe_t = optimize_portfolio(mu_t, Sigma_t, rf, max_variance)
        w_eq_t = equal_weight_portfolio(len(mu_t))
        out_sample_returns = returns.loc[current_lookback_end + pd.Timedelta(days=1):current_out_sample_end]
        if out_sample_returns.empty:
            break
        daily_mvo.extend(np.log1p(np.dot(out_sample_returns.values, w_mvo_t)))
        daily_maxsharpe.extend(np.log1p(np.dot(out_sample_returns.values, w_sharpe_t)))
        daily_eq.extend(np.log1p(np.dot(out_sample_returns.values, w_eq_t)))
        daily_bl.extend(np.log1p(np.dot(out_sample_returns.values, w_bl_t)))
        dates_list.extend(out_sample_returns.index)
        current_start += rebalance_frequency

    if len(dates_list) == 0:
        st.info("Not enough data for the selected backtest settings.")
    else:
        daily_mvo = pd.Series(daily_mvo, index=dates_list, name='MVO')
        daily_maxsharpe = pd.Series(daily_maxsharpe, index=dates_list, name='Max Sharpe')
        daily_eq = pd.Series(daily_eq, index=dates_list, name='Equal Weight')
        daily_bl = pd.Series(daily_bl, index=dates_list, name='Black-Litterman')

        daily_df = pd.concat([daily_mvo, daily_maxsharpe, daily_eq, daily_bl], axis=1)

        # Cumulative returns
        cumulative = np.exp(daily_df.cumsum())
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        for col in cumulative.columns:
            ax1.plot(cumulative.index, cumulative[col], label=col)
        ax1.set_title('Portfolio Cumulative Returns')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()
        st.pyplot(fig1)

        # Drawdowns overlay
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for col in cumulative.columns:
            series = cumulative[col]
            running_max = series.cummax()
            drawdown = series / running_max - 1
            ax2.plot(series.index, series, label=col)
            ax2.fill_between(series.index, series, running_max, where=drawdown < 0, alpha=0.1)
        ax2.set_title('Cumulative Returns with Drawdowns')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Return')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        st.pyplot(fig2)

        st.session_state['daily_df'] = daily_df

with tab4:
    st.subheader("Performance Metrics")

    if 'daily_df' not in st.session_state:
        st.info("Run the backtest in the previous tab first.")
    else:
        daily_df = st.session_state['daily_df']

        def max_drawdown(cumulative_returns: pd.Series):
            running_max = cumulative_returns.cummax()
            drawdown = cumulative_returns / running_max - 1
            return float(drawdown.min())

        def cagr_log(daily_log_returns: pd.Series, trading_days=252):
            if len(daily_log_returns) == 0:
                return np.nan
            total_log_return = daily_log_returns.sum()
            total_years = len(daily_log_returns) / trading_days
            return np.exp(total_log_return / max(total_years, 1e-12)) - 1

        def annualized_volatility(daily_log_returns: pd.Series, trading_days=252):
            return float(daily_log_returns.std() * np.sqrt(trading_days))

        def sharpe_ratio_log(daily_log_returns: pd.Series, rf_annual: float, trading_days=252):
            rf_daily = rf_annual / trading_days
            excess_returns = daily_log_returns - rf_daily
            return float(np.sqrt(trading_days) * excess_returns.mean() / (excess_returns.std() + 1e-12))

        metrics = {}
        for name in daily_df.columns:
            series = daily_df[name].dropna()
            cum_ret = np.exp(series.cumsum())
            metrics[name] = {
                "Max Drawdown": max_drawdown(cum_ret),
                "CAGR": cagr_log(series, trading_days=252),
                "Volatility": annualized_volatility(series, trading_days=252),
                "Sharpe Ratio": sharpe_ratio_log(series, rf_annual=rf_annual, trading_days=252),
            }
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df.style.format({
            "Max Drawdown": "{:.2%}",
            "CAGR": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.2f}",
        }))

        st.download_button(
            "Download Metrics CSV",
            data=metrics_df.to_csv().encode('utf-8'),
            file_name="metrics.csv",
            mime="text/csv",
        )

with tab5:
    st.subheader("Rolling Volatility & Sharpe")
    if 'daily_df' not in st.session_state:
        st.info("Run the backtest in the previous tabs first.")
    else:
        daily_df = st.session_state['daily_df']
        window = rolling_window_days

        # Rolling Volatility
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        for name in daily_df.columns:
            rolling_vol = daily_df[name].rolling(window).std() * np.sqrt(252)
            ax3.plot(rolling_vol.index, rolling_vol, label=name)
        ax3.set_title(f'Rolling {window}-Day Annualized Volatility')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Volatility')
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend()
        st.pyplot(fig3)

        # Rolling Sharpe (simple rolling mean/std)
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        for name in daily_df.columns:
            rolling_mean = daily_df[name].rolling(window).mean()
            rolling_std = daily_df[name].rolling(window).std()
            rolling_sharpe = (rolling_mean / (rolling_std + 1e-12)) * np.sqrt(252)
            ax4.plot(rolling_sharpe.index, rolling_sharpe, label=name)
        ax4.set_title(f'Rolling {window}-Day Sharpe Ratio')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.grid(True, linestyle='--', alpha=0.5)
        ax4.legend()
        st.pyplot(fig4)

with tab6:
    st.subheader("Correlation Heatmap (Daily Log Returns)")
    if 'daily_df' not in st.session_state:
        st.info("Run the backtest first to populate daily returns.")
    else:
        daily_df = st.session_state['daily_df']
        corr_matrix = daily_df.corr()
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax5, fmt='.2f')
        ax5.set_title("Portfolio Daily Returns Correlation")
        st.pyplot(fig5)
        st.dataframe(corr_matrix.style.format("{:.2f}"))

with tab7:
    st.subheader("Value at Risk (VaR) & Conditional VaR (CVaR)")
    if 'daily_df' not in st.session_state:
        st.info("Run the backtest first to compute VaR/CVaR.")
    else:
        daily_df = st.session_state['daily_df']
        # 1% VaR and CVaR (on log returns)
        var_1 = daily_df.quantile(0.01)
        cvar_1 = daily_df[daily_df.le(var_1, axis=1)].mean()

        left, right = st.columns(2)
        with left:
            st.write("**1% Daily VaR (log returns)**")
            st.dataframe(var_1.to_frame("VaR 1%"))
        with right:
            st.write("**1% Daily CVaR (Expected Shortfall)**")
            st.dataframe(cvar_1.to_frame("CVaR 1%"))

        # Plot VaR vs. CVaR
        x = np.arange(len(daily_df.columns))
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        ax6.bar(x, -cvar_1.values, alpha=0.6, label='1% Daily CVaR')
        ax6.plot(x, -var_1.values, marker='o', linewidth=2, label='1% Daily VaR')
        ax6.set_xticks(x)
        ax6.set_xticklabels(daily_df.columns)
        ax6.set_ylabel('Loss (log return)')
        ax6.set_title('1% Daily VaR vs CVaR for Portfolios')
        ax6.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax6.legend()
        st.pyplot(fig6)

