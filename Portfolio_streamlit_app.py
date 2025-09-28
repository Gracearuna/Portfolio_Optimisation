# ===============================
# Portfolio Optimization Streamlit App
# ===============================
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
import random
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ===============================
# Frequency Settings
# ===============================
FREQ_SETTINGS = {
    'daily': {'resample': None, 'rf_divisor': 252},
    'weekly': {'resample': 'W-FRI', 'rf_divisor': 52},
    'monthly': {'resample': 'M', 'rf_divisor': 12}
}

# ===============================
# Utility Functions
# ===============================
def resample_returns(stock_data, freq_key):
    rule = FREQ_SETTINGS[freq_key]['resample']
    if rule:
        stock_data = stock_data.resample(rule).last()
    returns = np.log(stock_data / stock_data.shift(1)).dropna()
    return returns

def max_drawdown(cum_returns):
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    return drawdown.min()

def performance_metrics_full(cum_returns_df, freq_key):
    n_days = (cum_returns_df.index[-1] - cum_returns_df.index[0]).days
    n_years = n_days / 365.25 if n_days > 0 else 0.0001

    metrics = {}
    for col in cum_returns_df.columns:
        total_return = cum_returns_df[col].iloc[-1] / cum_returns_df[col].iloc[0] - 1
        CAGR = (1 + total_return)**(1/n_years) - 1 if n_years > 0 else np.nan
        period_returns = cum_returns_df[col].pct_change().dropna()
        vol = period_returns.std() * np.sqrt(FREQ_SETTINGS[freq_key]['rf_divisor']) if len(period_returns) > 1 else np.nan
        sharpe = CAGR / vol if vol and vol != 0 else np.nan
        mdd = max_drawdown(cum_returns_df[col])
        metrics[col] = [CAGR, vol, sharpe, mdd]
    return pd.DataFrame(metrics, index=['CAGR','Volatility','Sharpe','Max Drawdown']).T

# ===============================
# LSTM Return Prediction
# ===============================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
tf.random.set_seed(42)

def predict_returns_lstm(returns, n_lags=3, epochs=50, batch_size=16):
    tickers = returns.columns
    n_tickers = len(tickers)
    X_all, y_all = [], []
    for i in range(n_lags, len(returns)-1):
        X_all.append(returns.iloc[i-n_lags:i].values)
        y_all.append(returns.iloc[i+1].values)
    X = np.array(X_all)
    Y = np.array(y_all)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, n_tickers)).reshape(X.shape)
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y)
    latest_input = returns.iloc[-n_lags:].values.reshape(1, n_lags, n_tickers)
    latest_input_scaled = scaler_X.transform(latest_input.reshape(-1, n_tickers)).reshape(1, n_lags, n_tickers)
    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=(n_lags, n_tickers)))
    model.add(Dense(n_tickers))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, Y_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
    pred_scaled = model.predict(latest_input_scaled, verbose=0)
    predicted_returns = scaler_Y.inverse_transform(pred_scaled)[0]
    return predicted_returns

# ===============================
# Portfolio Optimization
# ===============================
def optimize_portfolio(mu, Sigma, rf, tickers, max_variance=0.001):
    n = len(mu)
    shrinkage_factor = 0.5
    mu_shrunk = shrinkage_factor * mu + (1 - shrinkage_factor) * np.mean(mu)
    # MVO
    w_mvo = cp.Variable(n)
    lambda_reg = 0.01
    portfolio_return = mu_shrunk @ w_mvo - lambda_reg * cp.sum_squares(w_mvo)
    portfolio_variance = cp.quad_form(w_mvo, Sigma)
    max_weight = 0.3
    constraints = [cp.sum(w_mvo) == 1, w_mvo >= 0, w_mvo <= max_weight, portfolio_variance <= max_variance]
    prob = cp.Problem(cp.Maximize(portfolio_return), constraints)
    prob.solve()
    weights_mvo = w_mvo.value if w_mvo.value is not None else np.repeat(1/n, n)
    ret_mvo = mu @ weights_mvo
    vol_mvo = np.sqrt(weights_mvo.T @ Sigma @ weights_mvo)
    sharpe_mvo = (ret_mvo - rf)/vol_mvo
    # Max Sharpe
    def neg_sharpe(w):
        ret = np.dot(w, mu_shrunk)
        vol = np.sqrt(w.T @ Sigma @ w)
        return -(ret - rf)/vol + lambda_reg * np.sum(w**2)
    from scipy.optimize import minimize
    bounds = [(0, max_weight)]*n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1}]
    init_guess = np.repeat(1/n, n)
    result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    weights_sharpe = result.x if result.success else np.repeat(1/n, n)
    ret_sharpe = np.dot(weights_sharpe, mu)
    vol_sharpe = np.sqrt(weights_sharpe.T @ Sigma @ weights_sharpe)
    sharpe_sharpe = (ret_sharpe - rf)/vol_sharpe
    return weights_mvo, ret_mvo, vol_mvo, sharpe_mvo, weights_sharpe, ret_sharpe, vol_sharpe, sharpe_sharpe

def equal_weight_portfolio(mu, Sigma, rf):
    n = len(mu)
    w_eq = np.repeat(1/n, n)
    ret = np.dot(w_eq, mu)
    vol = np.sqrt(w_eq.T @ Sigma @ w_eq)
    sharpe = (ret - rf)/vol
    return w_eq, ret, vol, sharpe

# ===============================
# Black-Litterman
# ===============================
def _nearest_psd(A, eps=1e-10):
    B = 0.5*(A + A.T)
    w, V = np.linalg.eigh(B)
    w_clipped = np.clip(w, eps, None)
    return (V * w_clipped) @ V.T

def market_implied_delta(returns, rf, market_weights):
    mu_mkt = returns.mean().values @ market_weights
    var_mkt = market_weights.T @ returns.cov().values @ market_weights
    delta = (mu_mkt - rf) / max(var_mkt, 1e-12)
    return max(delta, 0.0)

def black_litterman(mu_view, Sigma, rf, tickers, returns, tau=0.5, omega_scalar=0.01):
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
    Sigma = _nearest_psd(np.asarray(Sigma))
    delta = market_implied_delta(returns, rf, market_weights)
    Pi = delta * (Sigma @ market_weights)
    P = np.eye(n)
    Omega = np.eye(n) * omega_scalar
    A = np.linalg.inv(tau*Sigma)
    post_prec = A + P.T @ np.linalg.inv(Omega) @ P
    post_mean = np.linalg.inv(post_prec) @ (A @ Pi + P.T @ np.linalg.inv(Omega) @ mu_view)
    w = cp.Variable(n)
    ret = post_mean @ w
    risk = cp.quad_form(w, Sigma)
    constraints = [cp.sum(w)==1, w>=0]
    prob = cp.Problem(cp.Maximize(ret - delta*risk), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    weights_bl = w.value if w.value is not None else np.repeat(1/n, n)
    ret_bl = np.dot(weights_bl, mu_view)
    vol_bl = np.sqrt(weights_bl.T @ Sigma @ weights_bl)
    sharpe_bl = (ret_bl - rf)/vol_bl
    return post_mean, weights_bl, ret_bl, vol_bl, sharpe_bl

# ===============================
# Streamlit Dashboard
# ===============================
st.title("📊 Portfolio Optimization Dashboard")
st.write("This dashboard uses LSTM-based return prediction, Mean-Variance, Max-Sharpe, Equal-Weight, and Black-Litterman optimization.")

# Sidebar inputs
tickers = st.text_area("Enter tickers (comma-separated)", value="JPM, GS, AAPL, MSFT, NVDA, GOOGL, META, AMZN, HD, KO, XOM, CVX, UNH, PFE, CAT, UNP, NFLX, DIS, NEE, PLD")
tickers = [t.strip().upper() for t in tickers.split(",")]
frequency = st.selectbox("Select frequency", options=['daily', 'weekly', 'monthly'], index=1)
n_lags = st.slider("LSTM Lookback Period", 1, 12, 3)
epochs = st.slider("LSTM Training Epochs", 10, 200, 50)

if st.button("Run Portfolio Analysis"):
    with st.spinner("Downloading stock data..."):
        stock_data = yf.download(tickers, start="2020-06-01", end="2025-06-01", auto_adjust=True)["Close"]
    returns = resample_returns(stock_data, frequency)
    mu = predict_returns_lstm(returns, n_lags=n_lags, epochs=epochs)
    Sigma = LedoitWolf().fit(returns).covariance_
    treasury = web.DataReader("DGS5", "fred", returns.index[0], returns.index[-1])
    rf_annual = treasury["DGS5"].mean()/100
    rf = rf_annual / FREQ_SETTINGS[frequency]['rf_divisor']

    w_mvo, ret_mvo, vol_mvo, sharpe_mvo, w_sharpe, ret_sharpe, vol_sharpe, sharpe_sharpe = optimize_portfolio(mu, Sigma, rf, tickers)
    w_eq, ret_eq, vol_eq, sharpe_eq = equal_weight_portfolio(mu, Sigma, rf)
    mu_bl, w_bl, ret_bl, vol_bl, sharpe_bl = black_litterman(mu, Sigma, rf, tickers, returns)

    portfolio_metrics = {
        "Equal Weight": (ret_eq, vol_eq, sharpe_eq),
        "Max Sharpe": (ret_sharpe, vol_sharpe, sharpe_sharpe),
        "MVO": (ret_mvo, vol_mvo, sharpe_mvo),
        "Black-Litterman": (ret_bl, vol_bl, sharpe_bl)
    }

    # Plot Efficient Frontier
    def plot_ef(mu, Sigma, rf, portfolios):
        n = len(mu)
        results = np.zeros((3, 5000))
        for i in range(5000):
            weights = np.random.dirichlet(np.ones(n))
            ret = np.dot(weights, mu)
            vol = np.sqrt(weights.T @ Sigma @ weights)
            sharpe = (ret - rf)/vol
            results[:, i] = [ret, vol, sharpe]
        ef_returns = np.linspace(min(mu), max(mu), 50)
        ef_vols, ef_sharpes = [], []
        for r_target in ef_returns:
            w = cp.Variable(n)
            portfolio_var = cp.quad_form(w, Sigma)
            constraints = [cp.sum(w)==1, w>=0, w<=0.3, mu @ w==r_target]
            prob = cp.Problem(cp.Minimize(portfolio_var), constraints)
            prob.solve(solver=cp.SCS, verbose=False)
            if w.value is not None:
                ef_vols.append(np.sqrt(w.value.T @ Sigma @ w.value))
                ef_sharpes.append((r_target - rf)/ef_vols[-1])
        fig, ax = plt.subplots(figsize=(12,7))
        sc = ax.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', alpha=0.4)
        ax.plot(ef_vols, ef_returns, 'r--', lw=2, label='Efficient Frontier')
        for label, metrics in portfolios.items():
            r, v, s = metrics
            ax.scatter(v, r, marker='X', s=200, label=f"{label} (Sharpe: {s:.3f})")
        ax.set_xlabel("Volatility (Std Dev)")
        ax.set_ylabel("Expected Return")
        ax.set_title("Efficient Frontier & Portfolio Positions")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Sharpe Ratio")
        ax.legend()
        st.pyplot(fig)

    st.subheader("📈 Portfolio Performance Metrics")
    df_metrics = pd.DataFrame({k: [v[0], v[1], v[2]] for k,v in portfolio_metrics.items()}, index=["Expected Return","Volatility","Sharpe Ratio"]).T
    st.dataframe(df_metrics.style.format("{:.4f}"))

    st.subheader("💹 Efficient Frontier")
    plot_ef(mu, Sigma, rf, portfolio_metrics)

    st.subheader("🧮 Portfolio Weights")
    weights_df = pd.DataFrame({
        "Equal Weight": w_eq,
        "Max Sharpe": w_sharpe,
        "MVO": w_mvo,
        "Black-Litterman": w_bl
    }, index=tickers)
    st.dataframe(weights_df.style.format("{:.2%}"))

    # Download buttons
    st.download_button("Download Portfolio Metrics CSV", df_metrics.to_csv().encode('utf-8'), file_name="portfolio_metrics.csv")
    st.download_button("Download Portfolio Weights CSV", weights_df.to_csv().encode('utf-8'), file_name="portfolio_weights.csv")

    st.success("Portfolio optimization completed!")

# ===============================
# End of App
# ===============================
