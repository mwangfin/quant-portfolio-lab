import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st
from quant_portfolio_lab.data import download_prices, simple_returns
from quant_portfolio_lab.metrics import performance_metrics
from quant_portfolio_lab.optimization import optimize_weights
from quant_portfolio_lab.backtest import walk_forward_backtest

st.set_page_config(page_title="Quant Portfolio Lab", layout="wide")
st.title("Quant Portfolio Lab")
st.caption("ETF portfolio optimization and out-of-sample research")

with st.sidebar:
    ticker_text = st.text_input("Tickers", "SPY, QQQ, IWM, EFA, TLT, GLD, VNQ")
    start = st.date_input("Start date", pd.Timestamp("2015-01-01"))
    end = st.date_input("End date", pd.Timestamp.today())
    method = st.selectbox("Portfolio method", ["Minimum variance", "Maximum Sharpe", "Inverse volatility", "Equal weight"])
    rf = st.number_input("Annual risk-free rate", 0.0, 0.20, 0.02, 0.005, format="%.3f")
    max_weight = st.slider("Maximum asset weight", 0.15, 1.0, 0.40, 0.05)
    lookback = st.selectbox("Estimation window", [252, 504, 756], index=1, format_func=lambda x: f"{x} trading days")
    costs = st.number_input("Transaction cost, bps", 0.0, 100.0, 5.0, 1.0)
    run = st.button("Run analysis", type="primary")

if not run:
    st.info("Choose parameters and click Run analysis.")
    st.stop()

try:
    tickers = [x.strip() for x in ticker_text.split(",")]
    prices = download_prices(tickers, str(start), str(end))
    returns = simple_returns(prices)
    weights = optimize_weights(returns, method, rf, max_weight)
    strategy, weight_history = walk_forward_backtest(returns, method, lookback, rf, max_weight, costs)
    benchmark = returns.loc[strategy.index].mean(axis=1)
except Exception as exc:
    st.error(str(exc))
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Current allocation")
    alloc = weights.rename_axis("Asset").reset_index()
    st.plotly_chart(px.bar(alloc, x="Asset", y="Weight", text_auto=".1%"), use_container_width=True)
with col2:
    st.subheader("Latest estimates")
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * (252 ** 0.5)
    stats = pd.DataFrame({"Expected return": annual_return, "Volatility": annual_vol})
    st.dataframe(stats.style.format("{:.2%}"), use_container_width=True)

st.subheader("Walk-forward performance")
wealth = pd.DataFrame({method: (1 + strategy).cumprod(), "Equal-weight benchmark": (1 + benchmark).cumprod()})
st.plotly_chart(px.line(wealth, labels={"value": "Growth of 1", "index": "Date", "variable": "Portfolio"}), use_container_width=True)
metric_table = pd.DataFrame({method: performance_metrics(strategy, rf), "Equal-weight benchmark": performance_metrics(benchmark, rf)}).T
percent_cols = ["CAGR", "Volatility", "Max drawdown", "Daily VaR 95%", "Daily CVaR 95%"]
formats = {c: "{:.2%}" for c in percent_cols}
formats.update({"Sharpe": "{:.2f}", "Sortino": "{:.2f}"})
st.dataframe(metric_table.style.format(formats), use_container_width=True)

st.subheader("Portfolio weights through time")
st.plotly_chart(px.area(weight_history, labels={"value": "Weight", "index": "Rebalance date", "variable": "Asset"}), use_container_width=True)

with st.expander("Methodology and limitations"):
    st.markdown("""The optimizer is long-only and fully invested. Parameters are estimated from historical daily returns. The backtest recalculates weights monthly using only data available at each rebalance date. Results are sensitive to the asset universe, estimation window and costs. This is educational research, not investment advice.""")
