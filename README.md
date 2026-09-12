# Quant Portfolio Lab

A portfolio-research project built for a quantitative investment role. It downloads adjusted ETF prices, calculates risk and performance metrics, constructs constrained portfolios, and performs a monthly walk-forward backtest.

## Methods

- Equal weight benchmark
- Long-only minimum-variance portfolio
- Long-only maximum-Sharpe portfolio
- Inverse-volatility allocation
- Rolling out-of-sample backtest with configurable lookback and transaction costs
- CAGR, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, historical VaR and CVaR

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
streamlit run app.py
```

The default universe is SPY, QQQ, IWM, EFA, TLT, GLD and VNQ. In the app you can enter other Yahoo Finance ticker symbols.

## Test

```bash
pytest -q
ruff check .
```

## Repository structure

```text
app.py                         Streamlit user interface
src/quant_portfolio_lab/
  data.py                      Market-data download and validation
  metrics.py                   Performance and downside-risk metrics
  optimization.py              Portfolio optimizers
  backtest.py                  Walk-forward backtest
tests/                         Offline unit tests with synthetic data
```

## Research design and limitations

- Adjusted daily closing prices are requested with `auto_adjust=True`.
- Expected returns and covariance are estimated from historical observations and are uncertain.
- The walk-forward test avoids using future observations in portfolio estimation.
- Transaction costs are simplified as basis points multiplied by one-way turnover.
- Taxes, bid-ask spreads, market impact, FX hedging and fund fees are not explicitly modeled.
- Yahoo Finance data accessed through yfinance is intended for research/personal use. Check the applicable terms before other use.
- This repository is educational research, not investment advice.

## Suggested next extensions

1. Shrinkage covariance estimator and Black-Litterman expected returns
2. Risk-parity optimization based on equal risk contributions
3. Momentum and trend signals with purged validation
4. Market-regime model evaluated strictly out of sample
5. EUR base-currency returns and FX-hedged ETF comparison
