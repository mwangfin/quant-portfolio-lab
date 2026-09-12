# Quant Portfolio Lab

A Python-based quantitative investment research platform for portfolio construction, risk analytics, systematic investing and machine learning. 

## Key Features 

- Portfolio Optimization 
  - Minimum Variance 
  - Maximum Sharpe Ratio 
  - Inverse Volatility 

- Risk Analytics 
  - Volatility 
  - Sharpe Ratio 
  - Sortino Ratio 
  - Maximum Drawdown 
  - VaR 
  - CVaR 
  
- Backtesting 
  - Monthly Walk-Forward Testing 
  - Transaction Cost Modelling 

- Interactive Dashboard 
  - Streamlit Visualization 
  - Portfolio Allocation Monitoring 
  - Performance Reporting

## Machine Learning

The repository includes a Random Forest example for market prediction.

Example:

```bash
python examples/market_prediction.py

### Random Forest Market Prediction

This project includes a machine learning module that uses a Random Forest classifier to predict short-term ETF market direction.

#### Features

The model is trained using technical and risk-related factors:

- 20-day momentum
- 60-day momentum
- 20-day rolling volatility
- Price-to-200-day moving average ratio

#### Target

The classifier predicts whether the ETF price will be higher over the next 20 trading days. 

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
quant-portfolio-lab/
├── app.py                         Streamlit dashboard
├── README.md
├── pyproject.toml
├── .github/
│   └── workflows/
│       └── tests.yml              CI pipeline
├── examples/
│   └── market_prediction.py       Random Forest market prediction example
├── src/
│   └── quant_portfolio_lab/
│       ├── __init__.py
│       ├── data.py                Market-data download and preprocessing
│       ├── metrics.py             Risk and performance metrics
│       ├── optimization.py        Portfolio optimization
│       ├── backtest.py            Walk-forward backtesting
│       └── ml_model.py            Random Forest prediction model
├── tests/
│   └── test_core.py               Unit tests
└── .gitignore
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

## Project Motivation

This project was developed to demonstrate the application of quantitative analysis, portfolio optimization, machine learning and risk management techniques to systematic investing.

The goal is to bridge expertise in quantitative risk modelling with practical asset management and investment decision-making.
