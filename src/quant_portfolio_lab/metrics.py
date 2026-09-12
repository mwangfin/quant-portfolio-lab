from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

def drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    return wealth / wealth.cummax() - 1.0

def performance_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("Return series is empty.")
    years = len(r) / TRADING_DAYS
    terminal = float((1 + r).prod())
    cagr = terminal ** (1 / years) - 1 if terminal > 0 and years > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    downside = float(r[r < 0].std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else np.nan
    sortino = (cagr - risk_free_rate) / downside if downside > 0 else np.nan
    var95 = float(r.quantile(0.05))
    tail = r[r <= var95]
    return {
        "CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "Sortino": sortino,
        "Max drawdown": float(drawdown(r).min()), "Daily VaR 95%": -var95,
        "Daily CVaR 95%": -float(tail.mean()),
    }
