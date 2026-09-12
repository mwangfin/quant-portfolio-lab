from __future__ import annotations

import pandas as pd
from .optimization import optimize_weights

def walk_forward_backtest(returns: pd.DataFrame, method: str, lookback_days: int = 504,
                         risk_free_rate: float = 0.0, max_weight: float = 1.0,
                         transaction_cost_bps: float = 5.0):
    if len(returns) <= lookback_days:
        raise ValueError("Not enough observations for the selected lookback.")
    rebal_dates = returns.iloc[lookback_days:].resample("ME").last().index
    strategy_parts, weight_rows = [], []
    previous = None
    for i, date in enumerate(rebal_dates[:-1]):
        position = returns.index.searchsorted(date, side="right")
        train = returns.iloc[max(0, position-lookback_days):position]
        weights = optimize_weights(train, method, risk_free_rate, max_weight)
        start = date
        end = rebal_dates[i+1]
        test = returns[(returns.index > start) & (returns.index <= end)]
        if test.empty:
            continue
        part = test @ weights
        turnover = float((weights - previous).abs().sum()) if previous is not None else 0.0
        part.iloc[0] -= turnover * transaction_cost_bps / 10000.0
        strategy_parts.append(part)
        weight_rows.append(pd.Series(weights, name=date))
        previous = weights
    if not strategy_parts:
        raise ValueError("Backtest produced no out-of-sample periods.")
    return pd.concat(strategy_parts).sort_index(), pd.DataFrame(weight_rows)
