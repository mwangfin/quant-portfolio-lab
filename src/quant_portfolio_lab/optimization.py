from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DAYS = 252

def _inputs(returns: pd.DataFrame):
    mu = returns.mean().to_numpy() * DAYS
    cov = returns.cov().to_numpy() * DAYS
    cov = cov + np.eye(len(cov)) * 1e-10
    return mu, cov

def _solve(objective, n: int, max_weight: float) -> np.ndarray:
    if max_weight * n < 1 - 1e-12:
        raise ValueError("max_weight is infeasible for the number of assets.")
    result = minimize(objective, np.repeat(1/n, n), method="SLSQP",
                      bounds=[(0.0, max_weight)] * n,
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                      options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    w = np.clip(result.x, 0, max_weight)
    return w / w.sum()

def optimize_weights(returns: pd.DataFrame, method: str, risk_free_rate: float = 0.0,
                     max_weight: float = 1.0) -> pd.Series:
    mu, cov = _inputs(returns)
    n = returns.shape[1]
    if method == "Equal weight":
        w = np.repeat(1/n, n)
    elif method == "Inverse volatility":
        inv = 1 / np.sqrt(np.diag(cov))
        w = inv / inv.sum()
        if w.max() > max_weight:
            w = _solve(lambda x: x @ cov @ x, n, max_weight)
    elif method == "Minimum variance":
        w = _solve(lambda x: x @ cov @ x, n, max_weight)
    elif method == "Maximum Sharpe":
        def negative_sharpe(x):
            vol = np.sqrt(x @ cov @ x)
            return -((x @ mu - risk_free_rate) / vol)
        w = _solve(negative_sharpe, n, max_weight)
    else:
        raise ValueError(f"Unknown method: {method}")
    return pd.Series(w, index=returns.columns, name="Weight")
