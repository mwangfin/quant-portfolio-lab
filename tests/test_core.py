import numpy as np
import pandas as pd
from quant_portfolio_lab.metrics import performance_metrics
from quant_portfolio_lab.optimization import optimize_weights
from quant_portfolio_lab.backtest import walk_forward_backtest

def sample_returns(n=900):
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(rng.normal([0.0003, 0.0002, 0.0001], [0.01, 0.007, 0.004], (n, 3)), index=idx, columns=list("ABC"))

def test_weights_are_feasible():
    w = optimize_weights(sample_returns(), "Minimum variance", max_weight=0.6)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-10).all()
    assert (w <= 0.6 + 1e-8).all()

def test_metrics_keys_and_drawdown():
    m = performance_metrics(sample_returns()["A"])
    assert "Sharpe" in m
    assert m["Max drawdown"] <= 0

def test_walk_forward_is_nonempty():
    r, weights = walk_forward_backtest(sample_returns(), "Inverse volatility", lookback_days=252)
    assert len(r) > 0
    assert not weights.empty
