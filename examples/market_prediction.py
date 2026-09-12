from quant_portfolio_lab.data import (
    download_prices
)

from quant_portfolio_lab.ml_model import (
    train_random_forest
)

prices = download_prices(
    ["SPY"],
    "2015-01-01",
    "2025-01-01"
)

model, acc = train_random_forest(
    prices["SPY"]
)

print(
    "Accuracy:",
    round(acc, 4)
)
