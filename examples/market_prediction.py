import pandas as pd

from quant_portfolio_lab.data import download_prices
from quant_portfolio_lab.ml_model import train_random_forest


def main() -> None:
    price_data = download_prices(
        ["SPY"],
        start="2015-01-01",
        end="2025-01-01",
    )

    spy_prices = price_data["SPY"]

    model, accuracy = train_random_forest(spy_prices)

    print(f"Accuracy: {accuracy:.3f}")

    feature_names = [
        "mom5",
        "mom20",
        "vol20",
    ]

    feature_importance = pd.Series(
        model.feature_importances_,
        index=feature_names,
        name="Importance",
    ).sort_values(ascending=False)

    print("\nFeature importance:")
    print(feature_importance)


if __name__ == "__main__":
    main()