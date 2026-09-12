from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted daily close prices."""

    symbols = list(
        dict.fromkeys(
            ticker.strip().upper()
            for ticker in tickers
            if ticker.strip()
        )
    )

    if len(symbols) < 1:
        raise ValueError(
            "Enter at least one valid ticker symbol."
        )

    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise ValueError(
            "No market data returned. Check tickers and dates."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])

    # With one ticker, yfinance may leave the column named "Close"
    if len(symbols) == 1 and close.shape[1] == 1:
        close.columns = [symbols[0]]

    close = (
        close
        .dropna(axis=1, how="all")
        .sort_index()
        .ffill()
        .dropna()
    )

    if close.shape[1] < 1:
        raise ValueError(
            "No valid ticker data remained after cleaning."
        )

    if len(close) < 60:
        raise ValueError(
            f"Insufficient price history: only {len(close)} observations."
        )

    return close.astype(float)


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily returns."""

    return prices.pct_change(
        fill_method=None
    ).dropna(how="any")