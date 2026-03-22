"""
data_loader.py — Market Data Fetcher
Pulls OHLCV data from Yahoo Finance for regime analysis.
Handles yfinance API quirks, hourly data limits, and column format changes.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Ticker mapping for common names / crypto shorthand (BTC, ETH, SOL, AVAX only)
TICKER_MAP = {
    "BTC": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD",
    "ETHEREUM": "ETH-USD",
    "SOL": "SOL-USD",
    "SOLANA": "SOL-USD",
    "AVAX": "AVAX-USD",
    "AVALANCHE": "AVAX-USD",
}


def resolve_ticker(symbol: str) -> str:
    """Resolve common names to Yahoo Finance tickers."""
    upper = symbol.upper().strip()
    return TICKER_MAP.get(upper, upper)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle all yfinance column format variations."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map whatever column names yfinance returns to our standard names."""
    required = ["Open", "High", "Low", "Close", "Volume"]
    if all(c in df.columns for c in required):
        return df[required].copy()

    col_map = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl == "open":
            col_map[c] = "Open"
        elif cl == "high":
            col_map[c] = "High"
        elif cl == "low":
            col_map[c] = "Low"
        elif cl == "close":
            col_map[c] = "Close"
        elif cl == "volume":
            col_map[c] = "Volume"
        elif cl in ("adj close", "adjclose"):
            col_map[c] = "Adj Close"

    df = df.rename(columns=col_map)

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    return df[required].copy()


def fetch_data(
    symbol: str,
    period_days: int = 730,
    interval: str = "1h",
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.
    Uses multiple fallback methods for reliability.
    """
    ticker = resolve_ticker(symbol)

    if start_date and end_date:
        start = str(start_date)
        end = str(end_date)
    else:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=period_days)
        start = start_dt.strftime("%Y-%m-%d")
        end = end_dt.strftime("%Y-%m-%d")

    # Yahoo limits hourly data to ~730 days
    if interval in ("1h", "60m"):
        start_dt = datetime.strptime(start, "%Y-%m-%d") if isinstance(start, str) else start
        end_dt = datetime.strptime(end, "%Y-%m-%d") if isinstance(end, str) else end
        max_start = end_dt - timedelta(days=729)
        if start_dt < max_start:
            start = max_start.strftime("%Y-%m-%d")
            print(f"[DataLoader] Clamped start to {start} (Yahoo 730-day hourly limit)")

    print(f"[DataLoader] Fetching {ticker} | {interval} | {start} -> {end}")

    df = None

    # Method 1: Ticker.history (most reliable in newer yfinance)
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, interval=interval)
        if df is not None and not df.empty:
            print(f"[DataLoader] Ticker.history returned {len(df)} rows")
    except Exception as e:
        print(f"[DataLoader] Ticker.history failed: {e}")
        df = None

    # Method 2: yf.download fallback
    if df is None or df.empty:
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if df is not None and not df.empty:
                print(f"[DataLoader] yf.download returned {len(df)} rows")
        except Exception as e:
            print(f"[DataLoader] yf.download failed: {e}")
            df = None

    # Method 3: Use period string instead of date range
    if df is None or df.empty:
        try:
            if interval == "1d":
                period_str = "2y"
            else:
                period_str = "730d"
            t = yf.Ticker(ticker)
            df = t.history(period=period_str, interval=interval)
            if df is not None and not df.empty:
                print(f"[DataLoader] period={period_str} returned {len(df)} rows")
        except Exception as e:
            print(f"[DataLoader] period method failed: {e}")

    # Method 4: Fall back to daily if hourly fails
    if (df is None or df.empty) and interval != "1d":
        print(f"[DataLoader] Hourly unavailable, falling back to daily")
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="2y", interval="1d")
            if df is not None and not df.empty:
                print(f"[DataLoader] Daily fallback returned {len(df)} rows")
        except Exception as e:
            print(f"[DataLoader] Daily fallback failed: {e}")

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for {ticker}. "
            f"Try a different symbol or switch to daily interval."
        )

    df = _flatten_columns(df)
    df = _standardize_columns(df)
    df.dropna(inplace=True)
    df = df[df["Volume"] > 0]

    if df.empty:
        raise ValueError(f"Data for {ticker} was all NaN/zero-volume after cleaning.")

    print(f"[DataLoader] Loaded {len(df)} clean candles for {ticker}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features for HMM training:
      - returns: log returns of Close
      - range: (High - Low) / Close
      - volume_change: log ratio of volume vs rolling mean
    """
    out = df.copy()
    out["returns"] = np.log(out["Close"] / out["Close"].shift(1))
    out["range"] = (out["High"] - out["Low"]) / out["Close"]
    vol_ma = out["Volume"].rolling(20, min_periods=1).mean()
    out["volume_change"] = np.log((out["Volume"] + 1) / (vol_ma + 1))
    out.dropna(inplace=True)
    return out
