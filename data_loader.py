"""
data_loader.py — Market Data Fetcher (Tradier + Yahoo Finance)
Pulls OHLCV data using Tradier API as primary source (fast, no rate limits)
and Yahoo Finance as fallback (crypto, or if Tradier unavailable).
"""

import os
import json
import threading
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# ─── Tradier config ───────────────────────────────────────────
TRADIER_SETTINGS_FILE = os.path.join(os.path.dirname(__file__), ".tradier_settings.json")
TRADIER_SANDBOX_BASE = "https://sandbox.tradier.com/v1"
TRADIER_PROD_BASE = "https://api.tradier.com/v1"


_tradier_config_cache = None


def _load_tradier_config() -> dict:
    """Load Tradier config (cached after first call)."""
    global _tradier_config_cache
    if _tradier_config_cache is not None:
        return _tradier_config_cache

    config = {
        "access_token": os.environ.get("TRADIER_ACCESS_TOKEN", ""),
        "account_id": os.environ.get("TRADIER_ACCOUNT_ID", ""),
        "sandbox": True,
    }

    # Load from .env if python-dotenv available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        config["access_token"] = os.environ.get("TRADIER_ACCESS_TOKEN", config["access_token"])
        config["account_id"] = os.environ.get("TRADIER_ACCOUNT_ID", config["account_id"])
    except ImportError:
        pass

    # Local settings file
    if os.path.exists(TRADIER_SETTINGS_FILE):
        try:
            with open(TRADIER_SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            config.update({k: v for k, v in saved.items() if v})
        except Exception:
            pass

    _tradier_config_cache = config
    return config


def _tradier_available() -> bool:
    """Check if Tradier API is configured."""
    config = _load_tradier_config()
    return bool(config.get("access_token"))


def _is_crypto(symbol: str) -> bool:
    """Check if symbol is crypto (Tradier doesn't support crypto)."""
    upper = symbol.upper().strip()
    return upper.endswith("-USD") or upper in TICKER_MAP


# ─── Ticker mapping ──────────────────────────────────────────
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
    """Resolve common names to tickers."""
    upper = symbol.upper().strip()
    return TICKER_MAP.get(upper, upper)


# ─── Tradier data fetcher ────────────────────────────────────

# Thread-local sessions so each worker gets its own connection pool
_tradier_local = threading.local()


def _get_tradier_session():
    """Get or create a thread-local requests.Session with Tradier auth."""
    session = getattr(_tradier_local, "session", None)
    if session is None:
        config = _load_tradier_config()
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {config['access_token']}",
            "Accept": "application/json",
        })
        _tradier_local.session = session
    return session


def _fetch_tradier(
    symbol: str,
    start: str,
    end: str,
    interval: str = "daily",
) -> pd.DataFrame:
    """
    Fetch OHLCV from Tradier Markets API.
    interval: 'daily', 'weekly', 'monthly'
    Returns DataFrame with Open, High, Low, Close, Volume or empty DF on failure.
    """
    config = _load_tradier_config()
    base = TRADIER_PROD_BASE if not config.get("sandbox", True) else TRADIER_SANDBOX_BASE

    # Map our intervals to Tradier's
    tradier_interval = "daily"
    if interval in ("1wk", "weekly"):
        tradier_interval = "weekly"
    elif interval in ("1mo", "monthly"):
        tradier_interval = "monthly"

    session = _get_tradier_session()

    try:
        r = session.get(
            f"{base}/markets/history",
            params={
                "symbol": symbol.upper(),
                "interval": tradier_interval,
                "start": start,
                "end": end,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()

        # Tradier returns: {"history": {"day": [...]}}
        history = data.get("history")
        if not history:
            return pd.DataFrame()

        days = history.get("day", [])
        if not days:
            return pd.DataFrame()

        # Handle single-day response (dict instead of list)
        if isinstance(days, dict):
            days = [days]

        df = pd.DataFrame(days)

        # Rename to standard columns
        col_map = {
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns=col_map)

        # Set date index
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # Ensure numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    except Exception as e:
        print(f"[DataLoader] Tradier fetch failed for {symbol}: {e}")
        return pd.DataFrame()


# ─── Yahoo data fetcher (fallback) ───────────────────────────

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


def _fetch_yahoo(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance with multiple fallback methods."""
    df = None

    # Method 1: Ticker.history
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, interval=interval)
        if df is not None and not df.empty:
            df = _flatten_columns(df)
            return _standardize_columns(df)
    except Exception:
        df = None

    # Method 2: yf.download
    if df is None or df.empty:
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if df is not None and not df.empty:
                df = _flatten_columns(df)
                return _standardize_columns(df)
        except Exception:
            df = None

    # Method 3: period string
    if df is None or df.empty:
        try:
            period_str = "2y" if interval == "1d" else "730d"
            t = yf.Ticker(ticker)
            df = t.history(period=period_str, interval=interval)
            if df is not None and not df.empty:
                df = _flatten_columns(df)
                return _standardize_columns(df)
        except Exception:
            pass

    # Method 4: daily fallback if hourly fails
    if (df is None or df.empty) and interval != "1d":
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="2y", interval="1d")
            if df is not None and not df.empty:
                df = _flatten_columns(df)
                return _standardize_columns(df)
        except Exception:
            pass

    return pd.DataFrame()


# ─── Alpha Vantage (free: 25 calls/day) ──────────────────────

def _fetch_alpha_vantage(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Alpha Vantage (free tier, 25 req/day)."""
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not api_key:
        return pd.DataFrame()

    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": api_key,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return pd.DataFrame()

        rows = []
        for date_str, vals in ts.items():
            rows.append({
                "Date": date_str,
                "Open": float(vals["1. open"]),
                "High": float(vals["2. high"]),
                "Low": float(vals["3. low"]),
                "Close": float(vals["4. close"]),
                "Volume": int(vals["5. volume"]),
            })

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # Filter to date range
        df = df.loc[start:end]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    except Exception as e:
        print(f"[DataLoader] Alpha Vantage failed for {symbol}: {e}")
        return pd.DataFrame()


# ─── Financial Modeling Prep (free: 250 calls/day) ───────────

def _fetch_fmp(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Financial Modeling Prep (free tier, 250 req/day)."""
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        return pd.DataFrame()

    try:
        r = requests.get(
            f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}",
            params={"from": start, "to": end, "apikey": api_key},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        historical = data.get("historical", [])
        if not historical:
            return pd.DataFrame()

        df = pd.DataFrame(historical)
        col_map = {"date": "Date", "open": "Open", "high": "High",
                    "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns=col_map)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    except Exception as e:
        print(f"[DataLoader] FMP failed for {symbol}: {e}")
        return pd.DataFrame()


# ─── Twelve Data (free: 800 calls/day, 8 per min) ───────────

def _fetch_twelve_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Twelve Data (free tier, 800 req/day)."""
    api_key = os.environ.get("TWELVE_DATA_API_KEY", "")
    if not api_key:
        return pd.DataFrame()

    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol,
                "interval": "1day",
                "start_date": start,
                "end_date": end,
                "outputsize": 5000,
                "apikey": api_key,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        values = data.get("values", [])
        if not values:
            return pd.DataFrame()

        df = pd.DataFrame(values)
        col_map = {"datetime": "Date", "open": "Open", "high": "High",
                    "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns=col_map)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    except Exception as e:
        print(f"[DataLoader] Twelve Data failed for {symbol}: {e}")
        return pd.DataFrame()


# ─── Main fetch_data (Tradier > Yahoo > Alpha Vantage > FMP > Twelve Data) ───

def fetch_data(
    symbol: str,
    period_days: int = 730,
    interval: str = "1h",
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data with multi-source fallback chain:
      1. Tradier API (primary, unlimited for stocks/ETFs)
      2. Yahoo Finance (free, reliable, occasionally rate-limited)
      3. Alpha Vantage (free tier: 25 calls/day)
      4. Financial Modeling Prep (free tier: 250 calls/day)
      5. Twelve Data (free tier: 800 calls/day)
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

    use_tradier = _tradier_available() and not _is_crypto(ticker) and interval in ("1d", "daily")
    df = pd.DataFrame()

    # ── 1. Tradier (stocks/ETFs, daily only, unlimited) ──
    if use_tradier:
        df = _fetch_tradier(ticker, start, end, interval="daily")
        if not df.empty:
            df.dropna(inplace=True)
            df = df[df["Volume"] > 0]
            if not df.empty:
                return df

    # ── 2. Yahoo Finance (free, reliable) ──
    df = _fetch_yahoo(ticker, start, end, interval)
    if not df.empty:
        df.dropna(inplace=True)
        df = df[df["Volume"] > 0]
        if not df.empty:
            return df

    # ── 3. Alpha Vantage (25 calls/day free) ──
    if interval in ("1d", "daily") and not _is_crypto(ticker):
        df = _fetch_alpha_vantage(ticker, start, end)
        if not df.empty:
            df.dropna(inplace=True)
            df = df[df["Volume"] > 0]
            if not df.empty:
                return df

    # ── 4. Financial Modeling Prep (250 calls/day free) ──
    if interval in ("1d", "daily") and not _is_crypto(ticker):
        df = _fetch_fmp(ticker, start, end)
        if not df.empty:
            df.dropna(inplace=True)
            df = df[df["Volume"] > 0]
            if not df.empty:
                return df

    # ── 5. Twelve Data (800 calls/day free) ──
    if interval in ("1d", "daily") and not _is_crypto(ticker):
        df = _fetch_twelve_data(ticker, start, end)
        if not df.empty:
            df.dropna(inplace=True)
            df = df[df["Volume"] > 0]
            if not df.empty:
                return df

    raise ValueError(
        f"No data returned for {ticker} from any source (Tradier, Yahoo, "
        f"Alpha Vantage, FMP, Twelve Data). Check symbol or try daily interval."
    )


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
