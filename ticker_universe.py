"""
ticker_universe.py — Dynamic US Ticker Universe
Fetches all NYSE + NASDAQ listed tickers from official NASDAQ data feeds.
Filters out OTC, test issues, warrants, rights, units, and preferred shares.
Caches locally to avoid repeated fetches.
"""

import requests
import json
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Set

CACHE_FILE = os.path.join(os.path.dirname(__file__), ".ticker_cache.json")
CACHE_MAX_AGE_HOURS = 24

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Suffixes that indicate non-common-stock instruments
EXCLUDE_SUFFIXES = re.compile(r"[.+\-=^#]|[A-Z]{1,5}(W|U|R|WS|UN|WT)$")

# Words in security name that indicate non-tradeable / non-common
EXCLUDE_NAME_PATTERNS = re.compile(
    r"warrant|rights|units|preferred|debenture|note\b|bond\b|%.*note|acquisition.*unit"
    r"|when.issued|depositary|trust.*series|convertible",
    re.IGNORECASE,
)


def _parse_nasdaq_listed(text: str) -> List[Dict]:
    """Parse nasdaqlisted.txt pipe-delimited format."""
    tickers = []
    lines = text.strip().split("\n")
    for line in lines[1:]:  # skip header
        line = line.strip().rstrip("\r")
        if not line or line.startswith("File Creation"):
            continue
        parts = line.split("|")
        if len(parts) < 8:
            continue

        symbol = parts[0].strip()
        name = parts[1].strip()
        test_issue = parts[3].strip()
        financial_status = parts[4].strip()
        is_etf = parts[6].strip()

        # Skip test issues
        if test_issue == "Y":
            continue
        # Skip deficient/delinquent (keep normal "N" and ETFs)
        if financial_status in ("D", "E", "Q", "G", "H", "J", "K"):
            continue

        tickers.append({
            "symbol": symbol,
            "name": name,
            "exchange": "NASDAQ",
            "etf": is_etf == "Y",
        })

    return tickers


def _parse_other_listed(text: str) -> List[Dict]:
    """Parse otherlisted.txt pipe-delimited format (NYSE, AMEX, ARCA, BATS)."""
    exchange_map = {
        "N": "NYSE",
        "A": "AMEX",
        "P": "ARCA",
        "Z": "BATS",
        "V": "IEXG",
    }
    tickers = []
    lines = text.strip().split("\n")
    for line in lines[1:]:  # skip header
        line = line.strip().rstrip("\r")
        if not line or line.startswith("File Creation"):
            continue
        parts = line.split("|")
        if len(parts) < 8:
            continue

        symbol = parts[0].strip()
        name = parts[1].strip()
        exchange_code = parts[2].strip()
        is_etf = parts[4].strip()
        test_issue = parts[6].strip()

        if test_issue == "Y":
            continue

        exchange = exchange_map.get(exchange_code, exchange_code)

        tickers.append({
            "symbol": symbol,
            "name": name,
            "exchange": exchange,
            "etf": is_etf == "Y",
        })

    return tickers


def _filter_tradeable(tickers: List[Dict]) -> List[Dict]:
    """Filter to common stocks and ETFs only. Remove warrants, units, rights, preferred."""
    filtered = []
    seen = set()

    for t in tickers:
        sym = t["symbol"]

        # Skip duplicates
        if sym in seen:
            continue

        # Skip symbols with special characters (class shares like BRK.A, warrants like ACHR+)
        if any(c in sym for c in ".+=-^#$"):
            continue

        # Skip symbols ending in warrant/unit/rights suffixes
        if len(sym) > 4 and EXCLUDE_SUFFIXES.search(sym):
            continue

        # Skip if name contains non-common-stock keywords
        if EXCLUDE_NAME_PATTERNS.search(t["name"]):
            continue

        # Skip very short names (likely special instruments)
        if len(t["name"]) < 3:
            continue

        seen.add(sym)
        filtered.append(t)

    return filtered


def fetch_universe(force_refresh: bool = False) -> List[Dict]:
    """
    Fetch all NYSE + NASDAQ tradeable tickers.
    Returns list of dicts with keys: symbol, name, exchange, etf
    Caches results locally for 24 hours.
    """
    # Check cache
    if not force_refresh and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            cached_at = datetime.fromisoformat(cache["timestamp"])
            if datetime.now() - cached_at < timedelta(hours=CACHE_MAX_AGE_HOURS):
                return cache["tickers"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    all_tickers = []

    # Fetch NASDAQ-listed
    try:
        r = requests.get(NASDAQ_URL, timeout=15)
        r.raise_for_status()
        all_tickers.extend(_parse_nasdaq_listed(r.text))
    except Exception as e:
        print(f"[Universe] NASDAQ fetch failed: {e}")

    # Fetch NYSE/AMEX/other
    try:
        r = requests.get(OTHER_URL, timeout=15)
        r.raise_for_status()
        all_tickers.extend(_parse_other_listed(r.text))
    except Exception as e:
        print(f"[Universe] Other exchanges fetch failed: {e}")

    # Filter
    filtered = _filter_tradeable(all_tickers)
    filtered.sort(key=lambda t: t["symbol"])

    # Cache
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "count": len(filtered),
                "tickers": filtered,
            }, f)
    except Exception as e:
        print(f"[Universe] Cache write failed: {e}")

    print(f"[Universe] Loaded {len(filtered)} tradeable US tickers ({len([t for t in filtered if t['exchange'] == 'NASDAQ'])} NASDAQ, {len([t for t in filtered if t['exchange'] == 'NYSE'])} NYSE, {len([t for t in filtered if t['etf']])} ETFs)")

    return filtered


def get_all_symbols(force_refresh: bool = False) -> List[str]:
    """Get sorted list of all tradeable ticker symbols."""
    return [t["symbol"] for t in fetch_universe(force_refresh)]


def get_symbols_by_exchange(exchange: str, force_refresh: bool = False) -> List[str]:
    """Get tickers for a specific exchange (NYSE, NASDAQ, AMEX, ARCA)."""
    return [t["symbol"] for t in fetch_universe(force_refresh) if t["exchange"] == exchange]


def get_etf_symbols(force_refresh: bool = False) -> List[str]:
    """Get all ETF symbols."""
    return [t["symbol"] for t in fetch_universe(force_refresh) if t["etf"]]


def get_stock_symbols(force_refresh: bool = False) -> List[str]:
    """Get all non-ETF stock symbols."""
    return [t["symbol"] for t in fetch_universe(force_refresh) if not t["etf"]]


def search_tickers(query: str, force_refresh: bool = False) -> List[Dict]:
    """Search tickers by symbol or name substring."""
    q = query.upper()
    return [t for t in fetch_universe(force_refresh)
            if q in t["symbol"] or q in t["name"].upper()]
