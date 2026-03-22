"""
screener.py — Multi-Ticker Regime Screener Engine
Scans a watchlist in parallel, returns regime + signal state for each ticker.
Designed to power a real-time dashboard/screener.
"""

import concurrent.futures
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from data_loader import fetch_data, engineer_features, resolve_ticker
from hmm_engine import RegimeDetector, REGIME_LABELS
from backtester import compute_confirmations, get_current_signal
from strategy_v2 import get_current_signal_v2


# ── Curated Watchlists (focused subsets for quick scans) ──
WATCHLISTS = {
    "Mag 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Semiconductors": [
        "NVDA", "AMD", "AVGO", "QCOM", "TSM", "MU", "MRVL", "INTC", "TXN", "LRCX",
        "KLAC", "AMAT", "ASML", "SMCI", "ADI", "NXPI", "ON", "MCHP", "SWKS", "MPWR", "ARM",
    ],
    "Software / Cloud": [
        "CRM", "NOW", "ADBE", "INTU", "SNOW", "DDOG", "CRWD", "ZS", "NET", "PANW",
        "PLTR", "MDB", "TEAM", "WDAY", "HUBS", "VEEV", "OKTA", "SHOP", "SQ", "PYPL",
    ],
    "AI / Quantum": [
        "NVDA", "PLTR", "AI", "SOUN", "BBAI", "IONQ", "RGTI", "QUBT", "ARM", "SMCI",
    ],
    "Small-Cap Defense / Space": [
        "RKLB", "RCAT", "LUNR", "ASTS", "KTOS", "AVAV", "JOBY", "ACHR", "PL", "AXON",
    ],
    "3x Leveraged Bull": [
        "TQQQ", "SPXL", "UPRO", "SOXL", "TNA", "LABU", "FNGU", "TECL", "FAS",
        "DFEN", "DRN", "NAIL", "KORU", "YINN", "NVDL", "TSLL", "BULZ",
    ],
    "3x Leveraged Bear": [
        "SQQQ", "SPXS", "SOXS", "TZA", "LABD", "FNGD", "TECS", "FAZ", "DRV", "YANG",
    ],
    "2x Leveraged": [
        "QLD", "SSO", "UWM", "ROM", "UYG", "DIG", "UCO", "AGQ", "NUGT", "BOIL",
        "QID", "SDS", "TWM", "SKF", "DUG", "SCO", "ZSL", "DUST", "KOLD",
    ],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"],
    "Crypto Stocks": ["COIN", "MSTR", "HOOD", "RIOT", "MARA", "CLSK"],
    "Index ETFs": [
        "SPY", "QQQ", "IWM", "DIA", "RSP", "VTI", "VOO", "MDY", "IJR",
    ],
    "Sector ETFs": [
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLB",
        "SMH", "SOXX", "IGV", "ARKK", "XBI", "GDX", "GLD", "TLT",
    ],
}


def _build_universe() -> dict:
    """
    Build ticker universes dynamically from NASDAQ/NYSE feeds.
    Returns dict of universe names -> symbol lists.
    """
    crypto = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]

    try:
        from ticker_universe import fetch_universe, get_stock_symbols, get_etf_symbols

        all_tickers = fetch_universe()
        all_symbols = [t["symbol"] for t in all_tickers]
        stocks = get_stock_symbols()
        etfs = get_etf_symbols()

        return {
            "ALL TICKERS": sorted(set(all_symbols + crypto)),
            "All Stocks (no ETFs)": sorted(set(stocks + crypto)),
            "All ETFs": sorted(etfs),
        }
    except Exception as e:
        print(f"[Screener] Universe fetch failed ({e}), using curated lists")
        _all = set()
        for _tickers in WATCHLISTS.values():
            _all.update(_tickers)
        return {"ALL TICKERS": sorted(_all)}


# Build universes on import (cached by ticker_universe.py for 24h)
_universes = _build_universe()
for _uname, _usyms in _universes.items():
    WATCHLISTS[_uname] = _usyms


def scan_single_ticker(
    symbol: str,
    interval: str = "1d",
    period_days: int = 730,
    n_regimes: int = 7,
    min_confirmations: int = 6,
    regime_confirm_bars: int = 2,
    strategy: str = "v2",
) -> Optional[Dict]:
    """
    Scan a single ticker: fetch data, train HMM, get current regime + signal.

    strategy: "v1" (original 8-conf) or "v2" (12-conf long-call optimized, default)
    Returns a dict with all screener-relevant fields, or None on failure.
    """
    try:
        ticker = resolve_ticker(symbol)

        # Fetch and prepare data
        raw_df = fetch_data(symbol=symbol, period_days=period_days, interval=interval)
        feat_df = engineer_features(raw_df)

        if len(feat_df) < 100:
            return None

        # Train HMM
        detector = RegimeDetector(n_regimes=n_regimes)
        regime_df = detector.train(feat_df)

        # Current regime
        current = detector.predict_current(regime_df)

        # Current signal with confirmations (V1 or V2)
        if strategy == "v2":
            signal_data = get_current_signal_v2(regime_df, min_confirmations=min_confirmations, regime_confirm_bars=regime_confirm_bars)
        else:
            signal_data = get_current_signal(regime_df, min_confirmations=min_confirmations, regime_confirm_bars=regime_confirm_bars)

        # Price change stats
        price_now = float(regime_df["Close"].iloc[-1])
        price_prev = float(regime_df["Close"].iloc[-2]) if len(regime_df) > 1 else price_now
        price_1d_ago = None
        price_5d_ago = None
        price_20d_ago = None

        # Estimate bar counts for lookbacks based on interval
        bars_per_day = 7 if interval in ("1h", "60m") else 1

        if len(regime_df) > bars_per_day:
            price_1d_ago = float(regime_df["Close"].iloc[-bars_per_day])
        if len(regime_df) > 5 * bars_per_day:
            price_5d_ago = float(regime_df["Close"].iloc[-5 * bars_per_day])
        if len(regime_df) > 20 * bars_per_day:
            price_20d_ago = float(regime_df["Close"].iloc[-20 * bars_per_day])

        def pct_change(old, new):
            return round((new - old) / old * 100, 2) if old and old != 0 else None

        # Regime history — how many bars in current regime
        regime_ids = regime_df["regime_id"].values
        current_regime_id = regime_ids[-1]
        streak = 0
        for r in reversed(regime_ids):
            if r == current_regime_id:
                streak += 1
            else:
                break

        # Transition probabilities for current regime
        trans_matrix = detector.get_transition_matrix()
        current_label = current["regime_label"]
        if current_label in trans_matrix.index:
            trans_from_current = trans_matrix.loc[current_label].to_dict()
        else:
            trans_from_current = {}

        # RSI / ADX / MACD from signal data
        return {
            "symbol": symbol.upper(),
            "ticker": ticker,
            "price": price_now,
            "change_bar": pct_change(price_prev, price_now),
            "change_1d": pct_change(price_1d_ago, price_now),
            "change_5d": pct_change(price_5d_ago, price_now),
            "change_20d": pct_change(price_20d_ago, price_now),
            "regime_id": current["regime_id"],
            "regime_label": current["regime_label"],
            "regime_confidence": round(current["confidence"], 3),
            "regime_streak": streak,
            "signal": signal_data["signal"],
            "action": signal_data["action"],
            "confirmations_met": signal_data["confirmations_met"],
            "confirmations_total": signal_data.get("confirmations_total", 8),
            "confirmation_detail": signal_data["confirmation_detail"],
            "regime_changed": signal_data["regime_changed"],
            "prev_regime": signal_data["prev_regime"],
            "regime_confirmed": signal_data.get("regime_confirmed", None),
            "rsi": round(signal_data["rsi"], 1) if signal_data["rsi"] is not None else None,
            "adx": round(signal_data["adx"], 1) if signal_data["adx"] is not None else None,
            "macd_hist": round(signal_data["macd_hist"], 4) if signal_data["macd_hist"] is not None else None,
            "hv_rank": signal_data.get("hv_rank"),
            "transition_probs": trans_from_current,
            "mean_return": current["mean_return"],
            "volatility": current["volatility"],
            "data_points": len(regime_df),
            "scan_time": datetime.now().isoformat(),
            "error": None,
            # Store full data for drill-down
            "_regime_df": regime_df,
            "_detector": detector,
        }

    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "ticker": resolve_ticker(symbol),
            "error": str(e),
            "price": None,
            "regime_id": None,
            "regime_label": "ERROR",
            "regime_confidence": None,
            "signal": "ERROR",
            "confirmations_met": 0,
            "scan_time": datetime.now().isoformat(),
        }


def scan_watchlist(
    symbols: List[str],
    interval: str = "1d",
    period_days: int = 730,
    n_regimes: int = 7,
    min_confirmations: int = 6,
    regime_confirm_bars: int = 2,
    max_workers: int = 6,
    strategy: str = "v2",
) -> List[Dict]:
    """
    Scan multiple tickers in parallel.

    strategy: "v1" or "v2" (default)
    Returns list of scan result dicts, sorted by signal priority.
    """
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                scan_single_ticker,
                sym,
                interval,
                period_days,
                n_regimes,
                min_confirmations,
                regime_confirm_bars,
                strategy,
            ): sym
            for sym in symbols
        }

        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result is not None:
                results.append(result)

    # Sort: actionable signals first
    signal_priority = {
        "LONG -- ENTER": 0,
        "EXIT -- REGIME FLIP": 1,
        "LONG -- CONFIRMING": 2,
        "LONG -- HOLD": 3,
        "CASH -- NEUTRAL": 4,
        "CASH -- BEARISH": 5,
        "ERROR": 99,
    }

    results.sort(key=lambda r: (signal_priority.get(r.get("signal", "ERROR"), 50), -(r.get("confirmations_met") or 0)))

    return results


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert scan results to a clean DataFrame for display (drops internal objects)."""
    display_fields = [
        "symbol", "price", "change_1d", "change_5d",
        "regime_label", "regime_confidence", "regime_streak",
        "signal", "confirmations_met", "rsi", "adx",
        "regime_changed", "prev_regime", "error",
    ]

    rows = []
    for r in results:
        row = {k: r.get(k) for k in display_fields}
        rows.append(row)

    return pd.DataFrame(rows)
