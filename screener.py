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

import os
import logging

from data_loader import fetch_data, engineer_features, resolve_ticker
from hmm_engine import RegimeDetector, REGIME_LABELS
from backtester import compute_confirmations, get_current_signal
from strategy_v2 import get_current_signal_v2
from strategy_leaps import get_current_signal_leaps

logger = logging.getLogger(__name__)

_IS_CLOUD = bool(os.environ.get("RENDER") or os.environ.get("PORT"))

# ── Ticker Info Cache (sector, industry, options availability) ──
_ticker_info_cache: Dict[str, Dict] = {}

# Sectors and industries to exclude from screening
EXCLUDED_SECTORS = {"Healthcare", "Health Technology"}
EXCLUDED_INDUSTRIES_KEYWORDS = {"Biotechnology", "Pharmaceutical", "Pharma", "Drug", "Biotech"}

# Minimum screening thresholds
MIN_PRICE = 1.0
MIN_AVG_VOLUME_30D = 500_000


def _fetch_ticker_info(symbol: str) -> Dict:
    """
    Fetch sector, industry, and options availability for a ticker via yfinance.
    Results are cached in-memory to avoid repeated API calls.
    """
    if symbol in _ticker_info_cache:
        return _ticker_info_cache[symbol]

    # Default has_options to True so tickers aren't filtered out when yfinance is unreachable
    info = {"sector": None, "industry": None, "has_options": True, "name": None}

    # Skip crypto symbols — they don't have sectors/options in the traditional sense
    if symbol.endswith("-USD"):
        info["sector"] = "Crypto"
        info["industry"] = "Cryptocurrency"
        info["has_options"] = False
        _ticker_info_cache[symbol] = info
        return info

    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)

        # Get sector and industry from info
        tk_info = tk.info or {}
        info["sector"] = tk_info.get("sector")
        info["industry"] = tk_info.get("industry")
        info["name"] = tk_info.get("shortName") or tk_info.get("longName")

        # Check if options are available — only set False on definitive empty, keep True on errors
        try:
            expiry_dates = tk.options
            if expiry_dates is not None:
                info["has_options"] = len(expiry_dates) > 0
            # If None, keep default True (can't determine availability)
        except Exception:
            pass  # Keep default True — don't filter on failure

        # ETFs often don't have sector — classify by fund type
        if not info["sector"] and tk_info.get("quoteType") == "ETF":
            info["sector"] = "ETF"
            info["industry"] = tk_info.get("category", "ETF")

    except Exception as e:
        logger.debug(f"[Screener] Info fetch failed for {symbol}: {e}")

    _ticker_info_cache[symbol] = info
    return info


def _is_excluded_sector_or_industry(info: Dict) -> bool:
    """Check if a ticker should be excluded based on sector/industry."""
    sector = (info.get("sector") or "").strip()
    industry = (info.get("industry") or "").strip()

    if sector in EXCLUDED_SECTORS:
        return True

    for keyword in EXCLUDED_INDUSTRIES_KEYWORDS:
        if keyword.lower() in industry.lower():
            return True

    return False


def _passes_prescreen(symbol: str, raw_df: pd.DataFrame, min_avg_volume: Optional[int] = None) -> bool:
    """
    Check if ticker passes pre-screening filters:
    - Price > $1
    - 30-day average volume > min_avg_volume (defaults to MIN_AVG_VOLUME_30D)
    Pass min_avg_volume=0 to skip the volume filter entirely.
    """
    if raw_df is None or raw_df.empty:
        return False

    # Current price check
    current_price = float(raw_df["Close"].iloc[-1])
    if current_price < MIN_PRICE:
        return False

    # 30-day average volume check
    vol_threshold = min_avg_volume if min_avg_volume is not None else MIN_AVG_VOLUME_30D
    if vol_threshold > 0:
        recent_volume = raw_df["Volume"].tail(30)
        avg_volume_30d = float(recent_volume.mean())
        if avg_volume_30d < vol_threshold:
            return False

    return True


# ── Curated Watchlists (sector-specific, optionable stocks only) ──
# Excludes Healthcare sector and Biotech/Pharmaceutical industries
WATCHLISTS = {
    # ── Sector: Technology ──
    "Technology — Mag 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Technology — Semiconductors": [
        "NVDA", "AMD", "AVGO", "QCOM", "TSM", "MU", "MRVL", "INTC", "TXN", "LRCX",
        "KLAC", "AMAT", "ASML", "SMCI", "ADI", "NXPI", "ON", "MCHP", "SWKS", "MPWR", "ARM",
    ],
    "Technology — Software / Cloud": [
        "CRM", "NOW", "ADBE", "INTU", "SNOW", "DDOG", "CRWD", "ZS", "NET", "PANW",
        "PLTR", "MDB", "TEAM", "WDAY", "HUBS", "VEEV", "OKTA", "SHOP", "SQ", "PYPL",
    ],
    "Technology — AI / Quantum": [
        "NVDA", "PLTR", "AI", "SOUN", "BBAI", "IONQ", "RGTI", "QUBT", "ARM", "SMCI",
    ],
    # ── Sector: Industrials ──
    "Industrials — Defense / Aerospace": [
        "BA", "LMT", "RTX", "NOC", "GD", "KTOS", "AVAV", "AXON",
    ],
    "Industrials — Manufacturing / Logistics": [
        "GE", "HON", "CAT", "DE", "UPS", "FDX", "UNP", "CSX", "NSC", "WM", "RSG",
    ],
    "Industrials — Space": [
        "RKLB", "RCAT", "LUNR", "ASTS", "JOBY", "ACHR", "PL",
    ],
    # ── Sector: Energy ──
    "Energy — Oil & Gas": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "DVN", "OXY",
        "HES", "HAL", "FANG", "BKR", "MRO", "APA",
    ],
    # ── Sector: Financials ──
    "Financials — Banks": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "USB", "PNC", "TFC", "COF",
    ],
    "Financials — Capital Markets / Insurance": [
        "BLK", "AXP", "ICE", "CME", "MCO", "SPGI", "AIG", "MET",
    ],
    "Financials — Fintech": [
        "PYPL", "SQ", "SOFI", "AFRM", "UPST", "LC", "NU", "HOOD",
    ],
    # ── Sector: Consumer Discretionary ──
    "Consumer Discretionary — Retail": [
        "AMZN", "WMT", "COST", "TGT", "HD", "LOW", "NKE", "ETSY",
    ],
    "Consumer Discretionary — Restaurants / Leisure": [
        "SBUX", "MCD", "DIS", "NFLX", "ABNB", "BKNG", "DKNG", "PENN",
    ],
    "Consumer Discretionary — Mobility": [
        "TSLA", "UBER", "LYFT", "DASH",
    ],
    # ── Sector: Consumer Staples ──
    "Consumer Staples": [
        "PG", "KO", "PEP", "CL", "MDLZ", "KHC", "PM", "MO", "WBA",
    ],
    # ── Sector: Communication Services ──
    "Communication Services": [
        "META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "CHTR", "T", "VZ", "TMUS",
    ],
    # ── Sector: Real Estate ──
    "Real Estate": [
        "AMT", "SPG", "DLR", "PSA", "O", "WELL", "AVB", "EQR",
    ],
    # ── Sector: Utilities ──
    "Utilities": [
        "NEE", "DUK", "SO", "EXC", "AEP", "SRE", "D", "ED",
    ],
    # ── Sector: Materials ──
    "Materials": [
        "LIN", "APD", "ECL", "SHW", "NEM", "FCX", "NUE", "DOW",
    ],
    # ── Crypto ──
    "Crypto — Digital Assets": ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"],
    "Crypto — Stocks": ["COIN", "MSTR", "HOOD", "RIOT", "MARA", "CLSK"],
    # ── Leveraged / Inverse ETFs ──
    "Leveraged — 3x Bull": [
        "TQQQ", "SPXL", "UPRO", "SOXL", "TNA", "FNGU", "TECL", "FAS",
        "DFEN", "DRN", "NAIL", "YINN", "NVDL", "TSLL", "BULZ",
        "HIBL", "WANT", "WEBL", "RETL", "MIDU", "TPOR",
        "EURL", "EDC", "BNKU", "INDL", "MEXX", "UBOT", "CONL", "DUSL",
    ],
    "Leveraged — 3x Bear": [
        "SQQQ", "SPXS", "SOXS", "TZA", "FNGD", "TECS", "FAZ", "DRV", "YANG",
        "HIBS", "WEBS", "ERY", "EDZ", "BERZ", "SH", "SPDN",
    ],
    "Leveraged — 3x Inverse Single-Stock": [
        "NVDD", "TSLS", "AMZD", "MSFD", "AAPLD",
    ],
    "Leveraged — 2x Bull": [
        "QLD", "SSO", "UWM", "ROM", "UYG", "DIG", "UCO", "AGQ", "NUGT", "BOIL",
        "UGE", "UCC", "UPW", "SAA", "URE", "UXI", "MVV", "CWEB", "JNUG",
    ],
    "Leveraged — 2x Bear": [
        "QID", "SDS", "TWM", "SKF", "DUG", "SCO", "ZSL", "DUST", "KOLD",
        "REK", "SZK", "SCC", "SRS", "SIJ", "SDD", "MZZ", "JDST",
    ],
    # ── Volatility ──
    "Volatility": [
        "UVXY", "SVXY", "VXX", "VIXY", "SVOL",
    ],
    # ── Broad Market ETFs ──
    "Index ETFs": [
        "SPY", "QQQ", "IWM", "DIA", "RSP", "VTI", "VOO", "MDY", "IJR",
    ],
    "Sector ETFs": [
        "XLK", "XLF", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLB",
        "SMH", "SOXX", "IGV", "ARKK", "GDX", "GLD", "TLT",
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
    min_avg_volume: Optional[int] = None,
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

        # Pre-screen: price > $1, 30-day avg volume check
        if not _passes_prescreen(symbol, raw_df, min_avg_volume=min_avg_volume):
            logger.debug(f"[Scan] {symbol} excluded: prescreen (price/volume)")
            return None

        feat_df = engineer_features(raw_df)

        if len(feat_df) < 100:
            logger.debug(f"[Scan] {symbol} excluded: insufficient data ({len(feat_df)} pts)")
            return None

        # Fetch ticker info (sector, name, options) — non-blocking, uses cache
        ticker_info = _fetch_ticker_info(symbol)

        # Exclude Healthcare / Biotechnology sectors
        if _is_excluded_sector_or_industry(ticker_info):
            logger.debug(f"[Scan] {symbol} excluded: healthcare/biotech sector")
            return None

        # Train HMM (fewer iterations + looser tolerance on cloud for speed)
        hmm_iter = 50 if _IS_CLOUD else 100
        hmm_tol = 1e-2 if _IS_CLOUD else 1e-4
        detector = RegimeDetector(n_regimes=n_regimes, n_iter=hmm_iter, tol=hmm_tol)
        regime_df = detector.train(feat_df)

        # Current regime
        current = detector.predict_current(regime_df)

        # Current signal with confirmations (V1, V2, or LEAPS)
        if strategy == "leaps":
            # LEAPS has 10 confs — auto-cap min_confirmations to sensible range
            leaps_min = min(min_confirmations, 6)
            signal_data = get_current_signal_leaps(regime_df, min_confirmations=leaps_min, regime_confirm_bars=regime_confirm_bars)
        elif strategy == "v2":
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
        # For daily data, 1 day back = iloc[-2] (previous bar)
        bars_per_day = 7 if interval in ("1h", "60m") else 1
        offset_1d = bars_per_day + 1 if bars_per_day == 1 else bars_per_day

        if len(regime_df) > offset_1d:
            price_1d_ago = float(regime_df["Close"].iloc[-offset_1d])
        if len(regime_df) > 5 * bars_per_day + 1:
            price_5d_ago = float(regime_df["Close"].iloc[-(5 * bars_per_day + 1)])
        if len(regime_df) > 20 * bars_per_day + 1:
            price_20d_ago = float(regime_df["Close"].iloc[-(20 * bars_per_day + 1)])

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

        # Compute EMA 10/20/50 for filtering
        close = regime_df["Close"]
        ema_10 = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
        ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

        # 30-day average volume
        avg_volume_30d = float(regime_df["Volume"].tail(30).mean())

        # RSI / ADX / MACD from signal data
        return {
            "symbol": symbol.upper(),
            "ticker": ticker,
            "name": ticker_info.get("name"),
            "sector": ticker_info.get("sector"),
            "industry": ticker_info.get("industry"),
            "has_options": ticker_info.get("has_options", True),
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
            "pct_52w": signal_data.get("pct_52w"),
            "ema_10": round(ema_10, 2),
            "ema_20": round(ema_20, 2),
            "ema_50": round(ema_50, 2),
            "ema_10_above_20": ema_10 > ema_20,
            "price_above_ema50": price_now > ema_50,
            "avg_volume_30d": round(avg_volume_30d),
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


def _scan_batch(
    symbols: List[str],
    interval: str,
    period_days: int,
    n_regimes: int,
    min_confirmations: int,
    regime_confirm_bars: int,
    max_workers: int,
    strategy: str,
    min_avg_volume: Optional[int] = None,
) -> List[Dict]:
    """Scan a single batch of tickers in parallel."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                scan_single_ticker, sym, interval, period_days,
                n_regimes, min_confirmations, regime_confirm_bars, strategy,
                min_avg_volume,
            ): sym
            for sym in symbols
        }
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result is not None:
                results.append(result)
    return results


SIGNAL_PRIORITY = {
    "LONG -- ENTER": 0,
    "LEAPS -- BUY": 0,
    "EXIT -- REGIME FLIP": 1,
    "LEAPS -- EXIT": 1,
    "LONG -- CONFIRMING": 2,
    "LEAPS -- WATCH": 2,
    "LONG -- HOLD": 3,
    "LEAPS -- HOLD": 3,
    "CASH -- NEUTRAL": 4,
    "LEAPS -- WAIT": 4,
    "CASH -- BEARISH": 5,
    "LEAPS -- AVOID": 5,
    "ERROR": 99,
}

BULLISH_SIGNALS = {"LONG -- ENTER", "LONG -- CONFIRMING", "LONG -- HOLD", "LEAPS -- BUY", "LEAPS -- WATCH", "LEAPS -- HOLD"}


def scan_watchlist(
    symbols: List[str],
    interval: str = "1d",
    period_days: int = 730,
    n_regimes: int = 7,
    min_confirmations: int = 6,
    regime_confirm_bars: int = 2,
    max_workers: int = 10,
    strategy: str = "v2",
    batch_size: int = 200,
    progress_callback=None,
    bullish_only: bool = False,
    min_avg_volume: Optional[int] = None,
) -> List[Dict]:
    """
    Scan multiple tickers in batches of `batch_size` (default 200).

    progress_callback: optional callable(batch_num, total_batches, running_results)
        called after each batch completes so the UI can show progressive results.
    bullish_only: if True, discard bearish/neutral results after each batch
        to save memory and only return bullish signals.

    Returns list of scan result dicts, sorted by signal priority.
    """
    results = []
    total = len(symbols)

    # Split into batches
    batches = [symbols[i:i + batch_size] for i in range(0, total, batch_size)]

    for batch_idx, batch in enumerate(batches):
        batch_results = _scan_batch(
            batch, interval, period_days, n_regimes,
            min_confirmations, regime_confirm_bars, max_workers, strategy,
            min_avg_volume,
        )

        if bullish_only:
            # Only keep bullish signals — discard bearish/neutral immediately
            batch_results = [r for r in batch_results if r.get("signal") in BULLISH_SIGNALS]

        results.extend(batch_results)

        # Notify caller of progress
        if progress_callback:
            progress_callback(batch_idx + 1, len(batches), results)

    # Sort: actionable signals first
    results.sort(key=lambda r: (SIGNAL_PRIORITY.get(r.get("signal", "ERROR"), 50), -(r.get("confirmations_met") or 0)))

    return results


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert scan results to a clean DataFrame for display (drops internal objects)."""
    display_fields = [
        "symbol", "sector", "industry", "price", "change_1d", "change_5d",
        "regime_label", "regime_confidence", "regime_streak",
        "signal", "confirmations_met", "rsi", "adx",
        "regime_changed", "prev_regime", "error",
    ]

    rows = []
    for r in results:
        row = {k: r.get(k) for k in display_fields}
        rows.append(row)

    return pd.DataFrame(rows)
