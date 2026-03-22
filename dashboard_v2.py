"""
dashboard_v2.py — Real-Time Regime Screener & Dashboard
Multi-ticker scanning with auto-refresh, filterable screener table, and drill-down.

Run with: streamlit run dashboard_v2.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

from data_loader import fetch_data, engineer_features, resolve_ticker
from hmm_engine import RegimeDetector, REGIME_LABELS
from backtester import run_backtest, get_current_signal, compute_confirmations
from strategy_v2 import run_backtest_v2, get_current_signal_v2
from screener import scan_watchlist, results_to_dataframe, WATCHLISTS
from options_picker import get_options_recommendations, scan_options_for_watchlist
from settings_manager import load_settings, save_settings, DEFAULT_SETTINGS
from alerts import process_alerts
from tradier_broker import (
    is_configured as tradier_configured, get_account_info, get_positions,
    place_option_order, place_equity_order, save_config as save_tradier_config,
)
from position_sizer import compute_position_size
from performance_tracker import (
    get_open_positions, get_closed_trades, get_performance_summary,
    log_entry, log_exit,
)
from roll_manager import check_roll_trigger, find_roll_target
from order_executor import execute_buy_calls, execute_sell_to_close, execute_roll as exec_roll

# ─── Page Config ───
st.set_page_config(
    page_title="RRJCAR Regime Scanner",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── PWA Meta Tags ───
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#101114">
""", unsafe_allow_html=True)

# ─── Suppress error details ───
try:
    st.set_option("client.showErrorDetails", False)
except Exception:
    pass

# ─── Palantir-Inspired Dark UI ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    #MainMenu, footer, header {visibility: hidden;}

    /* Hide all Streamlit chrome: deploy bar, toolbar, branding, badges, watermarks */
    .stStatusWidget, div[data-testid="stStatusWidget"],
    div[data-testid="stNotification"],
    .stSpinner > div > div:first-child,
    div[data-testid="stAppDeployButton"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    .stRunningMan,
    a[href*="streamlit.io"],
    ._profileContainer_gzau3_53,
    div[data-testid="stSidebarCollapsedControl"],
    section[data-testid="stSidebar"],
    iframe[src*="streamlit"],
    div[data-testid="manage-app-button"],
    .viewerBadge_container__r5tak,
    .viewerBadge_link__qRIco,
    ._container_gzau3_1,
    div[class*="viewerBadge"],
    div[class*="stBottomBlockContainer"] iframe,
    .stApp > div:last-child > div[style*="position: fixed"] { display: none !important; visibility: hidden !important; }
    .main .block-container { padding: 0.4rem 0.8rem 1rem; max-width: 1600px; }
    .stApp { background: #101114; color: #e5e7eb; }

    .metric-card { background: transparent; padding: 0.3rem 0; text-align: center; }
    .metric-card .label {
        font-family: 'JetBrains Mono', monospace; color: #6b7280;
        font-size: 0.55rem; font-weight: 400; text-transform: uppercase; letter-spacing: 1px;
    }
    .metric-card .value {
        font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 600;
        color: #f3f4f6; margin-top: 0;
    }
    .bull { color: #2dd4bf !important; }
    .bear { color: #f87171 !important; }
    .neutral { color: #94a3b8 !important; }
    .cash { color: #4b5563 !important; }

    .signal-banner {
        border-radius: 6px; padding: 0.5rem 1rem; text-align: center;
        font-family: 'Inter', sans-serif; font-size: 0.85rem; font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .signal-long-enter { background: #065f46; color: #2dd4bf; }
    .signal-long-hold { background: #1a2e2a; color: #2dd4bf; }
    .signal-exit { background: #7f1d1d; color: #fca5a5; }
    .signal-cash { background: #1f2937; color: #6b7280; }
    .signal-bearish { background: #451a1a; color: #f87171; }

    .regime-badge {
        display: inline-block; padding: 2px 8px; border-radius: 3px;
        font-size: 0.6rem; font-weight: 500; font-family: 'JetBrains Mono', monospace;
    }

    .alert-flash { animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.6;} }

    /* Sidebar — dark bg, force all text white */
    section[data-testid="stSidebar"] { background: #101114 !important; }
    section[data-testid="stSidebar"] > div { background: #101114 !important; }
    section[data-testid="stSidebar"] * { color: #f3f4f6 !important; }
    section[data-testid="stSidebar"] label { color: #d1d5db !important; }
    section[data-testid="stSidebar"] p { color: #e5e7eb !important; }
    section[data-testid="stSidebar"] h2 { font-size: 0.85rem; color: #ffffff !important; }
    section[data-testid="stSidebar"] .stCaption * { color: #9ca3af !important; }

    /* Sidebar inputs */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select,
    section[data-testid="stSidebar"] textarea {
        background: #1f2937 !important; color: #ffffff !important;
        border-color: #374151 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"],
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="select"] span {
        background: #1f2937 !important; color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="popover"],
    section[data-testid="stSidebar"] [role="listbox"],
    section[data-testid="stSidebar"] [role="option"],
    section[data-testid="stSidebar"] ul[role="listbox"] li {
        background: #1f2937 !important; color: #ffffff !important;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] button {
        background: #1f2937 !important; color: #f3f4f6 !important; border-color: #374151 !important;
    }
    section[data-testid="stSidebar"] button[kind="primary"],
    section[data-testid="stSidebar"] button[kind="primary"] span,
    section[data-testid="stSidebar"] button[kind="primary"] p {
        background: #2dd4bf !important; color: #101114 !important; border: none !important;
    }

    /* Sidebar collapse button — always visible */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="baseButton-headerNoPadding"] {
        color: #2dd4bf !important; opacity: 1 !important;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 0; background: #18191d; border-radius: 4px; padding: 2px; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 400;
        padding: 0.4rem 0.8rem; color: #6b7280 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #2dd4bf !important; background: #101114; border-radius: 3px; }

    h1 { font-family: 'Inter', sans-serif !important; color: #f3f4f6 !important; font-weight: 600 !important; font-size: 1.3rem !important; }
    h2,h3,h4 { font-family: 'Inter', sans-serif !important; color: #e5e7eb !important; font-weight: 500 !important; }
    p, span, label, div { font-family: 'Inter', sans-serif; }

    .stDataFrame { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; }

    .stSelectbox > div > div, .stSlider > div, .stNumberInput > div > div,
    .stTextInput > div > div { border-color: #374151 !important; background: #1f2937 !important; }

    .streamlit-expanderHeader { font-family: 'Inter', sans-serif; font-size: 0.75rem; font-weight: 500; color: #9ca3af !important; }

    .stButton > button { border-radius: 4px; font-weight: 500; font-size: 0.75rem;
        background: #1f2937 !important; color: #d1d5db !important; border: 1px solid #374151 !important; }
    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] p { background: #2dd4bf !important; color: #101114 !important; border: none !important; }
    .stButton > button:hover { background: #374151 !important; }
    .stButton > button[kind="primary"]:hover { background: #14b8a6 !important; color: #101114 !important; }

    .stProgress > div > div { background: #1f2937; }
    .stProgress > div > div > div { background: #2dd4bf; }

    .stCaption { color: #6b7280 !important; }
    .stMarkdown a { color: #2dd4bf !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #101114; }
    ::-webkit-scrollbar-thumb { background: #374151; border-radius: 2px; }

    /* ── HTML Grid Layouts (responsive without st.columns) ── */
    .mg { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.2rem; margin-bottom: 0.3rem; }
    .mg .mc { text-align: center; padding: 0.2rem 0; }
    .mg .mc .ml { font-family: 'JetBrains Mono', monospace; color: #6b7280; font-size: 0.55rem; font-weight: 400; text-transform: uppercase; letter-spacing: 1px; }
    .mg .mc .mv { font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 600; color: #f3f4f6; }

    .screener-row {
        display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.8rem;
        padding: 0.25rem 0; border-bottom: 1px solid #1f2937; font-size: 0.8rem;
    }
    .screener-row .sr-price { color: #e5e7eb; }
    .screener-row .sr-sig { font-weight: 600; font-size: 0.75rem; }
    .screener-row .sr-conf { font-size: 0.8rem; }
    .screener-row .sr-rsi, .screener-row .sr-adx { font-size: 0.8rem; }

    .opt-card {
        background: #18191d; border-radius: 6px; padding: 0.4rem 0.6rem;
        margin-bottom: 0.3rem; border: 1px solid #1f2937;
        display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem 0.6rem;
        font-size: 0.85rem;
    }

    /* ── Mobile ── */
    @media (max-width: 600px) {
        .main .block-container { padding: 0.2rem 0.2rem 0.5rem; max-width: 100vw; overflow-x: hidden; }
        .mg { grid-template-columns: repeat(3, 1fr); }
        .mg .mc .mv { font-size: 0.85rem; }
        .mg .mc .ml { font-size: 0.45rem; letter-spacing: 0.5px; }
        .stTabs [data-baseweb="tab-list"] { overflow-x: auto; -webkit-overflow-scrolling: touch; }
        .stTabs [data-baseweb="tab"] { font-size: 0.55rem; padding: 0.3rem 0.5rem; white-space: nowrap; flex-shrink: 0; }
        h1 { font-size: 1rem !important; }
        h2,h3 { font-size: 0.85rem !important; }
        .stDataFrame { font-size: 0.55rem; }
        .stButton > button { font-size: 0.7rem; padding: 0.4rem 0.6rem; min-height: 44px; border-radius: 6px; }
        .signal-banner { font-size: 0.75rem; padding: 0.4rem 0.6rem; }
        .streamlit-expanderHeader { font-size: 0.7rem; }
        iframe { max-height: 180px !important; }
        .stSelectbox { font-size: 0.7rem; }
        .screener-row { font-size: 0.75rem; gap: 0.3rem 0.6rem; }
        .opt-card { font-size: 0.8rem; }
    }
</style>
""", unsafe_allow_html=True)


# ─── Regime Colors (Palantir dark) ───
REGIME_COLORS = {
    0: "#2dd4bf",   # Bull Run — teal
    1: "#14b8a6",   # Bull Trend
    2: "#5eead4",   # Mild Bull
    3: "#6b7280",   # Neutral — gray
    4: "#9ca3af",   # Mild Bear
    5: "#f87171",   # Bear Trend
    6: "#ef4444",   # Crash
}

REGIME_BG_COLORS = {
    0: "#0d3d38",
    1: "#0d3330",
    2: "#1a3a35",
    3: "#1f2937",
    4: "#27272a",
    5: "#3b1818",
    6: "#451a1a",
}


def _get_option_quote_safe(symbol, option_symbol):
    """Get option bid/ask, suppress errors."""
    try:
        from order_executor import _get_option_quote
        return _get_option_quote(symbol, option_symbol)
    except Exception:
        return {"bid": 0, "ask": 0, "last": 0}


def render_metric(label: str, value: str, css_class: str = ""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {css_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics_grid(items):
    """Render metrics as a CSS grid (6 cols desktop, 3 cols mobile).
    items: list of (label, value, css_class) tuples.
    """
    cells = ""
    for label, value, cls in items:
        cells += f'<div class="mc"><div class="ml">{label}</div><div class="mv {cls}">{value}</div></div>'
    st.markdown(f'<div class="mg">{cells}</div>', unsafe_allow_html=True)


def signal_css_class(signal: str) -> str:
    if "ENTER" in signal:
        return "signal-long-enter"
    elif "CONFIRMING" in signal:
        return "signal-long-hold"
    elif "HOLD" in signal:
        return "signal-long-hold"
    elif "EXIT" in signal:
        return "signal-exit"
    elif "BEARISH" in signal:
        return "signal-bearish"
    return "signal-cash"


def signal_icon(signal: str) -> str:
    """No icons — return empty string."""
    return ""


def regime_badge_html(regime_id, regime_label, confidence=None):
    color = REGIME_COLORS.get(regime_id, "#666")
    bg = REGIME_BG_COLORS.get(regime_id, "#111")
    conf_str = f" ({confidence:.0%})" if confidence else ""
    return f'<span class="regime-badge" style="background:{bg}; color:{color}; border:1px solid {color};">{regime_label}{conf_str}</span>'


def _screener_signal_info(r):
    """Extract signal display info from a result row."""
    sig = r.get("signal", "")
    short_sig = sig.replace("LONG -- ", "").replace("CASH -- ", "").replace("EXIT -- ", "EXIT: ")
    sig_colors = {"ENTER": "#34d399", "CONFIRMING": "#5eead4", "HOLD": "#2dd4bf", "EXIT": "#f87171", "BEARISH": "#f87171"}
    sig_hex = next((v for k, v in sig_colors.items() if k in sig), "#6b7280")
    return short_sig, sig_hex


def render_screener_table(results, filter_signal="All"):
    """Render the screener — button per ticker + HTML info row (responsive)."""
    filtered = results
    if filter_signal != "All":
        filtered = [r for r in results if filter_signal.upper() in (r.get("signal") or "").upper()]

    if not filtered:
        st.info("No tickers match the current filter.")
        return None

    selected_symbol = None

    errored = [r for r in filtered if r.get("error") and r.get("price") is None]
    filtered = [r for r in filtered if not (r.get("error") and r.get("price") is None)]

    for r in filtered:
        # Button for drill-down
        if st.button(r["symbol"], key=f"btn_{r['symbol']}", use_container_width=True):
            selected_symbol = r["symbol"]

        # Build HTML info row
        price_str = f"${r['price']:,.2f}" if r.get("price") else "--"
        chg = r.get("change_1d")
        chg_hex = "#34d399" if chg and chg >= 0 else "#f87171" if chg else "#6b7280"
        chg_str = f'<span style="color:{chg_hex}">{chg:+.1f}%</span>' if chg is not None else ""

        rid = r.get("regime_id")
        regime_html = regime_badge_html(rid, r.get("regime_label", "")) if rid is not None else ""

        short_sig, sig_hex = _screener_signal_info(r)
        flash_style = "animation:pulse 2s ease-in-out infinite;" if "ENTER" in r.get("signal", "") or "EXIT" in r.get("signal", "") else ""

        cmet = r.get("confirmations_met", 0)
        conf_total = r.get("confirmations_total", 12)
        ct_ratio = cmet / max(conf_total, 1)
        ct_hex = "#34d399" if ct_ratio >= 0.6 else ("#5eead4" if ct_ratio >= 0.4 else "#f87171")

        rsi = r.get("rsi")
        rsi_html = ""
        if rsi is not None:
            rsi_hex = "#f87171" if rsi > 70 else ("#34d399" if rsi < 30 else "#9ca3af")
            rsi_html = f'<span class="sr-rsi" style="color:{rsi_hex}">RSI {rsi:.0f}</span>'

        adx = r.get("adx")
        adx_html = ""
        if adx is not None:
            adx_hex = "#34d399" if adx > 25 else "#6b7280"
            adx_html = f'<span class="sr-adx" style="color:{adx_hex}">ADX {adx:.0f}</span>'

        st.markdown(
            f'<div class="screener-row">'
            f'<span class="sr-price">{price_str}</span>'
            f'{chg_str}'
            f'{regime_html}'
            f'<span class="sr-sig" style="color:{sig_hex};{flash_style}">{short_sig}</span>'
            f'<span class="sr-conf" style="color:{ct_hex}">{cmet}/{conf_total}</span>'
            f'{rsi_html}'
            f'{adx_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if errored:
        with st.expander(f"{len(errored)} tickers failed to scan", expanded=False):
            for r in errored:
                st.caption(f"{r['symbol']}: {r.get('error', 'unknown')[:60]}")

    return selected_symbol


def plot_price_with_regimes(df, title=""):
    fig = go.Figure()
    for regime_id in sorted(df["regime_id"].unique()):
        mask = df["regime_id"] == regime_id
        subset = df[mask]
        label = REGIME_LABELS[regime_id] if regime_id < len(REGIME_LABELS) else f"State {regime_id}"
        color = REGIME_COLORS.get(regime_id, "#666")
        fig.add_trace(go.Scatter(
            x=subset.index, y=subset["Close"],
            mode="markers", marker=dict(size=3, color=color),
            name=label,
            hovertemplate=f"<b>{label}</b><br>Price: %{{y:,.2f}}<br>%{{x}}<extra></extra>",
        ))
    fig.update_layout(
        title=title, template="plotly_dark",
        paper_bgcolor="#101114", plot_bgcolor="#101114",
        height=260,
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937", title="Price"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def plot_equity_curve(equity_curve, df):
    bh_start = df["Close"].iloc[0]
    bh_equity = (df["Close"] / bh_start) * equity_curve.iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        mode="lines", name="HMM Strategy",
        line=dict(color="#2dd4bf", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=bh_equity.values,
        mode="lines", name="Buy & Hold",
        line=dict(color="#666", width=1, dash="dash"),
    ))
    fig.update_layout(
        title="Equity Curve - Strategy vs Buy & Hold",
        template="plotly_dark",
        paper_bgcolor="#101114", plot_bgcolor="#101114",
        height=240,
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937", title="Equity ($)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def plot_regime_heatmap(results):
    """Heatmap showing regime state across all scanned tickers. Bearish left → Bullish right."""
    symbols = []
    regime_ids = []
    for r in results:
        if r.get("regime_id") is not None:
            symbols.append(r["symbol"])
            regime_ids.append(r["regime_id"])

    if not symbols:
        return None

    # Inverted order: bearish (6) on left → bullish (0) on right
    inv_order = [6, 5, 4, 3, 2, 1, 0]
    inv_labels = [REGIME_LABELS[i] for i in inv_order]

    matrix = np.zeros((len(symbols), 7))
    for i, rid in enumerate(regime_ids):
        if rid < 7:
            col_idx = inv_order.index(rid)
            matrix[i, col_idx] = 1.0

    fig = go.Figure(go.Heatmap(
        z=matrix, x=inv_labels, y=symbols,
        colorscale=[[0, "#101114"], [0.5, "#1a1f2e"], [1, "#2dd4bf"]],
        showscale=False,
        hovertemplate="<b>%{y}</b><br>%{x}<extra></extra>",
    ))

    for i, (sym, rid) in enumerate(zip(symbols, regime_ids)):
        if rid < 7:
            col_idx = inv_order.index(rid)
            fig.add_trace(go.Scatter(
                x=[inv_labels[col_idx]], y=[sym], mode="markers",
                marker=dict(size=14, color=REGIME_COLORS.get(rid, "#666"), symbol="square"),
                showlegend=False,
                hovertemplate=f"<b>{sym}</b><br>{REGIME_LABELS[rid]}<extra></extra>",
            ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#101114", plot_bgcolor="#101114",
        height=min(400, max(200, len(symbols) * 22 + 60)),
        margin=dict(l=60, r=10, t=30, b=20),
        xaxis=dict(side="top", tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


def plot_signal_distribution(results):
    """Pie chart of signal types across the watchlist."""
    signals = [r.get("signal", "UNKNOWN") for r in results if r.get("signal")]
    if not signals:
        return None

    signal_counts = pd.Series(signals).value_counts()

    color_map = {
        "LONG -- ENTER": "#2dd4bf",
        "LONG -- HOLD": "#22c55e",
        "LONG -- CONFIRMING": "#88cc44",
        "EXIT -- REGIME FLIP": "#ff4444",
        "CASH — NEUTRAL": "#ffaa00",
        "CASH — BEARISH": "#ff6666",
    }
    colors = [color_map.get(s, "#666") for s in signal_counts.index]

    fig = go.Figure(go.Pie(
        labels=signal_counts.index,
        values=signal_counts.values,
        marker=dict(colors=colors),
        textinfo="label+value",
        hole=0.4,
    ))
    fig.update_layout(
        title="Signal Distribution",
        template="plotly_dark",
        paper_bgcolor="#101114",
        height=240,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


def render_drill_down(result):
    """Full analysis drill-down for a single ticker."""
    sym = result["symbol"]
    regime_df = result.get("_regime_df")
    detector = result.get("_detector")

    if regime_df is None or detector is None:
        st.warning(f"No detailed data available for {sym}. Re-run the scan.")
        return

    st.markdown(f"## {sym}")

    # Signal banner
    sig = result["signal"]
    css = signal_css_class(sig)
    icon = signal_icon(sig)
    st.markdown(f"""
    <div class="signal-banner {css}">
        {icon} &nbsp; {sig} &nbsp; {icon}
        <div style="font-size: 0.85rem; font-weight: 400; margin-top: 4px; opacity: 0.85;">
            {result.get('action', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top metrics
    css_r = "bull" if result["regime_id"] <= 1 else ("bear" if result["regime_id"] >= 5 else "neutral")
    chg = result.get("change_1d")
    chg_val = f"{chg:+.2f}%" if chg is not None else "--"
    chg_cls = ("bull" if chg >= 0 else "bear") if chg is not None else ""
    render_metrics_grid([
        ("Price", f"${result['price']:,.2f}", ""),
        ("Regime", result["regime_label"], css_r),
        ("Confidence", f"{result['regime_confidence']:.0%}", ""),
        ("Confirmations", f"{result['confirmations_met']}/8", ""),
        ("Streak", f"{result.get('regime_streak', '?')} bars", ""),
        ("1D Change", chg_val, chg_cls),
    ])

    # Confirmation breakdown
    with st.expander("Confirmation Breakdown", expanded=False):
        conf_detail = result.get("confirmation_detail", {})
        conf_cols = st.columns(4)
        for idx, (name, passed) in enumerate(conf_detail.items()):
            with conf_cols[idx % 4]:
                icon_c = "+" if passed else "-"
                st.markdown(f"{icon_c} **{name}**")

    # TradingView Chart
    tv_symbol = resolve_ticker(sym).replace("-", "")
    tv_html = f'''
    <div style="overflow:hidden;">
    <iframe
        src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval=D&hidesidetoolbar=1&symboledit=0&saveimage=0&toolbarbg=101114&studies=MAExp%407%7C10%7Cclose%7C0%7C0%7C0%7C%232dd4bf&studies=MAExp%407%7C20%7Cclose%7C0%7C0%7C0%7C%233b82f6&studies=MAExp%407%7C50%7Cclose%7C0%7C0%7C0%7C%23a855f7&theme=dark&style=1&timezone=America%2FChicago&withdateranges=1&hideideas=1&width=100%25&height=220"
        style="width:100%;height:420px;border:none;"
        allowfullscreen>
    </iframe>
    </div>
    '''
    st.components.v1.html(tv_html, height=225)

    # Quick Trade (Tradier)
    if tradier_configured():
        with st.expander("Place Trade", expanded=False):
            _t1, _t2 = st.columns(2)
            trade_type = _t1.selectbox("Type", ["Buy Calls", "Buy Shares"], key=f"tt_{sym}")
            trade_order = _t2.selectbox("Order", ["Market", "Limit"], key=f"to_{sym}")
            _t3, _t4 = st.columns(2)
            trade_qty = _t3.number_input("Qty", value=1, min_value=1, key=f"tq_{sym}")
            trade_limit = _t4.number_input("Limit $", value=0.0, step=0.05, key=f"tl_{sym}",
                                           disabled=trade_order != "Limit")

            if trade_type == "Buy Calls":
                # Show top option pick if available
                options_recs = st.session_state.get("options_recs", [])
                sym_rec = next((r for r in options_recs if r.get("symbol") == sym), None)
                picks = sym_rec.get("recommendations", []) if sym_rec else []
                if picks:
                    opt_labels = [f"{p['contractSymbol']} | ${p['strike']} | {p['dte']}d | ${p['mid']}" for p in picks]
                    selected_opt = st.selectbox("Contract", opt_labels, key=f"oc_{sym}")
                    opt_idx = opt_labels.index(selected_opt)
                    selected_contract = picks[opt_idx]["contractSymbol"]
                else:
                    selected_contract = st.text_input("Option Symbol (OCC)", key=f"os_{sym}",
                                                       placeholder="SPY260417C00650000")

                if st.button("Preview Order", key=f"prev_{sym}", type="secondary"):
                    order_type = "limit" if trade_order == "Limit" else "market"
                    preview = place_option_order(
                        symbol=sym, option_symbol=selected_contract,
                        side="buy_to_open", quantity=int(trade_qty),
                        order_type=order_type,
                        limit_price=trade_limit if order_type == "limit" else None,
                        preview=True,
                    )
                    if "error" in preview:
                        st.error(preview["error"])
                    else:
                        st.json(preview)
                        if st.button("CONFIRM & SEND", key=f"send_{sym}", type="primary"):
                            live = place_option_order(
                                symbol=sym, option_symbol=selected_contract,
                                side="buy_to_open", quantity=int(trade_qty),
                                order_type=order_type,
                                limit_price=trade_limit if order_type == "limit" else None,
                                preview=False,
                            )
                            if "error" in live:
                                st.error(live["error"])
                            else:
                                st.success(f"Order placed: {live}")
            else:
                # Buy shares
                if st.button("Preview Order", key=f"prev_eq_{sym}", type="secondary"):
                    order_type = "limit" if trade_order == "Limit" else "market"
                    preview = place_equity_order(
                        symbol=sym, side="buy", quantity=int(trade_qty),
                        order_type=order_type,
                        limit_price=trade_limit if order_type == "limit" else None,
                        preview=True,
                    )
                    if "error" in preview:
                        st.error(preview["error"])
                    else:
                        st.json(preview)

    # Charts — compact, side-by-side
    tab1, tab2, tab3 = st.tabs(["Chart", "Backtest", "Options"])

    # Run backtest (used by Backtest tab)
    _strat = st.session_state.get("strategy", "v2")
    _confs = st.session_state.get("min_confs", 6)
    _cool = st.session_state.get("cooldown", 3)
    _confirm = st.session_state.get("regime_confirm", 2)
    _cap = st.session_state.get("initial_capital", 100_000)
    try:
        if _strat == "v2":
            bt = run_backtest_v2(regime_df, min_confirmations=_confs, cooldown_bars=_cool, regime_confirm_bars=_confirm, initial_capital=_cap)
        else:
            bt = run_backtest(regime_df, min_confirmations=_confs, cooldown_bars=_cool, regime_confirm_bars=_confirm, initial_capital=_cap)
    except Exception:
        bt = None

    with tab1:
        pass

    with tab2:
        try:
            metrics = bt["metrics"]
            render_metrics_grid([
                ("Return", f"{metrics['total_return_pct']:.1f}%", "bull" if metrics["total_return_pct"] > 0 else "bear"),
                ("Alpha", f"{metrics['alpha_vs_buyhold']:.1f}%", "bull" if metrics["alpha_vs_buyhold"] > 0 else "bear"),
                ("Win Rate", f"{metrics['win_rate']:.0f}%", "bull" if metrics["win_rate"] > 50 else "bear"),
                ("Sharpe", f"{metrics['sharpe_ratio']:.2f}", ""),
                ("Max DD", f"{metrics['max_drawdown_pct']:.1f}%", "bear"),
                ("PF", f"{metrics['profit_factor']:.2f}", ""),
            ])

            if bt.get("trades"):
                trade_df = pd.DataFrame(bt["trades"])
                display_cols = [c for c in ["entry_date","exit_date","pnl_pct","exit_reason","roll_count"] if c in trade_df.columns]
                st.dataframe(trade_df[display_cols], use_container_width=True, hide_index=True, height=180)
        except Exception:
            pass

    with tab3:
        # Options for this specific ticker
        if result.get("regime_id") is not None and result["regime_id"] <= 2:
            with st.spinner(f"Finding best options for {sym}..."):
                opts = get_options_recommendations(
                    symbol=sym,
                    current_price=result["price"],
                    regime_id=result["regime_id"],
                    regime_label=result["regime_label"],
                    confirmations=result.get("confirmations_met", 0),
                    signal=result.get("signal", ""),
                    min_dte=st.session_state.get("min_dte", 14),
                    max_dte=st.session_state.get("max_dte", 60),
                    top_n=st.session_state.get("top_n_options", 5),
                )
            picks = opts.get("recommendations", [])
            if picks:
                st.markdown(f"#### Top {len(picks)} Call Options for {sym}")
                pick_rows = []
                for p in picks:
                    spread = p["ask"] - p["bid"] if p["ask"] and p["bid"] else 0
                    spread_pct = spread / p["mid"] * 100 if p["mid"] > 0 else 0
                    pick_rows.append({
                        "Contract": p["contractSymbol"],
                        "Exp": p["expiration"],
                        "DTE": p["dte"],
                        "Strike": p["strike"],
                        "Bid": p["bid"],
                        "Ask": p["ask"],
                        "Mid": p["mid"],
                        "Spread%": round(spread_pct, 1),
                        "Vol": p["volume"],
                        "OI": p["openInterest"],
                        "IV%": p["iv_pct"],
                        "Delta": p["delta"],
                        "Theta": p["theta"],
                        "Score": p["score"],
                    })
                st.dataframe(pd.DataFrame(pick_rows), use_container_width=True, hide_index=True)
                best = picks[0]
                st.success(
                    f"**Recommended:** {best['contractSymbol']} — "
                    f"${best['strike']:.2f} strike, {best['dte']} DTE, "
                    f"delta {best['delta']:.2f}, mid ${best['mid']:.2f}"
                )
            elif opts.get("error"):
                st.warning(opts["error"])
            else:
                st.info(f"No suitable options found for {sym}")
        else:
            pass



# ════════════════════════════════════════════════════════
#  LOAD SAVED SETTINGS
# ════════════════════════════════════════════════════════
_saved = load_settings()

#  SESSION STATE
# ════════════════════════════════════════════════════════
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None
if "options_recs" not in st.session_state:
    st.session_state.options_recs = []
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False


# ════════════════════════════════════════════════════════
#  HERO + SETTINGS TOGGLE
# ════════════════════════════════════════════════════════
results = st.session_state.scan_results

if not results:
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:12vh 1rem 2rem;">
        <div style="font-family:'Inter',sans-serif; font-weight:600; font-size:clamp(2.5rem, 10vw, 5rem); letter-spacing:0.4rem; color:#f3f4f6; line-height:1;">
            RRJCAR
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-weight:400; font-size:clamp(0.5rem, 2vw, 0.7rem); letter-spacing:0.4rem; color:#2dd4bf; text-transform:uppercase; margin-top:0.6rem;">
            regime scanner
        </div>
    </div>
    """, unsafe_allow_html=True)

# Toggle button — always visible
_toggle_label = "Close Settings" if st.session_state.show_settings else "Settings"
if st.button(_toggle_label, use_container_width=True):
    st.session_state.show_settings = not st.session_state.show_settings
    st.rerun()

# ════════════════════════════════════════════════════════
#  SETTINGS (hidden by default, toggled open)
# ════════════════════════════════════════════════════════
if st.session_state.show_settings:
    watchlist_keys = list(WATCHLISTS.keys())
    saved_wl = _saved.get("watchlist", "All Stocks (no ETFs)")
    default_idx = watchlist_keys.index(saved_wl) if saved_wl in watchlist_keys else 0
    wl1, wl2 = st.columns([3, 1])
    watchlist_name = wl1.selectbox("Watchlist", watchlist_keys, index=default_idx, label_visibility="collapsed")
    strategy = "v2" if wl2.selectbox("V", ["V2", "V1"], label_visibility="collapsed") == "V2" else "v1"
    custom_tickers = st.text_input("Add", value=_saved.get("custom_tickers", ""), placeholder="Add tickers...", label_visibility="collapsed")
else:
    watchlist_name = _saved.get("watchlist", "All Stocks (no ETFs)")
    strategy = _saved.get("strategy", "v2")
    custom_tickers = _saved.get("custom_tickers", "")

tickers = list(WATCHLISTS.get(watchlist_name, []))
if custom_tickers.strip():
    extras = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers + extras))

interval = "1d"

# SCAN button — only in settings panel
if st.session_state.show_settings:
    scan_btn = st.button("SCAN", type="primary", use_container_width=True)
else:
    scan_btn = False

if st.session_state.show_settings:
    # Advanced settings
    with st.expander(f"{len(tickers):,} tickers  |  Advanced Settings"):
        s1, s2 = st.columns(2)
        n_regimes = s1.number_input(f"Regimes ({_saved.get('n_regimes',7)}/10)", value=_saved.get("n_regimes",7), min_value=3, max_value=10)
        max_workers = s2.number_input(f"Speed ({_saved.get('max_workers',6)}/8)", value=_saved.get("max_workers",6), min_value=1, max_value=8)
        min_confs = s1.number_input(f"Confs ({_saved.get('min_confs',6)}/12)", value=_saved.get("min_confs",6), min_value=3, max_value=12)
        regime_confirm = s2.number_input(f"Confirm ({_saved.get('regime_confirm',2)}/10)", value=_saved.get("regime_confirm",2), min_value=1, max_value=10)
        cooldown = s1.number_input(f"Cooldown ({_saved.get('cooldown',3)}/20)", value=_saved.get("cooldown",3), min_value=1, max_value=20)
        initial_capital = s2.number_input("Capital $", value=_saved.get("initial_capital", 100000), min_value=1000, step=10000)
        risk_pct = s1.number_input(f"Risk ({_saved.get('risk_pct',10)}%/25%)", value=_saved.get("risk_pct",10), min_value=1, max_value=25)
        st.session_state.risk_pct = risk_pct
        options_enabled = st.checkbox("Options Picker", value=_saved.get("options_enabled", True))
        if options_enabled:
            d1, d2, d3 = st.columns(3)
            min_dte = d1.number_input(f"Min DTE", value=_saved.get("min_dte",21), min_value=7, max_value=30)
            max_dte = d2.number_input(f"Max DTE", value=_saved.get("max_dte",45), min_value=30, max_value=180)
            top_n_options = d3.number_input(f"Picks", value=_saved.get("top_n_options",3), min_value=1, max_value=10)
        else:
            min_dte, max_dte, top_n_options = 21, 45, 3
        auto_refresh = st.checkbox("Auto-Refresh", value=_saved.get("auto_refresh", False))
        refresh_minutes = 5
        if auto_refresh:
            refresh_minutes = st.slider("Min", 1, 30, _saved.get("refresh_minutes", 5))

    # Alerts & Scheduling
    with st.expander("Alerts & Schedule"):
        alerts_enabled = st.checkbox("Enable Alerts", value=_saved.get("alerts_enabled", False))
        if alerts_enabled:
            st.markdown("**Trigger On**")
            tc1, tc2, tc3 = st.columns(3)
            alert_on_regime_change = tc1.checkbox("Any Change", value=_saved.get("alert_on_regime_change", True))
            alert_on_bull_entry = tc2.checkbox("Bull Entry", value=_saved.get("alert_on_bull_entry", True))
            alert_on_bear_entry = tc3.checkbox("Bear Entry", value=_saved.get("alert_on_bear_entry", False))
            alert_min_confirmations = st.slider("Min Confirmations", 1, 10, _saved.get("alert_min_confirmations", 6))
            st.markdown("**Email (SMTP)**")
            alert_email = st.text_input("To Address", value=_saved.get("alert_email", ""))
            ec1, ec2 = st.columns(2)
            alert_smtp_server = ec1.text_input("SMTP Server", value=_saved.get("alert_smtp_server", "smtp.gmail.com"))
            alert_smtp_port = ec2.number_input("Port", value=_saved.get("alert_smtp_port", 587), min_value=1, max_value=65535)
            alert_smtp_user = st.text_input("SMTP User", value=_saved.get("alert_smtp_user", ""))
            alert_smtp_password = st.text_input("SMTP Password", value=_saved.get("alert_smtp_password", ""), type="password")
            st.markdown("**Telegram**")
            alert_telegram_enabled = st.checkbox("Enable Telegram", value=_saved.get("alert_telegram_enabled", False))
            if alert_telegram_enabled:
                alert_telegram_bot_token = st.text_input("Bot Token", value=_saved.get("alert_telegram_bot_token", ""), type="password")
                alert_telegram_chat_id = st.text_input("Chat ID", value=_saved.get("alert_telegram_chat_id", ""))
            else:
                alert_telegram_bot_token = _saved.get("alert_telegram_bot_token", "")
                alert_telegram_chat_id = _saved.get("alert_telegram_chat_id", "")
            st.markdown("**Scheduled Scans**")
            scheduled_scans_enabled = st.checkbox("Enable Scheduled Scans", value=_saved.get("scheduled_scans_enabled", False))
            if scheduled_scans_enabled:
                scheduled_scan_times = st.text_input("Scan Times (24h)", value=_saved.get("scheduled_scan_times", "09:30,12:00,15:30"))
                scheduled_scan_timezone = st.selectbox("Timezone",
                    ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles", "UTC"],
                    index=["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles", "UTC"].index(_saved.get("scheduled_scan_timezone", "America/Chicago")))
            else:
                scheduled_scan_times = _saved.get("scheduled_scan_times", "09:30,12:00,15:30")
                scheduled_scan_timezone = _saved.get("scheduled_scan_timezone", "America/Chicago")
            if st.button("Test Alert", use_container_width=True):
                from alerts import send_email_alert, send_telegram_alert
                test_change = [{"symbol": "TEST", "prev_regime": 3, "new_regime": 0, "regime_label": "Bull Run", "price": 100.00, "confirmations": 6}]
                test_settings = {**_saved, "alert_email": alert_email, "alert_smtp_server": alert_smtp_server,
                    "alert_smtp_port": alert_smtp_port, "alert_smtp_user": alert_smtp_user, "alert_smtp_password": alert_smtp_password,
                    "alert_telegram_enabled": alert_telegram_enabled, "alert_telegram_bot_token": alert_telegram_bot_token, "alert_telegram_chat_id": alert_telegram_chat_id}
                if alert_email: st.info(send_email_alert(test_change, test_settings))
                if alert_telegram_enabled: st.info(send_telegram_alert(test_change, test_settings))
                if not alert_email and not alert_telegram_enabled: st.warning("No channels configured.")
        else:
            alerts_enabled = False
            alert_email = _saved.get("alert_email", "")
            alert_smtp_server = _saved.get("alert_smtp_server", "smtp.gmail.com")
            alert_smtp_port = _saved.get("alert_smtp_port", 587)
            alert_smtp_user = _saved.get("alert_smtp_user", "")
            alert_smtp_password = _saved.get("alert_smtp_password", "")
            alert_telegram_enabled = _saved.get("alert_telegram_enabled", False)
            alert_telegram_bot_token = _saved.get("alert_telegram_bot_token", "")
            alert_telegram_chat_id = _saved.get("alert_telegram_chat_id", "")
            alert_on_regime_change = _saved.get("alert_on_regime_change", True)
            alert_on_bull_entry = _saved.get("alert_on_bull_entry", True)
            alert_on_bear_entry = _saved.get("alert_on_bear_entry", False)
            alert_min_confirmations = _saved.get("alert_min_confirmations", 6)
            scheduled_scans_enabled = _saved.get("scheduled_scans_enabled", False)
            scheduled_scan_times = _saved.get("scheduled_scan_times", "09:30,12:00,15:30")
            scheduled_scan_timezone = _saved.get("scheduled_scan_timezone", "America/Chicago")

    # Tradier
    if not tradier_configured():
        with st.expander("Connect Tradier"):
            t_token = st.text_input("Token", type="password")
            t_acct = st.text_input("Account ID")
            t_sandbox = st.checkbox("Sandbox", value=True)
            if st.button("Connect", use_container_width=True):
                if t_token and t_acct:
                    save_tradier_config(t_token, t_acct, t_sandbox)
                    st.rerun()

    # Save
    if st.button("Save Settings", use_container_width=True):
        save_settings({
            "watchlist": watchlist_name, "custom_tickers": custom_tickers,
            "strategy": strategy, "min_confs": min_confs,
            "regime_confirm": regime_confirm, "cooldown": cooldown,
            "initial_capital": initial_capital, "n_regimes": n_regimes,
            "max_workers": max_workers, "options_enabled": options_enabled,
            "min_dte": min_dte, "max_dte": max_dte, "top_n_options": top_n_options,
            "auto_refresh": auto_refresh, "refresh_minutes": refresh_minutes,
            "risk_pct": risk_pct, "alerts_enabled": alerts_enabled,
            "alert_email": alert_email, "alert_smtp_server": alert_smtp_server,
            "alert_smtp_port": alert_smtp_port, "alert_smtp_user": alert_smtp_user,
            "alert_smtp_password": alert_smtp_password,
            "alert_telegram_enabled": alert_telegram_enabled,
            "alert_telegram_bot_token": alert_telegram_bot_token,
            "alert_telegram_chat_id": alert_telegram_chat_id,
            "alert_on_regime_change": alert_on_regime_change,
            "alert_on_bull_entry": alert_on_bull_entry,
            "alert_on_bear_entry": alert_on_bear_entry,
            "alert_min_confirmations": alert_min_confirmations,
            "scheduled_scans_enabled": scheduled_scans_enabled,
            "scheduled_scan_times": scheduled_scan_times,
            "scheduled_scan_timezone": scheduled_scan_timezone,
        })
        st.toast("Saved")

else:
    # Settings closed — use saved defaults for all variables
    n_regimes = _saved.get("n_regimes", 7)
    max_workers = _saved.get("max_workers", 6)
    min_confs = _saved.get("min_confs", 6)
    regime_confirm = _saved.get("regime_confirm", 2)
    cooldown = _saved.get("cooldown", 3)
    initial_capital = _saved.get("initial_capital", 100000)
    risk_pct = _saved.get("risk_pct", 10)
    options_enabled = _saved.get("options_enabled", True)
    min_dte = _saved.get("min_dte", 21)
    max_dte = _saved.get("max_dte", 45)
    top_n_options = _saved.get("top_n_options", 3)
    auto_refresh = _saved.get("auto_refresh", False)
    refresh_minutes = _saved.get("refresh_minutes", 5)
    alerts_enabled = _saved.get("alerts_enabled", False)
    alert_email = _saved.get("alert_email", "")
    alert_smtp_server = _saved.get("alert_smtp_server", "smtp.gmail.com")
    alert_smtp_port = _saved.get("alert_smtp_port", 587)
    alert_smtp_user = _saved.get("alert_smtp_user", "")
    alert_smtp_password = _saved.get("alert_smtp_password", "")
    alert_telegram_enabled = _saved.get("alert_telegram_enabled", False)
    alert_telegram_bot_token = _saved.get("alert_telegram_bot_token", "")
    alert_telegram_chat_id = _saved.get("alert_telegram_chat_id", "")
    alert_on_regime_change = _saved.get("alert_on_regime_change", True)
    alert_on_bull_entry = _saved.get("alert_on_bull_entry", True)
    alert_on_bear_entry = _saved.get("alert_on_bear_entry", False)
    alert_min_confirmations = _saved.get("alert_min_confirmations", 6)
    scheduled_scans_enabled = _saved.get("scheduled_scans_enabled", False)
    scheduled_scan_times = _saved.get("scheduled_scan_times", "09:30,12:00,15:30")
    scheduled_scan_timezone = _saved.get("scheduled_scan_timezone", "America/Chicago")

# Store in session
st.session_state.strategy = strategy
st.session_state.min_confs = min_confs
st.session_state.regime_confirm = regime_confirm
st.session_state.cooldown = cooldown
st.session_state.initial_capital = initial_capital
st.session_state.options_enabled = options_enabled
st.session_state.min_dte = min_dte
st.session_state.max_dte = max_dte
st.session_state.top_n_options = top_n_options
st.session_state.risk_pct = risk_pct

if st.session_state.last_scan_time:
    elapsed = (datetime.now() - st.session_state.last_scan_time).seconds
    st.caption(f"Last scan: {elapsed}s ago")


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════
# Header only shows in landing state (below)

# ── Sell Signal Check (check open positions against current scan) ──
_open_pos = get_open_positions()
if _open_pos and results:
    sell_alerts = []
    for pos in _open_pos:
        sym = pos.get("symbol", "")
        scan_match = next((r for r in results if r.get("symbol") == sym), None)
        if scan_match and ("EXIT" in scan_match.get("signal", "") or "BEARISH" in scan_match.get("signal", "")):
            sell_alerts.append(f"{sym}: {scan_match['signal']}")
    if sell_alerts:
        st.markdown(
            f'<div class="signal-banner signal-exit">SELL SIGNALS: {" | ".join(sell_alerts)}</div>',
            unsafe_allow_html=True,
        )

# ── Auto-Scan at 1:00 PM CT ──
try:
    import pytz
    ct_tz = pytz.timezone("America/Chicago")
    ct_now = datetime.now(ct_tz)
    is_weekday = ct_now.weekday() < 5
    is_scan_time = ct_now.hour == 13 and ct_now.minute < 5
    last_scan = st.session_state.get("last_scan_time")
    already_scanned_today = last_scan and last_scan.date() == ct_now.date() if last_scan else False

    if is_weekday and is_scan_time and not already_scanned_today and not scan_btn:
        scan_btn = True  # trigger auto-scan
        st.toast("1:00 PM CT auto-scan triggered")
except Exception:
    pass

# ── Run Scan ──
if scan_btn:
    progress = st.progress(0, text=f"Scanning {len(tickers):,} tickers...")

    try:
        with st.spinner(f"Scanning {len(tickers):,} tickers..."):
            results = scan_watchlist(
                symbols=tickers,
                interval=interval,
                n_regimes=n_regimes,
                min_confirmations=min_confs,
                regime_confirm_bars=regime_confirm,
                max_workers=max_workers,
                strategy=strategy,
            )
    except Exception as e:
        st.error(f"Scan failed: {str(e)[:100]}")
        results = []

    progress.progress(80, text="Finding options...")

    # Options scan for all scanned tickers
    options_recs = []
    if options_enabled:
        with_price = [r for r in results if r.get("price") is not None]
        if with_price:
            with st.spinner(f"Finding options for {len(with_price)} tickers..."):
                options_recs = scan_options_for_watchlist(
                    with_price,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    top_n=top_n_options,
                )

    progress.progress(100, text="Done!")
    st.session_state.scan_results = results
    st.session_state.options_recs = options_recs
    st.session_state.last_scan_time = datetime.now()
    st.session_state.selected_ticker = None

    # Process alerts on regime changes
    if alerts_enabled and results:
        alert_statuses = process_alerts(results)
        for s in alert_statuses:
            st.toast(s)

    time.sleep(0.3)
    st.rerun()


# ── Display Results ──
results = st.session_state.scan_results

if results:
    # Summary bar
    total = len(results)
    n_bull = sum(1 for r in results if r.get("regime_id") is not None and r["regime_id"] <= 2)
    n_neutral = sum(1 for r in results if r.get("regime_id") is not None and 3 <= r["regime_id"] <= 4)
    n_bear = sum(1 for r in results if r.get("regime_id") is not None and r["regime_id"] >= 5)
    n_enter = sum(1 for r in results if "ENTER" in (r.get("signal") or ""))
    n_exit = sum(1 for r in results if "EXIT" in (r.get("signal") or ""))
    n_errors = sum(1 for r in results if r.get("error") and r.get("price") is None)

    # Summary metrics
    render_metrics_grid([
        ("Scanned", str(total), ""),
        ("Bullish", str(n_bull), "bull"),
        ("Neutral", str(n_neutral), "neutral"),
        ("Bearish", str(n_bear), "bear"),
        ("Entry", str(n_enter), "bull" if n_enter > 0 else "cash"),
        ("Exit", str(n_exit), "bear" if n_exit > 0 else "cash"),
    ])


    # Tabs: Screener | Options Picks | Regime Chart | Signal Overview | Drill-Down
    main_tabs = st.tabs(["Screener", "Options", "Holdings", "Performance", "Chart", "Drill-Down"])

    with main_tabs[0]:
        # Filter bar
        _fc1, _fc2 = st.columns(2)
        with _fc1:
            filter_signal = st.selectbox(
                "Filter",
                ["All", "ENTER", "CONFIRMING", "HOLD", "EXIT", "CASH", "BEARISH"],
            )
        with _fc2:
            sort_by = st.selectbox(
                "Sort",
                ["Top Buy First", "Confirmations", "RSI", "Regime (Bullish)", "1D Change"],
            )

        # Apply sorting
        display_results = list(results)
        if sort_by == "Top Buy First":
            # Composite score: signal priority + confirmations + regime + confidence
            def buy_score(r):
                sig = r.get("signal", "")
                regime = r.get("regime_id")
                if regime is None:
                    regime = 99
                confs = r.get("confirmations_met", 0)
                conf_total = r.get("confirmations_total", 12)
                confidence = r.get("regime_confidence", 0) or 0

                # Signal weight (lower = better)
                if "ENTER" in sig:
                    sig_w = 0
                elif "CONFIRMING" in sig:
                    sig_w = 100
                elif "HOLD" in sig:
                    sig_w = 200
                else:
                    sig_w = 500

                # Regime weight (bullish regimes first)
                regime_w = regime * 50

                # Confirmation bonus (higher = better, inverted)
                conf_w = -(confs / max(conf_total, 1)) * 100

                # Confidence bonus
                conf_bonus = -confidence * 30

                return sig_w + regime_w + conf_w + conf_bonus

            display_results.sort(key=buy_score)
        elif sort_by == "Confirmations":
            display_results.sort(key=lambda r: -(r.get("confirmations_met") or 0))
        elif sort_by == "RSI":
            display_results.sort(key=lambda r: r.get("rsi") or 50)
        elif sort_by == "Regime (Bullish)":
            display_results.sort(key=lambda r: r.get("regime_id") if r.get("regime_id") is not None else 99)
        elif sort_by == "1D Change":
            display_results.sort(key=lambda r: -(r.get("change_1d") or -999))

        # Ticker selector dropdown — pick any scanned ticker to see contracts
        available_syms = ["-- Select Ticker --"] + [r["symbol"] for r in display_results if r.get("price")]
        sel_idx = 0
        if st.session_state.get("selected_ticker") in available_syms:
            sel_idx = available_syms.index(st.session_state["selected_ticker"])

        picked = st.selectbox("Trade", available_syms, index=sel_idx, key="ticker_pick", label_visibility="collapsed")
        if picked != "-- Select Ticker --":
            st.session_state.selected_ticker = picked
        sel = st.session_state.get("selected_ticker")

        if sel and sel != "-- Select Ticker --":
            # ── CONTRACT VIEW ──
            sel_scan = next((r for r in results if r["symbol"] == sel), None)

            # Fetch options on-demand
            sel_opts = st.session_state.get("options_recs", [])
            sel_rec = next((r for r in sel_opts if r.get("symbol") == sel), None)
            picks = sel_rec.get("recommendations", []) if sel_rec else []

            if not picks:
                # Always fetch on-demand — use scan price or fetch live
                _price = sel_scan.get("price", 0) if sel_scan else 0
                _regime_id = sel_scan.get("regime_id", 3) if sel_scan else 3
                _regime_label = sel_scan.get("regime_label", "") if sel_scan else ""
                _confs = sel_scan.get("confirmations_met", 0) if sel_scan else 0
                _signal = sel_scan.get("signal", "") if sel_scan else ""

                if _price <= 0:
                    # Fetch price from yfinance directly
                    try:
                        import yfinance as yf
                        t = yf.Ticker(sel)
                        hist = t.history(period="2d")
                        if hist is not None and not hist.empty:
                            _price = float(hist["Close"].iloc[-1])
                    except Exception:
                        pass

                if _price > 0:
                    try:
                        with st.spinner(f"Loading {sel} options..."):
                            fresh = get_options_recommendations(
                                symbol=sel,
                                current_price=_price,
                                regime_id=_regime_id,
                                regime_label=_regime_label,
                                confirmations=_confs,
                                signal=_signal,
                                min_dte=int(st.session_state.get("min_dte", 21)),
                                max_dte=int(st.session_state.get("max_dte", 45)),
                                top_n=int(st.session_state.get("top_n_options", 5)),
                            )
                            picks = fresh.get("recommendations", [])
                    except Exception:
                        picks = []

            # Ticker info
            price_str = f"${sel_scan['price']:,.2f}" if sel_scan and sel_scan.get("price") else ""
            regime_str = sel_scan.get("regime_label", "") if sel_scan else ""
            sig_str = sel_scan.get("signal", "") if sel_scan else ""
            sig_hex = "#2dd4bf" if "ENTER" in sig_str or "HOLD" in sig_str else ("#f87171" if "EXIT" in sig_str else "#6b7280")
            st.markdown(
                f'<div style="padding:0.3rem 0">'
                f'<span style="font-size:1.1rem;font-weight:600;color:#f3f4f6">{sel}</span>'
                f'<span style="color:#6b7280;margin-left:0.8rem">{price_str}</span>'
                f'<span style="color:#2dd4bf;margin-left:0.8rem;font-size:0.75rem">{regime_str}</span>'
                f'<span style="color:{sig_hex};margin-left:0.8rem;font-size:0.75rem">{sig_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if picks:
                _risk_pct = st.session_state.get("risk_pct", 10) / 100

                for i, p in enumerate(picks):
                    atr_est = (sel_scan.get("price", 100) * 0.02) if sel_scan else 2
                    sizing = compute_position_size(
                        account_equity=initial_capital,
                        entry_price=sel_scan.get("price", 100) if sel_scan else 100,
                        atr=atr_est,
                        regime_confidence=sel_scan.get("regime_confidence", 0.5) if sel_scan else 0.5,
                        confirmations_met=sel_scan.get("confirmations_met", 6) if sel_scan else 6,
                        confirmations_total=sel_scan.get("confirmations_total", 12) if sel_scan else 12,
                        option_mid=p["mid"],
                        max_risk_pct=_risk_pct,
                    )

                    # Option card (responsive via CSS)
                    st.markdown(
                        f'<div class="opt-card">'
                        f'<span style="color:#e5e7eb;font-weight:600">${p["strike"]:.0f}</span>'
                        f'<span style="color:#6b7280">{p["dte"]}d</span>'
                        f'<span style="color:#9ca3af">d{p["delta"]:.2f}</span>'
                        f'<span style="color:#6b7280">{p["iv_pct"]:.0f}%</span>'
                        f'<span style="color:#2dd4bf;font-weight:600">${p["mid"]:.2f}</span>'
                        f'<span style="color:#e5e7eb">x{sizing["contracts"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    _buy_clicked = st.button("BUY", key=f"buy_{sel}_{i}", type="primary", use_container_width=True)
                    if _buy_clicked:
                        if tradier_configured():
                            with st.spinner(f"Buying {sizing['contracts']}x {p['contractSymbol']}..."):
                                result = execute_buy_calls(
                                    symbol=sel,
                                    option_symbol=p["contractSymbol"],
                                    quantity=sizing["contracts"],
                                    starting_bid=p.get("bid", p["mid"] - 0.10),
                                )
                            if result.get("success"):
                                log_entry(
                                    symbol=sel, contract=p["contractSymbol"],
                                    quantity=sizing["contracts"],
                                    entry_price=result["fill_price"],
                                    regime=sel_scan.get("regime_label", "") if sel_scan else "",
                                    signal=sel_scan.get("signal", "") if sel_scan else "",
                                    confidence_tier=sizing["confidence_tier"],
                                    risk_dollars=sizing["risk_dollars"],
                                )
                                st.success(f"Filled {sizing['contracts']}x @ ${result['fill_price']:.2f}")
                            else:
                                st.error(result.get("error", "Order failed"))
                        else:
                            st.warning("Connect Tradier in sidebar first")
            else:
                st.caption(f"No options available for {sel}")

        else:
            # ── SCREENER TABLE (default view) ──
            render_screener_table(display_results, filter_signal)

    with main_tabs[1]:
        # ── Options Picks ──
        options_recs = st.session_state.get("options_recs", [])
        if not options_recs:
            pass
        else:

            for rec in options_recs:
                sym = rec["symbol"]
                picks = rec.get("recommendations", [])
                err = rec.get("error")

                if err and not picks:
                    continue  # skip non-bullish or errored

                with st.expander(
                    f"**{sym}** — ${rec['price']:,.2f} | {rec['regime_label']} | {rec['signal']}",
                    expanded=bool(picks),
                ):
                    if err:
                        st.warning(err)
                    if not picks:
                                    continue

                    # Display top picks as a table
                    pick_rows = []
                    for p in picks:
                        spread = p["ask"] - p["bid"] if p["ask"] and p["bid"] else 0
                        spread_pct = spread / p["mid"] * 100 if p["mid"] > 0 else 0
                        pick_rows.append({
                            "Contract": p["contractSymbol"],
                            "Exp": p["expiration"],
                            "DTE": p["dte"],
                            "Strike": p["strike"],
                            "Bid": p["bid"],
                            "Ask": p["ask"],
                            "Mid": p["mid"],
                            "Spread%": round(spread_pct, 1),
                            "Vol": p["volume"],
                            "OI": p["openInterest"],
                            "IV%": p["iv_pct"],
                            "Delta": p["delta"],
                            "Gamma": p["gamma"],
                            "Theta": p["theta"],
                            "ITM": "Y" if p["inTheMoney"] else "N",
                            "Score": p["score"],
                        })

                    pick_df = pd.DataFrame(pick_rows)
                    st.dataframe(
                        pick_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                            "Bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
                            "Ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
                            "Mid": st.column_config.NumberColumn("Mid", format="$%.2f"),
                            "Delta": st.column_config.NumberColumn("Delta", format="%.3f"),
                            "Theta": st.column_config.NumberColumn("Theta", format="%.3f"),
                            "Score": st.column_config.NumberColumn("Score", format="%.1f"),
                        },
                    )

                    # Quick summary + position sizing
                    best = picks[0]
                    # Get scan data for position sizing
                    scan_r = next((r for r in results if r["symbol"] == sym), None)
                    if scan_r:
                        atr_val = scan_r.get("_regime_df")
                        atr_est = rec["price"] * 0.02  # fallback 2% ATR estimate
                        sizing = compute_position_size(
                            account_equity=st.session_state.get("initial_capital", 100000),
                            entry_price=rec["price"],
                            atr=atr_est,
                            regime_confidence=scan_r.get("regime_confidence", 0.5),
                            confirmations_met=scan_r.get("confirmations_met", 6),
                            confirmations_total=scan_r.get("confirmations_total", 12),
                            option_mid=best["mid"],
                        )
                        st.markdown(
                            f"**{best['contractSymbol']}** "
                            f"${best['strike']:.0f} | {best['dte']}d | "
                            f"delta {best['delta']:.2f} | ${best['mid']:.2f} | "
                            f"**{sizing['contracts']} contracts** ({sizing['confidence_tier']}, ${sizing['risk_dollars']:,.0f} risk)"
                        )
                    else:
                        st.markdown(
                            f"**{best['contractSymbol']}** "
                            f"${best['strike']:.0f} | {best['dte']}d | "
                            f"delta {best['delta']:.2f} | ${best['mid']:.2f}"
                        )


    with main_tabs[2]:
        # ── HOLDINGS — Open positions + sell/roll ──
        open_pos = get_open_positions()
        if open_pos:
            for pos in open_pos:
                sym = pos.get("symbol", "?")
                contract = pos.get("contract", "")
                qty = pos.get("quantity", 1)
                entry_px = pos.get("entry_price", 0)
                scan_match = next((r for r in results if r.get("symbol") == sym), None)
                current_signal = scan_match.get("signal", "") if scan_match else ""
                current_price = scan_match.get("price", 0) if scan_match else 0
                is_sell = "EXIT" in current_signal or "BEARISH" in current_signal

                # Current P&L estimate
                pnl_est = (current_price - entry_px) / entry_px * 100 if entry_px > 0 and current_price > 0 else 0

                # Row
                pnl_color = "bull" if pnl_est >= 0 else "bear"
                st.markdown(
                    f'<div class="screener-row">'
                    f'<span style="font-weight:600;color:#f3f4f6">{sym}</span>'
                    f'<span style="color:#9ca3af"><code>{contract or "shares"}</code> x{qty}</span>'
                    f'<span style="color:#e5e7eb">${entry_px:.2f}</span>'
                    f'<span class="{pnl_color}">{pnl_est:+.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                ac1, ac2, ac3 = st.columns(3)

                # Actions
                if is_sell:
                    if ac1.button("SELL", key=f"sell_{pos['id']}", type="primary"):
                        if tradier_configured() and contract:
                            quote = _get_option_quote_safe(sym, contract)
                            result = execute_sell_to_close(sym, contract, qty, quote.get("ask", 0))
                            if result.get("success"):
                                log_exit(pos["id"], result["fill_price"], "Sold via dashboard")
                                st.rerun()
                            else:
                                st.error(result.get("error", "Failed"))
                        else:
                            log_exit(pos["id"], current_price, current_signal)
                            st.rerun()
                else:
                    ac1.caption("HOLD")

                # Roll button — find recommended roll target
                if contract and scan_match:
                    roll_target = find_roll_target(
                        sym, current_price,
                        current_contract_bid=current_price * 0.01,  # estimate
                        same_expiry=contract[-15:-9] if len(contract) > 15 else None,
                    )
                    if roll_target:
                        credit_est = roll_target.get("credit", 0)
                        if ac2.button(f"ROLL +${credit_est:.2f}", key=f"roll_{pos['id']}"):
                            if tradier_configured():
                                result = exec_roll(sym, contract, roll_target["contractSymbol"], qty,
                                                  current_price * 0.01, roll_target["ask"])
                                if result.get("success"):
                                    from performance_tracker import log_roll
                                    log_roll(pos["id"], contract, roll_target["contractSymbol"],
                                            "roll_up", result.get("credit", 0))
                                    st.rerun()
                        ac3.caption(f"-> {roll_target['contractSymbol'][-10:]}")
        else:
            pass

    with main_tabs[3]:
        # ── PERFORMANCE ──
        perf = get_performance_summary()
        if perf["total_trades"] > 0:
            render_metrics_grid([
                ("Total P&L", f"${perf['total_pnl']:,.0f}", "bull" if perf["total_pnl"] > 0 else "bear"),
                ("Win Rate", f"{perf['win_rate']:.0f}%", "bull" if perf["win_rate"] > 50 else "bear"),
                ("Trades", str(perf["total_trades"]), ""),
                ("Open", str(perf["open_trades"]), ""),
                ("Rolls", str(perf["total_rolls"]), ""),
                ("Roll Credits", f"${perf['total_roll_credits']:,.0f}", "bull"),
            ])

            # Trade history table
            closed = get_closed_trades(50)
            if closed:
                ct_df = pd.DataFrame(closed)
                display_cols = [c for c in ["symbol", "contract", "quantity", "entry_price",
                                "exit_price", "pnl_dollars", "pnl_pct", "regime_at_entry",
                                "confidence_tier", "roll_count", "entry_date", "exit_date"] if c in ct_df.columns]
                st.dataframe(ct_df[display_cols], use_container_width=True, hide_index=True)

            # Regime performance breakdown
            if perf.get("regime_performance"):
                rp = perf["regime_performance"]
                for regime, stats in rp.items():
                    wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
                    st.markdown(f"**{regime}**: {stats['trades']} trades, {wr:.0f}% WR, ${stats['pnl']:+,.0f}")
        else:
            pass

    with main_tabs[4]:
        fig_map = plot_regime_heatmap(results)
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)

    with main_tabs[5]:
        # Drill-down selector
        available = [r["symbol"] for r in results if r.get("price") is not None]
        preselect = 0
        if st.session_state.selected_ticker and st.session_state.selected_ticker in available:
            preselect = available.index(st.session_state.selected_ticker)

        selected_sym = st.selectbox("Select Ticker", available, index=preselect)
        if selected_sym:
            st.session_state.selected_ticker = selected_sym
            match = next((r for r in results if r["symbol"] == selected_sym), None)
            if match:
                render_drill_down(match)

else:
    pass  # Hero already rendered at top


# ── Auto-Refresh Logic ──
if auto_refresh and st.session_state.last_scan_time:
    elapsed = (datetime.now() - st.session_state.last_scan_time).total_seconds()
    if elapsed >= refresh_minutes * 60:
        st.rerun()

# ── Scheduled Scan Logic ──
if scheduled_scans_enabled:
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    now_tz = datetime.now(ZoneInfo(scheduled_scan_timezone))
    now_hm = now_tz.strftime("%H:%M")
    scan_times = [t.strip() for t in scheduled_scan_times.split(",") if t.strip()]
    # Check if we're within 1 minute of a scheduled time and haven't scanned recently
    for st_time in scan_times:
        try:
            sched_h, sched_m = int(st_time.split(":")[0]), int(st_time.split(":")[1])
            diff = abs((now_tz.hour * 60 + now_tz.minute) - (sched_h * 60 + sched_m))
            if diff <= 1:
                last = st.session_state.last_scan_time
                if last is None or (datetime.now() - last).total_seconds() > 120:
                    st.rerun()
        except (ValueError, IndexError):
            pass
