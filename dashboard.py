"""
dashboard.py — Regime Terminal Dashboard
Streamlit-based UI for HMM regime detection, backtesting, and live signals.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_loader import fetch_data, engineer_features, resolve_ticker
from hmm_engine import RegimeDetector, REGIME_LABELS
from backtester import run_backtest, get_current_signal

# ─── Page Config ───
st.set_page_config(
    page_title="Regime Terminal — HMM Trading Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');

    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }

    .metric-card {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card .label {
        font-family: 'JetBrains Mono', monospace;
        color: #888;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .bull { color: #00ff88; }
    .bear { color: #ff4444; }
    .neutral { color: #ffaa00; }
    .cash { color: #888; }

    .signal-banner {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .signal-long { background: linear-gradient(135deg, #003322, #004d33); border: 2px solid #00ff88; color: #00ff88; }
    .signal-exit { background: linear-gradient(135deg, #330000, #4d0000); border: 2px solid #ff4444; color: #ff4444; }
    .signal-cash { background: linear-gradient(135deg, #1a1a00, #333300); border: 2px solid #ffaa00; color: #ffaa00; }

    div[data-testid="stSidebar"] { background: #0a0a0f; }
    .stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ─── Regime Color Map ───
REGIME_COLORS = {
    0: "#00ff88",   # Bull Run — bright green
    1: "#00cc66",   # Bull Trend — green
    2: "#88cc44",   # Mild Bull — yellow-green
    3: "#ffaa00",   # Neutral — amber
    4: "#ff7744",   # Mild Bear — orange
    5: "#ff4444",   # Bear Trend — red
    6: "#cc0000",   # Crash — dark red
}


def render_metric(label: str, value: str, css_class: str = ""):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {css_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_signal_banner(signal_data: dict):
    """Render the main trading signal banner."""
    sig = signal_data["signal"]
    if "LONG" in sig:
        css = "signal-long"
        icon = "🟢"
    elif "EXIT" in sig:
        css = "signal-exit"
        icon = "🔴"
    else:
        css = "signal-cash"
        icon = "🟡"

    st.markdown(f"""
    <div class="signal-banner {css}">
        {icon} &nbsp; {sig} &nbsp; {icon}
        <div style="font-size: 0.85rem; font-weight: 400; margin-top: 6px; opacity: 0.85;">
            {signal_data['action']}
        </div>
    </div>
    """, unsafe_allow_html=True)


def plot_price_with_regimes(df: pd.DataFrame, title: str = "Price Chart with Regime Overlay"):
    """Plotly chart: price line colored by regime."""
    fig = go.Figure()

    for regime_id in sorted(df["regime_id"].unique()):
        mask = df["regime_id"] == regime_id
        subset = df[mask]
        label = REGIME_LABELS[regime_id] if regime_id < len(REGIME_LABELS) else f"State {regime_id}"
        color = REGIME_COLORS.get(regime_id, "#666")

        fig.add_trace(go.Scatter(
            x=subset.index,
            y=subset["Close"],
            mode="markers",
            marker=dict(size=3, color=color),
            name=label,
            hovertemplate=f"<b>{label}</b><br>Price: %{{y:,.2f}}<br>Date: %{{x}}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        height=500,
        xaxis=dict(gridcolor="#1a1a2e"),
        yaxis=dict(gridcolor="#1a1a2e", title="Price"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def plot_equity_curve(equity_curve: pd.Series, df: pd.DataFrame):
    """Plotly chart: equity curve vs buy-and-hold."""
    bh_start = df["Close"].iloc[0]
    bh_equity = (df["Close"] / bh_start) * equity_curve.iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        mode="lines", name="HMM Strategy",
        line=dict(color="#00ff88", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=bh_equity.values,
        mode="lines", name="Buy & Hold",
        line=dict(color="#666", width=1, dash="dash"),
    ))
    fig.update_layout(
        title="Equity Curve — Strategy vs Buy & Hold",
        template="plotly_dark",
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        height=400,
        xaxis=dict(gridcolor="#1a1a2e"),
        yaxis=dict(gridcolor="#1a1a2e", title="Equity ($)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


def plot_regime_distribution(regime_stats: pd.DataFrame):
    """Bar chart of time spent in each regime."""
    fig = go.Figure(go.Bar(
        x=regime_stats["regime_label"],
        y=regime_stats["pct_of_total"],
        marker_color=[REGIME_COLORS.get(i, "#666") for i in regime_stats["regime_id"]],
        text=[f"{v:.1f}%" for v in regime_stats["pct_of_total"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Time Distribution by Regime",
        template="plotly_dark",
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        height=350,
        yaxis=dict(title="% of Time", gridcolor="#1a1a2e"),
        xaxis=dict(gridcolor="#1a1a2e"),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    return fig


# ════════════════════════════════════════════
#  SIDEBAR — Configuration
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    symbol = st.text_input("Ticker Symbol", value="BTC", help="e.g. BTC, SPY, NVDA, PLTR, AMZN")

    st.markdown("---")
    st.markdown("### 📅 Date Range")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start", value=datetime.now() - timedelta(days=730))
    with col_e:
        end_date = st.date_input("End", value=datetime.now())

    st.markdown("---")
    st.markdown("### 🧠 HMM Settings")
    n_regimes = st.slider("Number of Regimes", 3, 10, 7)
    interval = st.selectbox("Candle Interval", ["1h", "1d"], index=0)

    st.markdown("---")
    st.markdown("### 📊 Strategy Settings")
    leverage = st.slider("Leverage", 1.0, 6.0, 2.5, 0.5)
    min_confs = st.slider("Min Confirmations (of 8)", 3, 8, 7)
    cooldown = st.slider("Cooldown (bars after exit)", 6, 120, 48)
    initial_capital = st.number_input("Initial Capital ($)", value=100_000, step=10_000)
    aggressive = st.checkbox("🔥 Aggressive Mode", value=False,
                             help="4x leverage, 5/8 confs, 24-bar cooldown, trailing stop")

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)


# ════════════════════════════════════════════
#  MAIN — Dashboard
# ════════════════════════════════════════════
st.markdown("# 🔮 Regime Terminal")
st.markdown("*Hidden Markov Model — Regime Detection & Strategy Engine*")

if run_btn:
    with st.spinner("Fetching market data..."):
        try:
            raw_df = fetch_data(
                symbol=symbol,
                interval=interval,
                start_date=str(start_date),
                end_date=str(end_date),
            )
            feat_df = engineer_features(raw_df)
        except Exception as e:
            st.error(f"❌ Data fetch failed: {e}")
            st.stop()

    with st.spinner(f"Training HMM on {len(feat_df):,} data points..."):
        detector = RegimeDetector(n_regimes=n_regimes)
        regime_df = detector.train(feat_df)

    # ── Current Signal ──
    st.markdown("---")
    current = detector.predict_current(regime_df)
    signal_data = get_current_signal(regime_df, min_confirmations=min_confs)

    render_signal_banner(signal_data)

    # Top metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        regime_css = "bull" if current["regime_id"] <= 1 else ("bear" if current["regime_id"] >= 5 else "neutral")
        render_metric("Current Regime", current["regime_label"], regime_css)
    with c2:
        render_metric("Confidence", f"{current['confidence']:.1%}", "")
    with c3:
        render_metric("Confirmations", f"{signal_data['confirmations_met']}/8", "")
    with c4:
        render_metric("Price", f"${signal_data['price']:,.2f}", "")
    with c5:
        changed = "YES ⚡" if signal_data["regime_changed"] else "No"
        render_metric("Regime Changed?", changed, "bear" if signal_data["regime_changed"] else "cash")

    # Confirmation breakdown
    with st.expander("📋 Confirmation Breakdown", expanded=False):
        conf_cols = st.columns(4)
        for idx, (name, passed) in enumerate(signal_data["confirmation_detail"].items()):
            with conf_cols[idx % 4]:
                icon = "✅" if passed else "❌"
                st.markdown(f"{icon} **{name}**")

    # ── Backtest ──
    st.markdown("---")
    with st.spinner("Running backtest simulation..."):
        results = run_backtest(
            regime_df,
            leverage=leverage,
            min_confirmations=min_confs,
            cooldown_bars=cooldown,
            initial_capital=initial_capital,
            aggressive_mode=aggressive,
        )

    metrics = results["metrics"]

    # Performance metrics
    st.markdown("### 📈 Backtest Performance")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        css = "bull" if metrics["total_return_pct"] > 0 else "bear"
        render_metric("Total Return", f"{metrics['total_return_pct']:.1f}%", css)
    with m2:
        css = "bull" if metrics["alpha_vs_buyhold"] > 0 else "bear"
        render_metric("Alpha vs B&H", f"{metrics['alpha_vs_buyhold']:.1f}%", css)
    with m3:
        render_metric("Win Rate", f"{metrics['win_rate']:.0f}%", "bull" if metrics["win_rate"] > 50 else "bear")
    with m4:
        render_metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", "")
    with m5:
        render_metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.1f}%", "bear")
    with m6:
        render_metric("Profit Factor", f"{metrics['profit_factor']:.2f}", "")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        render_metric("Initial Capital", f"${metrics['initial_capital']:,.0f}", "cash")
    with mc2:
        render_metric("Final Equity", f"${metrics['final_equity']:,.0f}",
                      "bull" if metrics["final_equity"] > metrics["initial_capital"] else "bear")
    with mc3:
        render_metric("Total Trades", str(metrics["total_trades"]), "")
    with mc4:
        render_metric("Leverage", f"{metrics['leverage']}x", "")

    # ── Charts ──
    st.markdown("---")
    tabs = st.tabs(["🗺️ Regime Map", "💰 Equity Curve", "📊 Regime Distribution", "📝 Trade Log"])

    with tabs[0]:
        fig_regime = plot_price_with_regimes(results["df"], f"{resolve_ticker(symbol)} — Regime Overlay")
        st.plotly_chart(fig_regime, use_container_width=True)

    with tabs[1]:
        fig_equity = plot_equity_curve(results["equity_curve"], results["df"])
        st.plotly_chart(fig_equity, use_container_width=True)

    with tabs[2]:
        col_dist, col_stats = st.columns([1, 1])
        with col_dist:
            fig_dist = plot_regime_distribution(detector.regime_stats)
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_stats:
            st.markdown("#### Regime Statistics")
            stats_display = detector.regime_stats.copy()
            stats_display["mean_return"] = stats_display["mean_return"].apply(lambda x: f"{x:.4f}")
            stats_display["volatility"] = stats_display["volatility"].apply(lambda x: f"{x:.4f}")
            stats_display["pct_of_total"] = stats_display["pct_of_total"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(stats_display, use_container_width=True, hide_index=True)

    with tabs[3]:
        if results["trades"]:
            trade_df = pd.DataFrame(results["trades"])
            # Color PnL
            st.dataframe(
                trade_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "pnl_pct": st.column_config.NumberColumn("PnL %", format="%.2f%%"),
                    "entry_price": st.column_config.NumberColumn("Entry $", format="$%.2f"),
                    "exit_price": st.column_config.NumberColumn("Exit $", format="$%.2f"),
                },
            )
        else:
            st.info("No trades generated in this backtest window.")

    # ── Transition Matrix ──
    with st.expander("🔄 Regime Transition Probabilities"):
        trans = detector.get_transition_matrix()
        fig_heat = px.imshow(
            trans.values,
            x=trans.columns,
            y=trans.index,
            color_continuous_scale="RdYlGn",
            text_auto=".2f",
            aspect="auto",
        )
        fig_heat.update_layout(
            title="Probability of Transitioning Between Regimes",
            template="plotly_dark",
            paper_bgcolor="#0a0a0f",
            height=450,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.success(f"✅ Analysis complete — {len(feat_df):,} candles processed, {len(results['trades'])} trades simulated.")

else:
    # Landing state
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; opacity: 0.7;">
        <h2 style="font-family: 'Space Grotesk', sans-serif;">Configure & Run</h2>
        <p style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;">
            Set your ticker, date range, and strategy parameters in the sidebar.<br>
            Then click <strong>🚀 Run Analysis</strong> to train the HMM and generate signals.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick-start suggestions
    st.markdown("### Quick Start Tickers")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        st.markdown("**Crypto:** BTC, ETH")
    with qc2:
        st.markdown("**Index:** SPY, QQQ")
    with qc3:
        st.markdown("**Mega-cap:** NVDA, AMZN, MSFT")
    with qc4:
        st.markdown("**Growth:** PLTR, MRVL, MU, ZS")
