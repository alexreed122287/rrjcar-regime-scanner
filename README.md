# 🔮 Regime Terminal — HMM Trading Engine

Hidden Markov Model regime detection system inspired by Renaissance Technologies.
Identifies market regimes (Bull Run → Crash) and layers confirmation-based trading strategies on top.

## Architecture

```
data_loader.py     ← Fetches OHLCV data via Yahoo Finance, engineers features
hmm_engine.py      ← Gaussian HMM training, regime detection, transition matrix
backtester.py      ← 8-confirmation strategy engine, risk management, cooldowns
dashboard.py       ← Streamlit dashboard for visualization and live signals
```

## How It Works

1. **HMM Regime Detection** — Trains a Gaussian Hidden Markov Model on 3 features (returns, range, volume change) to discover 7 hidden market states
2. **Auto-Labeling** — States are ranked by mean return: Bull Run (highest) → Crash (lowest)
3. **Strategy Layer** — Only enters trades when regime is bullish AND 7/8 confirmation signals pass:
   - RSI < 90 (not overbought extreme)
   - RSI > 25 (momentum present)
   - Momentum positive (10-bar)
   - ATR rising (volatility expanding)
   - Volume above 20-period average
   - ADX > 20 (trending market)
   - Price above 50 EMA
   - MACD histogram positive
4. **Exit Rules** — Close immediately on regime flip to bearish
5. **Risk Management** — 48-bar cooldown after exits, configurable leverage

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Launch the dashboard
streamlit run dashboard.py

# Or run headless test
python3 -c "
from data_loader import fetch_data, engineer_features
from hmm_engine import RegimeDetector
from backtester import run_backtest, get_current_signal

df = fetch_data('SPY', period_days=730, interval='1d')
feat_df = engineer_features(df)

detector = RegimeDetector(n_regimes=7)
regime_df = detector.train(feat_df)

signal = get_current_signal(regime_df)
print(signal['signal'], signal['action'])
"
```

## Configuration

| Parameter | Default | Aggressive Mode |
|-----------|---------|-----------------|
| Leverage | 2.5x | 4.0x |
| Min Confirmations | 7/8 | 5/8 |
| Cooldown (bars) | 48 | 24 |
| Trailing Stop | None | -8% from peak |

## Tickers

Works with any Yahoo Finance symbol: BTC, ETH, SPY, QQQ, NVDA, PLTR, AMZN, MSFT, GOOGL, META, TSLA, AMD, MRVL, MU, SMCI, ZS, etc.

## Adapting Over Time

The HMM regime detection stays stable — regimes are regimes. But the **strategy layer** should evolve:

- Adjust confirmation thresholds as market efficiency changes
- Modify leverage/cooldown based on volatility regime
- Swap or add new confirmations (e.g., breadth, put/call ratio, GEX)
- Use Claude Code to iterate: "The drawdown is too high — tighten entry signals"

---

*Not financial advice. For educational and research purposes only.*
