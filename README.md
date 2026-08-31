# 🔮 Regime Terminal — HMM Regime Detection Engine

Hidden Markov Model regime detection over daily bars, with confirmation-gated strategy
layers, a FastAPI backend, and a browser UI. Includes an out-of-sample validation harness —
and the results that harness produced, which are not flattering.

## Validation status — read this first

This system has been tested out of sample eight times. **All eight results were negative.**
Full detail, with effect sizes and reproduction commands, is in
[`docs/validation-findings.md`](docs/validation-findings.md).

The short version:

| Question | Answer |
|---|---|
| Does it beat matched-exposure random entry? | **No.** Point estimates negative on 4 of 5 tickers, none significant |
| Do the regimes carry forward return information? | **No.** Flat null at every horizon tested (1/5/10/20 bars) |
| Can the entry filters be tuned into edge? | **No.** The best tuned setting decayed to exactly zero out of sample |
| Does it beat buy-and-hold? | **No.** Once option roll legs are priced honestly, buy-and-hold wins on 5 of 5 tickers |
| Is the regime ranking fixable (inverted, forward-scored)? | **No.** No ranking rule clears a Bonferroni-corrected bar |

**What it does appear to do** is preserve capital. In the 2022 window SPY lost **-4.01%
against buy-and-hold's -18.78%**, at 3-19% market exposure — it behaves as a defensive
filter, not an alpha source. Note that this observation is currently a single ticker in a
single year and has **not** been benchmarked against simpler alternatives such as a 200-day
moving-average filter, so do not treat it as established.

The genuinely useful artifact in this repo is the validation harness (`walk_forward.py` plus
`tools/`), which is capable of producing negative results about its own strategy. Treat the
trading layer as an unvalidated research prototype. **Not financial advice; do not trade
this with money you care about.**

## Architecture

```
data_loader.py       ← Fetches OHLCV via Tradier/Yahoo/Alpha Vantage, engineers features
hmm_engine.py        ← Gaussian HMM training, regime detection, transition matrix, regime_sets
backtester.py        ← v1 strategy: 8 confirmation signals, risk management, cooldowns
strategy_v2.py       ← v2 strategy (the API default): 12 confirmations, priced option rolls
strategy_leaps.py    ← LEAPS variant
strategy_bottoming.py← Mean-reversion variant
options_pricing.py   ← Black-Scholes price/delta, used to charge rolls their real cost
walk_forward.py      ← Out-of-sample walk-forward harness (the part that works)
app.py + api/        ← FastAPI backend
static/ + index.html ← Browser UI
tools/               ← One script per validation experiment; see docs/validation-findings.md
```

## How It Works

1. **HMM Regime Detection** — Trains a Gaussian HMM on 3 features (returns, range, volume
   change) to discover 7 hidden market states.
2. **Auto-Labeling** — States are ranked by their mean return and labelled Bull Run
   (highest) → Crash (lowest). **Caveat:** because the HMM is fitted on return and momentum
   features, this ranking is close to tautological — a state scores highest largely because
   it is *defined* by prices having risen. Test 8 in the findings doc shows the ranking
   carries no forward information, and that inverting it does not help either.
3. **Strategy Layer** — Enters only when the regime is bullish *and* N confirmation signals
   pass (8 signals in v1, 12 in v2).
4. **Exit Rules** — Closes on regime flip out of the bullish set.
5. **Risk Management** — Cooldown after exits and configurable leverage in both engines;
   `strategy_v2` adds an ATR trailing stop, time stop, profit target and RSI exit.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # optional: Tradier / Alpha Vantage keys
```

## Run

```bash
# API + browser UI
uvicorn app:app --reload        # then open http://127.0.0.1:8000

# Test suite
python -m pytest -q

# Reproduce the validation findings (see docs/validation-findings.md for all commands)
python walk_forward.py
python tools/regime_ranking.py --tickers SPY --save-obs obs/SPY.csv
```

Headless smoke test:

```bash
python3 -c "
from data_loader import fetch_data, engineer_features
from hmm_engine import RegimeDetector
from backtester import get_current_signal

df = fetch_data('SPY', period_days=730, interval='1d')
detector = RegimeDetector(n_regimes=7)
regime_df = detector.train(engineer_features(df))
signal = get_current_signal(regime_df)
print(signal['signal'], signal['action'])
"
```

## Configuration

Actual code defaults, as of this commit:

| Parameter | `backtester` (v1) | `strategy_v2` (API default) | Aggressive mode (v1) |
|-----------|-------------------|------------------------------|----------------------|
| Leverage | 1.0x | 1.0x | 4.0x |
| Min confirmations | 5 (of 8) | 6 (of 12) | 5 |
| Cooldown (bars) | 3 | 3 | 3 |
| Regime confirm bars | 2 | 2 | 2 |
| Min regime confidence | 0.5 | 0.5 | 0.5 |
| ATR trailing-stop multiple | n/a | 2.0 | n/a |
| Cost per side | **0 bps** | **0 bps** | 0 bps |

**Mind the cost default.** The backtest engines default `cost_bps_per_side=0.0`, so calling
them directly gives you a **cost-free** run — the strategy's best case, not a neutral one.
Friction consumed 18-37% of gross return in testing. The API route
(`GET /api/backtest/{symbol}`) overrides this to **5 bps** per side; pass `cost_bps=0` there
to reproduce the old cost-free numbers. Prefer passing an explicit value.

v1 (`backtester`) exits only on regime flip and cooldown. The multi-exit logic — ATR trailing
stop, time stop, profit target, RSI exit — is in `strategy_v2` only.

`strategy_v2` prices option roll legs with Black-Scholes by default (`roll_model="priced"`).
The older `roll_model="flat"` credited a fixed percentage on every roll, which turned out to
account for roughly half the strategy's headline return.

## Tickers

Works with any Yahoo Finance symbol: SPY, QQQ, NVDA, PLTR, AMZN, MSFT, GOOGL, META, TSLA,
AMD, MRVL, MU, SMCI, ZS, BTC-USD, ETH-USD, etc. Validation used SPY / QQQ / NVDA / AAPL / XLF.

## If you pick this up again

Eight negative results share one root cause: the three features carry no forward return
information. Tuning thresholds on a signal with no information can only fit noise, so the
two directions worth pursuing are:

- **Test forward *volatility* rather than forward returns.** Volatility clustering is a far
  more robust empirical regularity than return predictability, and the existing features
  include vol proxies. This would give the capital-preservation behaviour a mechanism.
- **Benchmark the drawdown property against simple alternatives** — a 200-day MA filter,
  naive vol targeting, matched-exposure random. If a moving average does the same job, the
  HMM is expensive machinery for no gain, and that is worth knowing.

---

*Not financial advice. For educational and research purposes only.*
