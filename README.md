# 🔮 Regime Terminal — HMM Regime Detection Engine

Hidden Markov Model regime detection over daily bars, with confirmation-gated strategy
layers, a FastAPI backend, and a browser UI. Includes an out-of-sample validation harness —
and the results that harness produced, which are not flattering.

## Validation status — read this first

This system has been tested out of sample ten times: **nine negative results and one
positive.** The positive one is about volatility, not returns. Full detail, with effect sizes
and reproduction commands, is in [`docs/validation-findings.md`](docs/validation-findings.md).

The short version:

| Question | Answer |
|---|---|
| Does it beat matched-exposure random entry? | **No.** Point estimates negative on 4 of 5 tickers, none significant |
| Do the regimes carry forward return information? | **No.** Flat null at every horizon tested (1/5/10/20 bars) |
| Can the entry filters be tuned into edge? | **No.** The best tuned setting decayed to exactly zero out of sample |
| Does it beat buy-and-hold? | **No.** Once option roll legs are priced honestly, buy-and-hold wins on 5 of 5 tickers |
| Is the regime ranking fixable (inverted, forward-scored)? | **No.** No ranking rule clears a Bonferroni-corrected bar |
| Do regimes separate forward **volatility**, beyond free trailing vol? | **Yes, at 5 bars.** Bearish set runs ~14.5% higher forward 5-day vol, bootstrap CI [+0.069, +0.208]. Null at 20 bars, and it explains only ~1.4% of variance |
| Does the capital preservation beat trivial alternatives? | **No.** Indistinguishable from a coin flip at the same exposure; a 200-day MA earns 10.1% vs the HMM's 2.8% for the same drawdown |

**The capital-preservation claim has now been benchmarked, and it did not survive.** The old
headline — SPY **-4.01%** against buy-and-hold's **-18.78%** in 2022 — is still arithmetically
true, and across 50 windows the filter does cut drawdown from -16.7% to -12.5%. But it is
**statistically indistinguishable from a coin flip holding the same average exposure**, so the
reduction is what being out of the market buys, not evidence that the model knows when to be
out. A 200-day moving average earns **10.1% against the HMM's 2.8%** for the same drawdown at
a fifteenth of the turnover, and naive volatility targeting beats it on both drawdown and
Sharpe. See test 10.

**The one thing that did work:** the regimes carry a little forward-*volatility* information
at a 5-bar horizon (test 9). That points at position sizing, not entry timing — and it is weak
enough that a plain EWMA volatility forecast may well do the job better, which nobody has
checked yet.

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

Nine negative results share one root cause: the three features carry no forward *return*
information. Tuning thresholds on a signal with no information can only fit noise, so the
entry-timing direction is exhausted. Both of the directions this section used to propose have
since been run — they are tests 9 and 10 — which leaves three open questions, in order of how
much they would change the picture:

- **Does a plain EWMA volatility forecast beat the HMM at forward vol?** Test 9's positive
  result is an increment over a trailing-vol control, but it was never compared head to head
  against a good vol model. EWMA explains far more variance and needs no HMM. If it wins, the
  last positive result is also redundant, and that is the cheapest experiment left.
- **Size positions inversely to predicted volatility** rather than switching in and out. Test
  9 says the model's only real output is a vol estimate; test 10 says the binary in/out filter
  is worthless. Sizing is the one use of that output nobody has tested.
- **Build purpose-made volatility features** (longer lookbacks, real `^VIX`/`^TNX` data) now
  that there is evidence the model picks up vol rather than direction.

---

*Not financial advice. For educational and research purposes only.*
