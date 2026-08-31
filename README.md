# 🔮 Regime Terminal — HMM Regime Detection Engine

Hidden Markov Model regime detection over daily bars, with confirmation-gated strategy
layers, a FastAPI backend, and a browser UI. Includes an out-of-sample validation harness —
and the results that harness produced, which are not flattering.

## Validation status — read this first

This system has been tested out of sample thirteen times: **twelve negative results and one
positive — and the positive one has since been shown to be useless.** Full detail, with effect sizes
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
| Does that volatility signal beat a **free EWMA** forecast? | **No.** Every comparison a statistical tie; the regime label alone ranks 4th of 6 forecasts, below one line of pandas |
| Does **sizing** off predicted vol rescue the model? | **Sizing yes, the model no.** Vol targeting lifts Sharpe 0.36 → 0.88 — but adding the regime *significantly worsens* it (−0.079 Sharpe) |
| Would purpose-built **volatility features** fix it? | **No.** They performed *worse* than the shipped features. The constraint is the 7-state discrete architecture, not the inputs |

**The capital-preservation claim has now been benchmarked, and it did not survive.** The old
headline — SPY **-4.01%** against buy-and-hold's **-18.78%** in 2022 — is still arithmetically
true, and across 50 windows the filter does cut drawdown from -16.7% to -12.5%. But it is
**statistically indistinguishable from a coin flip holding the same average exposure**, so the
reduction is what being out of the market buys, not evidence that the model knows when to be
out. A 200-day moving average earns **10.1% against the HMM's 2.8%** for the same drawdown at
a fifteenth of the turnover, and naive volatility targeting beats it on both drawdown and
Sharpe. See test 10.

**The one thing that worked has now also been closed out.** The regimes do carry a little
forward-*volatility* information at a 5-bar horizon (test 9), which pointed at position sizing.
Three follow-ups settled it: an ordinary EWMA forecast matches the regime label head to head
(test 11), sizing positions off predicted vol is a large improvement over the shipped filter
but the *EWMA* delivers it while adding the regime makes it **significantly worse** through
turnover (test 12), and purpose-built volatility features perform worse than the shipped ones
(test 13). **No component of this model currently has a demonstrated use.**

The one positive by-product: naive volatility targeting — a 20-bar rolling standard deviation,
no regime model — earns Sharpe **0.89** against the shipped filter's **0.36** at a
twenty-fourth of the turnover. It is still below buy-and-hold on Sharpe, so it is not an edge,
but it is the best-performing thing measured in this repo.

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
since been run — they are tests 9 and 10 — as have the three that replaced them (tests 11, 12
and 13). What is left is no longer about tuning this model:

- **Try a different model class.** Every test here measures *this* HMM. A one-parameter EWMA
  keeps matching or beating it at its only surviving task, which points at GARCH, HAR, or a
  plain rolling standard deviation as a *replacement* rather than a benchmark.
- **Tune the volatility overlay** that fell out of test 12. `size_trail20` is the best result in
  the repo and has been run at exactly one target vol, one exposure cap and one cost assumption.
- **Real index data** (`^VIX`, `^TNX`), which no configured source provides. Test 13 makes this
  much less promising than it looked: better vol inputs made the model worse, not better.

---

*Not financial advice. For educational and research purposes only.*
