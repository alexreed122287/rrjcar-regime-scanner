# Out-of-sample validation findings

Everything here comes from `walk_forward.py` and `tools/confidence_sweep.py` on real
daily bars (Yahoo fallback, ~3000 calendar days, roughly 2019→2026), 14 non-overlapping
126-bar out-of-sample windows per ticker, tickers SPY / QQQ / NVDA / AAPL / XLF.

**The benchmark that matters is random entry at matched exposure.** Losing to
buy-and-hold while in cash ~90% of the time is expected and says nothing. The question is
whether the regime signal beats coin-flip entries held for the same fraction of bars.

---

## Summary

Three hypotheses tested, three negative results.

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | The strategy beats matched-exposure random entry | **No.** Point estimates negative on 4 of 5 tickers, none significant |
| 2 | 7 regimes is over-parameterized; a converged 3-4 regime model does better | **No.** Level with 7, worse on NVDA |
| 3 | The entry filters (confidence × confirmations) can be tuned into edge | **No.** Best tune setting decayed to exactly zero out of sample |

The system's one demonstrated virtue is capital preservation, not alpha. In the 2022
window SPY lost **-4.01% against buy-and-hold's -18.78%**. It runs at 3-19% market
exposure and behaves as a defensive filter.

---

## 1. Baseline: does it beat matched-exposure random?

At the shipped default (7 regimes, confidence 0.5, 5 confirmations):

| Ticker | Trades | Mean OOS | vs buy-hold | vs random | Win vs rnd | Worst DD | Exposure |
|---|---|---|---|---|---|---|---|
| SPY | 43 | +1.14% | -6.83% | -0.86% | 36% | -5.44% | 10.4% |
| QQQ | 51 | +0.84% | -10.27% | -0.84% | 36% | -11.00% | 10.0% |
| NVDA | 43 | +3.07% | -38.46% | -1.43% | 36% | -9.61% | 10.8% |
| AAPL | 40 | +2.59% | -13.11% | +0.49% | 36% | -8.23% | 6.3% |
| XLF | 42 | -0.46% | -7.40% | -0.89% | 50% | -9.28% | 6.0% |

Significance of per-window excess vs random:

| Series | n | mean | sd | t | p | wins | p (sign) |
|---|---|---|---|---|---|---|---|
| SPY | 14 | -0.90 | 2.19 | -1.54 | 0.149 | 4 | 0.180 |
| QQQ | 14 | -0.55 | 5.49 | -0.38 | 0.712 | 5 | 0.424 |
| NVDA | 14 | -0.92 | 5.99 | -0.58 | 0.574 | 5 | 0.424 |
| AAPL | 14 | +0.49 | 11.30 | 0.16 | 0.873 | 5 | 0.873 |
| XLF | 14 | -0.79 | 2.91 | -1.01 | 0.329 | 7 | 1.000 |
| **Pooled** | 70 | -0.53 | 6.27 | -0.71 | 0.478 | 26 | **0.041** |

No individual ticker is distinguishable from zero. The pooled mean isn't either. Only the
pooled sign test is marginal — and pooling is optimistic here, because SPY/QQQ/AAPL/NVDA
are all US equities over identical windows, so those 70 observations are far from
independent.

**Defensible claim:** no evidence of positive edge, and the sample is too small to prove
harm. 14 windows cannot resolve a ~1% per-window effect against a ~6% standard deviation.

---

## 2. Regime count is not the binding constraint

`hmmlearn` fails to converge on most windows at 7 regimes (7 full-covariance Gaussians
over 3 features is ~120 free parameters against 252 samples), and BIC prefers 3-4 on
every ticker. That made over-parameterization a plausible culprit.

It is not. After fixing the regime-set derivation (see below), auto-selection performs
about the same as the fixed 7:

| Ticker | Mode | Regimes | vs random | Exposure |
|---|---|---|---|---|
| SPY | 7 / auto | — / 3-4 | -0.86% / -0.02% | 10.4% / 14.2% |
| QQQ | 7 / auto | — / 3-4 | -0.84% / -1.23% | 10.0% / 13.4% |
| NVDA | 7 / auto | — / 3-4 | -1.43% / -3.84% | 10.8% / 12.0% |
| AAPL | 7 / auto | — / 3-4 | +0.49% / +0.16% | 6.3% / 10.1% |
| XLF | 7 / auto | — / 3-4 | -0.89% / -1.13% | 6.0% / 19.4% |

### The bug this uncovered

Before the fix, `run_backtest` hardcoded `bullish_regimes=[0, 1, 2]` and
`bearish_regimes=[5, 6]`, which only describes a 7-state model. With a 3-regime model
**every state was bullish**, so the strategy was permanently long and silently became
buy-and-hold — 82-96% exposure with -28% to -37% drawdowns, and an apparent "+16% vs
random" on NVDA that was pure beta.

The same truncation meant `labels_for(5)` produced no bearish label at all, so on the
cloud scan path (`routes_scan.py` uses `n_regimes=5`) the bearish exit could never fire.

Fixed by `hmm_engine.regime_sets(n)`, which maps any regime count proportionally onto the
7-state reference shape. `n_regimes=7` is unchanged.

---

## 3. Entry filters cannot be tuned into edge

`tools/confidence_sweep.py` sweeps `min_confidence` × `min_confirmations` (30 settings)
with a **chronological tune/verify split**: windows 1-7 select one setting, windows 8-14
score it exactly once.

A note on what the knobs actually are: `scheduled_scan.MIN_CONFIDENCE = 0.80` is **not**
the strategy's entry gate. It only filters which scan hits get emailed. The real entry
gate was hardcoded at `0.5` in `backtester.py`, so every earlier result was produced by a
much looser filter than the documented 0.80 implies. Both gates are now parameters.

### Result

- Best setting on tune (0.40 confidence / 4 confirmations): **+1.71%**, p=0.258
- The same setting on held-out windows: **-0.00%**, p=1.000
- Tune → verify decay: **+1.71 percentage points**

That is textbook noise-fitting. The shipped default (0.50 / 5) went from +0.95% on tune
to **-1.46%** on verify (p=0.105).

Nothing on the surface is significant. The best p-value anywhere is 0.258, against a
Bonferroni threshold of 0.0017 for 30 comparisons. Sign agreement between the two halves
is 19/30 (63%), barely above a coin flip — the surface is mostly noise.

### The one pattern that does replicate

Collapsing the 6 confidence levels reduces 30 comparisons to 5 and asks a single
structural question: does demanding more confirmations help?

| min_confirmations | TUNE mean | VERIFY mean | verify win% | Exposure |
|---|---|---|---|---|
| 3 | +0.88% | -0.03% | 46% | 8.1% |
| 4 | +1.38% | -0.04% | 45% | 7.8% |
| 5 (shipped) | +0.79% | -0.87% | 41% | 7.1% |
| 6 | -0.30% | -1.57% | 38% | 5.0% |
| 7 | -1.16% | -1.70% | 22% | 2.5% |

This is monotone on the held-out half and directionally consistent across both halves.
Requiring more confirmations does not merely fail to add value — it appears to subtract
it. At 3-4 confirmations the strategy is at parity with random; at 6-7 it is clearly
worse.

**No default was changed on the strength of this.** The best available outcome is parity
with random entry, and no cell is individually significant, so lowering
`min_confirmations` from 5 to 4 would be a judgment call rather than a demonstrated
improvement. It is recorded here for a human decision.

---

## Known measurement bugs found along the way

1. **Annualization** — `data_loader.fetch_data` defaults to **hourly** bars, but
   `walk_forward` annualized with a hardcoded 252, overstating benchmark Sharpe by
   √6.5 ≈ 2.5x. Fixed with interval-aware `periods_per_year()`.
2. **Still open:** `backtester.py:439` hardcodes `np.sqrt(252)` for the strategy's own
   `sharpe_ratio`. On non-daily bars the strategy and benchmark Sharpes use different
   annualization factors, and the live dashboard inherits the error. Left unchanged
   because it affects displayed output.

---

## What has not been tested

- **The feature set.** The HMM uses only 3 features. If they cannot separate regimes,
  nothing downstream can work. This is the last untested hypothesis and the most likely
  remaining explanation.
- **More windows.** 14 per ticker is too few for ~1% effects. Shorter OOS steps, more
  tickers, or overlapping windows with corrected standard errors would all help.
- **Transaction costs.** Not modeled anywhere. Including them would move every result
  above in the wrong direction.

## How to reproduce

```bash
# Baseline walk-forward on one ticker
.venv/bin/python walk_forward.py SPY --interval 1d --period-days 3000 --regimes 7

# Entry-filter sweep with tune/verify split
.venv/bin/python tools/confidence_sweep.py \
  --tickers SPY,QQQ,NVDA,AAPL,XLF \
  --confidences 0.4,0.5,0.6,0.7,0.8,0.9 \
  --confirmations 3,4,5,6,7
```
