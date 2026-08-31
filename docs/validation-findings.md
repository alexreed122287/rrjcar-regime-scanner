# Out-of-sample validation findings

Everything here comes from `walk_forward.py` and `tools/confidence_sweep.py` on real
daily bars (Yahoo fallback, ~3000 calendar days, roughly 2019→2026), 14 non-overlapping
126-bar out-of-sample windows per ticker, tickers SPY / QQQ / NVDA / AAPL / XLF.

**The benchmark that matters is random entry at matched exposure.** Losing to
buy-and-hold while in cash ~90% of the time is expected and says nothing. The question is
whether the regime signal beats coin-flip entries held for the same fraction of bars.

---

## Summary

Fourteen hypotheses tested: **thirteen negative, one positive -- and the positive one has since
been shown to be useless.** Test 9 found the regimes carry forward-*volatility* information.
Tests 11, 12 and 13 then established that the information is redundant against a free EWMA,
that acting on it makes performance significantly *worse*, and that purpose-built volatility
features do not fix it. The honest bottom line is that this model has no demonstrated use.

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | The strategy beats matched-exposure random entry | **No.** Point estimates negative on 4 of 5 tickers, none significant |
| 2 | 7 regimes is over-parameterized; a converged 3-4 regime model does better | **No.** Level with 7, worse on NVDA |
| 3 | The entry filters (confidence × confirmations) can be tuned into edge | **No.** Best tune setting decayed to exactly zero out of sample |
| 4 | The 3 features separate regimes in a way that predicts forward returns | **No.** Regimes carry no forward information; labelled-bullish mildly underperforms |
| 5 | Regimes are informative at longer horizons (5/10/20 bars) even if not at 1 | **No.** Flat null at every horizon tested |
| 6 | Transaction costs are a second-order detail | **No.** They cost 18-37% of gross return and *widen* the gap to random |
| 7 | Cross-asset features (credit, rates, breadth) give regimes forward information | **No.** Macro-only regimes score exactly at the null |
| 8 | The state ranking rule can be fixed (inverted, or scored on forward returns) | **No.** Rank 0 is the weakest state but at p=0.50; no rule clears Bonferroni |
| 9 | Regimes separate forward **volatility**, beyond what trailing vol already gives free | **Yes, at 5 bars.** Bearish set runs ~14% higher forward vol, bootstrap CI [+0.069, +0.208], 5/5 tickers. Null at 20 bars |
| 10 | The capital-preservation claim survives benchmarking against trivial alternatives | **No.** Real vs buy-and-hold (-12.5% vs -16.7% maxDD) but **indistinguishable from a coin flip with the same exposure**, and a 200-day MA earns 10.1% against the HMM's 2.8% for the same drawdown |
| 11 | The test-9 volatility signal beats, or adds to, a free EWMA forecast | **No.** Every comparison a statistical tie. The regime label alone is *worse* than EWMA on point estimate; adding it to EWMA improves R² 0.484 -> 0.497 but the interval brackets zero |
| 12 | Sizing positions off predicted volatility rescues the model | **Sizing yes, the model no.** Vol-targeted sizing beats the shipped filter massively (Sharpe 0.88 vs 0.36) -- but the *EWMA* forecast does that, and adding the regime makes it **significantly worse** (-0.079 Sharpe) |
| 13 | The nulls are an input problem; purpose-built volatility features would fix it | **No.** Vol-specific features are *worse* than the shipped ones, and none beats a free EWMA. The limitation is the architecture, not the inputs |
| 14 | The volatility overlay from #12 is robust enough to replace the shipped filter | **No.** Robust in sign across 240 configurations, but it never beats a *constant position at its own average exposure* on drawdown -- and most of its edge over the filter is simply the filter's 49.6 turnover |

**Nothing in this repo now has a demonstrated edge, defensive or otherwise.** The one positive
result survived exactly two more tests before collapsing: it is not better than an exponential
moving average of squared returns, and trading on it loses money to turnover. See #11-13.

**The capital-preservation claim, previously the system's one demonstrated virtue, does not
survive #10.** The drawdown reduction is real relative to buy-and-hold, but it is what being
out of the market buys you, not evidence of timing: a random filter holding the same average
exposure achieves the same drawdown, and both a 200-day moving average and naive volatility
targeting dominate the HMM on return, Sharpe and turnover. The headline SPY 2022 figure
(-4.01% vs -18.78%) still stands as arithmetic; it just is not attributable to the model.

What remains, after ten tests, is: the regimes carry no forward *return* information at any
horizon, ranking, feature set or cost assumption tested; they do carry a small amount of
forward *volatility* information at a 5-bar horizon; and the defensive behaviour that
information might justify is available more cheaply elsewhere.

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

## 4. The root cause: regimes carry no forward information

The first three tests all ran through the backtest, so entry rules, confirmations and
cooldowns sat between the signal and the result. `tools/regime_separability.py` removes
all of it and tests the premise directly.

`RegimeDetector` assigns regime ids by ranking states on their **in-sample** mean return,
so id 0 is the most bullish state *on the training window*, and `run_backtest` treats ids
0..k as bullish. That is only meaningful if the ranking persists on unseen data. So, with
causal labels, per window: rank-correlate regime id against the return realized **after**
the regime is observed (`Close.pct_change().shift(-1)`), and test whether the
forward-return distributions differ across regimes at all.

### Do regimes separate forward returns?

No. Kruskal-Wallis across regimes, fraction of windows significant at p<0.05:

| Model | KW p<0.05 | vs 5% null | Verdict |
|---|---|---|---|
| 7 regimes | 10% of 70 windows | p=0.088 | not significant |
| auto (3-4, converged) | **4% of 69 windows** | p=1.000 | **exactly the null** |

With converged models the regimes are indistinguishable from a random partition of bars
with respect to forward returns. This is the finding that explains all three results
above: there was never any forward information for the downstream layers to exploit.

### The little signal there is points the wrong way

For the labeling to be tradeable, rank correlation between regime id and forward return
must be reliably **negative** (low id → high return). It is positive at every level of
aggregation and in every ticker:

| Aggregation | 7 regimes | auto (3-4) |
|---|---|---|
| Mean rho (should be negative) | **+0.032** | **+0.026** |
| Time-clustered t-test (n=14 periods) | p=0.038 | p=0.064 |
| Tickers with intended sign | **0 of 5** (p=0.062) | **0 of 5** (p=0.062) |
| Windows where regime 0 beat window avg | 43% (p=0.585) | — |

0 of 5 tickers show the intended direction, at both regime counts — and with n=5, 0.062
is the smallest p-value attainable. So the direction is consistent, though no single
test clears 0.05.

The mechanism is straightforward. States are ranked on trailing mean return over 252 bars,
then the top-ranked state is bought. On daily equity bars that statistic is dominated by
noise and mildly mean-reverting, so the strategy systematically buys what has just run.
That is a coherent explanation for losing to matched-exposure random by roughly 1% per
window while never being significantly bad.

**Effect size caveat:** rho ≈ +0.03 explains about 0.1% of forward-return variance. The
correct read is not "this is a good short signal" but "this is noise with a faint adverse
tilt".

### What this implies

Further tuning of thresholds, confirmations, cooldowns or regime counts cannot help,
because those layers all consume a signal with no demonstrated forward information. The
two honest paths are to change the inputs — the features are `returns`, `range`,
`volume_change`, all short-horizon and largely restatements of realized volatility and
momentum — or to reposition the system as the defensive exposure filter it demonstrably
is.

---

## 5. No information at any holding horizon

Test 4 measured next-bar returns, which left a fair objection open: regime models are
usually motivated as *medium*-horizon descriptions, so a 1-bar null could be the wrong
question rather than a real absence. `tools/regime_horizons.py` reruns the same causal
setup at h = 1, 5, 10, 20 bars.

Overlapping forward returns are strongly autocorrelated, so observations are **subsampled
with stride h** to be non-overlapping. That leaves too few per window at long h, so the
primary test pools non-overlapping observations across windows within each ticker (regime
ids are comparable across windows by construction) and treats tickers as the independent
unit.

| h | mean rho (want negative) | tickers with intended sign | tickers with KW p<0.05 |
|---|---|---|---|
| 1 | +0.009 | 2/5 | 1/5 |
| 5 | -0.022 | 4/5 | 0/5 |
| 10 | +0.014 | 1/5 | 0/5 |
| 20 | -0.009 | 3/5 | 0/5 |

A flat null. Mean rho oscillates around zero with no trend in horizon, sign counts are
coin flips, and the single nominally significant Kruskal-Wallis result (QQQ at h=1,
p=0.027) is exactly what 20 ticker × horizon tests produce by chance — expected count
under the null is 1.0.

This closes the horizon loophole. The features carry no forward information at any holding
period from one day to one month.

### Secondary observation, deliberately underweighted

Regime 0 — the state labeled most bullish — underperformed the window average at longer
horizons:

| h | tickers where regime 0 < average | mean gap | sign p |
|---|---|---|---|
| 1 | 2/5 | +0.01 pp | 1.000 |
| 5 | 2/5 | -0.05 pp | 1.000 |
| 10 | **5/5** | -0.78 pp | 0.062 |
| 20 | **5/5** | -1.24 pp | 0.062 |

Consistent with the adverse tilt in test 4, and with the mean-reversion mechanism: rank
states on trailing mean return and regime 0 tends to mark a local top. But this should not
be traded on. The five tickers are correlated US equities over identical windows, so 5/5 is
nowhere near five independent successes; 0.062 is the floor for n=5 and does not clear
0.05; and rho is ~0 at exactly these horizons, so the rank ordering does not corroborate a
regime-0-specific effect. It is a hypothesis for a future test with independent assets, not
a finding.

---

## 6. Transaction costs, finally modelled

Every result above was originally produced with zero friction, which made them all
optimistic by an unknown amount. `run_backtest` now takes `cost_bps_per_side`
(default 0.0, preserving prior behaviour exactly) and `benchmark_random_entry` takes the
same parameter — **charging only the strategy would hand the benchmark a free edge and
make the comparison meaningless.**

Reference points for liquid US equities: ~1-2 bps per side is optimistic, 5 bps is a fair
central estimate, 10+ bps applies to wider spreads or size. Tradier charges no equity
commission, so this is spread plus slippage.

| bps/side | strategy | random | excess | p | win% | cost paid |
|---|---|---|---|---|---|---|
| 0 | +2.01% | +2.12% | -0.12 | 0.896 | 42% | 0.00 |
| 1 | +1.93% | +2.10% | -0.16 | 0.854 | 40% | 0.07 |
| 2 | +1.86% | +2.07% | -0.21 | 0.814 | 40% | 0.14 |
| **5** | +1.64% | +1.98% | **-0.34** | 0.694 | 35% | 0.35 |
| 10 | +1.27% | +1.84% | -0.57 | 0.510 | 35% | 0.71 |
| 20 | +0.55% | +1.56% | -1.02 | 0.233 | 31% | 1.42 |

Cost of the gross result: **18% at 5 bps, 37% at 10 bps, 73% at 20 bps.** Per ticker, at
10 bps XLF (-0.37%) turns negative and QQQ (+0.05%) rounds to nothing; at 20 bps SPY,
QQQ and XLF are all negative outright.

### Costs do not cancel — turnover is asymmetric

The intuitive expectation is that friction cancels in the comparison, since both sides pay
it. It does not, and the reason is turnover:

| | round trips / 126-bar window | avg hold |
|---|---|---|
| strategy | 3.55 at 9.1% exposure | **3.2 bars** |
| benchmark | 1.15 (fixed 10-bar holds) | 10 bars |

At *matched exposure* the strategy gets there with ~3.1x as many round trips, so it pays
roughly 3.1x the friction for the same time in market. Excess therefore degrades
monotonically with cost, from -0.12 to -1.02 pp. None of these are significant, but the
direction is structural rather than noise: short holds are expensive, and the benchmark's
10-bar holds are not.

This also means the earlier zero-cost results were the strategy's **best case**, and the
comparison was tilted in its favour rather than against it.

---

## 7. Cross-asset features do not rescue it

Tests 4 and 5 ruled out the three core features, all of which are short-horizon
restatements of the target's own volatility and momentum. The obvious follow-up is regimes
built on *macro* state instead. `RegimeDetector` now takes an optional `feature_columns`
argument (defaulting to the core three, so production is unaffected) and
`tools/cross_asset_features.py` compares three sets on identical windows and identical
tests:

| set | features | count |
|---|---|---|
| core | returns, range, volume_change | 3 |
| cross | credit, rates, breadth | 3 |
| both | all of the above | 6 |

Cross-asset series are ETF proxies, because index tickers (`^VIX`, `^TNX`) are not
available from any configured data source:

- **credit** — log(HYG/LQD) change, high-yield vs investment-grade, i.e. credit appetite
- **rates** — TLT return, long-duration Treasuries
- **breadth** — log(RSP/SPY) change, equal- vs cap-weighted S&P, i.e. participation

### Result

| set | n features | mean rho (want negative) | KW p<0.05 rate | vs 5% null | regime 0 > avg |
|---|---|---|---|---|---|
| core | 3 | +0.026 | 4% | p=1.000 | 56% |
| cross | 3 | **+0.000** | 5% | p=1.000 | 52% |
| both | 6 | +0.040 | 10% | p=0.050 | **36%** |

**`cross` is the fair comparison** — same feature count as `core`, so any difference
cannot be attributed to parameter count. It scores a mean rho of exactly 0.000 and a
Kruskal-Wallis significance rate of 5%, which is precisely the null. Macro state, as
encoded by these three proxies, carries no information about this target's forward returns.

`both` is the one ambiguous cell: 10% KW significance against a 5% null, nominally
p=0.050. Three reasons it is not a finding:

1. Three feature sets were compared, so the Bonferroni threshold is 0.017. p=0.050 does
   not clear it.
2. Mean rho is **+0.040** — the wrong sign. Whatever separation exists runs opposite to
   the labelling, so it is not tradeable as built.
3. Regime 0 beat the window average only **36%** of the time, worse than a coin flip and
   consistent with the inverted ordering already seen in tests 4 and 5.
4. It is concentrated in 2 of 5 tickers (SPY 23%, XLF 21%; AAPL and NVDA both 0%).

It also fits 6 features with full covariance on 252 samples, so it is the most
over-parameterized configuration tested.

The consistent story across tests 4, 5 and 7 is that the labelled-bullish regime mildly
*under*performs out of sample, whatever features are used to define it.

---

## 8. The regime ranking rule carries no forward information

The last idea in this repo with a mechanism rather than a hunch behind it.

`RegimeDetector` ranks raw HMM states by the mean return of the bars **during** each state
and calls rank 0 "most bullish". Since the HMM is fitted on return and momentum features,
that is close to tautological: a state scores highest largely because it is *defined* by
prices having risen. Whether the **next** bar is also good is a separate question, and tests
4, 5 and 7 each showed the labelled-bullish regime mildly *under*performing — consistent with
rank 0 marking local tops.

Correction to an earlier note in this file: the ranking uses the **full in-sample mean**
return per state, not a trailing 252-bar window. There is no 252-bar lookback in the ranking
code. The out-of-sample path is nonetheless clean — `filtered_regimes` uses the forward
algorithm and `state_order` is fitted on training data only.

### Setup

Five ranking rules, all scored on training data only, evaluated on causally-labelled OOS
windows (expanding train ≥756 bars, 126-bar OOS, 50 windows across 5 tickers, 6300
observations per rule):

| rule | ranks states by |
|---|---|
| `return` | mean return during the state (**the shipped rule**) |
| `inverted` | the same, ascending — is the rule anti-predictive? |
| `forward_return` | mean return of the **next** bar |
| `forward_sharpe` | next-bar mean / next-bar sd |
| `persistence` | self-transition probability |

A rule passes only if it clears four bars: right direction, negative Spearman rho, KW p below
Bonferroni alpha (0.05/5 = 0.01), and consistency across windows and tickers.

### Result: nothing passes

Unconditional mean forward return is **+7.91 bps/bar**.

| rule | rank0 edge | bullish-set edge | rho | KW p | rank0 beat window mean | verdict |
|---|---|---|---|---|---|---|
| `return` | **-4.58 bps** | +0.07 | -0.003 | 0.496 | 17/37 | no |
| `inverted` | +3.83 | +1.50 | +0.003 | 0.496 | 20/34 | no |
| `forward_return` | +5.01 | -3.25 | +0.024 | 0.228 | 20/36 | no |
| `forward_sharpe` | -4.08 | -2.97 | +0.026 | 0.105 | 19/41 | no |
| `persistence` | +3.49 | +5.41 | -0.028 | 0.174 | 24/43 | no |

The shipped rule's rank 0 does have the **lowest** forward return of all seven states
(+3.3 bps against a +7.91 bps unconditional mean; the best state is +11.7). So the direction
of the suspicion is confirmed. But KW p = 0.50, and the per-ticker spread is enormous — SPY
-29.4 bps, NVDA -15.0, AAPL -4.4, QQQ +5.6, XLF **+60.9**. That is noise with a sign, not an
effect.

Critically, **inverting the rule does not harvest it**: +3.83 bps, p = 0.50, and Spearman rho
flips to the wrong sign. `forward_return` — the theoretically correct rule, ranking states by
what happens *after* them — gets rank 0 to +5.01 bps but has a *negative* bullish-set edge
(-3.25) and p = 0.228. `persistence` has the best window consistency (24/43) and the best
bullish-set edge (+5.41) but p = 0.174.

Effect sizes are all ~3-5 bps/bar against a +7.91 bps/bar baseline and cross-ticker dispersion
an order of magnitude larger. Nothing here survives multiplicity correction.

### Why no strategy-level run followed

This test measures **information content**, which is upstream of any strategy. If forward
returns are not separated by the labels, no entry rule, confirmation count or position sizer
built on those labels can harvest what is not there. Running a walk-forward on a rule that
fails the information test would only measure noise. What would change that: a rule clearing
the Bonferroni bar with a consistent sign across a majority of tickers.

### A trap worth naming

Kruskal-Wallis is invariant to group relabelling, so I expected identical p-values across all
five rules and treated the mismatch as a bug in the tool. It is not. The invariance holds **per
window**; pooled group sizes legitimately differ because each rule promotes a *different* raw
state to id 0 in each window, and pooling mixes them. `return` and `inverted` do match exactly
(0.4963 both), which is the tell — a strict reversal preserves the pooled multiset. The tool now
asserts per-window partition invariance on every run and says so in its output.

Reproduce with `tools/regime_ranking.py`; raw output in `docs/regime_ranking.json`.

### Unrelated fragility found while building this

`GaussianHMM` with full covariance on three features fails to fit on most **synthetic** data.
Of 16 (bars, seed, vol, n_regimes) combinations probed, **3 fit**; the rest raise
`LinAlgError` or "covars must be symmetric, positive-definite". `RegimeDetector` has no
fallback, so a bad fit propagates as an exception rather than a degraded result.

**Scope, measured rather than assumed:** on real data this did not occur once --
`tools/regime_ranking.py` fitted **50 of 50 windows across 5 tickers with zero failures**.
The failures are a property of small, near-degenerate synthetic frames, not of the
production path. The practical consequence is confined to test fixtures, which is why the
ranking tests pin a verified-good fit configuration.

**Correction.** An earlier revision of this section claimed the API does not catch a failed
fit. That was wrong: `api/routes_backtest.py` wraps its handler in `except Exception` and
always did. The real defect was the opposite shape -- it returned the error with an **HTTP
200** status. See "Silent failure modes" below.

---

## 9. Regimes DO separate forward volatility -- the first positive result

`tools/regime_volatility.py`. Ten hypotheses in, this is the only one that came back
positive, and it is worth being precise about how narrow the claim is.

**Why ask.** Tests 4, 5, 7 and 8 all asked whether regimes predict forward *returns*. They do
not. But volatility clustering is among the most robust regularities in empirical finance,
while return predictability is fragile and mostly absent -- and the three features the HMM is
fitted on (returns, high-low range, volume change) are largely volatility proxies. So the
model may well encode a volatility state while failing entirely to encode a return state.

**The trap this test is built around.** A naive version of this experiment passes trivially
and means nothing. Realized vol is strongly autocorrelated -- trailing 20-bar vol alone has
Spearman rho **+0.67** against forward 5-bar vol and **+0.77** at 20 bars, for free, with no
model. Any label correlated with current vol therefore "predicts" forward vol. So every test
here is run twice: raw, and with a **trailing-vol family regressed out** (20-bar, 5-bar and
EWMA log vol, coefficients fitted on training bars only). Only the residual result can
support a claim.

Three further controls, because a bare p-value has already fooled this repo once:

* **Non-overlapping observations.** Forward vol at bar t and bar t+1 share h-1 of their h
  returns, so 6300 overlapping observations are nothing like 6300 independent draws. Keeping
  every h-th bar costs power and buys honesty. This is the control that matters most: the
  overlapping residual test reports p=1.6e-11, which is arithmetically correct and
  scientifically meaningless.
* **Block bootstrap over whole (ticker, window) blocks**, so window-to-window variation is in
  the interval.
* **Within-quintile comparison** of trailing vol, which assumes no functional form at all.

**Result, 5 tickers x 10 windows, 6300 observations, Bonferroni alpha = 0.05/4 = 0.0125:**

| horizon | free benchmark rho | residual KW p (non-overlapping) | epsilon-sq | bear-bull gap | bootstrap 95% CI | tickers | windows | quintiles | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 5 bars | +0.671 | **6.1e-04** | 0.0141 | **+0.136** | **[+0.069, +0.208]** | 5/5 | 31/42 | 20/25 | **YES** |
| 20 bars | +0.775 | 0.258 | 0.0056 | +0.053 | [-0.033, +0.133] | 4/5 | 7/11 | 21/25 | no |

The 5-bar effect clears every bar: significant after decimation, bootstrap CI excluding zero
with P(gap>0) = 1.000, right direction in all five tickers, and positive in 20 of 25
trailing-vol quintiles. The gap is in log units, so **+0.136 means the bearish regime set
runs roughly 14.5% higher forward 5-day volatility** than the bullish set, over and above
what trailing volatility already implied.

**How small it is.** Epsilon-squared of 0.0141 means the regime label explains about **1.4%
of the variance** in residual forward vol. Trailing vol alone explains vastly more. And the
per-ticker spread is wide: SPY +0.191, XLF +0.197, NVDA +0.182, QQQ +0.111, **AAPL +0.005** --
on AAPL the effect is absent, so "5/5 tickers" is directionally true but leans on four.

**The 20-bar null is informative, not just an absence.** Decimating a 20-bar horizon leaves
only 315 observations, so power is genuinely low; but the point estimate also halves. The
honest reading is that this is a short-horizon effect, consistent with vol clustering decaying
over a few days.

**What this does and does not license.** It does not imply tradeable return edge; nine other
tests say there is none, and knowing that volatility will be higher tells you nothing about
sign. What it does license is treating the regime label as a weak *volatility* signal --
relevant to position sizing or option-premium timing, where forward vol is the input that
matters. Anyone building on this should first check whether an EWMA vol estimate does the job
better, since it explains far more variance and needs no HMM.

---

## 10. The capital-preservation claim does not survive benchmarking

`tools/drawdown_benchmark.py`. This is the test that retires the last surviving claim.

**Why ask.** The headline defensive result -- SPY **-4.01%** against buy-and-hold's
**-18.78%** in 2022 at 3-19% exposure -- was one ticker, one year, with no benchmark. And the
question was framed wrongly. Any filter that spends half its time in cash reduces drawdown;
being flat is not a skill. The real question is whether the HMM reduces drawdown *more than
trivial alternatives that cost nothing to build*.

Four comparisons, walk-forward, 50 windows x 5 tickers, exposure at bar t earning bar t+1's
return, 5 bps charged on every exposure change:

| strategy | return | max DD | vol | Sharpe | exposure | turnover | DD saved per unit of return given up |
|---|---|---|---|---|---|---|---|
| hmm | 2.80% | -12.53% | 17.0% | 0.36 | 45% | 49.6 | **0.47** |
| sma200 | **10.08%** | -12.06% | 19.7% | 0.61 | 76% | 3.4 | **2.94** |
| vol_target | 6.29% | **-10.32%** | 15.3% | **0.84** | 71% | 2.7 | 1.19 |
| random_matched | 2.62% | -12.01% | 17.5% | 0.21 | 45% | 56.9 | 0.52 |
| buy_hold | 11.66% | -16.68% | 26.5% | 0.97 | 100% | 1.0 | -- |

`random_matched` is the load-bearing benchmark: flat/long at random with the **same average
exposure as the HMM**, averaged over 200 draws per window so no single lucky sequence decides
it.

**Paired per-window differences, bootstrapped over windows** (positive maxDD difference
favours the HMM):

| comparison | max drawdown | total return | Sharpe |
|---|---|---|---|
| hmm vs buy_hold | **+0.0416** [+0.026, +0.058], better in 76% of windows | **-0.0886** [-0.152, -0.033] | **-0.61** [-0.93, -0.29] |
| hmm vs sma200 | -0.0047 [-0.021, +0.010], n.s. | **-0.0729** [-0.137, -0.015] | -0.25, n.s. |
| hmm vs vol_target | **-0.0221** [-0.045, -0.001] | -0.0349, n.s. | **-0.48** [-0.81, -0.15] |
| hmm vs random_matched | -0.0052 [-0.017, +0.007], n.s. | +0.0018, n.s. | +0.15, n.s. |

**Three conclusions, in descending order of how much they hurt:**

1. **The drawdown reduction is real against buy-and-hold** -- +4.2pp shallower, significant,
   better in 76% of windows. That part of the original claim holds up.
2. **It is indistinguishable from a coin flip with the same exposure budget.** Against
   `random_matched` every metric's interval brackets zero. So the drawdown reduction is what
   being out of the market buys, not evidence that the model knows *when* to be out. A
   smaller constant position size achieves the same thing without 50x the turnover.
3. **Both trivial alternatives dominate it.** A 200-day moving average delivers the same
   drawdown while earning **10.08% against the HMM's 2.80%** -- 6.3x more drawdown saved per
   unit of return surrendered (2.94 vs 0.47), at 3.4 turnover against 49.6. Naive vol
   targeting is *significantly better* on both drawdown and Sharpe. The HMM is elaborate
   machinery losing to one line of pandas.

**Caveats.** Drawdown is measured within 126-bar windows, so these are not multi-year
peak-to-trough figures. The HMM path here is the raw bullish-regime filter, not the full
confirmation-gated strategy -- deliberately, since the question is what the *regime signal*
contributes. And the 200-day MA is itself a filter with well-known weaknesses; the point is
not that it is good, only that it is cheaper and better here.

---

## 11. The volatility signal loses to an exponential average

`tools/vol_forecast_shootout.py`. Test 9 measured the regime label as an increment over a
trailing-vol *control* -- a nuisance regressor whose job was to absorb the obvious. It was
never asked to compete. This asks it to compete.

Six forecasts of forward 5-bar realized vol, all calibrated on training bars only, all scored
on the same 1260 non-overlapping OOS bars. QLIKE is the standard volatility loss (it punishes
under-prediction asymmetrically, as a risk manager would); MSE is on log vol.

| forecast | QLIKE | MSE(log) | R-sq | corr | what it is |
|---|---|---|---|---|---|
| `ewma_hmm` | **0.55334** | 0.23389 | 0.497 | 0.711 | EWMA + per-regime offset |
| `ewma94` | 0.55487 | 0.23955 | 0.484 | 0.696 | free EWMA, lambda=0.94 |
| `combo_vol` | 0.55696 | 0.23799 | 0.488 | 0.699 | three vol models, no regime info |
| `hmm` | 0.56302 | 0.24938 | 0.463 | 0.689 | the regime label alone |
| `trail20` | 0.56916 | 0.24167 | 0.480 | 0.693 | one line of pandas |
| `trail5` | 0.59280 | 0.26242 | 0.435 | 0.662 | one noisier line of pandas |

Paired loss differentials, bootstrapped over (ticker, window) blocks -- **every single
comparison is a tie:**

| comparison | MSE difference | QLIKE difference | verdict |
|---|---|---|---|
| `hmm` vs `ewma94` | +0.0098 [-0.0072, +0.0260] | +0.0082 [-0.0356, +0.0490] | tie |
| `ewma_hmm` vs `ewma94` | -0.0057 [-0.0153, +0.0037] | -0.0015 [-0.0391, +0.0329] | tie |
| `hmm` vs `trail20` | +0.0077 [-0.0072, +0.0220] | -0.0061 [-0.0483, +0.0354] | tie |
| `ewma_hmm` vs `combo_vol` | -0.0041 [-0.0130, +0.0047] | -0.0036 [-0.0343, +0.0280] | tie |

**Two readings, and the difference matters.** The regime label standing alone ranks *fourth of
six*, below a plain EWMA and below a three-model vol combination that uses no regime
information at all -- its point estimates are worse, though not significantly. Adding it on top
of EWMA does improve the point estimate (R-sq 0.484 -> 0.497, MSE -0.0057) but the interval
comfortably brackets zero.

So the strictly correct statement is: **no detectable difference either way**, with 50 blocks
and wide intervals, and a favourable-but-insignificant point estimate for the increment. This
is a null result, not proof of uselessness. What it does establish is that test 9's positive
result buys nothing you cannot get free -- the EWMA already contains it. Test 12 then asks what
happens if you act on the favourable point estimate anyway.

---

## 12. Sizing beats switching -- but the HMM is the wrong input for it

`tools/vol_sizing.py`. Test 10 killed the in/out filter; test 9 said the model's only output is
a vol estimate. Sizing is the obvious remaining use: hold a continuous position inversely
proportional to predicted vol instead of switching between fully invested and flat. All sizers
target 15% annualized vol, cap exposure at 1.0 (no leverage, so nothing can win by quietly
taking more risk), and pay 5 bps on every exposure change.

| strategy | return | max DD | vol | Sharpe | exposure | turnover | ret/vol |
|---|---|---|---|---|---|---|---|
| `size_trail20` | 6.68% | -11.38% | 16.9% | **0.89** | 76% | 2.1 | 0.40 |
| `size_ewma` | 6.58% | -11.54% | 17.0% | 0.88 | 77% | 2.0 | 0.39 |
| `size_hmm` | 5.82% | -11.24% | 16.3% | 0.81 | 75% | 7.9 | 0.36 |
| `size_ewma_hmm` | 6.15% | -11.37% | 16.5% | 0.80 | 76% | 6.3 | 0.37 |
| `hmm_filter` (shipped) | 2.80% | -12.53% | 17.0% | **0.36** | 45% | 49.6 | 0.16 |
| `buy_hold` | 11.66% | -16.68% | 26.5% | 0.97 | 100% | 1.0 | 0.44 |

| comparison | Sharpe | max drawdown | total return |
|---|---|---|---|
| `size_ewma` vs `hmm_filter` | **+0.522** [+0.201, +0.850] | +0.0099, n.s. | **+3.78pp** [+0.26, +7.01] |
| `size_ewma_hmm` vs `size_ewma` | **-0.079** [-0.114, -0.044] | +0.0016, n.s. | **-0.43pp** [-0.81, -0.08] |
| `size_hmm` vs `size_ewma` | **-0.077** [-0.145, -0.005] | +0.0029, n.s. | **-0.76pp** [-1.48, -0.03] |
| `size_ewma` vs `size_trail20` | -0.011, n.s. | -0.0016 [-0.0027, -0.0005] | -0.10pp, n.s. |
| `size_ewma` vs `buy_hold` | **-0.084** [-0.127, -0.043] | **+5.15pp** [+3.27, +7.32] | **-5.08pp** [-11.53, -0.07] |

**Three findings:**

1. **Sizing massively beats the shipped filter.** +0.52 Sharpe, +3.78pp return, and turnover of
   2.0 against 49.6. This is the largest improvement any test in this document has produced --
   and it comes entirely from replacing the regime signal with a volatility estimate.
2. **Adding the regime makes sizing significantly WORSE.** `size_ewma_hmm` loses 0.079 Sharpe
   and 0.43pp of return against plain `size_ewma`, with intervals excluding zero, and is better
   in only 24% of windows. `size_hmm` is worse still. **The mechanism is turnover:** the regime
   offset jitters the position (6.3 turnover vs 2.0) for a forecast improvement too small to
   detect. This is the cleanest lesson in the whole document -- *a statistically undetectable
   forecast gain became a statistically significant performance loss once you had to pay to
   trade on it.* Test 11's favourable point estimate was worth acting on only if free.
3. **Even EWMA is unnecessary.** `size_trail20` -- a 20-bar rolling standard deviation -- ties
   `size_ewma` on Sharpe and is *significantly better* on drawdown.

**And sizing is not free either.** Against buy-and-hold it cuts drawdown by 5.15pp but gives up
5.08pp of return and 0.084 of Sharpe, both significant. It is a risk-reduction tool, not an
improvement in risk-adjusted return, on this sample.

---

## 13. It is not the features. It is the architecture.

`tools/vol_features.py`. The standing objection to tests 9, 11 and 12 is that the HMM was never
given volatility features -- `returns`, `range` and `volume_change` are only incidentally vol
proxies. So: build proper ones and re-run the only question that matters.

`RegimeDetector` already accepts `feature_columns`, so no engine change was needed. Three sets,
each fitted per window and used to add a per-regime offset to the EWMA forecast:

* `baseline` -- returns, range, volume_change (shipped)
* `vol` -- log EWMA vol, log 60-bar vol, vol-of-vol, downside-variance share, normalized range
* `vol_ret` -- the vol set plus returns

All 50 windows fitted successfully for all three sets, so nothing below is a convergence
artifact.

| forecast | QLIKE | MSE | R-sq | vs free EWMA (QLIKE) |
|---|---|---|---|---|
| `ewma94` (no HMM at all) | 0.55487 | 0.23955 | 0.484 | -- |
| `baseline` features | **0.55332** | 0.23388 | **0.497** | -0.0016 [-0.039, +0.033], tie |
| `vol` features | 0.56106 | 0.24549 | 0.472 | +0.0062 [-0.032, +0.041], tie |
| `vol_ret` features | 0.57315 | 0.24078 | 0.482 | +0.0183 [-0.026, +0.057], tie |

**The purpose-built volatility features are worse than the shipped ones**, and neither beats a
free EWMA. Not significantly worse -- every interval brackets zero -- but the ordering is the
opposite of the objection's prediction, across both loss functions.

The reading is architectural rather than about inputs. Forward volatility is continuous, highly
persistent, and already well estimated by an exponential average with one parameter. A 7-state
*discrete* latent variable has to quantize it, and quantization discards exactly the gradations
that make a vol forecast useful. Feeding better volatility measurements into a container that
rounds them to seven buckets does not help, and adding a returns feature (`vol_ret`) makes it
worse by spending states on direction, which tests 4, 5, 7 and 8 established carries no
information at all.

**What would actually be needed** is a different model class -- GARCH, HAR, or simply the EWMA
that keeps winning -- at which point nothing of the HMM remains. That is the finding.

---

## 14. The overlay does not survive its own control either

`tools/vol_overlay_sweep.py`. Test 12 produced the largest improvement in this document: vol
targeting beat the shipped filter by +0.52 Sharpe. It was measured at **one** target vol, one
exposure cap, one cost assumption, and no rebalancing band. A result that exists at a single
point in a four-dimensional space is not a finding, so this scores the same overlay across the
grid -- 5 target vols x 2 caps x 4 cost levels x 3 deadbands x 2 forecasts = 240
configurations, 50 walk-forward windows each. One HMM pass saves the forward returns, the
calibrated forecasts and the filter's exposure path; every configuration is re-scored from
those, so the grid costs one pass rather than 240.

### It does beat the shipped filter, robustly in sign

The overlay's Sharpe exceeds the filter's in **120 of 120** unlevered configurations, is
significantly better in **90 of 120**, and is **never significantly worse**. The advantage
ranges from +0.092 to +1.725, median +0.713.

But the 30 configurations that fail significance are not scattered -- they are *exactly* the 30
zero-cost ones. That is the tell:

| cost charged | `hmm_filter` Sharpe | `buy_hold` Sharpe |
|---|---|---|
| 0 bps | 0.74 | 0.97 |
| 5 bps | **0.36** | 0.97 |
| 10 bps | **-0.02** | 0.96 |
| 20 bps | **-0.78** | 0.95 |

**Most of the overlay's advantage over the filter is not the overlay. It is the filter's
turnover.** Charge the regime filter nothing and it recovers to 0.74; charge it 10 bps and it
stops making money at all; charge it 20 and it destroys capital. Turnover 49.6 against the
overlay's 1.5 does that. The comparison in test 12 was fair -- 5 bps is generous, not harsh --
but the mechanism deserves naming: this is a cost result at least as much as a forecasting one.

### And against the correct control it adds nothing at all

Test 10 established the discipline: a strategy that reduces drawdown by holding less must be
compared against **simply holding less**. So each configuration is scored against a constant
position at its own realized average exposure -- same average risk, no timing whatsoever.

| comparison (120 unlevered configs) | significantly better | significantly worse | median diff |
|---|---|---|---|
| max drawdown vs constant-exposure | **0 / 120** | 51 / 120 | **-0.17pp (deeper)** |
| max drawdown vs buy-and-hold | 120 / 120 | 0 / 120 | +5.28pp |
| Sharpe vs buy-and-hold | 0 / 120 | 100 / 120 | -0.072 |

**The overlay never produces a shallower drawdown than a constant position holding the same
average exposure -- in any of 240 configurations -- and in 51 of them it is significantly
deeper.** Every bit of the 5.28pp drawdown reduction against buy-and-hold is bought by holding
less on average. None of it comes from knowing *when* to hold less. This is the same verdict
test 10 delivered on the regime filter, reached independently, on the thing that was supposed
to replace it.

*(A note on the Sharpe row of that control: Sharpe is scale-invariant, so a constant-exposure
control has the same Sharpe as buy-and-hold by construction, up to cost drag. That row is
identical to the buy-and-hold row and carries no independent information. Drawdown is not
scale-invariant, which is why it is the informative comparison. The tool prints this caveat
itself so the row cannot be mistaken for a second piece of evidence.)*

### The sensitivity table says the same thing a third way

| target vol | Sharpe | return | max DD | turnover |
|---|---|---|---|---|
| 10% | 0.83 | 4.56% | -8.34% | 1.6 |
| 12% | 0.84 | 5.39% | -9.75% | 1.6 |
| 15% | 0.89 | 6.60% | -11.42% | 1.6 |
| 20% | 0.92 | 8.04% | -13.33% | 1.4 |
| 25% | **0.95** | 9.13% | -14.43% | 1.3 |

Sharpe rises monotonically with target vol, and the only 20 configurations that are
statistically indistinguishable from buy-and-hold are the highest-exposure ones (20-25%
target). **The overlay gets better the more it resembles doing nothing.** Extrapolate the
column and the optimum is buy-and-hold, which is exactly what the constant-exposure control
already said.

Two minor positives, recorded because they are real even though the headline is negative:

* **Cost decay is mild.** Mean Sharpe falls only 0.90 -> 0.87 across 0 to 20 bps. At turnover
  1.5 the overlay is genuinely cheap to run, unlike the filter.
* **A rebalancing band is free.** A 10pp deadband cuts turnover from 1.9 to 1.2 and *slightly
  improves* Sharpe (0.88 -> 0.89). Anyone building a vol overlay for other reasons should use
  one.

### Recommendation

**Do not ship the overlay as a replacement for the regime filter, and do not change any
default.** It is not an edge: it loses to buy-and-hold on Sharpe in 100 of 120 configurations
and beats it in none. It is not even a better *de-risking* tool than the trivial alternative of
holding a constant smaller position. The honest description is "a way to hold less stock, with
extra steps."

What test 14 does establish, and this is worth stating plainly: **the case for the shipped
regime filter is now weaker than the case for doing nothing at all.** At realistic costs it has
a negative Sharpe. That is a live recommendation to remove it from the default path -- which is
a decision about the product, not a research finding, and so is left to the maintainer.

---

## Reporting fixes (not hypotheses)

Two defects made reported performance wrong rather than merely optimistic. Both are fixed;
neither changes any conclusion above, because every result above was produced on daily bars
via `walk_forward`, which was already interval-aware for its benchmarks.

### Sharpe was annualized with a hardcoded 252

`backtester._compute_metrics` and `strategy_v2._compute_metrics_v2` both multiplied by
`sqrt(252)` regardless of bar interval. `data_loader.fetch_data` defaults to **hourly**
bars, where the correct factor is `252 * 6.5 = 1638`, so Sharpe was understated by
`sqrt(1638/252) = 2.55x` on the default path. Worse, `walk_forward`'s *benchmark* Sharpe
already used the interval-aware `periods_per_year()`, so strategy and benchmark Sharpe were
computed with different factors and were never comparable.

`periods_per_year` now lives in `backtester.py` (walk_forward re-exports it) and
`run_backtest`/`run_backtest_v2` take a `periods_per_year` argument. The default stays 252
so daily-bar results are unchanged; `walk_forward` passes its own `self.ppy`.

### v2 ignored transaction costs, and the benchmark went uncharged

- `strategy_v2` — the API's **default** strategy — had no cost model at all. It now mirrors
  backtester's, charging `cost_frac` against compounding capital on entry and exit. Charging
  only the trade rows would have left `total_return_pct`, the number the dashboard shows,
  cost-free. On SPY 2020-2026 daily, 37 trades: **+12.07% gross → +8.00% at 5 bps → +4.07%
  at 10 bps → -3.36% at 20 bps.** Friction takes **34% of gross at 5 bps** on this path,
  roughly double the 18% the v1 walk-forward path pays.
- `WalkForwardEngine` previously had no cost parameter; the only way to charge the strategy
  was to pass `cost_bps_per_side` inside `backtest_kwargs`, which left
  `benchmark_random_entry` **cost-free**. The strategy paid friction its own control did
  not, biasing the comparison *against* the strategy. A single `cost_bps_per_side` argument
  now feeds both sides, and an explicit `backtest_kwargs` entry still wins.

Verified end to end on SPY daily, 7 regimes: excess return over matched random entry moves
from **+0.04% at 0 bps to -0.44% at 5 bps**, and strategy Sharpe from +1.08 to +0.60.

### Defaults

| path | cost default | rationale |
|---|---|---|
| `run_backtest`, `run_backtest_v2` | 0.0 | library primitives; keeps existing results reproducible |
| `WalkForwardEngine(...)` | 0.0 | same, for programmatic callers |
| `walk_forward.py` CLI `--cost-bps` | **5.0** | anything a human reads should not be cost-free |
| `GET /backtest/{symbol}` `cost_bps` | **5.0** | same; pass `cost_bps=0` for the old numbers |

---

## Known measurement bugs found along the way

1. **Annualization** — `data_loader.fetch_data` defaults to **hourly** bars, but
   `walk_forward` annualized with a hardcoded 252, overstating benchmark Sharpe by
   √6.5 ≈ 2.5x. Fixed with interval-aware `periods_per_year()`.
2. ~~`backtester.py` hardcodes `np.sqrt(252)` for the strategy's own `sharpe_ratio`.~~
   **Fixed** — `run_backtest`/`run_backtest_v2` now take `periods_per_year`, and
   `walk_forward` passes its own factor so strategy and benchmark match. See
   "Reporting fixes" above.
3. **A regression I introduced.** The commit "Derive bullish/bearish regime sets from
   n_regimes" added `n_regimes=` to *both* call sites in `api/routes_backtest.py` but added
   the parameter only to `run_backtest`. `strategy_v2.run_backtest_v2` never accepted it,
   and v2 is that route's **default** strategy -- so `GET /backtest/{symbol}` raised
   `TypeError: unexpected keyword argument 'n_regimes'` on every default request from that
   commit until it was fixed. No test exercised the route's kwargs against the engines'
   signatures, so nothing caught it.

   Fixing it exposed a second defect underneath: v2's fallbacks were hardcoded
   `[0, 1, 2]` bullish / `[5, 6]` bearish, correct only at 7 regimes. `regime_sets(3)`
   returns bearish `[2]` and `regime_sets(4)` returns `[3]`, so below 7 the hardcoded set
   matched **no state at all** -- nothing was ever bearish and the regime-flip exit was
   unreachable. This is the same bug that commit fixed for `run_backtest`; v2 was missed.
4. **`strategy_v2`'s roll credits** — now measured, see "Roll credits" below. A median
   **41%** of v2's reported return comes from them.
5. ~~**Latent:** `regime_sets(1)` returns an **empty bullish list**.~~ **Fixed.** Any frame
   whose `regime_id` is constant makes `run_backtest_v2` infer `n_regimes=1`, so no entry
   could ever fire and the backtest silently returned zero trades rather than erroring. This
   produced a set of vacuously-passing tests before it was noticed. `run_backtest_v2` now
   raises `ValueError` when the resolved bullish set is empty. See "Silent failure modes".
6. ~~**Dormant:** `hv_20` annualized with a hardcoded `sqrt(252)`.~~ **Fixed.** Worth being
   precise about severity, because it was initially overstated: this produced **no wrong
   numbers**. Its only two consumers were `hv_rank`, a rolling percentile and therefore
   scale-invariant, and `strategy_v2._sigma`, which compensated explicitly. It was a trap
   for the next consumer, not a live defect. Now annualized with the actual bar frequency at
   the source, with the compensation in `_sigma` removed so nothing scales twice. Daily
   results are bit-identical, as verified against `main`.

---

## Silent failure modes

Three defects that shared a shape: **the system reported success while failing.** None
changed a published number; all three could have hidden a future one.

### A backtest that could not trade returned 0.00% instead of erroring

`regime_sets(1)` has an empty bullish list, so a constant-`regime_id` frame made
`run_backtest_v2` infer `n_regimes=1`, enter nothing, and return `trades: 0`,
`total_return_pct: 0` — indistinguishable from a strategy that legitimately stayed in cash.
Four tests in #24 passed vacuously this way. `run_backtest_v2` now raises `ValueError`
naming the cause and the fix. Explicit `bullish_regimes=[...]` still bypasses the guard, so
deliberate single-regime experiments remain possible.

### Failed API requests returned HTTP 200

Route handlers caught every exception and returned `{"error": str(e)}`, which FastAPI
serializes with a **200** status. Any client checking `response.ok` or `status_code == 200`
read a failed backtest as a successful one; only a client that inspected the body for an
`"error"` key would notice. Six sites now return truthful codes via `api/errors.py` — 500
for internal errors, 502 for upstream/data-provider failures, 404 for a missing order —
while keeping the `"error"` body key so existing front-end checks keep working.

Three sites were left as 200 deliberately: an empty scan (`"No tickers to scan"` with
`results: []`) is an empty success, not an error, and `"Tradier not configured"` is a
configuration state the UI drives a flow from.

### hv_20's annualization

Documented under "Known measurement bugs" item 6. Dormant, now fixed at the source.

---

## Roll credits: a median 41% of v2's reported return

`strategy_v2` is the default strategy behind `GET /backtest/{symbol}`, so its numbers are
what the dashboard shows. It hands out two credits not derived from any option pricing:

| signal | credit | trigger |
|---|---|---|
| `ROLL_UP` | +0.5% of stock price | `price >= effective_entry + 1 ATR`, still bullish |
| `ROLL_OUT` | +0.3% of stock price | past the time stop **at a loss**, still bullish |

up to 3 per trade. Neither depends on strike, moneyness, implied vol, or time to expiry,
and nothing is surrendered in exchange. `tools/roll_credit_sensitivity.py` measures four
configurations on identical data (5 bps/side, 7 regimes, ~2060 daily bars).

### Result

| ticker | as shipped | credits off | delta | credit share | rolls | buy & hold |
|---|---|---|---|---|---|---|
| SPY | +129.75% | +68.64% | 61.11 pp | **47%** | 62 | +200.55% |
| QQQ | +66.76% | +39.35% | 27.41 pp | **41%** | 36 | +320.27% |
| NVDA | +1074.08% | +753.24% | 320.84 pp | **30%** | 64 | +3209.06% |
| AAPL | +36.07% | +19.52% | 16.55 pp | **46%** | 26 | +504.77% |
| XLF | +40.84% | +30.04% | 10.80 pp | **26%** | 16 | +138.80% |

Mean return falls **+269.4% → +182.1%** when the credits are switched off. That is 32% of
the mean, but the mean is dominated by NVDA's compounding; the **per-ticker median is 41%**,
ranging 26-47%. Every ticker is affected.

### The wrong-signed credit never fires

`ROLL_OUT` has the wrong sign — rolling a long call to a later expiry buys time value and
costs a **debit**, and it fires only on a losing position, so it pays cash *and* defers
realizing the loss. In practice it is **unreachable**: setting
`roll_out_credit_pct` to +0.3, +99 or -99 gives byte-identical results, so all credits come
from `ROLL_UP`. The reason is exit ordering — the regime-flip check precedes the time-stop
roll attempt, and 85 of 97 SPY exits are regime flips (mean hold 5.5 bars), so positions
close before they can qualify. The sign error is inert, not merely small. A characterization
test pins this down so it starts failing if a future change makes the path reachable.

`ROLL_UP` has the right sign — closing a long call and reopening higher genuinely collects
a credit — but the model keeps compounding **full capital** on the underlying afterwards. A
real roll up cuts delta, so upside participation should fall, and here it never does. The
credit is collected with no offsetting cost.

### The larger finding

**Buy-and-hold beats v2 on 5 of 5 tickers, with or without the credits.** NVDA returns
+1074% against +3209% for simply holding. A +269% mean looks impressive in isolation, and
roughly two fifths of it is fabricated, and even the gross figure loses to doing nothing on
every name tested.

These are in-sample full-history backtests, the same basis the API uses — the right basis
for "how much of the displayed number is this", but **not** out-of-sample evidence. They say
nothing about whether the strategy works; test 4 already answered that.

Raw output: `docs/roll_credit.json`.

### Fixed: the legs are now priced

`roll_up_credit_pct` and `roll_out_credit_pct` now default to **0**, and
`roll_model="priced"` is the default. `options_pricing.py` (Black-Scholes, no scipy) prices
both legs of every roll:

* **Roll up** — sell the held call, buy the same contract count struck back at `itm_depth`
  below spot. The difference is real cash to the cash side; the new leg carries less delta.
  Cash and lost participation are the same transaction, so no return is manufactured.
* **Roll out** — same strike, more time. That is a **debit**, and the sign now comes out of
  the pricing rather than being asserted.
* Both legs pay `cost_bps_per_side`. Rolls used to be free.
* Positions are sized so **dollar delta equals capital**, balance in cash. Spending all
  capital on premium is ~5x levered at these strikes, and comparing a 5x book to
  buy-and-hold would flatter the strategy for reasons unrelated to the signal.

| config | mean return | mean Sharpe | mean maxDD | beat B&H |
|---|---|---|---|---|
| `legacy_flat` (old default) | +269.50% | 1.06 | -12.23% | 0/5 |
| `flat_zero` (credits off, stock accounting) | +182.16% | 0.82 | -12.98% | 0/5 |
| `priced_norolls` (priced legs, no rolling) | +153.79% | 0.71 | -14.00% | 0/5 |
| **`priced` (new default)** | **+129.04%** | **0.68** | **-13.69%** | **0/5** |

Per ticker under the priced model: SPY +53.81% (B&H +200.55%), QQQ +31.92% (+320.27%),
NVDA +516.78% (+3209.07%), AAPL +22.81% (+504.77%), XLF +19.88% (+138.80%).

**52% of the legacy headline was roll-model artifact** — +269.50% → +129.04%. The flat credit
accounted for 87 pp of that; the remaining ~53 pp was never charging theta or the delta given
up at each roll. Sharpe falls 1.06 → 0.68.

Rolling is now worth **-24.75 pp** rather than a large positive: taking cash off the table
caps participation in the very move that triggered the roll. That is the correct sign, and it
means the roll rule as designed is a drag in a trending market.

Buy-and-hold still wins 5/5 under every configuration. Reproduce with
`tools/roll_model_compare.py`; raw output in `docs/roll_model.json`.

Remaining simplifications, stated so nobody over-reads this: one vol input (rescaled
`hv_20`) rather than a real surface, no smile or skew, no early exercise, no dividends, no
interest earned on the cash side, and IV tracks trailing realized vol. These bias the result
*optimistically* on the vol path but are second-order next to the effects above.

---

## What has not been tested

- **Better features.** Tests 4, 5 and 7 rule out the core three *and* credit/rates/breadth
  proxies; test 13 now rules out a purpose-built volatility feature set as well -- it performed
  *worse* than the shipped features. Still untested: real index data (`^VIX`, `^TNX`), which no
  configured source provides. Given test 13's result, this is no longer a promising direction:
  the constraint appears to be the discrete 7-state architecture, not the inputs.
- **A different model class.** Every test here measures *this* HMM. The repeated finding that a
  one-parameter EWMA matches or beats it at its only surviving task points at GARCH, HAR, or a
  plain rolling standard deviation. None has been tried as a *replacement* rather than a
  benchmark, and on the evidence one of them should be.
- ~~**Vol targeting as a product in its own right.**~~ Closed by test 14: stress-tested across
  240 configurations, it is robust in sign against the shipped filter but adds nothing over a
  constant position at the same average exposure, and never beats buy-and-hold on Sharpe.
- **Whether the default path should include the regime filter at all.** Test 14 shows it has a
  negative Sharpe at 10 bps and worse at 20. Nothing here has measured the *product* decision
  of removing it, and no default has been changed.
- **More windows.** 10-14 per ticker is too few for ~1% effects. Shorter OOS steps, more
  tickers, or overlapping windows with corrected standard errors would all help.
- **Realistic slippage on the entry bar.** Costs are charged as a flat bps figure on the
  close. Real fills on a regime flip may be worse than that, and the 3.2-bar average hold
  makes the strategy unusually sensitive to it.

## How to reproduce

```bash
# Does the regime labeling predict anything out-of-sample?
.venv/bin/python tools/regime_separability.py --regimes auto

# ...at any holding horizon?
.venv/bin/python tools/regime_horizons.py --regimes auto --horizons 1,5,10,20

# Do cross-asset features help?
.venv/bin/python tools/cross_asset_features.py --regimes auto

# How much of v2's return is the unconditional roll credit?
.venv/bin/python tools/roll_credit_sensitivity.py

# What changes once the roll legs are actually priced?
.venv/bin/python tools/roll_model_compare.py

# What do transaction costs do to all of the above?
.venv/bin/python tools/cost_sensitivity.py --regimes 7 --costs 0,1,2,5,10,20

# Do regimes separate forward VOLATILITY? (test 9 -- the one positive result)
for T in SPY QQQ NVDA AAPL XLF; do
  python tools/regime_volatility.py --tickers $T --save-obs volobs/$T.csv
done
python tools/regime_volatility.py --from-obs 'volobs/*.csv' --json docs/regime_volatility.json

# Does the capital preservation beat a 200-day MA or a coin flip? (test 10)
for T in SPY QQQ NVDA AAPL XLF; do
  python tools/drawdown_benchmark.py --tickers $T --save-obs ddobs/$T.csv
done
python tools/drawdown_benchmark.py --from-obs 'ddobs/*.csv' --json docs/drawdown_benchmark.json

# Does the vol signal beat a free EWMA? (test 11)
for T in SPY QQQ NVDA AAPL XLF; do
  python tools/vol_forecast_shootout.py --tickers $T --save-obs vfobs/$T.csv
done
python tools/vol_forecast_shootout.py --from-obs 'vfobs/*.csv' --json docs/vol_forecast_shootout.json

# Does sizing off predicted vol rescue it? (test 12)
for T in SPY QQQ NVDA AAPL XLF; do
  python tools/vol_sizing.py --tickers $T --save-obs vsobs/$T.csv
done
python tools/vol_sizing.py --from-obs 'vsobs/*.csv' --json docs/vol_sizing.json

# Would purpose-built volatility features fix it? (test 13)
for T in SPY QQQ NVDA AAPL XLF; do
  python tools/vol_features.py --tickers $T --save-obs fsobs/$T.csv
done
python tools/vol_features.py --from-obs 'fsobs/*.csv' --json docs/vol_features.json

# Is the vol overlay robust, and does it beat its own control? (test 14)
for T in SPY QQQ NVDA AAPL XLF; do
  python tools/vol_overlay_sweep.py --tickers $T --save-obs swobs/$T.csv
done
python tools/vol_overlay_sweep.py --from-obs 'swobs/*.csv' --json docs/vol_overlay_sweep.json

# Baseline walk-forward on one ticker
.venv/bin/python walk_forward.py SPY --interval 1d --period-days 3000 --regimes 7

# Entry-filter sweep with tune/verify split
.venv/bin/python tools/confidence_sweep.py \
  --tickers SPY,QQQ,NVDA,AAPL,XLF \
  --confidences 0.4,0.5,0.6,0.7,0.8,0.9 \
  --confirmations 3,4,5,6,7
```
