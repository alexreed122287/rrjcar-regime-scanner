# Bottoming Strategy + Per-Strategy Recommended Defaults — Design

**Date:** 2026-04-21
**Status:** Approved for implementation planning
**Scope:** Two coupled features delivered in one spec.

---

## 1. Summary

Add two features to `rrjcar-regime-scanner`:

1. **`strategy_bottoming.py`** — a new 12-confirmation strategy module alongside V1 (`backtester.py`), V2 (`strategy_v2.py`), and LEAPS (`strategy_leaps.py`). Targets "bottoming stocks" — previously-beaten-down names that have begun recovering and are breaking out of a base. Based on a Minervini-style base-and-breakout pattern (B) with a trend-reclaim overlay (C).
2. **Per-strategy recommended defaults** — each strategy module exports a `RECOMMENDED_SETTINGS` constant. A new `GET /api/strategy-defaults` endpoint aggregates them. A new button under the Strategy dropdown in `index.html` fetches the defaults for the currently-selected strategy and applies them to the settings form.

These features are coupled because the new strategy needs its own defaults as part of the same UX affordance.

---

## 2. Non-goals

- **No new backtester.** The strategy ships as a live-signal module; historical backtesting is a follow-up.
- **No changes to `options_picker.py`.** The existing picker works for any long-call strategy and can be reused unchanged.
- **No new alerting.** Existing alert infrastructure will route `BOTTOM — BUY` / `BOTTOM — WATCH` signals automatically by name.
- **No "user-saved presets" (MY defaults).** The button only loads the module-provided recommended defaults. Users can still use `Save Settings` to persist their own customized values.
- **No changes to HMM regime detection.** The new strategy consumes existing `regime_id` / `regime_label` columns.

---

## 3. The Bottoming strategy (`strategy_bottoming.py`)

### 3.1 Design intent

Identify stocks that meet **all** of:
- Were meaningfully beaten down (≥35% off 52-week high).
- Have already begun recovering (≥15% off 52-week low).
- Have formed a tight base (low-volatility consolidation with higher lows).
- Are breaking out of that base on above-average volume.
- Show trend-reclaim evidence (above 50 EMA, stacked short-term EMAs, rising MACD).

HMM regime is treated as a **confidence tier, not a gate** (per design Q2) — the strategy fires regardless of regime, but labels the signal `BUY` vs `WATCH` based on regime quality, and explicitly blocks `BUY` when the regime is `Crash / Capitulation`.

### 3.2 The 12 confirmations

Grouped into four layers. Layer 1 is a **hard gate**: if either signal fails, the output is `BOTTOM — N/A` (the ticker isn't a bottoming candidate).

**Layer 1 — Drawdown gate (2 signals, both required):**

| # | Column | Logic |
|---|---|---|
| 1 | `conf_01_drawdown_depth` | Close ≤ 65% of trailing 252-bar high (≥35% off 52w high) |
| 2 | `conf_02_off_lows`       | Close ≥ 115% of trailing 252-bar low (≥15% off 52w low) |

**Layer 2 — Base formation (4 signals):**

| # | Column | Logic |
|---|---|---|
| 3 | `conf_03_tight_base`        | 20-day ATR / Close ≤ 0.08 |
| 4 | `conf_04_range_contraction` | (20-day High − 20-day Low) / Close < 0.20 |
| 5 | `conf_05_higher_lows`       | 10-day rolling Low > 10-day rolling Low shifted 10 bars |
| 6 | `conf_06_volume_dryup`      | 5-day avg Volume < 20-day avg Volume |

**Layer 3 — Breakout trigger (3 signals):**

| # | Column | Logic |
|---|---|---|
| 7 | `conf_07_breakout_day` | Close > prior 20-day High |
| 8 | `conf_08_volume_surge` | Today's Volume > 1.5 × 20-day avg Volume |
| 9 | `conf_09_strong_close` | Close > (High + Low) / 2 (upper half of day's range) |

**Layer 4 — Trend reclaim overlay (3 signals):**

| # | Column | Logic |
|---|---|---|
| 10 | `conf_10_above_50ema` | Close > 50 EMA |
| 11 | `conf_11_ema_stack`   | 10 EMA > 20 EMA |
| 12 | `conf_12_macd_rising` | MACD histogram > 0 AND MACD histogram > prior bar |

All signals are computed as vectorized pandas operations on the OHLCV DataFrame returned by `data_loader.fetch_data` / `engineer_features`. No new dependencies.

### 3.3 Signal output logic

```
if NOT (conf_01 AND conf_02):
    → "BOTTOM — N/A"           # not a bottoming candidate

confs = sum(conf_01 ... conf_12)   # includes gate signals

regime_label = latest regime label from HMM

if regime_label == "Crash / Capitulation":
    → "BOTTOM — AVOID"         # never catch falling knives in crash
elif confs >= 9 AND regime_label ∈ {Bull Run, Bull Trend, Mild Bull, Neutral / Chop}:
    → "BOTTOM — BUY"           # high confidence: setup passes + regime supports
elif confs >= min_confirmations:
    → "BOTTOM — WATCH"         # setup passes but regime is weak (or confs 8)
else:
    → "BOTTOM — WAIT"          # below threshold
```

Default `min_confirmations = 8` (mirrors V2's 8/12 ratio). Adjustable via the `Min Confs` UI field.

### 3.4 Public interface

Mirrors `strategy_leaps.get_current_signal_leaps` signature:

```python
def compute_bottoming_confirmations(df: pd.DataFrame) -> pd.DataFrame: ...

def get_current_signal_bottoming(
    df: pd.DataFrame,
    min_confirmations: int = 8,
    regime_confirm_bars: int = 2,
) -> dict:
    """
    Returns:
        {
          "signal": str,                    # e.g. "BOTTOM -- BUY"
          "action": str,                    # human-readable description
          "regime_id": int,
          "regime_label": str,
          "confidence": float,              # from HMM regime_confidence
          "confirmations_met": int,
          "confirmations_required": int,
          "confirmations_total": 12,
          "confirmation_detail": dict[str, bool],
          "price": float,
          "pct_off_52w_high": float,
          "pct_off_52w_low": float,
          ...                               # other standard fields matching V2/LEAPS
        }
    """
```

### 3.5 Edge cases

- **Fewer than 252 bars of history:** treat `conf_01` and `conf_02` as False (can't compute 52w high/low reliably) → returns `BOTTOM — N/A`.
- **Healthcare / biotech sectors:** excluded upstream by existing `_is_excluded_sector_or_industry` filter in `screener.py`. No change needed.
- **NaN confirmations from insufficient rolling-window data:** fill with False before aggregation (matches pattern in `strategy_leaps.py`).

---

## 4. Per-strategy recommended defaults

### 4.1 `RECOMMENDED_SETTINGS` constant

Each of the four strategy modules exports a flat dict with the same key set. Missing keys mean "don't touch this UI field."

**`backtester.py` — V1 (8-conf momentum):**
```python
RECOMMENDED_SETTINGS = {
    "min_confs": 7,
    "regime_confirm": 2,
    "cooldown": 48,
    "min_dte": 14,
    "max_dte": 45,
    "min_avg_volume": 1_000_000,
    "min_price": 5,
    "max_price": None,
    "price_above_ema50": True,
    "ema10_above_20": False,
}
```

**`strategy_v2.py` — V2 (12-conf call-optimized pullback):**
```python
RECOMMENDED_SETTINGS = {
    "min_confs": 8,
    "regime_confirm": 2,
    "cooldown": 3,
    "min_dte": 30,
    "max_dte": 60,
    "min_avg_volume": 500_000,
    "min_price": 10,
    "max_price": 1000,
    "price_above_ema50": False,
    "ema10_above_20": False,
}
```

**`strategy_leaps.py` — LEAPS (10-conf long-dated):**
```python
RECOMMENDED_SETTINGS = {
    "min_confs": 7,
    "regime_confirm": 3,
    "cooldown": 10,
    "min_dte": 270,
    "max_dte": 540,
    "min_avg_volume": 1_000_000,
    "min_price": 20,
    "max_price": None,
    "price_above_ema50": True,
    "ema10_above_20": False,
}
```

**`strategy_bottoming.py` — Bottoming (12-conf):**
```python
RECOMMENDED_SETTINGS = {
    "min_confs": 8,
    "regime_confirm": 2,
    "cooldown": 5,
    "min_dte": 30,
    "max_dte": 60,
    "min_avg_volume": 500_000,
    "min_price": 10,
    "max_price": None,
    "price_above_ema50": False,   # implicit via conf_10
    "ema10_above_20": False,      # implicit via conf_11
}
```

### 4.2 API endpoint

New route added to `api/routes_settings.py` (the app uses FastAPI routers split by concern — `app.py` only wires routers together):

```python
# In api/routes_settings.py

@router.get("/strategy-defaults")
async def get_strategy_defaults():
    from backtester import RECOMMENDED_SETTINGS as V1_DEFAULTS
    from strategy_v2 import RECOMMENDED_SETTINGS as V2_DEFAULTS
    from strategy_leaps import RECOMMENDED_SETTINGS as LEAPS_DEFAULTS
    from strategy_bottoming import RECOMMENDED_SETTINGS as BOTTOMING_DEFAULTS
    return {
        "v1": V1_DEFAULTS,
        "v2": V2_DEFAULTS,
        "leaps": LEAPS_DEFAULTS,
        "bottoming": BOTTOMING_DEFAULTS,
    }
```

The `/api` prefix is applied by `app.py`'s `include_router(settings_router, prefix="/api")`, so the final path is `/api/strategy-defaults`.

Response is static per deploy. Safe to cache client-side for the session.

### 4.3 `settings_manager.py` changes

Extend `DEFAULT_SETTINGS` to cover fields previously only persisted by the client:

```python
# New keys added to DEFAULT_SETTINGS:
"min_avg_volume": 500_000,
"min_price": 1,
"max_price": None,
"price_above_ema50": False,
"ema10_above_20": False,
```

`load_settings`'s existing merge behavior ensures backward compatibility with old `.dashboard_settings.json` files (they simply don't include the new keys and fall back to defaults).

### 4.4 `api/routes_settings.py` — extend `SettingsUpdate` Pydantic model

The existing `POST /api/settings` endpoint validates incoming bodies through `SettingsUpdate`. To allow the new fields to be persisted when the user clicks Save Settings, add the following optional fields:

```python
class SettingsUpdate(BaseModel):
    # ... existing fields ...
    min_avg_volume: Optional[int] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    price_above_ema50: Optional[bool] = None
    ema10_above_20: Optional[bool] = None
```

Without this change, the new fields would be silently dropped by `exclude_none=True` during `model_dump`.

### 4.5 `screener.py` dispatch

Add the new strategy to the dispatch block in `_scan_single_ticker`:

```python
from strategy_bottoming import get_current_signal_bottoming
...
elif strategy == "bottoming":
    signal_data = get_current_signal_bottoming(
        regime_df,
        min_confirmations=min_confirmations,
        regime_confirm_bars=regime_confirm_bars,
    )
```

---

## 5. Frontend wiring

### 5.1 `index.html` changes

**Add strategy option** (in `#setting-strategy`):
```html
<option value="bottoming">Bottoming (12-conf)</option>
```

**Add button** directly below the dropdown's `.setting-group`:
```html
<button type="button"
        id="btn-load-strategy-defaults"
        class="btn-load-defaults"
        onclick="Settings.loadRecommendedDefaults()"
        title="Overwrite current settings with recommended defaults for the selected strategy">
    ↻ Load recommended defaults
</button>
```

**Add signal filter options** in `#filter-signal`:
```html
<option value="BOTTOM -- BUY">BOTTOM BUY</option>
<option value="BOTTOM -- WATCH">BOTTOM WATCH</option>
```

CSS for `.btn-load-defaults` follows existing `.btn-scan-go btn-copy-grid` pattern — small, teal accent, not a primary action.

### 5.2 `js/api.js` addition

```js
async getStrategyDefaults() {
    if (this._strategyDefaultsCache) return this._strategyDefaultsCache;
    const res = await fetch('/api/strategy-defaults');
    if (!res.ok) throw new Error(`strategy-defaults ${res.status}`);
    this._strategyDefaultsCache = await res.json();
    return this._strategyDefaultsCache;
}
```

Cached in-memory for the session — defaults are static per deploy.

### 5.3 `js/settings.js` additions

```js
async loadRecommendedDefaults() {
    const strategy = this.getVal('setting-strategy');
    if (!strategy) return;
    try {
        const allDefaults = await API.getStrategyDefaults();
        const defaults = allDefaults[strategy];
        if (!defaults) return;

        const FIELD_MAP = {
            min_confs:       'setting-min-confs',
            regime_confirm:  'setting-regime-confirm',
            cooldown:        'setting-cooldown',
            min_dte:         'setting-min-dte',
            max_dte:         'setting-max-dte',
            min_avg_volume:  'filter-min-volume',
            min_price:       'filter-min-price',
            max_price:       'filter-max-price',
        };
        const TOGGLE_MAP = {
            price_above_ema50: 'filter-price-above-ema50',
            ema10_above_20:    'filter-ema10-above-20',
        };

        Object.entries(FIELD_MAP).forEach(([key, id]) => {
            if (!(key in defaults)) return;
            const val = defaults[key];
            this.setVal(id, val === null ? '' : val);
        });

        Object.entries(TOGGLE_MAP).forEach(([key, id]) => {
            if (!(key in defaults)) return;
            const el = document.getElementById(id);
            if (el) el.checked = !!defaults[key];
        });

        // Re-run client-side filter pipeline (filter-* fields skip 'change' when set programmatically)
        if (window.Screener?.render) {
            Screener.minPrice        = parseFloat(this.getVal('filter-min-price')) || 0;
            Screener.maxPrice        = parseFloat(this.getVal('filter-max-price')) || 0;
            Screener.minVolume       = parseFloat(this.getVal('filter-min-volume')) || 0;
            Screener.priceAboveEma50 = document.getElementById('filter-price-above-ema50').checked;
            Screener.ema10Above20    = document.getElementById('filter-ema10-above-20').checked;
            Screener.render(Screener.results, document.getElementById('screener-content'));
        }

        this.flashButton('btn-load-strategy-defaults', 'Loaded');
    } catch (err) {
        console.error('Load recommended defaults failed:', err);
    }
},

flashButton(id, msg) {
    const btn = document.getElementById(id);
    if (!btn) return;
    const orig = btn.textContent;
    btn.textContent = msg;
    setTimeout(() => { btn.textContent = orig; }, 1500);
},
```

### 5.4 UX flow

1. User selects a strategy from the dropdown.
2. User clicks `↻ Load recommended defaults`.
3. Fields populate; button flashes `Loaded` for 1.5 seconds.
4. User can still tweak anything before clicking Scan or Save Settings.
5. If the user never clicks the button, the app behaves exactly as before.

---

## 6. Error handling

| Failure mode | Behavior |
|---|---|
| `/api/strategy-defaults` fetch fails | Log error to console. No field changes. Button stays normal. |
| Strategy key missing from response | Early `return` in `loadRecommendedDefaults`. No-op. |
| Strategy module fails to import on backend startup | `app.py` endpoint returns 500. Frontend falls through to fetch-fail path. Caught at deploy time. |
| User runs scan with unknown `strategy` value | **Behavior change:** `screener.py` dispatch will raise a clear `ValueError("unknown strategy: ...")` rather than the current silent fall-through to V1. Surfaces misconfiguration early. |
| Bottoming strategy on ticker with <252 bars | Returns `BOTTOM — N/A`. Matches existing "insufficient data" pattern. |
| Healthcare / biotech ticker | Filtered upstream by existing sector exclusion logic. |

**Principles:**
- Defaults system is purely additive. Never clicking the button = no behavior change.
- Bottoming strategy is isolated. Failures don't affect V1/V2/LEAPS.
- `DEFAULT_SETTINGS` extension is backward-compatible — old settings files still load.

---

## 7. Testing

### 7.1 Manual smoke tests (minimum bar before merge)

1. A known bottoming candidate (stock 35%+ off highs, 15%+ off lows, breaking out) → `BOTTOM — BUY` or `BOTTOM — WATCH`.
2. A stock at 52w highs (e.g. a strong performer during rally) → `BOTTOM — N/A`.
3. A stock making new lows → `BOTTOM — N/A`.
4. Click "Load recommended defaults" for each of V1 / V2 / LEAPS / Bottoming → every mapped field updates to the values in §4.1.
5. Switch strategy mid-session and re-click button → fields update correctly; fields not in the target strategy's dict are untouched.
6. `curl /api/strategy-defaults` → returns all four strategy dicts as JSON.

### 7.2 Unit tests (not blocking, recommended)

- `tests/test_strategy_bottoming.py` — synthetic OHLCV fixtures producing each of BUY / WATCH / AVOID / N/A / WAIT outcomes. Mirror existing test patterns if present.
- `tests/test_api_strategy_defaults.py` — assert endpoint returns dict with keys `v1`, `v2`, `leaps`, `bottoming`, each with expected subset of keys.

### 7.3 Regression checks

- Run V1/V2/LEAPS scans on a fixed ticker list before and after the change. Results must be identical (this change is additive).
- `/api/scan` endpoint signature unchanged.

---

## 8. File change summary

| File | Change type | Notes |
|---|---|---|
| `strategy_bottoming.py` | **new** | Full module: `compute_bottoming_confirmations`, `get_current_signal_bottoming`, `RECOMMENDED_SETTINGS` |
| `backtester.py` | edit | Append `RECOMMENDED_SETTINGS` constant |
| `strategy_v2.py` | edit | Append `RECOMMENDED_SETTINGS` constant |
| `strategy_leaps.py` | edit | Append `RECOMMENDED_SETTINGS` constant |
| `screener.py` | edit | Import bottoming module; add dispatch branch; raise on unknown strategy |
| `api/routes_settings.py` | edit | Add `GET /strategy-defaults` route; extend `SettingsUpdate` Pydantic model with 5 new optional fields |
| `settings_manager.py` | edit | Extend `DEFAULT_SETTINGS` with 5 new keys |
| `index.html` | edit | Add strategy `<option>`, button, signal filter options |
| `js/api.js` | edit | Add `getStrategyDefaults()` |
| `js/settings.js` | edit | Add `loadRecommendedDefaults()` and `flashButton()` |
| `tests/test_strategy_bottoming.py` | **new** (optional) | Unit coverage for the new module |
| `tests/test_api_strategy_defaults.py` | **new** (optional) | Endpoint shape test |
| `docs/superpowers/specs/2026-04-21-bottoming-strategy-and-per-strategy-defaults-design.md` | **new** | This file |

---

## 9. Open questions

None at spec time. All design questions from brainstorming were resolved:

- **Q1 (what is bottoming):** B + C hybrid — base-and-breakout with trend-reclaim overlay.
- **Q2 (regime filtering):** Confidence tier, not a gate. Two-tier BUY / WATCH output.
- **Q3 (drawdown thresholds):** ≥35% off 52w high AND ≥15% off 52w low.
- **Q4 (defaults UX):** Explicit button ("Approach B"), no auto-apply.
- **Implementation approach:** Backend-owned `RECOMMENDED_SETTINGS` per strategy module, exposed via single API endpoint ("Approach 2").
