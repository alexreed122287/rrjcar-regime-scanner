"""Tests for tools/vol_overlay_sweep.py.

The sweep is what stopped a one-setting result from being promoted into the repo, so its
machinery needs to be trustworthy: the rebalancing band, the grid, the significance rule, and
the constant-exposure control that produced the decisive finding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.drawdown_benchmark import _metrics, _paired_bootstrap  # noqa: E402
from tools.vol_overlay_sweep import (  # noqa: E402
    CAPS, COSTS, DEADBANDS, FORECASTS, TARGET_VOLS, apply_deadband,
)


# ------------------------------------------------------------------- deadband

def test_zero_deadband_is_the_identity():
    """The no-band column of the sweep must reproduce test 12 exactly."""
    t = np.array([0.3, 0.9, 0.1, 0.55])
    assert apply_deadband(t, 0.0) == pytest.approx(t)


def test_deadband_holds_position_until_drift_exceeds_it():
    t = np.array([0.50, 0.52, 0.54, 0.70])
    out = apply_deadband(t, 0.10)
    # First bar breaches from a flat start, then small drifts are ignored, then 0.70 breaches.
    assert out[0] == pytest.approx(0.50)
    assert out[1] == pytest.approx(0.50)
    assert out[2] == pytest.approx(0.50)
    assert out[3] == pytest.approx(0.70)


def test_deadband_never_increases_turnover():
    rng = np.random.default_rng(0)
    t = np.clip(rng.normal(0.6, 0.2, 500), 0, 1)
    f_ret = rng.normal(0, 0.01, 500)
    base = _metrics(apply_deadband(t, 0.0), f_ret, 5.0)["turnover"]
    for db in (0.05, 0.10, 0.25):
        assert _metrics(apply_deadband(t, db), f_ret, 5.0)["turnover"] <= base + 1e-9


def test_deadband_output_stays_within_the_input_range():
    rng = np.random.default_rng(1)
    t = np.clip(rng.normal(0.5, 0.3, 300), 0, 1)
    out = apply_deadband(t, 0.08)
    assert out.min() >= t.min() - 1e-12
    assert out.max() <= t.max() + 1e-12


def test_deadband_treats_a_broken_forecast_as_flat():
    out = apply_deadband(np.array([0.8, np.nan, np.nan]), 0.05)
    assert out[0] == pytest.approx(0.8)
    assert out[1] == pytest.approx(0.0)


def test_deadband_is_finite_everywhere():
    out = apply_deadband(np.array([np.nan, 0.4, np.inf, 0.4]), 0.1)
    assert np.isfinite(out).all()


# ----------------------------------------------------------------------- grid

def test_grid_covers_a_zero_cost_and_a_punitive_cost():
    """The cost sweep is the axis that revealed the filter's turnover problem."""
    assert 0.0 in COSTS and max(COSTS) >= 20.0


def test_grid_includes_an_unlevered_cap():
    assert 1.0 in CAPS


def test_grid_includes_a_zero_deadband_baseline():
    assert 0.0 in DEADBANDS


def test_grid_forecasts_use_no_regime_model():
    """Both sweep forecasts must be HMM-free, or Q1 is circular."""
    assert set(FORECASTS) == {"trail20", "ewma94"}


def test_grid_size_is_what_the_docstring_claims():
    n = len(TARGET_VOLS) * len(CAPS) * len(COSTS) * len(DEADBANDS) * len(FORECASTS)
    assert n == 240


# -------------------------------------------------------- significance rule

def _paired(diff_mean: float, noise: float, n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n):
        base = float(rng.normal(0.5, 0.2))
        rows.append({"ticker": "A", "window": w, "strategy": "a",
                     "sharpe": base + diff_mean + rng.normal(0, noise)})
        rows.append({"ticker": "A", "window": w, "strategy": "b", "sharpe": base})
    return pd.DataFrame(rows)


def test_ci_excluding_zero_is_the_significance_rule():
    """This is the rule the sweep applies; an earlier draft read a key that does not exist."""
    t = _paired_bootstrap(_paired(0.5, 0.05), "sharpe", "a", "b", n_boot=500)
    assert t["usable"]
    assert t["ci_low"] > 0, "a clear positive effect must have a CI above zero"
    t2 = _paired_bootstrap(_paired(0.0, 0.5), "sharpe", "a", "b", n_boot=500)
    assert t2["ci_low"] < 0 < t2["ci_high"], "a null must have a CI bracketing zero"


def test_paired_bootstrap_exposes_no_significant_key():
    """Guards the exact bug this sweep hit: there is no ready-made significance flag."""
    t = _paired_bootstrap(_paired(0.5, 0.05), "sharpe", "a", "b", n_boot=200)
    assert "significant" not in t
    assert {"ci_low", "ci_high", "mean_diff"} <= set(t)


# ------------------------------------------------- the constant-exposure control

def test_sharpe_is_scale_invariant():
    """Why the constant-exposure control is uninformative on Sharpe, and informative on DD.

    A constant exposure multiplies every return by the same number, which cannot change
    Sharpe. The sweep reports this explicitly so the row is not mistaken for evidence.
    """
    rng = np.random.default_rng(2)
    f_ret = rng.normal(0.0004, 0.01, 400)
    full = _metrics(np.ones(400), f_ret, 0.0)
    half = _metrics(np.full(400, 0.5), f_ret, 0.0)
    assert half["sharpe"] == pytest.approx(full["sharpe"], rel=1e-6)


def test_constant_exposure_does_reduce_drawdown():
    """Drawdown is NOT scale-invariant -- which is what makes the control meaningful."""
    rng = np.random.default_rng(3)
    f_ret = rng.normal(-0.001, 0.02, 400)
    full = _metrics(np.ones(400), f_ret, 0.0)
    half = _metrics(np.full(400, 0.5), f_ret, 0.0)
    assert half["max_drawdown"] > full["max_drawdown"], "less exposure must drawdown less"


def test_constant_control_matches_the_overlay_average_exposure():
    """The control is only fair if its average exposure equals the overlay's."""
    rng = np.random.default_rng(4)
    e = np.clip(rng.normal(0.7, 0.2, 300), 0, 1)
    const = np.full(len(e), float(np.mean(e)))
    f_ret = rng.normal(0, 0.01, 300)
    assert _metrics(const, f_ret, 0.0)["exposure"] == pytest.approx(
        _metrics(e, f_ret, 0.0)["exposure"], rel=1e-9)


def test_constant_control_has_no_turnover_beyond_entry():
    rng = np.random.default_rng(5)
    f_ret = rng.normal(0, 0.01, 200)
    assert _metrics(np.full(200, 0.6), f_ret, 0.0)["turnover"] == pytest.approx(0.6, abs=1e-9)


# ------------------------------------------------------------- cost mechanics

def test_higher_costs_punish_higher_turnover_harder():
    """The mechanism behind the filter's collapse from Sharpe 0.74 to -0.78."""
    rng = np.random.default_rng(6)
    f_ret = rng.normal(0.0003, 0.01, 500)
    churn = np.tile([1.0, 0.0], 250)
    calm = np.full(500, 0.5)
    churn_hit = _metrics(churn, f_ret, 0.0)["total_return"] - \
        _metrics(churn, f_ret, 20.0)["total_return"]
    calm_hit = _metrics(calm, f_ret, 0.0)["total_return"] - \
        _metrics(calm, f_ret, 20.0)["total_return"]
    assert churn_hit > calm_hit * 10
