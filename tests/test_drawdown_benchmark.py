"""Tests for tools/drawdown_benchmark.py.

The comparison is only fair if costs are charged on every exposure change and if exposure at
bar t earns bar t+1's return. Both are pinned here, along with the paired bootstrap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.drawdown_benchmark import _metrics, _paired_bootstrap, evaluate  # noqa: E402


def test_flat_exposure_earns_and_costs_nothing():
    m = _metrics(np.zeros(50), np.full(50, 0.01), cost_bps=5.0)
    assert m["total_return"] == pytest.approx(0.0)
    assert m["max_drawdown"] == pytest.approx(0.0)
    assert m["turnover"] == pytest.approx(0.0)
    assert m["exposure"] == pytest.approx(0.0)


def test_full_exposure_compounds_the_return_stream():
    r = np.full(10, 0.01)
    m = _metrics(np.ones(10), r, cost_bps=0.0)
    assert m["total_return"] == pytest.approx(1.01 ** 10 - 1)


def test_cost_is_charged_on_every_exposure_change():
    """Ten round trips at 5 bps a side must cost measurably more than holding."""
    r = np.zeros(20)
    hold = _metrics(np.ones(20), r, cost_bps=5.0)
    flip = _metrics(np.tile([1.0, 0.0], 10), r, cost_bps=5.0)
    assert hold["turnover"] == pytest.approx(1.0)      # one entry
    assert flip["turnover"] == pytest.approx(20.0)     # 10 entries + 10 exits from flat
    assert flip["total_return"] < hold["total_return"]


def test_entering_from_flat_is_charged_once():
    """Turnover must count the initial entry; a path starting long is not free."""
    m = _metrics(np.ones(5), np.zeros(5), cost_bps=100.0)
    assert m["turnover"] == pytest.approx(1.0)
    assert m["total_return"] == pytest.approx(-0.01, abs=1e-6)


def test_max_drawdown_is_negative_and_matches_hand_calc():
    # +10% then -20% -> peak 1.10, trough 0.88, drawdown = 0.88/1.10 - 1 = -20%
    m = _metrics(np.ones(2), np.array([0.10, -0.20]), cost_bps=0.0)
    assert m["max_drawdown"] == pytest.approx(-0.20)


def test_nan_exposure_is_treated_as_flat_not_dropped():
    e = np.array([np.nan, 1.0, np.nan, 1.0])
    m = _metrics(e, np.full(4, 0.01), cost_bps=0.0)
    assert np.isfinite(m["total_return"])
    assert m["exposure"] == pytest.approx(0.5)


def test_vol_is_annualized():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 500)
    m = _metrics(np.ones(500), r, cost_bps=0.0)
    assert m["vol_ann"] == pytest.approx(np.std(r, ddof=1) * np.sqrt(252), rel=0.02)


def _pair_frame(diff: float, n: int = 40, seed: int = 0, noise: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n):
        base = rng.normal(-0.15, 0.05)
        jitter = rng.normal(0.0, noise) if noise else 0.0
        rows.append({"ticker": "A", "window": w, "strategy": "hmm",
                     "max_drawdown": base + diff + jitter, "total_return": 0.05, "sharpe": 0.5,
                     "vol_ann": 0.2, "exposure": 0.5, "turnover": 20.0})
        rows.append({"ticker": "A", "window": w, "strategy": "sma200",
                     "max_drawdown": base, "total_return": 0.05, "sharpe": 0.5,
                     "vol_ann": 0.2, "exposure": 0.8, "turnover": 3.0})
        rows.append({"ticker": "A", "window": w, "strategy": "buy_hold",
                     "max_drawdown": base - 0.05, "total_return": 0.10, "sharpe": 0.9,
                     "vol_ann": 0.26, "exposure": 1.0, "turnover": 1.0})
    return pd.DataFrame(rows)


def test_paired_bootstrap_detects_no_difference():
    """A true null: differences vary window to window but average zero."""
    t = _paired_bootstrap(_pair_frame(0.0, noise=0.03), "max_drawdown", "hmm", "sma200",
                          n_boot=500)
    assert t["usable"]
    assert t["ci_low"] < 0 < t["ci_high"]


def test_paired_bootstrap_on_identical_inputs_gives_a_degenerate_interval():
    """Byte-identical strategies must yield exactly zero, not a spurious interval."""
    t = _paired_bootstrap(_pair_frame(0.0), "max_drawdown", "hmm", "sma200", n_boot=200)
    assert t["mean_diff"] == pytest.approx(0.0)
    assert t["ci_low"] == pytest.approx(0.0) and t["ci_high"] == pytest.approx(0.0)


def test_paired_bootstrap_detects_a_real_difference():
    t = _paired_bootstrap(_pair_frame(0.04), "max_drawdown", "hmm", "sma200", n_boot=500)
    assert t["ci_low"] > 0
    assert t["mean_diff"] == pytest.approx(0.04, abs=0.005)
    assert t["frac_a_better"] == pytest.approx(1.0)


def test_paired_bootstrap_missing_strategy_is_unusable():
    t = _paired_bootstrap(_pair_frame(0.0), "max_drawdown", "hmm", "nope", n_boot=100)
    assert t["usable"] is False


def test_evaluate_reports_drawdown_saved_per_return_given_up():
    res = evaluate(_pair_frame(0.04))
    hmm = res["pooled"]["hmm"]
    # HMM gives up 5pp of return and saves 9pp of drawdown vs buy and hold.
    assert hmm["return_given_up"] == pytest.approx(0.05, abs=1e-6)
    assert hmm["dd_saved"] == pytest.approx(0.09, abs=0.005)
    assert hmm["dd_saved_per_return_given_up"] == pytest.approx(1.8, abs=0.1)


def test_evaluate_survives_zero_return_difference():
    """A strategy that gives up no return must not divide by zero."""
    df = _pair_frame(0.0)
    df.loc[df.strategy == "buy_hold", "total_return"] = 0.05
    res = evaluate(df)
    assert res["pooled"]["hmm"]["dd_saved_per_return_given_up"] is None
