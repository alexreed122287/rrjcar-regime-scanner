"""Tests for tools/regime_volatility.py.

The claims this experiment makes are only as good as its lookahead discipline and its
controls, so that is what these tests pin down. No HMM fitting here -- the statistical
helpers are tested directly on constructed data where the right answer is known.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from tools.regime_volatility import (  # noqa: E402
    ANNUALIZE, _bootstrap_gap, _decimate, _epsilon_squared, _ewma_vol, _fwd_vol,
    _quintile_check, _test_one, _trail_vol,
)


def test_fwd_vol_uses_only_future_bars():
    """A spike at bar k must raise forward vol BEFORE k and never at or after it."""
    rets = np.zeros(30)
    rets[20] = 0.10
    fv = _fwd_vol(rets, 5)
    # Bars 15..19 look forward far enough to contain bar 20.
    assert np.all(fv[15:20] > 0)
    # Bar 20 itself looks at 21..25, which are flat. Its own spike must not leak backwards.
    assert fv[20] == pytest.approx(0.0)
    assert np.all(fv[21:25] == pytest.approx(0.0))


def test_fwd_vol_tail_is_nan_not_zero():
    """The last h bars have no future; they must be NaN so they drop out, not count as calm."""
    fv = _fwd_vol(np.random.default_rng(0).normal(0, 0.01, 50), 5)
    assert np.isnan(fv[-5:]).all()
    assert np.isfinite(fv[:-5]).all()


def test_fwd_vol_matches_manual_stdev():
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, 40)
    fv = _fwd_vol(rets, 5)
    expected = np.std(rets[11:16], ddof=1) * ANNUALIZE
    assert fv[10] == pytest.approx(expected)


def test_trail_vol_is_inclusive_and_backward_looking():
    """Trailing vol at bar i uses i-19..i, so a spike at i shows up at i, not before."""
    rets = np.zeros(40)
    rets[25] = 0.08
    tv = _trail_vol(rets, 20)
    assert np.isnan(tv[:19]).all()          # not enough history
    assert tv[24] == pytest.approx(0.0)     # spike not yet seen
    assert tv[25] > 0                       # inclusive of the current bar


def test_ewma_vol_responds_faster_than_flat_window():
    rets = np.concatenate([np.zeros(60), np.full(5, 0.03)])
    ew = _ewma_vol(rets)
    tv = _trail_vol(rets, 20)
    # Immediately after the shock the EWMA should have moved more than the flat average,
    # which dilutes the shock over 20 bars.
    assert ew[62] > tv[62]


def test_decimate_removes_overlap():
    df = pd.DataFrame({"bar": np.arange(100)})
    out = _decimate(df, 5)
    assert len(out) == 20
    assert np.all(np.diff(out["bar"].values) == 5)


def test_epsilon_squared_zero_for_identical_groups():
    g = [np.arange(50.0), np.arange(50.0), np.arange(50.0)]
    assert _epsilon_squared(g) == pytest.approx(0.0, abs=0.02)


def test_epsilon_squared_large_for_separated_groups():
    g = [np.zeros(50), np.ones(50) * 10, np.ones(50) * 20]
    assert _epsilon_squared(g) > 0.5


def _frame(bear_shift: float, n: int = 60, n_win: int = 10, seed: int = 0) -> pd.DataFrame:
    """Synthetic observations: bearish regimes get `bear_shift` added to forward vol."""
    rng = np.random.default_rng(seed)
    rows = []
    for tkr in ("A", "B", "C"):
        for w in range(n_win):
            for i in range(n):
                rid = int(rng.integers(0, 7))
                is_bear = rid in (5, 6)
                is_bull = rid in (0, 1)
                rows.append({
                    "ticker": tkr, "window": w, "bar": w * n + i, "regime_id": rid,
                    "is_bear": int(is_bear), "is_bull": int(is_bull),
                    "trail_vol": float(abs(rng.normal(0.2, 0.05))),
                    "resid_5": float(rng.normal(0, 0.3) + (bear_shift if is_bear else 0.0)),
                    "fwd_vol_5": float(abs(rng.normal(0.2, 0.05))
                                       + (bear_shift if is_bear else 0.0)),
                })
    return pd.DataFrame(rows)


def test_test_one_finds_no_effect_when_there_is_none():
    res = _test_one(_frame(0.0), "resid_5", 0.0125)
    assert res["usable"]
    assert res["verdict"] == "no"


def test_test_one_finds_a_real_planted_effect():
    res = _test_one(_frame(0.5), "resid_5", 0.0125)
    assert res["verdict"] == "yes"
    assert res["bear_minus_bull"] > 0
    assert res["bars"]["direction"] and res["bars"]["separation"]


def test_bootstrap_ci_brackets_zero_under_the_null():
    bs = _bootstrap_gap(_frame(0.0), "resid_5", n_boot=300)
    assert bs["usable"]
    assert bs["ci_low"] < 0 < bs["ci_high"]


def test_bootstrap_ci_excludes_zero_for_a_planted_effect():
    bs = _bootstrap_gap(_frame(0.6), "resid_5", n_boot=300)
    assert bs["ci_low"] > 0
    assert bs["frac_positive"] > 0.95


def test_quintile_check_reports_a_fraction():
    qc = _quintile_check(_frame(0.5), "fwd_vol_5")
    assert qc["frac"] is not None and qc["frac"] > 0.5
    assert "/" in qc["quintiles_positive"]


def test_test_one_handles_single_regime_gracefully():
    df = _frame(0.0)
    df["regime_id"] = 0
    df["is_bear"] = 0
    df["is_bull"] = 1
    res = _test_one(df, "resid_5", 0.0125)
    assert res["usable"] is False
