"""Tests for tools/vol_forecast_shootout.py, tools/vol_sizing.py and tools/vol_features.py.

These three experiments decide whether the HMM has any remaining use, so the machinery that
scores them needs to be right. The loss functions, the calibration, the sizing map and the
feature construction are all pinned here on constructed data where the answer is known.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.vol_features import FEATURE_SETS, add_vol_features  # noqa: E402
from tools.vol_forecast_shootout import (  # noqa: E402
    _apply_linear, _decimate, _dm_test, _fit_linear, _fwd_vol, _losses,
)
from tools.vol_sizing import sized_exposure  # noqa: E402


# --------------------------------------------------------------------------- losses

def _lf(y, f):
    return pd.DataFrame({"y": np.asarray(y, float), "m": np.asarray(f, float)})


def test_perfect_forecast_has_zero_loss():
    y = np.log(np.array([0.10, 0.20, 0.30]))
    L = _losses(_lf(y, y), "m")
    assert np.allclose(L["mse"], 0.0)
    assert np.allclose(L["qlike"], 0.0)


def test_qlike_is_never_negative():
    rng = np.random.default_rng(0)
    y = np.log(np.abs(rng.normal(0.2, 0.05, 500)) + 0.01)
    f = np.log(np.abs(rng.normal(0.2, 0.05, 500)) + 0.01)
    assert (_losses(_lf(y, f), "m")["qlike"] >= -1e-12).all()


def test_qlike_punishes_under_prediction_harder_than_over():
    """A risk manager cares more about forecasting calm and getting a storm."""
    y = np.log(np.array([0.20]))
    under = _losses(_lf(y, np.log([0.10])), "m")["qlike"][0]
    over = _losses(_lf(y, np.log([0.40])), "m")["qlike"][0]
    assert under > over


def test_mse_is_symmetric_in_log_space():
    y = np.log(np.array([0.20]))
    lo = _losses(_lf(y, y - 0.5), "m")["mse"][0]
    hi = _losses(_lf(y, y + 0.5), "m")["mse"][0]
    assert lo == pytest.approx(hi)


# --------------------------------------------------------------------- calibration

def test_fit_linear_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 400)
    y = 0.7 + 1.3 * x
    coef = _fit_linear(x.reshape(-1, 1), y)
    assert coef[0] == pytest.approx(0.7, abs=1e-6)
    assert coef[1] == pytest.approx(1.3, abs=1e-6)
    assert _apply_linear(coef, x.reshape(-1, 1)) == pytest.approx(y)


def test_fit_linear_refuses_tiny_samples():
    """Too few bars must return None rather than a wildly overfitted line."""
    assert _fit_linear(np.arange(10.0).reshape(-1, 1), np.arange(10.0)) is None


def test_apply_linear_with_no_model_yields_nan():
    out = _apply_linear(None, np.zeros((5, 1)))
    assert np.isnan(out).all()


# ------------------------------------------------------------------ model comparison

def _two_models(better_first: bool, n_win: int = 20, n: int = 30, seed: int = 0):
    """Model 'a' is deliberately better or worse than 'b' by a known margin."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_win):
        for i in range(n):
            y = float(np.log(abs(rng.normal(0.2, 0.04)) + 0.02))
            good = y + rng.normal(0, 0.05)
            bad = y + rng.normal(0, 0.30)
            rows.append({"ticker": "A", "window": w, "bar": w * n + i, "y": y,
                         "a": good if better_first else bad,
                         "b": bad if better_first else good})
    return pd.DataFrame(rows)


def test_dm_test_identifies_the_better_model():
    t = _dm_test(_two_models(True), "a", "b", "mse", n_boot=400)
    assert t["usable"]
    assert t["verdict"] == "a_better"
    assert t["ci_high"] < 0


def test_dm_test_identifies_the_worse_model():
    t = _dm_test(_two_models(False), "a", "b", "mse", n_boot=400)
    assert t["verdict"] == "a_worse"
    assert t["ci_low"] > 0


def test_dm_test_calls_a_tie_when_models_are_identical():
    df = _two_models(True)
    df["b"] = df["a"]
    t = _dm_test(df, "a", "b", "mse", n_boot=200)
    assert t["verdict"] == "tie"
    assert t["mean_diff"] == pytest.approx(0.0)


def test_dm_test_needs_enough_blocks():
    df = _two_models(True, n_win=3)
    assert _dm_test(df, "a", "b", "mse", n_boot=50)["usable"] is False


def test_decimate_leaves_non_overlapping_bars():
    df = pd.DataFrame({"bar": np.arange(50), "y": 0.0})
    assert len(_decimate(df, 5)) == 10


# ------------------------------------------------------------------------- sizing

def test_sizing_is_inversely_proportional_to_predicted_vol():
    """Twice the predicted vol must mean half the position."""
    e = sized_exposure(np.log([0.15, 0.30, 0.60]), target_vol=0.15, max_exposure=10.0)
    assert e[0] == pytest.approx(1.0)
    assert e[1] == pytest.approx(0.5)
    assert e[2] == pytest.approx(0.25)


def test_sizing_respects_the_cap():
    e = sized_exposure(np.log([0.01]), target_vol=0.15, max_exposure=1.0)
    assert e[0] == pytest.approx(1.0)


def test_sizing_never_goes_short():
    e = sized_exposure(np.log([0.5, 5.0, 50.0]), target_vol=0.15, max_exposure=1.0)
    assert (e >= 0).all()


def test_failed_forecast_sits_in_cash():
    e = sized_exposure(np.array([np.nan, np.inf, -np.inf]), 0.15, 1.0)
    assert (e == 0).all()


# ------------------------------------------------------------------- vol features

def _frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, n)
    return pd.DataFrame({"returns": r, "range": np.abs(r) * 2 + 0.001,
                         "volume_change": rng.normal(0, 0.1, n)})


def test_add_vol_features_creates_every_declared_column():
    out = add_vol_features(_frame())
    for c in FEATURE_SETS["vol"]:
        assert c in out.columns, c
        assert np.isfinite(out[c].dropna()).all()


def test_downside_share_is_a_proportion():
    d = add_vol_features(_frame())["downside_share"].dropna()
    assert (d >= 0).all() and (d <= 1).all()


def test_downside_share_is_one_when_all_returns_are_negative():
    df = _frame()
    df["returns"] = -np.abs(df["returns"])
    d = add_vol_features(df)["downside_share"].dropna()
    assert d.iloc[-1] == pytest.approx(1.0)


def test_vol_features_are_strictly_trailing():
    """A shock at bar k must not change any feature value BEFORE bar k."""
    base = _frame()
    shocked = base.copy()
    shocked.loc[300, "returns"] = 0.25
    shocked.loc[300, "range"] = 0.5
    a = add_vol_features(base)
    b = add_vol_features(shocked)
    for c in FEATURE_SETS["vol"]:
        left = a[c].values[:300]
        right = b[c].values[:300]
        both = np.isfinite(left) & np.isfinite(right)
        assert np.allclose(left[both], right[both]), f"{c} leaked information backwards"


def test_range_norm_centres_near_one():
    rn = add_vol_features(_frame(800))["range_norm"].dropna()
    assert 0.7 < rn.mean() < 1.4


def test_feature_sets_are_disjoint_in_purpose():
    """vol_ret must be the vol set plus returns -- guards against silent drift."""
    assert set(FEATURE_SETS["vol"]).issubset(set(FEATURE_SETS["vol_ret"]))
    assert "returns" in FEATURE_SETS["vol_ret"]
    assert "returns" not in FEATURE_SETS["vol"]


def test_fwd_vol_still_forward_only_in_this_module():
    rets = np.zeros(30)
    rets[20] = 0.10
    fv = _fwd_vol(rets, 5)
    assert fv[20] == pytest.approx(0.0)
    assert np.all(fv[15:20] > 0)
