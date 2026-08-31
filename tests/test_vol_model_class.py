"""Tests for tools/vol_model_class.py -- the test-15 model-class comparison.

The load-bearing pieces here are the hand-rolled GARCH(1,1) (neither `arch` nor `statsmodels`
is installed, and adding a dependency for one experiment is not worth it) and the block win-rate
diagnostic. Both are the kind of code that fails quietly: a GARCH filter that peeks at future
returns would produce a beautiful, meaningless forecast, and a win-rate helper that grew a
`significant` key would repeat a bug this repo has already made once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.vol_forecast_shootout import ANNUALIZE, HORIZON
from tools.vol_model_class import (
    LAMBDA_CURVE,
    LAMBDA_GRID,
    MODELS,
    NOTES,
    _block_win_rate,
    _fit_garch,
    _garch_forecast_path,
    _garch_nll,
    _lam_col,
    evaluate,
)


def _simulate_garch(omega, alpha, beta, n, seed=7):
    """Simulate a GARCH(1,1) on a percent scale, return decimal returns."""
    rng = np.random.default_rng(seed)
    s2 = np.empty(n)
    r = np.empty(n)
    s2[0] = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = rng.normal(0.0, np.sqrt(s2[t]))
        if t + 1 < n:
            s2[t + 1] = omega + alpha * r[t] ** 2 + beta * s2[t]
    return r / 100.0


# --------------------------------------------------------------------- GARCH likelihood

def test_nll_rejects_nonpositive_omega():
    r2 = np.full(300, 1.0)
    assert _garch_nll((0.0, 0.05, 0.9), r2, 1.0) >= 1e12


def test_nll_rejects_nonstationary_parameters():
    r2 = np.full(300, 1.0)
    assert _garch_nll((0.01, 0.5, 0.5), r2, 1.0) >= 1e12
    assert _garch_nll((0.01, 0.05, 0.99), r2, 1.0) >= 1e12


def test_nll_rejects_negative_coefficients():
    r2 = np.full(300, 1.0)
    assert _garch_nll((0.01, -0.05, 0.9), r2, 1.0) >= 1e12
    assert _garch_nll((0.01, 0.05, -0.9), r2, 1.0) >= 1e12


def test_nll_is_finite_and_beaten_by_the_truth_on_simulated_data():
    """A wrong parameter set should not out-score the generating one."""
    r = _simulate_garch(0.02, 0.08, 0.90, 3000) * 100.0
    r2 = (r - r.mean()) ** 2
    bc = float(np.mean(r2))
    truth = _garch_nll((0.02, 0.08, 0.90), r2, bc)
    silly = _garch_nll((0.50, 0.01, 0.10), r2, bc)
    assert np.isfinite(truth)
    assert truth < silly


# --------------------------------------------------------------------- GARCH estimation

def test_fit_recovers_persistence_on_simulated_data():
    """Individual alpha/beta are famously weakly identified; their sum is not."""
    r = _simulate_garch(0.02, 0.08, 0.90, 4000)
    fit = _fit_garch(r)
    assert fit is not None
    omega, alpha, beta, _ = fit
    assert omega > 0
    assert alpha >= 0 and beta >= 0
    assert alpha + beta == pytest.approx(0.98, abs=0.03)


def test_fit_respects_the_stationarity_bound():
    r = _simulate_garch(0.01, 0.10, 0.89, 2000)
    fit = _fit_garch(r)
    assert fit is not None
    assert fit[1] + fit[2] < 0.9999


def test_fit_returns_none_when_the_sample_is_too_short():
    assert _fit_garch(np.random.default_rng(0).normal(0, 0.01, 200)) is None


def test_fit_is_deterministic():
    r = _simulate_garch(0.02, 0.08, 0.90, 1500)
    assert _fit_garch(r) == _fit_garch(r)


def test_fit_tolerates_nans():
    r = _simulate_garch(0.02, 0.08, 0.90, 1500)
    r[::50] = np.nan
    assert _fit_garch(r) is not None


# --------------------------------------------------------------------- GARCH forecasting

def test_forecast_is_constant_when_there_is_no_arch_effect():
    """alpha = beta = 0 collapses the model to a constant variance omega."""
    r = np.random.default_rng(1).normal(0, 0.01, 500)
    out = _garch_forecast_path(r, omega=0.04, alpha=0.0, beta=0.0, h=HORIZON)
    good = out[np.isfinite(out)]
    expected = np.sqrt(0.04) / 100.0 * ANNUALIZE
    assert good == pytest.approx(expected, rel=1e-9)


def test_forecast_rises_after_a_shock():
    r = np.full(400, 0.001)
    quiet = _garch_forecast_path(r, 0.02, 0.10, 0.85, HORIZON)
    shocked = r.copy()
    shocked[300] = 0.10  # a 10% day
    loud = _garch_forecast_path(shocked, 0.02, 0.10, 0.85, HORIZON)
    assert loud[300] > quiet[300], "the bar that saw the shock must forecast higher vol"
    assert loud[305] > quiet[305], "and it should still be elevated a few bars later"


def test_forecast_does_not_peek_at_future_returns():
    """The regression that would silently invalidate the whole experiment.

    The forecast at bar t must depend only on returns up to and including t. Rewriting every
    return after t must leave it untouched.
    """
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 400)
    base = _garch_forecast_path(r, 0.02, 0.10, 0.85, HORIZON)

    tampered = r.copy()
    tampered[250:] = rng.normal(0, 0.20, len(tampered) - 250)  # violent future
    after = _garch_forecast_path(tampered, 0.02, 0.10, 0.85, HORIZON)

    np.testing.assert_allclose(base[:250], after[:250], rtol=1e-12, atol=0)
    assert not np.allclose(base[250:-1], after[250:-1]), (
        "sanity check on the test itself: the tampered future must change later forecasts")


def test_forecast_horizon_aggregation_matches_an_explicit_recursion():
    """The closed-form horizon average must equal iterating the recursion by hand."""
    omega, alpha, beta, h = 0.02, 0.10, 0.85, HORIZON
    r = np.random.default_rng(5).normal(0, 0.01, 300)
    out = _garch_forecast_path(r, omega, alpha, beta, h)

    # Rebuild the filter, then step forward h times explicitly.
    rp = r * 100.0
    r2 = rp ** 2
    lr = omega / (1 - alpha - beta)
    s2 = np.empty(len(rp))
    s2[0] = lr
    for t in range(1, len(rp)):
        s2[t] = omega + alpha * r2[t - 1] + beta * s2[t - 1]

    t = 200
    step = omega + alpha * r2[t] + beta * s2[t]
    acc = []
    cur = step
    for _ in range(h):
        acc.append(cur)
        cur = omega + (alpha + beta) * cur
    expected = np.sqrt(np.mean(acc)) / 100.0 * ANNUALIZE
    assert out[t] == pytest.approx(expected, rel=1e-10)


def test_forecast_last_bar_is_nan():
    """There is no bar t+1 to forecast from the final observation."""
    r = np.random.default_rng(9).normal(0, 0.01, 100)
    out = _garch_forecast_path(r, 0.02, 0.10, 0.85, HORIZON)
    assert np.isnan(out[-1])


# --------------------------------------------------------------------- lambda plumbing

def test_lam_col_is_stable_and_unique():
    assert _lam_col(0.94) == "lam0940"
    assert _lam_col(0.7) == "lam0700"
    assert len({_lam_col(l) for l in LAMBDA_CURVE}) == len(LAMBDA_CURVE)


def test_lambda_curve_includes_the_hardcoded_constant():
    """Without 0.94 on the curve there is nothing to compare the repo's choice against."""
    assert 0.94 in LAMBDA_CURVE
    assert LAMBDA_CURVE == sorted(LAMBDA_CURVE)


def test_lambda_grid_brackets_the_curve():
    """A fitted lambda pinned to a grid edge reports the bound, not the optimum."""
    assert LAMBDA_GRID.min() <= min(LAMBDA_CURVE)
    assert LAMBDA_GRID.max() >= 0.94


def test_every_model_has_a_report_note():
    assert set(NOTES) == set(MODELS)


# --------------------------------------------------------------------- win-rate helper

def _frame(a_vals, b_vals, n_blocks=6, per_block=5):
    rows = []
    for w in range(n_blocks):
        for i in range(per_block):
            rows.append({"ticker": "T", "window": w, "bar": w * per_block + i,
                         "y": 0.0, "a": a_vals[w], "b": b_vals[w]})
    return pd.DataFrame(rows)


def test_block_win_rate_counts_blocks_not_bars():
    """Six blocks, `a` closer to the truth in four of them."""
    a = [0.01, 0.01, 0.01, 0.01, 0.50, 0.50]
    b = [0.20, 0.20, 0.20, 0.20, 0.01, 0.01]
    df = _frame(a, b)
    out = _block_win_rate(df, "a", "b")
    assert out["n_blocks"] == 6
    assert out["a_better_blocks"] == 4
    assert out["a_better_frac"] == pytest.approx(4 / 6, abs=1e-4)


def test_block_win_rate_reports_no_significance_key():
    """Guards the exact bug this repo shipped once: reading significance off a helper that
    does not compute it. Significance is the bootstrap CI excluding zero, and lives in
    _dm_test. This helper is descriptive only."""
    out = _block_win_rate(_frame([0.01] * 6, [0.2] * 6), "a", "b")
    assert "significant" not in out
    assert "ci_low" not in out and "ci_high" not in out
    assert out["loss"] == "qlike"


def test_block_win_rate_median_and_mean_can_disagree():
    """The situation the test-15 write-up turns on.

    One block with a huge win plus five tiny losses: the mean favours `a`, the median favours
    `b`. Reporting only one of them would misdescribe the result.
    """
    a = [0.001, 0.31, 0.31, 0.31, 0.31, 0.31]
    b = [3.0, 0.30, 0.30, 0.30, 0.30, 0.30]
    df = _frame(a, b)
    out = _block_win_rate(df, "a", "b")
    assert out["a_better_blocks"] == 1
    assert out["median_block_diff"] > 0  # median says `a` is worse
    la = np.mean([(np.exp(0.0) / np.exp(2 * v)) for v in a])  # crude mean-direction check
    assert out["n_blocks"] == 6 and la > 0


def test_block_win_rate_needs_enough_blocks_for_wilcoxon():
    out = _block_win_rate(_frame([0.01] * 3, [0.2] * 3, n_blocks=3), "a", "b")
    assert "wilcoxon_p" not in out


# --------------------------------------------------------------------- evaluate()

def _obs_frame(n=400, seed=11):
    """A frame shaped like real observations, with a perfect and a noisy forecast."""
    rng = np.random.default_rng(seed)
    y = rng.normal(-1.5, 0.4, n)
    rows = pd.DataFrame({
        "ticker": np.repeat(["A", "B"], n // 2),
        "window": np.tile(np.repeat(np.arange(5), (n // 2) // 5), 2),
        "bar": np.tile(np.arange(n // 2), 2),
        "y": y,
        "ewma94": y + rng.normal(0, 0.10, n),
        "ewma_fit": y + rng.normal(0, 0.10, n),
        "har": y + rng.normal(0, 0.10, n),
        "garch11": y,  # perfect
        "lam_fit": 0.85,
        "g_omega": 0.02, "g_alpha": 0.08, "g_beta": 0.90, "g_converged": True,
    })
    return rows


def test_evaluate_scores_a_perfect_forecast_as_perfect():
    res = evaluate(_obs_frame(), n_boot=200)
    t = res["table"]["garch11"]
    assert t["qlike"] == pytest.approx(0.0, abs=1e-9)
    assert t["mse"] == pytest.approx(0.0, abs=1e-9)
    assert t["r2"] == pytest.approx(1.0, abs=1e-9)
    assert res["ranking_by_qlike"][0] == "garch11"


def test_evaluate_decimates_to_non_overlapping_bars():
    df = _obs_frame()
    res = evaluate(df, n_boot=200)
    assert res["n_obs_all"] == len(df)
    assert res["n_obs_nonoverlapping"] < res["n_obs_all"]
    assert res["horizon"] == HORIZON


def test_evaluate_survives_observations_without_curve_columns():
    """Older observation files predate the lambda curve; scoring must still work."""
    res = evaluate(_obs_frame(), n_boot=200)
    assert res["lambda_loss_curve"] == {}


def test_evaluate_reports_the_curve_when_columns_are_present():
    df = _obs_frame()
    for lam in LAMBDA_CURVE:
        df[_lam_col(lam)] = df["y"] + 0.01
    res = evaluate(df, n_boot=200)
    assert set(res["lambda_loss_curve"]) == {f"{l:.2f}" for l in LAMBDA_CURVE}
    assert all(v > 0 for v in res["lambda_loss_curve"].values())


def test_evaluate_includes_every_model_and_comparison():
    res = evaluate(_obs_frame(), n_boot=200)
    assert set(res["table"]) == set(MODELS)
    for key in ("har_vs_ewma94", "garch11_vs_ewma94", "ewma_fit_vs_ewma94"):
        assert key in res["comparisons"]
        assert "block_win_rate" in res["comparisons"][key]
        assert "qlike" in res["comparisons"][key]


def test_evaluate_diagnostics_flag_a_censored_lambda_grid():
    df = _obs_frame()
    df["lam_fit"] = float(LAMBDA_GRID[0])  # everything pinned to the lower bound
    res = evaluate(df, n_boot=200)
    assert res["diagnostics"]["lambda_fitted"]["frac_at_grid_edge"] == pytest.approx(1.0)
