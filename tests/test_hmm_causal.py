"""
Tests for causal (look-ahead-free) regime inference and automatic regime selection.

The headline test is test_appending_bars_does_not_relabel_history: it is the property
that was violated by the old predict_current(), which ran full-sequence Viterbi over all
history and therefore let tomorrow's bar change yesterday's regime.
"""

import numpy as np
import pandas as pd
import pytest

from data_loader import engineer_features
from hmm_engine import (
    RegimeDetector,
    forward_filtered_posteriors,
    select_n_regimes,
    labels_for,
    REGIME_LABELS,
)


@pytest.fixture
def featured(make_ohlcv):
    """Synthetic OHLCV with HMM features attached."""
    return engineer_features(make_ohlcv(n_bars=320, trend_slope=0.0004, seed=7))


@pytest.fixture
def trained(featured):
    det = RegimeDetector(n_regimes=4, n_iter=30, random_state=0)
    det.train(featured)
    return det


# ── causality: the core regression test ──

def test_appending_bars_does_not_relabel_history(trained, featured):
    """
    Causal labeling must be append-only.

    Labeling the first 200 bars, then labeling the first 250, must produce identical
    regimes for the overlapping first 200. The old implementation failed this.
    """
    short = trained.filtered_regimes(featured.iloc[:200])
    long = trained.filtered_regimes(featured.iloc[:250])

    assert list(short["regime_id"]) == list(long["regime_id"].iloc[:200])
    assert list(short["regime_label"]) == list(long["regime_label"].iloc[:200])
    np.testing.assert_allclose(
        short["regime_confidence"].values,
        long["regime_confidence"].values[:200],
        rtol=1e-9, atol=1e-12,
    )


def test_predict_current_is_stable_as_history_grows(trained, featured):
    """
    predict_current() on bars 0..N must equal the causal label at bar N regardless of
    how many bars come after it in the frame we happen to pass later.
    """
    cut = 240
    now = trained.predict_current(featured.iloc[:cut])
    later = trained.filtered_regimes(featured.iloc[:cut + 40])

    assert now["regime_id"] == int(later["regime_id"].iloc[cut - 1])
    assert now["confidence"] == pytest.approx(
        float(later["regime_confidence"].iloc[cut - 1]), rel=1e-9
    )


def test_predict_current_differs_from_smoothed_labels(trained, featured):
    """
    Sanity check that the causal path is genuinely different from the smoothed one.

    If filtered and forward-backward labels were identical everywhere, the causal fix
    would be a no-op and this test should make that visible.
    """
    X = trained.scaler.transform(featured[["returns", "range", "volume_change"]].values)
    filtered = forward_filtered_posteriors(trained.model, X).argmax(axis=1)
    smoothed = trained.model.predict_proba(X).argmax(axis=1)
    # Early bars are where smoothing borrows the most from the future.
    assert not np.array_equal(filtered, smoothed)


# ── forward algorithm mechanics ──

def test_filtered_posteriors_are_valid_distributions(trained, featured):
    X = trained.scaler.transform(featured[["returns", "range", "volume_change"]].values)
    post = forward_filtered_posteriors(trained.model, X)

    assert post.shape == (len(featured), trained.n_regimes)
    assert np.all(np.isfinite(post))
    assert np.all(post >= 0)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, rtol=1e-9, atol=1e-9)


def test_first_bar_posterior_uses_only_startprob_and_emission(trained, featured):
    """At t=0 the filtered posterior is exactly the normalized prior x emission."""
    X = trained.scaler.transform(featured[["returns", "range", "volume_change"]].values)
    post = forward_filtered_posteriors(trained.model, X)

    logp = trained.model._compute_log_likelihood(X[:1])[0]
    expected = np.exp(np.log(trained.model.startprob_) + logp)
    expected = expected / expected.sum()

    np.testing.assert_allclose(post[0], expected, rtol=1e-8, atol=1e-10)


def test_filtered_regimes_requires_training(featured):
    det = RegimeDetector(n_regimes=3)
    with pytest.raises(RuntimeError):
        det.filtered_regimes(featured)


def test_filtered_regimes_preserves_index_and_row_count(trained, featured):
    out = trained.filtered_regimes(featured)
    assert len(out) == len(featured)
    assert list(out.index) == list(featured.index)
    for col in ("raw_state", "regime_id", "regime_label", "regime_confidence"):
        assert col in out.columns


def test_missing_feature_columns_raise(trained, featured):
    bad = featured.drop(columns=["range"])
    with pytest.raises(ValueError, match="range"):
        trained.filtered_regimes(bad)


# ── automatic regime count selection ──

def test_select_n_regimes_returns_a_candidate(featured):
    result = select_n_regimes(featured, candidates=range(2, 5), n_iter=15)
    assert result["best_n"] in (2, 3, 4)
    assert result["criterion"] == "bic"
    assert "reason" in result

    table = result["table"]
    assert set(table["n_regimes"]) == {2, 3, 4}
    # BIC must be computable for at least one candidate on 300+ bars.
    assert table["bic"].notna().any()


def test_select_n_regimes_bic_penalizes_complexity(featured):
    """More states must cost more parameters, otherwise BIC is not doing its job."""
    table = select_n_regimes(featured, candidates=range(2, 6), n_iter=15)["table"]
    valid = table.dropna(subset=["n_params"]).sort_values("n_regimes")
    assert valid["n_params"].is_monotonic_increasing


def test_select_n_regimes_holdout_criterion(featured):
    result = select_n_regimes(
        featured, candidates=range(2, 5), n_iter=15, criterion="holdout"
    )
    assert result["best_n"] in (2, 3, 4)
    assert result["criterion"] in ("holdout", "bic")  # may fall back


def test_select_n_regimes_rejects_bad_criterion(featured):
    with pytest.raises(ValueError):
        select_n_regimes(featured, criterion="vibes")


def test_select_n_regimes_rejects_empty_candidates(featured):
    with pytest.raises(ValueError):
        select_n_regimes(featured, candidates=[])


def test_short_series_degrades_gracefully(featured):
    """Too few bars must not raise — it should skip candidates and pick a fallback."""
    result = select_n_regimes(featured.iloc[:8], candidates=range(3, 8), n_iter=5)
    assert isinstance(result["best_n"], int)
    assert result["best_n"] >= 3


# ── auto mode on the detector ──

def test_auto_mode_resolves_regime_count_at_train(featured):
    det = RegimeDetector(n_regimes="auto", auto_candidates=range(2, 5), n_iter=15)
    assert det.n_regimes is None
    out = det.train(featured)

    assert det.n_regimes in (2, 3, 4)
    assert det.regime_selection is not None
    assert det.regime_selection["best_n"] == det.n_regimes
    # Labels and stats must line up with the resolved count.
    assert out["regime_id"].max() < det.n_regimes
    assert len(det.regime_stats) <= det.n_regimes

    current = det.predict_current(featured)
    assert 0 <= current["regime_id"] < det.n_regimes


def test_auto_mode_accepts_case_insensitive_string(featured):
    det = RegimeDetector(n_regimes="AUTO", auto_candidates=[3], n_iter=10)
    assert det.auto is True


def test_invalid_string_regime_count_raises():
    with pytest.raises(ValueError):
        RegimeDetector(n_regimes="seven")


def test_explicit_count_does_not_run_selection(featured):
    det = RegimeDetector(n_regimes=3, n_iter=15)
    det.train(featured)
    assert det.auto is False
    assert det.regime_selection is None
    assert det.n_regimes == 3


# ── label handling ──

def test_labels_for_default_is_unchanged():
    """Default 7-regime labeling must match the original constant exactly."""
    assert labels_for(7) == REGIME_LABELS


def test_labels_for_spans_full_range_below_seven():
    """Previously truncated from the bullish end, leaving no bearish label at all."""
    for n in range(2, 8):
        labels = labels_for(n)
        assert len(labels) == n
        assert labels[0] == REGIME_LABELS[0]
        assert labels[-1] == REGIME_LABELS[-1], f"n={n} has no bearish label: {labels}"


def test_labels_for_five_includes_a_bear_label():
    labels = labels_for(5)
    assert len(labels) == 5
    assert any("Bear" in l or "Crash" in l for l in labels)


def test_labels_for_handles_more_than_seven():
    labels = labels_for(9)
    assert len(labels) == 9
    assert len(set(labels)) == 9, "labels must stay unique"
    assert labels[0] == REGIME_LABELS[0]
    assert labels[-1] == REGIME_LABELS[-1]


def test_labels_for_rejects_zero():
    with pytest.raises(ValueError):
        labels_for(0)


def test_transition_matrix_shape_matches_regimes(trained):
    tm = trained.get_transition_matrix()
    assert tm.shape == (trained.n_regimes, trained.n_regimes)
    np.testing.assert_allclose(tm.values.sum(axis=1), 1.0, rtol=1e-6)
