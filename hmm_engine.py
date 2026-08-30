"""
hmm_engine.py — Hidden Markov Model Regime Detector
Trains a Gaussian HMM on market features to identify regime states.
Inspired by Renaissance Technologies / Jim Simons approach.

Look-ahead bias
---------------
train() uses full-sequence Viterbi decoding, which is fine for historical labeling and
in-sample fitting. Anything used for a LIVE decision or an out-of-sample evaluation must
be causal, and uses the forward algorithm only — see forward_filtered_posteriors() and
predict_current(). Do not replace those with model.predict() or model.predict_proba().
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

FEATURE_COLUMNS = ["returns", "range", "volume_change"]

# Regime labels ordered from most bullish to most bearish
REGIME_LABELS = [
    "Bull Run",           # Strongest positive returns
    "Bull Trend",         # Moderate positive returns
    "Mild Bull",          # Slight positive returns
    "Neutral / Chop",     # Near-zero returns, noise
    "Mild Bear",          # Slight negative returns
    "Bear Trend",         # Moderate negative returns
    "Crash / Capitulation"  # Extreme negative returns
]

# The 7-label scheme is the reference shape: 3 bullish states, 2 neutral, 2 bearish.
# Every other regime count is mapped proportionally onto it, so the invariant
# "regime 0 is most bullish, regime n-1 is most bearish" always holds.
REFERENCE_N = 7
REFERENCE_BULLISH = 3
REFERENCE_BEARISH = 2


def labels_for(n_regimes: int) -> list:
    """
    Ordered labels (most bullish first) for a given regime count.

    Labels are interpolated across the full bullish-to-bearish span of REGIME_LABELS so
    the last regime is always the most bearish. n_regimes=7 reproduces REGIME_LABELS
    exactly.

    This replaces the previous behavior of truncating the list from the bullish end,
    which left no bearish label at all for n_regimes < 7 -- a 5-regime model could never
    report a bear regime. See regime_sets() for the matching index sets.
    """
    if n_regimes < 1:
        raise ValueError("n_regimes must be >= 1")
    if n_regimes == 1:
        return [REGIME_LABELS[REFERENCE_N // 2]]

    span = REFERENCE_N - 1
    labels = [REGIME_LABELS[round(i * span / (n_regimes - 1))] for i in range(n_regimes)]

    # More states than reference labels causes collisions; disambiguate by occurrence so
    # labels stay unique and still read in bullish-to-bearish order.
    if len(set(labels)) != len(labels):
        counts = {l: labels.count(l) for l in labels}
        seen, out = {}, []
        for l in labels:
            if counts[l] == 1:
                out.append(l)
            else:
                seen[l] = seen.get(l, 0) + 1
                out.append(f"{l} ({seen[l]})")
        labels = out
    return labels


def regime_sets(n_regimes: int) -> dict:
    """
    Which regime ids count as bullish, neutral, and bearish for a given regime count.

    Regime ids are rank-ordered by mean return (0 = most bullish), so these sets are
    derived proportionally from the 7-regime reference shape rather than hardcoded.

    n_regimes=7 returns exactly the historical defaults -- bullish [0, 1, 2] and
    bearish [5, 6]. This function exists because those literals were hardcoded in
    run_backtest, which meant a 3-regime model treated EVERY state as bullish
    (ids 0, 1, 2) and was therefore always in the market. That silently turned the
    strategy into buy-and-hold whenever n_regimes was reduced or auto-selected.

    Always leaves at least one neutral state so there is a genuine "do nothing" band.

    Returns
    -------
    dict with keys ``bullish``, ``neutral``, ``bearish`` (lists of int), ``n_regimes``.
    """
    n = int(n_regimes)
    if n < 1:
        raise ValueError("n_regimes must be >= 1")
    if n == 1:
        return {"bullish": [], "neutral": [0], "bearish": [], "n_regimes": 1}
    if n == 2:
        return {"bullish": [0], "neutral": [], "bearish": [1], "n_regimes": 2}

    n_bull = max(1, round(n * REFERENCE_BULLISH / REFERENCE_N))
    n_bear = max(1, round(n * REFERENCE_BEARISH / REFERENCE_N))

    while n_bull + n_bear > n - 1:
        if n_bull >= n_bear and n_bull > 1:
            n_bull -= 1
        elif n_bear > 1:
            n_bear -= 1
        else:
            break

    bullish = list(range(n_bull))
    bearish = list(range(n - n_bear, n))
    neutral = [i for i in range(n) if i not in bullish and i not in bearish]
    return {"bullish": bullish, "neutral": neutral, "bearish": bearish, "n_regimes": n}


def _frame_log_likelihood(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Per-bar, per-state emission log-likelihood, shape (n_samples, n_components).

    Prefers hmmlearn's internal hook and falls back to computing the multivariate
    normal density directly, so this keeps working across hmmlearn versions that
    rename or remove the private method.
    """
    try:
        return model._compute_log_likelihood(X)
    except Exception:
        pass

    from scipy.stats import multivariate_normal

    n_components = model.means_.shape[0]
    out = np.empty((X.shape[0], n_components))
    covars = model.covars_
    for c in range(n_components):
        cov = covars[c]
        cov = np.diag(cov) if np.ndim(cov) == 1 else cov
        out[:, c] = multivariate_normal.logpdf(
            X, mean=model.means_[c], cov=cov, allow_singular=True
        )
    return out


def forward_filtered_posteriors(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Causal (filtered) state probabilities using the forward algorithm ONLY.

    Computes ``P(state_t | observations_0..t)`` so bar ``t`` never sees bar ``t+1``.

    This exists because both ``model.predict()`` (full-sequence Viterbi) and
    ``model.predict_proba()`` (forward-BACKWARD smoothing) allow future observations to
    change the label assigned to a past bar. Using either for live signals or
    out-of-sample evaluation is look-ahead bias and inflates measured performance.

    Do not "simplify" this into a library call — see .github/copilot-instructions.md.

    Returns
    -------
    np.ndarray, shape (n_samples, n_components), rows summing to 1.
    """
    X = np.asarray(X, dtype=float)
    framelogprob = _frame_log_likelihood(model, X)
    n_samples, n_components = framelogprob.shape

    with np.errstate(divide="ignore"):
        log_startprob = np.log(model.startprob_)
        log_transmat = np.log(model.transmat_)

    fwd = np.full((n_samples, n_components), -np.inf)
    fwd[0] = log_startprob + framelogprob[0]

    for t in range(1, n_samples):
        # log P(state_t = j, obs_0..t) = logsumexp_i[fwd[t-1,i] + logA[i,j]] + b_j(o_t)
        fwd[t] = logsumexp(fwd[t - 1][:, None] + log_transmat, axis=0) + framelogprob[t]

    norm = logsumexp(fwd, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        posteriors = np.exp(fwd - norm)

    # Guard against numerical underflow producing a degenerate row.
    bad = ~np.isfinite(posteriors).all(axis=1) | (posteriors.sum(axis=1) <= 0)
    if bad.any():
        posteriors[bad] = 1.0 / n_components
    return posteriors


def _hmm_param_count(n_components: int, n_features: int) -> int:
    """Free parameter count for a full-covariance Gaussian HMM (used for BIC)."""
    startprob = n_components - 1
    transmat = n_components * (n_components - 1)
    means = n_components * n_features
    covars = n_components * n_features * (n_features + 1) // 2
    return startprob + transmat + means + covars


def select_n_regimes(
    df: pd.DataFrame,
    candidates=range(3, 8),
    n_iter: int = 100,
    random_state: int = 42,
    tol: float = 1e-4,
    criterion: str = "bic",
    feature_columns=None,
) -> dict:
    """
    Choose the number of regimes from the data instead of hardcoding it.

    Fits a full-covariance Gaussian HMM for each candidate count and scores it. The
    winner is the lowest BIC (``criterion="bic"``) or the highest held-out
    log-likelihood on the last 20% of bars (``criterion="holdout"``).

    Returns
    -------
    dict with keys ``best_n`` (int), ``criterion`` (str), ``table`` (pd.DataFrame with
    one row per candidate) and ``reason`` (str).
    """
    candidates = [int(c) for c in candidates]
    if not candidates:
        raise ValueError("candidates must not be empty")
    if criterion not in ("bic", "holdout"):
        raise ValueError("criterion must be 'bic' or 'holdout'")

    cols = list(feature_columns) if feature_columns else list(FEATURE_COLUMNS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"df missing feature columns: {missing}")

    X_raw = df[cols].values
    n_samples, n_features = X_raw.shape

    X = StandardScaler().fit_transform(X_raw)

    split = int(n_samples * 0.8)
    holdout_usable = split >= 2 and (n_samples - split) >= 2

    def _blank(n):
        return {
            "n_regimes": n, "log_likelihood": np.nan, "n_params": np.nan,
            "bic": np.nan, "holdout_log_likelihood": np.nan,
            "converged": False, "skipped": True,
        }

    rows = []
    for n in candidates:
        # A full-covariance HMM needs meaningfully more bars than states.
        if n < 2 or n_samples <= n * 2:
            rows.append(_blank(n))
            continue
        try:
            model = GaussianHMM(
                n_components=n, covariance_type="full", n_iter=n_iter,
                tol=tol, random_state=random_state, verbose=False,
            )
            model.fit(X)
            logl = float(model.score(X))
            k = _hmm_param_count(n, n_features)
            bic = -2.0 * logl + k * float(np.log(n_samples))

            holdout_logl = np.nan
            if holdout_usable:
                try:
                    ho = GaussianHMM(
                        n_components=n, covariance_type="full", n_iter=n_iter,
                        tol=tol, random_state=random_state, verbose=False,
                    )
                    ho.fit(X[:split])
                    holdout_logl = float(ho.score(X[split:]))
                except Exception:
                    holdout_logl = np.nan

            rows.append({
                "n_regimes": n,
                "log_likelihood": logl,
                "n_params": k,
                "bic": bic,
                "holdout_log_likelihood": holdout_logl,
                "converged": bool(getattr(model.monitor_, "converged", False)),
                "skipped": False,
            })
        except Exception as exc:
            print(f"[HMM] n_regimes={n} failed to fit: {exc}")
            rows.append(_blank(n))

    table = pd.DataFrame(rows)

    score_col = "bic" if criterion == "bic" else "holdout_log_likelihood"
    valid = table.dropna(subset=[score_col])
    if valid.empty and criterion == "holdout":
        score_col, criterion = "bic", "bic"
        valid = table.dropna(subset=[score_col])

    if valid.empty:
        best_n = min(candidates)
        reason = (
            f"No candidate could be scored (only {n_samples} bars available); "
            f"defaulting to n_regimes={best_n}."
        )
    else:
        idx = valid[score_col].idxmin() if score_col == "bic" else valid[score_col].idxmax()
        best_row = valid.loc[idx]
        best_n = int(best_row["n_regimes"])
        reason = (
            f"Selected n_regimes={best_n} by {criterion} "
            f"({score_col}={best_row[score_col]:.2f}) from candidates "
            f"{sorted(candidates)} over {n_samples} bars."
        )

    print(f"[HMM] {reason}")
    return {"best_n": best_n, "criterion": criterion, "table": table, "reason": reason}


class RegimeDetector:
    """
    Gaussian HMM-based market regime detector.

    Trains on [returns, range, volume_change] features to discover n_regimes hidden
    states, then labels them from most bullish to most bearish by mean return.

    ``n_regimes`` may be the string ``"auto"``, in which case the count is chosen from
    the data at train() time by select_n_regimes().
    """

    def __init__(
        self,
        n_regimes=7,
        n_iter: int = 100,
        random_state: int = 42,
        tol: float = 1e-4,
        auto_candidates=range(3, 8),
        auto_criterion: str = "bic",
        feature_columns=None,
    ):
        if isinstance(n_regimes, str):
            if n_regimes.strip().lower() != "auto":
                raise ValueError("n_regimes must be an int or 'auto'")
            self.auto = True
            self.n_regimes = None
        else:
            self.auto = False
            self.n_regimes = int(n_regimes)

        # Which columns the HMM is fitted on. Defaults to the three core features so
        # every existing caller is unaffected. Overridden by the cross-asset feature
        # experiments in tools/, which need to fit on rates/credit/breadth columns.
        self.feature_columns = list(feature_columns) if feature_columns else list(FEATURE_COLUMNS)

        self.auto_candidates = auto_candidates
        self.auto_criterion = auto_criterion
        self.regime_selection = None   # populated when auto-selection runs

        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.state_order = None      # Maps raw state → rank (0=most bullish)
        self.regime_stats = None     # Summary stats per labeled regime
        self.labels = None           # Ordered labels for the resolved regime count
        self.is_trained = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract the configured feature columns (the 3 core features by default)."""
        cols = self.feature_columns
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"df missing feature columns: {missing}")
        return df[cols].values

    def _label(self, regime_id: int) -> str:
        labels = self.labels or REGIME_LABELS
        return labels[regime_id] if regime_id < len(labels) else f"State {regime_id}"

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train the HMM on feature data and label regimes.

        Uses full-sequence Viterbi decoding for the historical label column, which is
        appropriate for in-sample labeling. Live/out-of-sample decisions must use
        predict_current() or filtered_regimes() instead.

        Returns
        -------
        pd.DataFrame
            Original df with added columns:
            - raw_state: raw HMM state id
            - regime_id: ordered regime (0=most bullish)
            - regime_label: human-readable label
            - regime_confidence: posterior probability of assigned state
        """
        # Resolve "auto" regime count from the data before fitting.
        if self.auto:
            selection = select_n_regimes(
                df,
                candidates=self.auto_candidates,
                n_iter=self.n_iter,
                random_state=self.random_state,
                tol=self.tol,
                criterion=self.auto_criterion,
                feature_columns=self.feature_columns,
            )
            self.regime_selection = selection
            self.n_regimes = int(selection["best_n"])

        self.labels = labels_for(self.n_regimes)

        X_raw = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(X_raw)

        print(f"[HMM] Training on {len(X_scaled)} samples with {self.n_regimes} regimes...")

        # Train Gaussian HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=False,
        )
        self.model.fit(X_scaled)

        # Decode most likely state sequence (Viterbi) — in-sample labeling only.
        raw_states = self.model.predict(X_scaled)

        # Get posterior probabilities
        posteriors = self.model.predict_proba(X_scaled)

        # Compute mean return per raw state, rank from most bullish → most bearish
        state_returns = {}
        for s in range(self.n_regimes):
            mask = raw_states == s
            if mask.sum() > 0:
                state_returns[s] = df["returns"].values[mask].mean()
            else:
                state_returns[s] = 0.0

        # Sort states by descending mean return (highest return = most bullish = regime 0)
        sorted_states = sorted(state_returns.keys(), key=lambda s: state_returns[s], reverse=True)
        self.state_order = {raw: rank for rank, raw in enumerate(sorted_states)}

        # Apply to dataframe
        result = df.copy()
        result["raw_state"] = raw_states
        result["regime_id"] = result["raw_state"].map(self.state_order)
        result["regime_label"] = result["regime_id"].map(self._label)

        # Confidence = probability of the assigned state
        result["regime_confidence"] = [
            posteriors[i, raw_states[i]] for i in range(len(raw_states))
        ]

        # Build regime summary stats
        self._build_regime_stats(result)

        self.is_trained = True
        print(f"[HMM] Training complete. Log-likelihood: {self.model.score(X_scaled):.2f}")
        return result

    def filtered_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Causal regime labels for every bar, using the forward algorithm only.

        The regime assigned to bar ``t`` depends solely on bars ``0..t``, so appending
        new bars can never change a label already assigned to an earlier bar. This is
        the correct labeling for live decisions and out-of-sample evaluation.

        Returns
        -------
        pd.DataFrame
            Copy of df with raw_state, regime_id, regime_label, regime_confidence,
            computed causally.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X_scaled = self.scaler.transform(self._prepare_features(df))
        posteriors = forward_filtered_posteriors(self.model, X_scaled)

        raw_states = posteriors.argmax(axis=1)
        confidences = posteriors[np.arange(len(raw_states)), raw_states]

        out = df.copy()
        out["raw_state"] = raw_states
        out["regime_id"] = pd.Series(raw_states, index=out.index).map(self.state_order)
        out["regime_label"] = out["regime_id"].map(self._label)
        out["regime_confidence"] = confidences
        return out

    def predict_current(self, df: pd.DataFrame) -> dict:
        """
        Get the current regime for the most recent bar, computed causally.

        Uses the forward algorithm only, so the returned regime reflects information
        available up to the last bar and nothing after it. Previously this called
        model.predict() over the whole sequence, which retroactively relabeled bars as
        new data arrived (look-ahead bias).

        Returns
        -------
        dict with keys:
            regime_id, regime_label, confidence, mean_return, volatility, sample_count
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X_scaled = self.scaler.transform(self._prepare_features(df))
        posteriors = forward_filtered_posteriors(self.model, X_scaled)

        last_raw = int(posteriors[-1].argmax())
        last_conf = float(posteriors[-1, last_raw])
        last_regime = int(self.state_order[last_raw])

        label = self._label(last_regime)

        stats = self.regime_stats
        regime_row = stats[stats["regime_id"] == last_regime] if stats is not None else None
        has_row = regime_row is not None and len(regime_row)

        return {
            "regime_id": last_regime,
            "regime_label": label,
            "confidence": last_conf,
            "mean_return": float(regime_row["mean_return"].values[0]) if has_row else 0.0,
            "volatility": float(regime_row["volatility"].values[0]) if has_row else 0.0,
            "sample_count": int(regime_row["count"].values[0]) if has_row else 0,
        }

    def _build_regime_stats(self, df: pd.DataFrame):
        """Build summary statistics table for each regime."""
        stats = []
        for rid in range(self.n_regimes):
            mask = df["regime_id"] == rid
            subset = df[mask]
            if len(subset) == 0:
                continue
            stats.append({
                "regime_id": rid,
                "regime_label": self._label(rid),
                "mean_return": subset["returns"].mean(),
                "volatility": subset["returns"].std(),
                "mean_range": subset["range"].mean(),
                "mean_vol_change": subset["volume_change"].mean(),
                "count": len(subset),
                "pct_of_total": len(subset) / len(df) * 100,
            })
        self.regime_stats = pd.DataFrame(stats)

    def regime_sets(self) -> dict:
        """Bullish / neutral / bearish regime ids for this detector's regime count."""
        if self.n_regimes is None:
            raise RuntimeError("n_regimes unresolved. Call train() first for auto mode.")
        return regime_sets(self.n_regimes)

    def get_transition_matrix(self) -> pd.DataFrame:
        """Return the regime transition probability matrix (labeled)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        # Raw transition matrix from HMM
        raw_trans = self.model.transmat_

        # Reorder rows and columns by our regime ranking
        inv_order = {v: k for k, v in self.state_order.items()}
        order = [inv_order[i] for i in range(self.n_regimes)]

        reordered = raw_trans[np.ix_(order, order)]

        labels = [self._label(i) for i in range(self.n_regimes)]

        return pd.DataFrame(reordered, index=labels, columns=labels)
