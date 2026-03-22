"""
hmm_engine.py — Hidden Markov Model Regime Detector
Trains a Gaussian HMM on market features to identify regime states.
Inspired by Renaissance Technologies / Jim Simons approach.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

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


class RegimeDetector:
    """
    Gaussian HMM-based market regime detector.

    Trains on [returns, range, volume_change] features
    to discover n_regimes hidden states, then labels them
    from most bullish to most bearish by mean return.
    """

    def __init__(self, n_regimes: int = 7, n_iter: int = 100, random_state: int = 42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.state_order = None      # Maps raw state → rank (0=most bullish)
        self.regime_stats = None     # Summary stats per labeled regime
        self.is_trained = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and scale the 3 core features."""
        features = ["returns", "range", "volume_change"]
        X = df[features].values
        return X

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train the HMM on feature data and label regimes.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: returns, range, volume_change

        Returns
        -------
        pd.DataFrame
            Original df with added columns:
            - raw_state: raw HMM state id
            - regime_id: ordered regime (0=most bullish, 6=most bearish)
            - regime_label: human-readable label
            - regime_confidence: posterior probability of assigned state
        """
        X_raw = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(X_raw)

        print(f"[HMM] Training on {len(X_scaled)} samples with {self.n_regimes} regimes...")

        # Train Gaussian HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        self.model.fit(X_scaled)

        # Decode most likely state sequence (Viterbi)
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
        result["regime_label"] = result["regime_id"].map(
            lambda x: REGIME_LABELS[x] if x < len(REGIME_LABELS) else f"State {x}"
        )

        # Confidence = probability of the assigned state
        result["regime_confidence"] = [
            posteriors[i, raw_states[i]] for i in range(len(raw_states))
        ]

        # Build regime summary stats
        self._build_regime_stats(result)

        self.is_trained = True
        print(f"[HMM] Training complete. Log-likelihood: {self.model.score(X_scaled):.2f}")
        return result

    def predict_current(self, df: pd.DataFrame) -> dict:
        """
        Get the current regime for the most recent candle.

        Returns
        -------
        dict with keys:
            regime_id, regime_label, confidence, mean_return, volatility
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X_raw = self._prepare_features(df)
        X_scaled = self.scaler.transform(X_raw)

        raw_states = self.model.predict(X_scaled)
        posteriors = self.model.predict_proba(X_scaled)

        last_raw = raw_states[-1]
        last_regime = self.state_order[last_raw]
        last_conf = posteriors[-1, last_raw]

        label = REGIME_LABELS[last_regime] if last_regime < len(REGIME_LABELS) else f"State {last_regime}"

        stats = self.regime_stats
        regime_row = stats[stats["regime_id"] == last_regime]

        return {
            "regime_id": int(last_regime),
            "regime_label": label,
            "confidence": float(last_conf),
            "mean_return": float(regime_row["mean_return"].values[0]) if len(regime_row) else 0.0,
            "volatility": float(regime_row["volatility"].values[0]) if len(regime_row) else 0.0,
            "sample_count": int(regime_row["count"].values[0]) if len(regime_row) else 0,
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
                "regime_label": REGIME_LABELS[rid] if rid < len(REGIME_LABELS) else f"State {rid}",
                "mean_return": subset["returns"].mean(),
                "volatility": subset["returns"].std(),
                "mean_range": subset["range"].mean(),
                "mean_vol_change": subset["volume_change"].mean(),
                "count": len(subset),
                "pct_of_total": len(subset) / len(df) * 100,
            })
        self.regime_stats = pd.DataFrame(stats)

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

        labels = [REGIME_LABELS[i] if i < len(REGIME_LABELS) else f"State {i}"
                  for i in range(self.n_regimes)]

        return pd.DataFrame(reordered, index=labels, columns=labels)
