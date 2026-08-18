"""Imbalance-handling bakeoff: baseline vs class weights vs SMOTE.

The deliverable here is the comparison table, not the winner. "SMOTE is usually
wrong on tabular fraud" is a claim you have to earn by running it, so this module
implements SMOTE rather than importing it, and the training script reports all
three on the same out-of-time split.

SMOTE is implemented here (imbalanced-learn is not a dependency) so the
interpolation semantics are visible: synthetic minority points are convex
combinations of a minority sample and one of its k minority neighbours. That
construction is exactly why it tends to disappoint on fraud data -- fraud
minority points are not a smooth manifold, they are scattered modes, so the
interpolated points land in regions where no fraud actually lives and the model
learns a blurred boundary. Calibration is the first casualty.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors


def smote(X: np.ndarray, y: np.ndarray, target_ratio: float = 0.25,
          k: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class up to `target_ratio` of the majority."""
    rng = np.random.default_rng(seed)
    minority = X[y == 1]
    n_major = int((y == 0).sum())
    n_needed = int(target_ratio * n_major) - len(minority)
    if n_needed <= 0 or len(minority) <= k:
        return X, y

    nn = NearestNeighbors(n_neighbors=k + 1).fit(minority)
    _, idx = nn.kneighbors(minority)
    base = rng.integers(0, len(minority), n_needed)
    neigh = idx[base, rng.integers(1, k + 1, n_needed)]
    lam = rng.random((n_needed, 1))
    synthetic = minority[base] + lam * (minority[neigh] - minority[base])

    X_out = np.vstack([X, synthetic])
    y_out = np.concatenate([y, np.ones(n_needed, dtype=y.dtype)])
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def make_model(seed: int = 0) -> HistGradientBoostingClassifier:
    """One model config across all three strategies -- otherwise the comparison
    measures hyperparameter luck instead of the imbalance treatment.

    HistGradientBoosting stands in for LightGBM (same histogram-based algorithm);
    swapping in LightGBM/XGBoost is a one-line change and is listed in the
    remaining work, along with focal loss, which needs a custom objective this
    estimator does not expose.
    """
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=200, l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.15,
        random_state=seed)


def fit_strategy(name: str, X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 0):
    """Returns (fitted_model, note). Strategies share the model config above."""
    model = make_model(seed)
    if name == "baseline":
        model.fit(X_tr, y_tr)
        return model, "no imbalance handling"
    if name == "class_weight":
        pos = float((y_tr == 1).sum())
        neg = float((y_tr == 0).sum())
        w = np.where(y_tr == 1, neg / pos, 1.0)
        model.fit(X_tr, y_tr, sample_weight=w)
        return model, "sample_weight = neg/pos = {:.1f} on positives".format(neg / pos)
    if name == "smote":
        Xs, ys = smote(X_tr, y_tr, target_ratio=0.25, seed=seed)
        model.fit(Xs, ys)
        return model, "minority resampled to 25% of majority ({} -> {} rows)".format(
            len(y_tr), len(ys))
    raise ValueError(name)


STRATEGIES = ["baseline", "class_weight", "smote"]
