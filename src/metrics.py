"""The metrics a fraud/risk reviewer actually asks for.

Accuracy is not in this file and never will be: at 0.58% prevalence, always
predicting "legitimate" scores 99.42% accurate and blocks zero fraud. The
operating question is "how much fraud loss do I stop per unit of customer
friction", which is what precision-at-fixed-FPR and the cost curve answer.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def gini(auc: float) -> float:
    return 2 * auc - 1


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    """Max separation between the cumulative distributions of good and bad.
    Standard credit/fraud reporting metric; reported alongside AUC, not instead."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def precision_at_fpr(y_true: np.ndarray, score: np.ndarray,
                     target_fpr: float = 0.01) -> dict:
    """Precision and recall at the threshold that yields `target_fpr` on legits.

    Fixed-FPR is the review-capacity-constrained view: the false-positive rate is
    the customer-friction budget the business has agreed to, so the model is
    judged at that budget rather than at an arbitrary 0.5 cutoff.
    """
    fpr, tpr, thr = roc_curve(y_true, score)
    idx = int(np.searchsorted(fpr, target_fpr, side="right") - 1)
    idx = max(idx, 0)
    t = float(thr[idx])
    pred = score >= t
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {
        "threshold": t,
        "fpr": float(fpr[idx]),
        "recall": float(tpr[idx]),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "alerts_per_10k": 10_000 * (tp + fp) / len(y_true),
    }


def brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((prob - y_true) ** 2))


def calibration_table(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> list[dict]:
    """Reliability curve as a table. A model whose 2% bucket defaults at 6%
    cannot be used for $-weighted decisioning, however good its AUC is."""
    q = np.quantile(prob, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    idx = np.digitize(prob, q[1:-1])
    out = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        out.append({"bin": b, "n": int(m.sum()),
                    "mean_predicted": float(prob[m].mean()),
                    "observed": float(y_true[m].mean())})
    return out


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index against a reference window.

    Standard bands used throughout this repo (stated because unlabeled PSI is
    meaningless): <0.10 stable | 0.10-0.25 monitor | >0.25 investigate.
    """
    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    cuts = np.unique(cuts)
    e = np.histogram(expected, bins=cuts)[0].astype(float)
    a = np.histogram(actual, bins=cuts)[0].astype(float)
    e = np.clip(e / e.sum(), 1e-6, None)
    a = np.clip(a / a.sum(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_band(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "monitor"
    return "INVESTIGATE"


def full_report(y_true: np.ndarray, prob: np.ndarray, target_fpr: float = 0.01) -> dict:
    auc = float(roc_auc_score(y_true, prob))
    return {
        "auc": auc,
        "gini": gini(auc),
        "ks": ks_statistic(y_true, prob),
        "brier": brier(y_true, prob),
        "prevalence": float(y_true.mean()),
        "at_fpr": precision_at_fpr(y_true, prob, target_fpr),
    }
