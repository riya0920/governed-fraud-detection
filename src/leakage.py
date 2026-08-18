"""Leakage audit.

Two layers, because either one alone is insufficient:

  1. A denylist of fields known to be populated after the authorization decision.
     Knowing *why* each is post-decision is the whole exercise; the list lives in
     docs/LEAKAGE_AUDIT.md with a justification per field.
  2. A univariate screen that flags any feature whose single-feature AUC is
     implausibly high. This catches the leaks nobody thought to deny -- the ones
     that arrive when an upstream team adds a column.

The screen produces suspicion, not verdicts. A feature can legitimately be
strong; the response to a flag is to trace its lineage, not to drop it reflexively.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Populated only after the decision exists -> unusable at scoring time.
POST_DECISION_FIELDS = {
    "chargeback_filed":  "Filed by the issuer days-to-months after settlement.",
    "review_queue_flag": "Set by the queue our own model feeds. Circular.",
    "card_blocked_24h":  "A consequence of the decline, not an input to it.",
    "is_fraud":          "Target.",
}

SUSPICION_AUC = 0.80


def audit(df: pd.DataFrame, target: str = "is_fraud",
          exclude: set[str] | None = None) -> pd.DataFrame:
    exclude = (exclude or set()) | {target, "txn_id", "card_id", "day"}
    y = df[target].to_numpy()
    rows = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        if not np.issubdtype(s.dtype, np.number):
            s = s.astype("category").cat.codes
        try:
            auc = roc_auc_score(y, s)
        except ValueError:
            continue
        auc = max(auc, 1 - auc)   # direction-agnostic
        rows.append({
            "feature": col,
            "univariate_auc": round(float(auc), 4),
            "denylisted": col in POST_DECISION_FIELDS,
            "flagged": auc >= SUSPICION_AUC,
            "reason": POST_DECISION_FIELDS.get(col, ""),
        })
    return (pd.DataFrame(rows)
            .sort_values("univariate_auc", ascending=False)
            .reset_index(drop=True))


class LeakageError(AssertionError):
    pass


def assert_clean(feature_names) -> None:
    """Fails loudly if a post-decision field reached the model matrix. Wired into
    the training script so the build cannot silently regress."""
    bad = sorted(set(feature_names) & set(POST_DECISION_FIELDS))
    if bad:
        raise LeakageError(
            "post-decision fields present in the feature matrix: {}".format(bad))
