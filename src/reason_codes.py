"""Per-decision reason codes -- the adverse-action pattern, on TreeSHAP.

Regulatory framing: under FCRA 615(a) / ECOA Reg B 1002.9, a consumer who is
declined is entitled to the *principal reasons*, in terms they can act on, not a
model score. Card-fraud declines sit in a softer place than credit denials, but
issuers converge on the same machinery, because the alternative -- an agent
reading "score 0.87" to a customer standing at a checkout -- is indefensible, and
the regulator's question is identical either way: which specific facts drove this
decision?

ATTRIBUTION. This now uses TreeSHAP (`shap.TreeExplainer`), which computes exact
Shapley values for tree ensembles in polynomial time. Shapley values are the
unique attribution satisfying efficiency (contributions sum to the prediction
minus the base value), symmetry, dummy and additivity -- which matters here
because an adverse-action reason has to survive the question "why THAT feature
and not this one?".

The previous implementation was occlusion-against-median: re-score with one
feature reset to its reference and call the drop that feature's contribution.
It is cheap and exactly reproducible, and it is wrong in a specific way --
interaction credit goes entirely to whichever feature is occluded, so two
features that only matter together each look individually decisive. On a fraud
model where amount and velocity interact strongly, that is not a rounding error.

`OcclusionCoder` is kept for comparison, and `compare_attributions` reports how
often the two disagree on the top reason. That number is the argument for using
SHAP rather than a claim that it is better.
"""
from __future__ import annotations

import numpy as np

PLAIN_ENGLISH = {
    "amount_minor": "Transaction amount is unusual for this account",
    "velocity_24h": "Unusually many transactions on this card in the last 24 hours",
    "cross_border": "Transaction originated outside the cardholder's usual country",
    "device_change": "Transaction came from a device not seen on this account before",
    "mcc_risk": "Merchant category carries elevated fraud rates",
    "hour": "Transaction occurred at an unusual hour for this account",
    "card_tenure_days": "Account is newer than most accounts we approve at this amount",
    "amount_per_velocity": "Spending pace on this card is unusual",
    "is_night": "Transaction occurred overnight",
}


class ReasonCoder:
    """TreeSHAP attribution, top-k contributors per decline."""

    def __init__(self, model, feature_names: list[str], X_reference: np.ndarray,
                 calibrator=None):
        import shap

        self.model = model
        # The calibrator travels WITH the model. A served score on a different
        # scale from the threshold it is compared against is not a rounding
        # problem, it is a different decision.
        self.calibrator = calibrator
        self.features = list(feature_names)
        self.reference = np.asarray(X_reference, dtype=float)
        # TreeExplainer on the raw margin: contributions then sum exactly to
        # (logit - base_value), which is the additivity property the reason
        # codes lean on. Explaining the probability instead breaks that, because
        # the sigmoid is not additive.
        # Unwrap any classifier wrapper: TreeExplainer needs the tree ensemble
        # itself, not a Python object that forwards predict_proba to one.
        inner = getattr(model, "model", model)
        self.explainer = shap.TreeExplainer(inner)
        self.method = "TreeSHAP (exact Shapley values on the log-odds margin)"

    def _score(self, X: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(np.asarray(X, dtype=float))[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return p

    def shap_values(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        vals = self.explainer.shap_values(x)
        if isinstance(vals, list):          # older API: one array per class
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:                  # (n, features, classes)
            vals = vals[:, :, -1]
        return vals.reshape(-1)

    def explain(self, x: np.ndarray, top_k: int = 3) -> list[dict]:
        contrib = self.shap_values(x)
        x = np.asarray(x, dtype=float).reshape(-1)
        # Only features that pushed the score UP can be reasons for a decline.
        # Listing a feature that argued for approval as a "principal reason" is
        # not a technicality -- it is a false statement to the customer.
        order = np.argsort(-contrib)
        out = []
        for rank, j in enumerate([i for i in order if contrib[i] > 0][:top_k], 1):
            name = self.features[j]
            out.append({
                "rank": rank,
                "feature": name,
                "contribution": round(float(contrib[j]), 5),
                "statement": PLAIN_ENGLISH.get(name, name.replace("_", " ")),
                "observed_value": float(x[j]),
            })
        return out

    def decision_record(self, x: np.ndarray, threshold: float,
                        model_version: str) -> dict:
        """Exactly what the scoring API returns and what an agent reads out."""
        score = float(self._score(np.asarray(x, dtype=float).reshape(1, -1))[0])
        declined = score >= threshold
        return {
            "score": round(score, 6),
            "decision": "decline" if declined else "approve",
            "threshold": threshold,
            "model_version": model_version,
            "reason_codes": self.explain(x) if declined else [],
            "attribution_method": self.method,
        }


class OcclusionCoder:
    """The previous method, retained so the two can be compared.

    Occlusion-against-median: reset one feature to its reference and call the
    score drop that feature's contribution. Blind to interactions -- credit for a
    joint effect goes entirely to whichever feature is occluded.
    """

    def __init__(self, model, feature_names: list[str], X_reference: np.ndarray):
        self.model = model
        self.features = list(feature_names)
        self.reference = np.median(np.asarray(X_reference, dtype=float), axis=0)
        self.method = "occlusion-vs-median (interaction-blind)"

    def explain(self, x: np.ndarray, top_k: int = 3) -> list[dict]:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        base = float(self.model.predict_proba(x)[:, 1][0])
        occluded = np.repeat(x, len(self.features), axis=0)
        for j in range(len(self.features)):
            occluded[j, j] = self.reference[j]
        drops = base - self.model.predict_proba(occluded)[:, 1]
        order = np.argsort(-drops)[:top_k]
        return [{"rank": r, "feature": self.features[j],
                 "contribution": round(float(drops[j]), 5),
                 "statement": PLAIN_ENGLISH.get(self.features[j], self.features[j]),
                 "observed_value": float(x[0, j])}
                for r, j in enumerate(order, 1)]


def compare_attributions(shap_coder: ReasonCoder, occ_coder: OcclusionCoder,
                         X: np.ndarray, n: int = 200) -> dict:
    """How often do the two methods disagree on the PRINCIPAL reason?

    This is the number that justifies the switch. If they always agreed, the
    cheaper method would be the right choice.
    """
    rows = np.asarray(X, dtype=float)[:n]
    top_disagree = 0
    set_disagree = 0
    for row in rows:
        s = shap_coder.explain(row, top_k=3)
        o = occ_coder.explain(row, top_k=3)
        if not s or not o:
            continue
        if s[0]["feature"] != o[0]["feature"]:
            top_disagree += 1
        if {d["feature"] for d in s} != {d["feature"] for d in o}:
            set_disagree += 1
    total = max(len(rows), 1)
    return {
        "n": total,
        "top_reason_disagreement": top_disagree / total,
        "top3_set_disagreement": set_disagree / total,
    }
