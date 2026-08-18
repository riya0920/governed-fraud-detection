"""Per-decision reason codes -- the adverse-action pattern.

Regulatory framing: under FCRA §615(a) / ECOA Reg B §1002.9, a consumer who is
declined is entitled to the *principal reasons*, in terms they can act on, not a
model score. Card-fraud declines sit in a softer place than credit denials, but
issuers converge on the same machinery because the alternative -- an agent
reading "score 0.87" to a customer at a checkout -- is indefensible and the
regulator's question is identical: which specific facts drove this decision?

What this module computes, stated precisely so nobody mistakes it for SHAP:
an OCCLUSION attribution. For each feature we re-score the transaction with that
feature reset to its reference (training median / mode) and take the drop in
score as the contribution. It is a one-at-a-time counterfactual: cheap, exactly
reproducible, and blind to interactions.

SHAP's advantage over this is that it splits interaction credit fairly instead of
attributing it to whichever feature is occluded. Replacing this with TreeSHAP is
in the remaining work; the reason-code *plumbing* -- top-3 into the API response,
in plain English -- is what is built here.
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
    def __init__(self, model, feature_names: list[str], X_reference: np.ndarray):
        self.model = model
        self.features = list(feature_names)
        # Reference vector = column medians of the training population.
        self.reference = np.median(X_reference, axis=0)

    def _score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def explain(self, x: np.ndarray, top_k: int = 3) -> list[dict]:
        x = np.asarray(x, dtype=float).reshape(1, -1)
        base = float(self._score(x)[0])
        occluded = np.repeat(x, len(self.features), axis=0)
        for j in range(len(self.features)):
            occluded[j, j] = self.reference[j]
        drops = base - self._score(occluded)      # positive => feature raised risk
        order = np.argsort(-drops)[:top_k]
        out = []
        for rank, j in enumerate(order, 1):
            name = self.features[j]
            out.append({
                "rank": rank,
                "feature": name,
                "contribution": round(float(drops[j]), 5),
                "statement": PLAIN_ENGLISH.get(name, name.replace("_", " ")),
                "observed_value": float(x[0, j]),
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
            "attribution_method": "occlusion-vs-median (not SHAP; see reason_codes.py)",
        }
