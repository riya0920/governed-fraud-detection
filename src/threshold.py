"""Threshold economics: the operating point comes from a cost matrix, not from
0.5 and not from maximising F1.

Cost assumptions (stated so they can be argued with -- that is the point of
writing them down rather than hard-coding a cutoff):

  FN  a fraudulent transaction approved -> we eat the transaction amount plus a
      fixed handling cost. Amount-weighted, so a $2,000 miss is not a $20 miss.
  FP  a legitimate transaction declined -> friction cost: a support contact, a
      damaged relationship, and some fraction of lifetime value. Modelled as a
      fixed insult cost plus a share of the declined amount as lost margin.
  TP  a caught fraud still costs the review/handling overhead.
  TN  free.

Every number here is a policy input owned by the fraud/risk business owner, not
by the modeller. The code's job is to make the tradeoff explicit and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    fraud_handling_minor: int = 1_500        # $15 ops cost per fraud case
    insult_cost_minor: int = 2_500           # $25 per wrongly declined customer
    lost_margin_bps: int = 200               # 2% of a declined legit amount
    review_cost_minor: int = 400             # $4 per alert worked

    def expected_cost(self, y_true: np.ndarray, amount_minor: np.ndarray,
                      decline: np.ndarray) -> dict:
        fraud = y_true == 1
        legit = ~fraud
        fn = fraud & ~decline
        tp = fraud & decline
        fp = legit & decline

        loss_fn = int(amount_minor[fn].sum() + self.fraud_handling_minor * fn.sum())
        loss_tp = int(self.review_cost_minor * tp.sum())
        loss_fp = int(self.insult_cost_minor * fp.sum()
                      + (amount_minor[fp].sum() * self.lost_margin_bps) // 10_000)
        return {
            "fraud_loss_minor": loss_fn,
            "review_cost_minor": loss_tp,
            "friction_cost_minor": loss_fp,
            "total_cost_minor": loss_fn + loss_tp + loss_fp,
            "n_declined": int(decline.sum()),
            "caught_fraud": int(tp.sum()),
            "missed_fraud": int(fn.sum()),
        }


def sweep(y_true: np.ndarray, score: np.ndarray, amount_minor: np.ndarray,
          cost: CostModel, n_points: int = 200) -> list[dict]:
    """Total expected cost across candidate thresholds. The chosen operating
    point is the argmin, reported next to its FPR so the friction implied by the
    'optimal' point is visible rather than buried."""
    lo, hi = np.quantile(score, 0.50), score.max()
    rows = []
    for t in np.linspace(lo, hi, n_points):
        decline = score >= t
        r = cost.expected_cost(y_true, amount_minor, decline)
        legit = y_true == 0
        r["threshold"] = float(t)
        r["fpr"] = float((decline & legit).sum() / legit.sum())
        r["recall"] = float(r["caught_fraud"] / max((y_true == 1).sum(), 1))
        rows.append(r)
    return rows


def choose(rows: list[dict]) -> dict:
    return min(rows, key=lambda r: r["total_cost_minor"])


def rules_only_baseline(amount_minor: np.ndarray, velocity: np.ndarray,
                        cross_border: np.ndarray, device_change: np.ndarray) -> np.ndarray:
    """The incumbent every fraud model is actually compared against: a handful of
    hard rules a fraud analyst wrote. Beating AUC 0.5 is not the bar; beating
    THIS on $-weighted cost is."""
    return ((amount_minor > 150_000)
            | (velocity >= 8)
            | ((cross_border == 1) & (amount_minor > 40_000))
            | ((device_change == 1) & (amount_minor > 60_000)))
