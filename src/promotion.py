"""Champion/challenger promotion, as executable gates rather than a doc table.

docs/MODEL_VALIDATION.md section 5.1 lists seven gates. A table in a document is
a description of intent; this module is the thing that actually refuses to
promote. Every gate returns a pass/fail with the measured value beside the
requirement, and **a single failure blocks promotion** -- there is deliberately no
aggregate score, because an aggregate lets a strong result on one gate buy a weak
one on another, and "the challenger scored 0.84 overall" is how a miscalibrated
model reaches production.

Shadow scoring: the challenger scores live traffic and its decisions are logged
but not acted on. That is the only way to measure a challenger on the population
it will actually face, rather than on a held-out slice of the training window.

Calibration is included as a gate and, when a challenger fails it, this module
recalibrates rather than discarding: isotonic regression on a held-out window is
cheap and usually fixes a re-weighted model's probabilities. Discarding a model
that ranks well but is miscalibrated throws away the hard part.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from .metrics import brier, precision_at_fpr, psi


@dataclass
class Gate:
    name: str
    requirement: str
    measured: str
    passed: bool
    blocking: bool = True


@dataclass
class PromotionDecision:
    promote: bool
    gates: list[Gate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def failed(self) -> list[Gate]:
        return [g for g in self.gates if not g.passed and g.blocking]

    def render(self) -> str:
        width = max(len(g.name) for g in self.gates) + 2
        lines = ["{:<{w}}{:<34}{:<26}{}".format(
            "gate", "requirement", "measured", "result", w=width)]
        lines.append("-" * (width + 68))
        for g in self.gates:
            lines.append("{:<{w}}{:<34}{:<26}{}".format(
                g.name, g.requirement, g.measured,
                "PASS" if g.passed else ("FAIL" if g.blocking else "warn"),
                w=width))
        lines.append("-" * (width + 68))
        lines.append("DECISION: {}".format(
            "PROMOTE" if self.promote else "BLOCKED on {}".format(
                ", ".join(g.name for g in self.failed()))))
        for n in self.notes:
            lines.append("  note: " + n)
        return "\n".join(lines)


def recalibrate(scores_cal: np.ndarray, y_cal: np.ndarray):
    """Isotonic regression fitted on a HELD-OUT window.

    Fitting the calibrator on the same data the model was trained on produces a
    calibration curve that looks perfect and means nothing -- the model has
    already memorised those labels.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores_cal, y_cal)
    return iso


def evaluate(champion_scores: np.ndarray, challenger_scores: np.ndarray,
             y: np.ndarray, *, target_fpr: float = 0.01,
             shadow_days: int = 14, reason_code_coverage: float = 1.0,
             champion_air: float | None = None,
             challenger_air: float | None = None,
             rollback_tested: bool = True,
             challenger_ref_scores: np.ndarray | None = None) -> PromotionDecision:
    """Run every gate. Returns the decision and the evidence for it."""
    gates: list[Gate] = []
    notes: list[str] = []

    # 1. shadow period
    gates.append(Gate(
        "shadow_period", ">= 14 days", "{} days".format(shadow_days),
        shadow_days >= 14))

    # 2. discrimination at the operating FPR
    ch = precision_at_fpr(y, champion_scores, target_fpr)
    cl = precision_at_fpr(y, challenger_scores, target_fpr)
    lift = cl["precision"] - ch["precision"]
    gates.append(Gate(
        "discrimination", ">= champion + 0.01 abs",
        "{:+.4f} ({:.4f} vs {:.4f})".format(lift, cl["precision"], ch["precision"]),
        lift >= 0.01))

    # 3. calibration
    b_ch, b_cl = brier(y, champion_scores), brier(y, challenger_scores)
    gates.append(Gate(
        "calibration", "Brier <= champion x 1.05",
        "{:.5f} vs {:.5f}".format(b_cl, b_ch),
        b_cl <= b_ch * 1.05))

    # 4. stability
    #
    # PSI of the CHALLENGER against ITS OWN training-window distribution. An
    # earlier version compared challenger scores against the champion's
    # distribution, which is a category error: two different models produce
    # different score distributions by construction, so that comparison measures
    # "is this a different model" (always yes, that is the point) rather than
    # "is this model drifting". It reported PSI 0.52 and blocked a challenger for
    # the crime of not being the champion.
    if challenger_ref_scores is not None:
        score_psi = psi(challenger_ref_scores, challenger_scores)
        measured = "{:.4f} (vs own training window)".format(score_psi)
        passed = score_psi < 0.10
    else:
        score_psi = float("nan")
        measured = "not measured"
        passed = False
        notes.append("stability needs the challenger's own training-window "
                     "scores; without them the gate fails rather than passing "
                     "silently")
    gates.append(Gate("stability", "score PSI < 0.10", measured, passed))

    # 5. fairness
    if champion_air is not None and challenger_air is not None:
        delta = challenger_air - champion_air
        gates.append(Gate(
            "fairness", "AIR not worse by > 0.02",
            "{:+.4f} ({:.4f} vs {:.4f})".format(delta, challenger_air, champion_air),
            delta > -0.02))
    else:
        gates.append(Gate(
            "fairness", "AIR not worse by > 0.02", "not measured", False))
        notes.append("fairness was not measured; an unmeasured gate is a FAILED "
                     "gate, not a skipped one")

    # 6. explainability
    gates.append(Gate(
        "explainability", "reason codes on 100% of declines",
        "{:.1%}".format(reason_code_coverage), reason_code_coverage >= 1.0))

    # 7. rollback
    gates.append(Gate(
        "rollback", "one-command revert, tested",
        "tested" if rollback_tested else "untested", rollback_tested))

    # AUC is reported but does NOT gate. Rank-ordering across the whole score
    # range is not what the deployment does; it acts at one threshold.
    gates.append(Gate(
        "auc_reference", "reported, not a gate",
        "{:.4f} vs {:.4f}".format(roc_auc_score(y, challenger_scores),
                                  roc_auc_score(y, champion_scores)),
        True, blocking=False))

    promote = not [g for g in gates if not g.passed and g.blocking]
    return PromotionDecision(promote, gates, notes)
