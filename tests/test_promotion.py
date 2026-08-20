"""Promotion gates. The point of these tests is that a gate actually BLOCKS."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.promotion import evaluate, recalibrate


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.02).astype(int)
    # Champion: informative. Challenger: same signal, slightly better.
    base = rng.random(n) * 0.3 + y * 0.4
    champion = np.clip(base + rng.normal(0, 0.05, n), 0, 1)
    challenger = np.clip(base + y * 0.15 + rng.normal(0, 0.04, n), 0, 1)
    return y, champion, challenger


def _kw(**over):
    kw = {"shadow_days": 15, "champion_air": 0.95, "challenger_air": 0.95,
          "reason_code_coverage": 1.0, "rollback_tested": True}
    kw.update(over)
    return kw


def test_a_single_failed_gate_blocks_promotion(data):
    """No aggregate score. One failure is a block, however good the rest look."""
    y, ch, cl = data
    d = evaluate(ch, cl, y, challenger_ref_scores=cl, **_kw(rollback_tested=False))
    assert d.promote is False
    assert [g.name for g in d.failed()] == ["rollback"]


def test_short_shadow_period_blocks(data):
    y, ch, cl = data
    d = evaluate(ch, cl, y, challenger_ref_scores=cl, **_kw(shadow_days=3))
    assert not d.promote and "shadow_period" in [g.name for g in d.failed()]


def test_worse_fairness_blocks(data):
    y, ch, cl = data
    d = evaluate(ch, cl, y, challenger_ref_scores=cl,
                 **_kw(champion_air=0.95, challenger_air=0.90))
    assert not d.promote and "fairness" in [g.name for g in d.failed()]


def test_unmeasured_fairness_fails_rather_than_passing_silently(data):
    """An unmeasured gate is a FAILED gate. Treating 'not measured' as 'fine'
    is how a model ships with no fairness analysis at all."""
    y, ch, cl = data
    d = evaluate(ch, cl, y, challenger_ref_scores=cl,
                 **_kw(champion_air=None, challenger_air=None))
    assert not d.promote
    assert "fairness" in [g.name for g in d.failed()]
    assert any("unmeasured" in n for n in d.notes)


def test_unmeasured_stability_fails_rather_than_passing_silently(data):
    y, ch, cl = data
    d = evaluate(ch, cl, y, challenger_ref_scores=None, **_kw())
    assert "stability" in [g.name for g in d.failed()]


def test_missing_reason_codes_block(data):
    """A decline we cannot explain is not shippable regardless of AUC."""
    y, ch, cl = data
    d = evaluate(ch, cl, y, challenger_ref_scores=cl,
                 **_kw(reason_code_coverage=0.97))
    assert not d.promote and "explainability" in [g.name for g in d.failed()]


def test_auc_is_reported_but_does_not_gate(data):
    """The deployment acts at one threshold, not across the whole score range."""
    y, ch, cl = data
    d = evaluate(ch, cl, y, challenger_ref_scores=cl, **_kw())
    auc_gate = [g for g in d.gates if g.name == "auc_reference"][0]
    assert auc_gate.blocking is False


def test_miscalibrated_challenger_is_blocked_then_rescued_by_isotonic(data):
    """Isotonic is monotone, so it fixes the probabilities without touching the
    ranking -- which is the whole reason it is the right tool."""
    from sklearn.metrics import roc_auc_score
    y, ch, cl = data
    inflated = np.clip(cl * 6, 0, 1)          # ranks identically, wildly miscalibrated

    before = evaluate(ch, inflated, y, challenger_ref_scores=inflated, **_kw())
    assert "calibration" in [g.name for g in before.failed()]

    half = len(y) // 2
    iso = recalibrate(inflated[half:], y[half:])
    fixed = iso.predict(inflated)

    assert roc_auc_score(y, fixed) == pytest.approx(
        roc_auc_score(y, inflated), abs=1e-6), "isotonic changed the ranking"

    after = evaluate(ch[:half], fixed[:half], y[:half],
                     challenger_ref_scores=fixed, **_kw())
    assert "calibration" not in [g.name for g in after.failed()]


def test_every_gate_reports_its_measured_value(data):
    """A gate that says only PASS/FAIL cannot be argued with."""
    y, ch, cl = data
    d = evaluate(ch, cl, y, challenger_ref_scores=cl, **_kw())
    for g in d.gates:
        assert g.measured and g.measured != "", g.name
        assert g.requirement
