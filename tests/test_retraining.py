"""The retraining pipeline's state machine and the control it refuses to skip."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import retraining
from src.intersectional import intersectional_table
from src.promotion import Gate, PromotionDecision

PAGE = {"status": "page", "score_psi": 0.51,
        "findings": [{"signal": "alert_rate", "severity": "page",
                      "detail": "decline rate 34% vs 2%"}]}
WARN = {"status": "warn", "score_psi": 0.14,
        "findings": [{"signal": "score_psi", "severity": "warn",
                      "detail": "score PSI 0.1400"}]}
QUIET = {"status": "healthy", "score_psi": 0.01, "findings": []}
THIN = {"status": "insufficient_data", "n": 40}


def _passing():
    return PromotionDecision(promote=True, gates=[
        Gate("discrimination", ">= +0.01", "+0.02", True)])


def _failing():
    return PromotionDecision(promote=False, gates=[
        Gate("calibration", "<= 1.05x", "worse", False)])


# ------------------------------------------------------------------ triggers
def test_a_page_fires_the_retrain():
    rec = retraining.new_run(PAGE, days_since_last_fit=3)
    assert rec.fired == "alert_rate" and rec.state == "RETRAIN"


def test_a_warn_does_not_fire():
    """Retraining on a warn burns the clean comparison window a page will need."""
    rec = retraining.new_run(WARN, days_since_last_fit=3)
    assert rec.fired is None and rec.state == "NO_ACTION"


def test_a_quiet_monitor_does_nothing():
    assert retraining.new_run(QUIET, days_since_last_fit=3).state == "NO_ACTION"


def test_the_calendar_backstop_fires_on_its_own():
    rec = retraining.new_run(QUIET, days_since_last_fit=200)
    assert rec.fired == "calendar"


def test_an_evidenced_trigger_outranks_the_calendar():
    """Both are eligible; the record must name the one that can page someone."""
    rec = retraining.new_run(PAGE, days_since_last_fit=200)
    assert rec.fired == "alert_rate"


def test_insufficient_data_never_triggers_a_retrain():
    rec = retraining.new_run(THIN, days_since_last_fit=999)
    assert rec.state == "NO_ACTION"
    assert any("insufficient_data" in n for n in rec.notes)


def test_every_trigger_is_recorded_even_when_it_did_not_fire():
    rec = retraining.new_run(QUIET, days_since_last_fit=3)
    assert [t["name"] for t in rec.triggers] == retraining.TRIGGER_PRIORITY
    assert all(t["evidence"] for t in rec.triggers)


# ------------------------------------------------------------ the hard stop
def test_passing_every_gate_does_not_promote():
    """The terminal state of a successful run is AWAITING_VALIDATION. A pipeline
    that promotes here has removed the SR 11-7 control, not automated it."""
    rec = retraining.record_gates(
        retraining.new_run(PAGE, 3), _passing(), "v2")
    assert rec.state == "AWAITING_VALIDATION"
    assert rec.state != "PROMOTED"


def test_promotion_without_a_named_validator_is_refused():
    rec = retraining.record_gates(retraining.new_run(PAGE, 3), _passing(), "v2")
    for bad in (None, "", "   "):
        with pytest.raises(ValueError, match="validator|unsigned"):
            retraining.promote(rec, bad)


def test_promotion_with_a_signature_records_who_signed():
    rec = retraining.record_gates(retraining.new_run(PAGE, 3), _passing(), "v2")
    rec = retraining.promote(rec, "model-risk: A. Validator")
    assert rec.state == "PROMOTED"
    assert rec.signed_off_by == "model-risk: A. Validator"


def test_a_blocked_run_cannot_be_promoted_even_with_a_signature():
    rec = retraining.record_gates(retraining.new_run(PAGE, 3), _failing(), "v2")
    assert rec.state == "BLOCKED"
    with pytest.raises(ValueError, match="AWAITING_VALIDATION"):
        retraining.promote(rec, "model-risk: A. Validator")


def test_the_run_log_is_append_only_and_reloadable(tmp_path):
    path = tmp_path / "runs.jsonl"
    for _ in range(3):
        retraining.append_run(retraining.new_run(PAGE, 3), path)
    runs = retraining.load_runs(path)
    assert len(runs) == 3 and all(r["fired"] == "alert_rate" for r in runs)


def test_render_names_the_trigger_and_the_state():
    text = retraining.render(retraining.new_run(PAGE, 3))
    assert "alert_rate" in text and "RETRAIN" in text


# ------------------------------------------------------- intersectional
def test_a_failing_cell_hides_behind_passing_marginals():
    """The whole reason the cross exists, on data constructed to contain it.

    Both attributes are balanced and each marginal approval rate is close, but
    the (1,1) cell is approved less often. One-attribute-at-a-time closes the
    file; the cross does not.

    The cell rate has to be chosen inside a band, and that band is the point.
    With a best-cell rate of 0.95, the cross fails below 0.76 and the marginal
    still passes above 0.57 -- so masking happens for cell rates in roughly
    [0.57, 0.76) and nowhere else. A disparity harsher than that shows up in the
    marginals anyway; a milder one is not a finding. Intersectional analysis
    earns its place in that band, which is exactly where a single-attribute
    review is most confident it has cleared the model.
    """
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    both = (a == 1) & (b == 1)
    approved = rng.random(n) < np.where(both, 0.68, 0.95)

    res = intersectional_table(approved, a, b)
    assert res["masked_by_marginals"], res["verdict"]
    assert all(m["air"] >= 0.80 for m in res["marginals"])
    assert any(c["status"] == "below 0.80" for c in res["cells"])


def test_small_cells_are_refused_rather_than_estimated():
    a = np.array([1] * 20 + [0] * 2000)
    b = np.array([1] * 20 + [0] * 2000)
    approved = np.array([False] * 20 + [True] * 2000)
    res = intersectional_table(approved, a, b, min_n=200)
    assert any(c["status"] == "insufficient" for c in res["cells"])


def test_no_disparity_produces_no_failing_cell():
    rng = np.random.default_rng(1)
    n = 4000
    a, b = rng.integers(0, 2, n), rng.integers(0, 2, n)
    approved = rng.random(n) < 0.9
    res = intersectional_table(approved, a, b)
    assert not any(c["status"] == "below 0.80" for c in res["cells"])
    assert not res["masked_by_marginals"]
