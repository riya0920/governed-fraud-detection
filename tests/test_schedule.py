"""Monitoring on a cadence, without paging anyone three times for one event."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schedule import (CADENCE_MINUTES, ScheduleState, detection_delay_minutes,
                          due, observe, record_retrain, retrain_allowed)

T0 = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def state(tmp_path):
    return ScheduleState.load(tmp_path / "schedule.json")


# ------------------------------------------------------------- cadences
def test_cadence_follows_signal_latency():
    """The label-free signals run fastest because they CAN. Running the
    label-dependent check hourly recomputes the same stale answer sixty times a
    day and teaches whoever reads it that the number never moves."""
    assert CADENCE_MINUTES["alert_rate"] < CADENCE_MINUTES["score_psi"]
    assert CADENCE_MINUTES["score_psi"] < CADENCE_MINUTES["feature_psi"]
    assert CADENCE_MINUTES["feature_psi"] < CADENCE_MINUTES["performance"]


def test_everything_is_due_on_a_cold_start(state):
    d = due(state, T0)
    assert all(v["due"] for v in d.values())
    assert all(v["reason"] == "never run" for v in d.values())


def test_a_check_is_not_due_again_inside_its_cadence(state):
    observe(state, "alert_rate", False, T0)
    d = due(state, T0 + timedelta(minutes=30))
    assert d["alert_rate"]["due"] is False
    assert "next due" in d["alert_rate"]["reason"]


def test_a_check_becomes_due_after_its_cadence(state):
    observe(state, "alert_rate", False, T0)
    assert due(state, T0 + timedelta(minutes=61))["alert_rate"]["due"] is True


def test_the_slow_check_is_not_due_when_the_fast_one_is(state):
    """The whole point of per-signal cadence."""
    for name in ("alert_rate", "score_psi"):
        observe(state, name, False, T0)
    d = due(state, T0 + timedelta(minutes=90))
    assert d["alert_rate"]["due"] is True
    assert d["score_psi"]["due"] is False


# --------------------------------------------------- missed runs coalesce
def test_missed_runs_coalesce_into_one_rather_than_backfilling(state):
    """The opposite of SE-2's settlement catch-up, and deliberately so.

    A settlement cycle must process every missed date because the state is
    cumulative. Monitoring is not: replaying three days of it pages the on-call
    for drift that has already resolved and buries the current state under
    history.
    """
    observe(state, "alert_rate", False, T0)
    d = due(state, T0 + timedelta(hours=72))
    assert d["alert_rate"]["due"] is True
    assert d["alert_rate"]["missed"] == 71, "expected 72 intervals, 1 run"


def test_the_missed_count_is_reported_rather_than_hidden(state):
    """"We have not monitored for three days" is itself a finding, and silently
    catching up conceals it."""
    observe(state, "alert_rate", False, T0)
    d = due(state, T0 + timedelta(hours=72))
    out = observe(state, "alert_rate", False, T0 + timedelta(hours=72),
                  missed=d["alert_rate"]["missed"])
    assert out["missed_runs"] == 71


def test_a_missed_run_does_not_manufacture_breach_history(state):
    """Coalescing must not count the skipped runs as breaches -- that would page
    on the first run back purely because the box was down."""
    observe(state, "alert_rate", True, T0)
    out = observe(state, "alert_rate", True, T0 + timedelta(hours=72), missed=71)
    assert out["consecutive_breaches"] == 2
    assert out["page_now"] is False


# ------------------------------------------------------------ hysteresis
def test_one_breach_does_not_page(state):
    out = observe(state, "score_psi", True, T0)
    assert out["paging"] is False and out["page_now"] is False


def test_three_consecutive_breaches_page(state):
    for i in range(2):
        out = observe(state, "score_psi", True, T0 + timedelta(hours=i))
        assert out["page_now"] is False
    out = observe(state, "score_psi", True, T0 + timedelta(hours=2))
    assert out["page_now"] is True and out["paging"] is True


def test_a_flapping_signal_never_reaches_the_page_threshold(state):
    """PSI over a rolling window crosses a threshold, drops back, crosses again.
    Each crossing was a page; three pages for one event trains the on-call to
    acknowledge without looking."""
    pages = 0
    for i in range(12):
        out = observe(state, "score_psi", i % 2 == 0, T0 + timedelta(hours=i))
        pages += out["page_now"]
    assert pages == 0


def test_staying_breached_does_not_page_again(state):
    """The on-call was already told. Telling them every hour until they fix it
    is how a page becomes noise."""
    pages = 0
    for i in range(20):
        out = observe(state, "score_psi", True, T0 + timedelta(hours=i))
        pages += out["page_now"]
    assert pages == 1


def test_the_signal_resolves_after_enough_clean_runs(state):
    for i in range(3):
        observe(state, "score_psi", True, T0 + timedelta(hours=i))
    out = observe(state, "score_psi", False, T0 + timedelta(hours=3))
    assert out["paging"] is True, "one clean run should not resolve it"
    out = observe(state, "score_psi", False, T0 + timedelta(hours=4))
    assert out["paging"] is False and out["resolved_now"] is True


def test_resolution_fires_once_not_on_every_clean_run(state):
    for i in range(3):
        observe(state, "score_psi", True, T0 + timedelta(hours=i))
    resolutions = 0
    for i in range(3, 12):
        resolutions += observe(state, "score_psi", False,
                               T0 + timedelta(hours=i))["resolved_now"]
    assert resolutions == 1


def test_firing_is_slower_than_forgiving_is_deliberate(state):
    """Asymmetric on purpose: three breaches to page, two clean runs to clear.
    A control that forgives as slowly as it fires stays lit long after the
    incident and stops meaning anything."""
    from src.schedule import BREACHES_TO_PAGE, CLEAN_TO_RESET
    assert BREACHES_TO_PAGE > CLEAN_TO_RESET


def test_the_cost_of_hysteresis_is_computable(state):
    """It DELAYS detection by (N-1) intervals. That is the trade, not a free
    improvement, so a reader can compute it for their own settings."""
    assert detection_delay_minutes("alert_rate") == 2 * 60
    assert detection_delay_minutes("alert_rate", breaches_to_page=1) == 0
    assert detection_delay_minutes("score_psi") == 2 * 360


# -------------------------------------------------------------- cooldown
def test_a_first_retrain_is_allowed(state):
    assert retrain_allowed(state, T0)["allowed"] is True


def test_a_second_retrain_inside_the_cooldown_is_refused(state):
    record_retrain(state, T0)
    out = retrain_allowed(state, T0 + timedelta(hours=10))
    assert out["allowed"] is False
    assert out["hours_remaining"] == pytest.approx(62, abs=0.1)


def test_the_cooldown_expires(state):
    record_retrain(state, T0)
    assert retrain_allowed(state, T0 + timedelta(hours=73))["allowed"] is True


def test_a_fired_trigger_does_not_override_the_cooldown(state):
    """A fired trigger is a reason to CONSIDER retraining. Two fits inside the
    cooldown means the second trains on a window the first already reacted to,
    and neither can be judged against a clean comparison."""
    record_retrain(state, T0)
    for i in range(5):
        observe(state, "alert_rate", True, T0 + timedelta(hours=i))
    assert retrain_allowed(state, T0 + timedelta(hours=5))["allowed"] is False


# ----------------------------------------------------------- persistence
def test_state_survives_a_restart(tmp_path):
    """A scheduler that forgets its breach counts on restart never reaches the
    page threshold, because a restarting box resets the very counter that was
    accumulating."""
    p = tmp_path / "sched.json"
    s = ScheduleState.load(p)
    for i in range(2):
        observe(s, "score_psi", True, T0 + timedelta(hours=i))
    s.save()

    s2 = ScheduleState.load(p)
    assert s2.state_for("score_psi").consecutive_breaches == 2
    out = observe(s2, "score_psi", True, T0 + timedelta(hours=2))
    assert out["page_now"] is True


def test_the_paging_flag_survives_a_restart(tmp_path):
    """Otherwise a restart re-pages for an incident already acknowledged."""
    p = tmp_path / "sched.json"
    s = ScheduleState.load(p)
    for i in range(3):
        observe(s, "score_psi", True, T0 + timedelta(hours=i))
    s.save()

    s2 = ScheduleState.load(p)
    out = observe(s2, "score_psi", True, T0 + timedelta(hours=3))
    assert out["paging"] is True
    assert out["page_now"] is False, "a restart re-paged an acknowledged alert"


def test_the_retrain_clock_survives_a_restart(tmp_path):
    p = tmp_path / "sched.json"
    s = ScheduleState.load(p)
    record_retrain(s, T0)
    s.save()
    assert retrain_allowed(ScheduleState.load(p),
                           T0 + timedelta(hours=1))["allowed"] is False


# ------------------------------------------------- the Evidently caveat
def test_evidently_is_installed_and_does_not_import_on_this_python():
    """Pinned so the caveat is revisited rather than believed forever.

    `monitor.py` says Evidently "would produce a prettier version of 2 and 3"
    and would not produce signal 1. That is a claim about a library this
    environment cannot load: it is installed, and importing it raises a pydantic
    ConfigError under Python 3.14 because Evidently still depends on pydantic
    v1 compatibility shims.

    The fix is not available here -- pinning an older pydantic to satisfy it
    would drag the whole environment backwards, and this project has already
    seen a dependency downgrade for one library break another.

    If this test starts FAILING, Evidently now imports and the comparison in
    monitor.py can finally be made against the real thing instead of asserted.
    """
    import importlib.util

    if importlib.util.find_spec("evidently") is None:
        pytest.skip("evidently is not installed in this environment")

    with pytest.raises(Exception) as exc:
        import evidently  # noqa: F401
    assert "ConfigError" in type(exc.value).__name__ or "pydantic" in str(exc.value), (
        "evidently now fails for a different reason than the recorded pydantic "
        "v1 incompatibility -- the caveat in monitor.py needs rechecking"
    )


def test_the_fastest_signal_is_the_one_evidently_would_not_provide():
    """The substantive half of the same claim, which does not need the library.

    Signal 1 is the alert rate at the FROZEN OPERATING THRESHOLD. A drift
    library compares distributions; it does not know the threshold, because that
    is a fact about the deployment rather than about the data. So the signal
    that catches an adversary probing the boundary is the one no
    distribution-comparison tool can produce -- and it is also the one this
    schedule runs most often.
    """
    from src.schedule import CADENCE_MINUTES

    assert CADENCE_MINUTES["alert_rate"] == min(CADENCE_MINUTES.values()), (
        "the signal a drift library cannot provide should be the one run most "
        "frequently, since it is both the fastest and the least replaceable")
