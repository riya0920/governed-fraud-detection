"""Running the monitor on a cadence, without paging anyone three times.

`monitor.py` produces the report and `src/retraining.py` turns it into ranked
triggers. Nothing ran either of them. A monitoring plan that executes when
somebody remembers to execute it is a document, and the specific failure is that
it gets run *after* an incident, to explain it, rather than before one, to catch
it.

FOUR THINGS THIS DECIDES, and the first is the one that separates a scheduler
from a cron line.

1. CADENCE FOLLOWS SIGNAL LATENCY, NOT CONVENIENCE.

   `monitor.py` ranks its signals by how fast they can possibly fire: alert rate
   and score PSI need no labels, feature PSI needs none either, and precision
   decay waits weeks for chargebacks. Running all four on one cadence throws
   away the fast ones. Running the label-dependent check hourly is worse than
   useless -- it recomputes the same stale answer sixty times a day and teaches
   whoever reads it that the number never moves.

2. A MISSED MONITORING RUN IS NOT BACKFILLED, AND THIS IS THE OPPOSITE OF
   SE-2's SETTLEMENT CATCH-UP.

   A settlement cycle must process every missed date: the state is cumulative
   and skipping one leaves a hole. Monitoring is not cumulative. If the box was
   down for three days, replaying three days of monitor runs pages the on-call
   for drift that has already resolved and buries the current state under
   history. What matters is the answer NOW.

   So missed runs COALESCE into one. The count of skipped runs is reported
   rather than hidden, because "we have not monitored for three days" is itself
   a finding and silently catching up conceals it.

3. HYSTERESIS, BECAUSE A FLAPPING CHECK PAGES REPEATEDLY.

   PSI computed over a rolling window crosses a threshold, drops back, crosses
   again. Each crossing is a page. Three pages for one event trains the on-call
   to acknowledge without looking, which is the state in which the fourth page
   -- the real one -- is also ignored. A signal must breach on N CONSECUTIVE
   runs before it pages, and must clear for M before it resets.

   The cost is honest and stated: hysteresis DELAYS detection by (N-1) intervals.
   That is the trade being made, not a free improvement.

4. RETRAINING HAS A COOLDOWN THAT THE ALERT CANNOT OVERRIDE.

   A fired trigger is a reason to consider retraining, not a reason to retrain
   now. Two retrains in a day means the second is fitted on a window the first
   already reacted to, and there is no clean comparison left to judge either.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Cadences, set from how fast each signal CAN move rather than from taste.
CADENCE_MINUTES = {
    # Needs no labels and is the only signal that catches an adversary probing
    # the threshold. Fastest thing available, so it runs fastest.
    "alert_rate": 60,
    # Needs no labels. Distribution shift is slower than a rate change and a
    # short window is mostly noise.
    "score_psi": 360,          # 6 hours
    # Needs no labels, least sensitive of the three.
    "feature_psi": 1440,       # daily
    # Needs chargebacks, which arrive over 4-8 weeks. Running it hourly
    # recomputes the same stale answer and teaches the reader it never moves.
    "performance": 10080,      # weekly
}

# Consecutive breaches before a signal pages, and consecutive clean runs before
# it resets. Asymmetric on purpose: slow to fire, slower to forgive.
BREACHES_TO_PAGE = 3
CLEAN_TO_RESET = 2

RETRAIN_COOLDOWN_HOURS = 72


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CheckState:
    name: str
    last_run: str | None = None
    consecutive_breaches: int = 0
    consecutive_clean: int = 0
    paging: bool = False
    skipped_runs: int = 0


@dataclass
class ScheduleState:
    path: Path
    checks: dict = field(default_factory=dict)
    last_retrain: str | None = None

    @classmethod
    def load(cls, path: Path | str) -> "ScheduleState":
        p = Path(path)
        if not p.exists():
            return cls(p)
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls(p,
                   {k: CheckState(**v) for k, v in raw.get("checks", {}).items()},
                   raw.get("last_retrain"))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "checks": {k: vars(v) for k, v in self.checks.items()},
            "last_retrain": self.last_retrain,
        }, indent=1), encoding="utf-8")

    def state_for(self, name: str) -> CheckState:
        return self.checks.setdefault(name, CheckState(name))


# ------------------------------------------------------------------ due
def due(state: ScheduleState, now: datetime,
        cadence: dict | None = None) -> dict:
    """Which checks are due, and how many runs each one missed.

    The missed count is returned rather than turned into extra runs. Monitoring
    is not cumulative -- replaying three days of it pages for drift that has
    already resolved -- so the runs coalesce into one and the skip count becomes
    a finding in its own right.
    """
    cadence = cadence or CADENCE_MINUTES
    out = {}
    for name, minutes in cadence.items():
        cs = state.state_for(name)
        if cs.last_run is None:
            out[name] = {"due": True, "missed": 0, "reason": "never run"}
            continue
        elapsed = now - datetime.fromisoformat(cs.last_run)
        intervals = elapsed / timedelta(minutes=minutes)
        if intervals < 1:
            out[name] = {"due": False, "missed": 0,
                         "reason": "next due in {:.0f} min".format(
                             minutes - elapsed.total_seconds() / 60)}
        else:
            out[name] = {"due": True, "missed": int(intervals) - 1,
                         "reason": "{:.1f} intervals elapsed".format(intervals)}
    return out


# ------------------------------------------------------------ hysteresis
def observe(state: ScheduleState, name: str, breached: bool, now: datetime,
            breaches_to_page: int = BREACHES_TO_PAGE,
            clean_to_reset: int = CLEAN_TO_RESET,
            missed: int = 0) -> dict:
    """Record one run's result and say whether it pages.

    A signal pages on the run where it reaches `breaches_to_page` consecutive
    breaches, and ONLY on that run -- staying breached does not page again. The
    on-call was already told; telling them every hour until they fix it is how a
    page becomes noise.
    """
    cs = state.state_for(name)
    cs.last_run = now.isoformat()
    cs.skipped_runs += missed

    was_paging = cs.paging
    if breached:
        cs.consecutive_breaches += 1
        cs.consecutive_clean = 0
    else:
        cs.consecutive_clean += 1
        cs.consecutive_breaches = 0

    if not cs.paging and cs.consecutive_breaches >= breaches_to_page:
        cs.paging = True
    elif cs.paging and cs.consecutive_clean >= clean_to_reset:
        cs.paging = False

    return {
        "check": name,
        "breached": breached,
        "consecutive_breaches": cs.consecutive_breaches,
        "consecutive_clean": cs.consecutive_clean,
        "paging": cs.paging,
        # The EDGES are the events. A page fires on the transition into paging,
        # and a resolution on the transition out. Anything that reports "paging"
        # on every run has reinvented the flapping it was built to stop.
        "page_now": cs.paging and not was_paging,
        "resolved_now": was_paging and not cs.paging,
        "missed_runs": cs.skipped_runs,
    }


def detection_delay_minutes(name: str, breaches_to_page: int = BREACHES_TO_PAGE,
                            cadence: dict | None = None) -> int:
    """What the hysteresis costs, in minutes of delayed detection.

    Stated as a function rather than a comment because it is the price of the
    feature and a reader should be able to compute it for their own settings.
    """
    cadence = cadence or CADENCE_MINUTES
    return (breaches_to_page - 1) * cadence[name]


# -------------------------------------------------------------- cooldown
def retrain_allowed(state: ScheduleState, now: datetime,
                    cooldown_hours: int = RETRAIN_COOLDOWN_HOURS) -> dict:
    """A fired trigger is a reason to CONSIDER retraining, not to retrain now.

    Two retrains inside a cooldown means the second is fitted on a window the
    first already reacted to, and neither can be judged against a clean
    comparison.
    """
    if state.last_retrain is None:
        return {"allowed": True, "reason": "no previous retrain on record"}
    elapsed = now - datetime.fromisoformat(state.last_retrain)
    hours = elapsed.total_seconds() / 3600
    if hours >= cooldown_hours:
        return {"allowed": True,
                "reason": "{:.1f}h since the last retrain (cooldown {}h)".format(
                    hours, cooldown_hours)}
    return {"allowed": False,
            "reason": "only {:.1f}h since the last retrain; cooldown is {}h. A "
                      "second fit inside the cooldown trains on a window the "
                      "first already reacted to.".format(hours, cooldown_hours),
            "hours_remaining": cooldown_hours - hours}


def record_retrain(state: ScheduleState, now: datetime) -> None:
    state.last_retrain = now.isoformat()
