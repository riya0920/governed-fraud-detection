"""One scheduled monitoring pass. This is what the systemd timer invokes.

    python run_schedule_tick.py

`run_schedule.py` drives the schedule over simulated time to show what it does.
This is the real thing: it takes the clock as it is, asks `schedule.due()` what
is owed, runs those checks against the live decision log, and records the result.

EXIT CODES ARE THE INTERFACE, because the caller is a timer and not a person:

    0   ran, nothing paging
    10  ran, something PAGED -- the unit lists this as a success (see
        SuccessExitStatus in ops/install_timers.sh) because a page is a signal
        rather than a crash, and a unit marked failed for a working alert is a
        unit somebody disables
    1   the pass itself failed -- could not read the model, the log, or the
        state. A real failure, and the one that should page about ITSELF.

The distinction matters: "the model has drifted" and "the monitoring is broken"
are different incidents with different responders, and collapsing them into one
non-zero exit hides the second behind the first.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.schedule import (ScheduleState, due, observe, record_retrain,
                          retrain_allowed)

STATE = ROOT / "artifacts" / "schedule_state.json"
LOG = ROOT / "artifacts" / "schedule_runs.jsonl"

EXIT_OK = 0
EXIT_PAGED = 10
EXIT_BROKEN = 1


def _monitor_report() -> dict:
    """Run the real monitor. Returns its report, or raises.

    Deliberately NOT wrapped in a try/except that returns an empty report: a
    monitoring pass that cannot read the decision log must fail loudly, because
    an empty report looks exactly like a healthy one.
    """
    import monitor

    return monitor.run()


def _breached(report: dict, check: str) -> bool:
    """Did this check breach, according to the monitor's own findings?

    `monitor.run()` emits only the signals that breached, so absence IS the
    negative result -- the same convention `retraining.evaluate_triggers` reads.
    A `warn` is present but does not count as a breach here, for the reason
    that module gives: reacting to a warn burns the clean comparison window a
    page will need later.
    """
    for f in report.get("findings", []):
        if f.get("signal") == check and f.get("severity") == "page":
            return True
    return False


def main() -> int:
    now = datetime.now(timezone.utc)
    state = ScheduleState.load(STATE)

    try:
        report = _monitor_report()
    except Exception as exc:                                 # noqa: BLE001
        # The pass itself is broken. This is a DIFFERENT incident from drift and
        # a different responder, so it gets its own exit code rather than being
        # folded in with a page.
        record = {"at": now.isoformat(), "outcome": "broken",
                  "error": "{}: {}".format(type(exc).__name__, exc)}
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print("MONITORING PASS FAILED: {}".format(record["error"]),
              file=sys.stderr)
        print("This is not drift. The monitor could not run at all, which is a "
              "separate incident -- an empty report looks exactly like a "
              "healthy one, so it must never be reported as one.",
              file=sys.stderr)
        return EXIT_BROKEN

    owed = due(state, now)
    ran, paged, resolved = [], [], []

    for check, info in owed.items():
        if not info["due"]:
            continue
        out = observe(state, check, _breached(report, check), now,
                      missed=info["missed"])
        ran.append(check)
        if out["page_now"]:
            paged.append(out)
        if out["resolved_now"]:
            resolved.append(out)

    retrained = None
    if paged:
        allowed = retrain_allowed(state, now)
        if allowed["allowed"]:
            record_retrain(state, now)
            retrained = "started"
        else:
            retrained = "refused: {}".format(allowed["reason"])

    state.save()

    record = {
        "at": now.isoformat(),
        "outcome": "paged" if paged else "ok",
        "checks_run": ran,
        "checks_skipped": [c for c, i in owed.items() if not i["due"]],
        "missed_runs": {c: i["missed"] for c, i in owed.items() if i["missed"]},
        "paged": [p["check"] for p in paged],
        "resolved": [r["check"] for r in resolved],
        "retrain": retrained,
        "status": report.get("status"),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    print("{}  ran={} skipped={} paged={} resolved={}".format(
        now.isoformat(timespec="seconds"), len(ran),
        len(record["checks_skipped"]), len(paged), len(resolved)))
    for c, n in record["missed_runs"].items():
        print("  {} came back after {} missed run(s) -- coalesced, not "
              "back-filled".format(c, n))
    for p in paged:
        print("  PAGE {} after {} consecutive breach(es)".format(
            p["check"], p["consecutive_breaches"]))
    for r in resolved:
        print("  resolved {}".format(r["check"]))
    if retrained:
        print("  retrain: {}".format(retrained))

    return EXIT_PAGED if paged else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
