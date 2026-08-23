"""The retraining pipeline: monitor -> trigger -> retrain -> gate -> STOP.

`monitor.py` proposes retraining and `run_promotion.py` evaluates the gates, but
until now nothing joined them and nothing acted on the verdict. This is the
orchestration, and the interesting design decision is where it deliberately
refuses to go.

THE PIPELINE ENDS AT A HUMAN, ON PURPOSE. Under SR 11-7 a model change requires
independent validation before production use. A pipeline that retrains, gates,
and then deploys has automated the validator out of the loop -- which is not a
faster process, it is a missing control. So the terminal state of a successful
run is `AWAITING_VALIDATION`, not `PROMOTED`, and promotion requires a signature
this code cannot produce. `promote()` refuses to run without one and names who
signed.

Everything else IS automated, because everything else is mechanical: deciding
whether the trigger fired, cutting the training window, fitting the challenger,
running it in shadow, evaluating every gate, and writing an immutable record of
what was decided and on what evidence.

TRIGGERS, and why they are ranked. Retraining on a schedule retrains a model
that was fine and leaves a broken one broken for the rest of its cycle. These
fire on evidence, in the order they can actually page someone:

  1. alert_rate      label-free, same day. The only signal that moves before
                     chargebacks arrive.
  2. score_psi       label-free, same day, but blind to the failure mode where
                     the score distribution holds and the ranking rots.
  3. performance     needs labels, so it is 4-8 weeks late by construction.
  4. calendar        the backstop, and the weakest reason to retrain.

A run records WHICH trigger fired. "The model was retrained in March" is not an
audit record; "the alert rate moved +64% against a frozen threshold on 12 March,
above the +50% tolerance" is.

WHAT THIS DOES NOT DO. There is no scheduler -- something still has to invoke
`run_retraining.py` nightly, and that something is cron, Airflow or a systemd
timer. The refusal to embed one is deliberate: a scheduler inside the
application is a scheduler nobody can see the state of.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

TRIGGER_PRIORITY = ["alert_rate", "score_psi", "performance", "calendar"]

CALENDAR_DAYS = 90


@dataclass
class Trigger:
    name: str
    fired: bool
    evidence: str
    latency: str


@dataclass
class RunRecord:
    run_id: str
    started_at: str
    triggers: list = field(default_factory=list)
    fired: str | None = None
    state: str = "NO_ACTION"
    gates: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    champion_version: str | None = None
    challenger_version: str | None = None
    signed_off_by: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def evaluate_triggers(monitor_report: dict, days_since_last_fit: int,
                      calendar_days: int = CALENDAR_DAYS) -> list[Trigger]:
    """Turn a monitor report into ranked, evidenced triggers."""
    # `monitor.run()` emits only the signals that breached, so absence IS the
    # negative result. A `warn` is recorded but does not fire: retraining on a
    # warn burns the clean comparison window a page will need later.
    found = {f["signal"]: f for f in monitor_report.get("findings", [])}

    def state(name, quiet):
        f = found.get(name)
        if f is None:
            return False, quiet
        return f["severity"] == "page", "{} [{}]".format(f["detail"], f["severity"])

    alert_fired, alert_why = state(
        "alert_rate", "decline rate within {:+.0%} of reference".format(0.50))
    psi_fired, psi_why = state(
        "score_psi", "score PSI {:.4f}, below the 0.25 page band".format(
            monitor_report.get("score_psi", float("nan"))))

    out = [
        Trigger("alert_rate", alert_fired, alert_why,
                "same day, no labels needed"),
        Trigger("score_psi", psi_fired, psi_why,
                "same day, no labels needed"),
        Trigger("performance", False,
                str(monitor_report.get("labelled_metrics",
                                       "no labelled performance signal")),
                "4-8 weeks, gated on chargeback arrival"),
        Trigger("calendar", days_since_last_fit >= calendar_days,
                "{} days since the last fit (policy: {})".format(
                    days_since_last_fit, calendar_days),
                "fires whether or not anything is wrong"),
    ]
    return sorted(out, key=lambda t: TRIGGER_PRIORITY.index(t.name))


def first_fired(triggers: list[Trigger]) -> Trigger | None:
    for t in triggers:
        if t.fired:
            return t
    return None


def new_run(monitor_report: dict, days_since_last_fit: int,
            champion_version: str | None = None,
            now: datetime | None = None) -> RunRecord:
    now = now or datetime.now(timezone.utc)
    triggers = evaluate_triggers(monitor_report, days_since_last_fit)
    fired = first_fired(triggers)
    rec = RunRecord(
        run_id="retrain-{}".format(now.strftime("%Y%m%dT%H%M%SZ")),
        started_at=now.isoformat(),
        triggers=[asdict(t) for t in triggers],
        fired=fired.name if fired else None,
        champion_version=champion_version,
        state="NO_ACTION" if fired is None else "RETRAIN",
    )
    if monitor_report.get("status") == "insufficient_data":
        rec.state = "NO_ACTION"
        rec.fired = None
        rec.notes.append(
            "monitor returned insufficient_data -- retraining on a monitor that "
            "cannot see is worse than not retraining, because it consumes the "
            "one thing a challenger needs, which is a clean comparison window")
    return rec


def record_gates(rec: RunRecord, decision, challenger_version: str) -> RunRecord:
    """Attach the promotion decision. Success means AWAITING_VALIDATION."""
    rec.gates = [{"name": g.name, "requirement": g.requirement,
                  "measured": g.measured, "passed": g.passed,
                  "blocking": g.blocking} for g in decision.gates]
    rec.challenger_version = challenger_version
    if decision.promote:
        rec.state = "AWAITING_VALIDATION"
        rec.notes.append(
            "every gate passed. This is NOT a promotion: SR 11-7 requires "
            "independent validation before production use, and no pipeline can "
            "sign its own model off.")
    else:
        rec.state = "BLOCKED"
        rec.notes.append("blocked on: " + ", ".join(
            g.name for g in decision.failed()))
    return rec


def promote(rec: RunRecord, validated_by: str | None) -> RunRecord:
    """Move AWAITING_VALIDATION -> PROMOTED. Requires a named validator.

    The signature is a string this module cannot manufacture. That is the whole
    control: an automated pipeline that can promote without one has not
    automated validation, it has removed it.
    """
    if rec.state != "AWAITING_VALIDATION":
        raise ValueError(
            "cannot promote from state {!r} -- only AWAITING_VALIDATION".format(
                rec.state))
    if not validated_by or not str(validated_by).strip():
        raise ValueError(
            "promotion requires a named independent validator; refusing to "
            "promote an unsigned model")
    rec.signed_off_by = validated_by
    rec.state = "PROMOTED"
    rec.notes.append("promoted under signature: {}".format(validated_by))
    return rec


def append_run(rec: RunRecord, path: Path) -> None:
    """Append-only run log. A retraining history you can edit is a changelog."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(rec.to_json() + "\n")


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def render(rec: RunRecord) -> str:
    lines = ["run {}  ({})".format(rec.run_id, rec.started_at),
             "-" * 78,
             "{:<14}{:<10}{:<30}{}".format("trigger", "fired", "evidence",
                                           "detection latency")]
    for t in rec.triggers:
        lines.append("{:<14}{:<10}{:<30}{}".format(
            t["name"], "YES" if t["fired"] else "no",
            (t["evidence"] or "")[:29], t["latency"]))
    lines.append("-" * 78)
    lines.append("fired : {}".format(rec.fired or "nothing -- no action"))
    lines.append("state : {}".format(rec.state))
    if rec.gates:
        lines.append("")
        lines.append("{:<22}{:<34}{}".format("gate", "measured", "result"))
        for g in rec.gates:
            lines.append("{:<22}{:<34}{}".format(
                g["name"], g["measured"][:33],
                "PASS" if g["passed"] else ("FAIL" if g["blocking"] else "warn")))
    for n in rec.notes:
        lines.append("  note: " + n)
    return "\n".join(lines)
