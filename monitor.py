"""Production monitoring job: runs against the DECISION LOG, not a data dump.

This is the executable half of the monitoring plan in docs/MODEL_VALIDATION.md.
It reads what the API actually scored (`artifacts/decisions.jsonl`) and compares
it against the reference window frozen into the model artifact at training time.

The signal ordering here is the finding from RESULTS.md section 6, turned into
code. In priority order:

  1. ALERT RATE at the frozen threshold      -- needs no labels. Fastest signal.
  2. SCORE distribution PSI                  -- needs no labels.
  3. FEATURE PSI                             -- needs no labels, least sensitive.
  4. Precision decay                         -- needs labels, so it is weeks late
                                                and cannot page anyone in time.

Evidently would produce a prettier version of 2 and 3. It would not produce 1,
which is the one that actually catches an adversary, because 1 requires knowing
the operating threshold -- a fact about the deployment, not about the data.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.metrics import psi, psi_band

ARTIFACT = ROOT / "artifacts" / "model.pkl"
AUDIT_LOG = ROOT / "artifacts" / "decisions.jsonl"

# Alert thresholds. These are the numbers an on-call engineer is woken by, so
# they are declared here rather than discovered in a dashboard config.
ALERT_RATE_REL_TOLERANCE = 0.50   # +/-50% relative move in decline rate
SCORE_PSI_WARN = 0.10
SCORE_PSI_PAGE = 0.25
MIN_SAMPLE = 200                  # below this, say "insufficient data", never "ok"


def load_decisions(path: Path = AUDIT_LOG) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run(decisions: list[dict] | None = None) -> dict:
    with ARTIFACT.open("rb") as fh:
        art = pickle.load(fh)
    rows = decisions if decisions is not None else load_decisions()

    if len(rows) < MIN_SAMPLE:
        return {
            "status": "insufficient_data",
            "n": len(rows),
            "required": MIN_SAMPLE,
            "note": ("Not enough production traffic to make a call. This is "
                     "deliberately not reported as 'healthy' -- an empty monitor "
                     "that shows green is worse than one that shows nothing."),
        }

    scores = np.array([r["score"] for r in rows], dtype=float)
    declines = np.array([r["decision"] == "decline" for r in rows])
    ref_scores = np.asarray(art["reference_scores"], dtype=float)
    ref_decline_rate = float((ref_scores >= art["threshold"]).mean())
    live_decline_rate = float(declines.mean())

    findings = []

    # -- 1. alert rate at the frozen threshold ------------------------------
    if ref_decline_rate > 0:
        rel = (live_decline_rate - ref_decline_rate) / ref_decline_rate
    else:
        rel = 0.0
    alert_breach = abs(rel) > ALERT_RATE_REL_TOLERANCE
    if alert_breach:
        findings.append({
            "signal": "alert_rate", "severity": "page",
            "detail": ("decline rate {:.4%} vs reference {:.4%} ({:+.1%} relative). "
                       "No labels needed: something upstream changed -- feature "
                       "pipeline, traffic mix, or an adversary."
                       .format(live_decline_rate, ref_decline_rate, rel)),
        })

    # -- 2. score PSI --------------------------------------------------------
    score_psi = psi(ref_scores, scores)
    if score_psi >= SCORE_PSI_PAGE:
        findings.append({"signal": "score_psi", "severity": "page",
                         "detail": "score PSI {:.4f} ({})".format(score_psi,
                                                                 psi_band(score_psi))})
    elif score_psi >= SCORE_PSI_WARN:
        findings.append({"signal": "score_psi", "severity": "warn",
                         "detail": "score PSI {:.4f} ({})".format(score_psi,
                                                                 psi_band(score_psi))})

    # -- 3. feature PSI ------------------------------------------------------
    ref_X = np.asarray(art["reference_X"], dtype=float)
    feature_psi = {}
    for i, name in enumerate(art["features"]):
        live = np.array([r["features"][name] for r in rows], dtype=float)
        v = psi(ref_X[:, i], live)
        feature_psi[name] = v
        if v >= SCORE_PSI_PAGE:
            findings.append({"signal": "feature_psi", "severity": "warn",
                             "detail": "{} PSI {:.4f} ({})".format(name, v, psi_band(v))})

    return {
        "status": "page" if any(f["severity"] == "page" for f in findings)
        else ("warn" if findings else "healthy"),
        "n": len(rows),
        "model_version": art["model_version"],
        "threshold": art["threshold"],
        "decline_rate_live": live_decline_rate,
        "decline_rate_reference": ref_decline_rate,
        "decline_rate_relative_change": rel,
        "score_psi": score_psi,
        "feature_psi": feature_psi,
        "findings": findings,
        "labelled_metrics": ("not computed: fraud labels arrive weeks late via "
                             "chargeback. Precision decay is a retrospective "
                             "confirmation, never an early warning."),
    }


def render(report: dict) -> str:
    if report["status"] == "insufficient_data":
        return "MONITOR: insufficient data ({}/{} decisions)\n{}".format(
            report["n"], report["required"], report["note"])
    lines = [
        "MONITOR  status={}  n={:,}  model={}".format(
            report["status"].upper(), report["n"], report["model_version"]),
        "-" * 72,
        "decline rate   live {:.4%}   reference {:.4%}   change {:+.1%}".format(
            report["decline_rate_live"], report["decline_rate_reference"],
            report["decline_rate_relative_change"]),
        "score PSI      {:.4f}  ({})".format(report["score_psi"],
                                             psi_band(report["score_psi"])),
        "-" * 72,
    ]
    for name, v in sorted(report["feature_psi"].items(), key=lambda kv: -kv[1]):
        lines.append("  feature PSI  {:<22} {:.4f}  {}".format(name, v, psi_band(v)))
    lines.append("-" * 72)
    if report["findings"]:
        for f in report["findings"]:
            lines.append("  [{}] {}: {}".format(
                f["severity"].upper(), f["signal"], f["detail"]))
    else:
        lines.append("  no findings")
    lines.append("-" * 72)
    lines.append("  " + report["labelled_metrics"])
    return "\n".join(lines)


if __name__ == "__main__":
    print(render(run()))
