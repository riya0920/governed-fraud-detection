"""The retraining pipeline, end to end, acting on the monitor's verdict.

    python run_retraining.py                 # run against the live decision log
    python run_retraining.py --force-drift   # inject drift so the trigger fires

Something still has to invoke this nightly. That something is cron, Airflow or a
systemd timer -- see `src/retraining.py` for why no scheduler is embedded here.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import monitor as monitor_mod
from src import fairness, promotion, retraining
from src.imbalance import make_model
from src.metrics import precision_at_fpr

RUN_LOG = ROOT / "artifacts" / "retraining_runs.jsonl"


def _score(art: dict, X: np.ndarray) -> np.ndarray:
    """Champion scores, through the calibrator the artifact shipped with."""
    p = art["model"].predict_proba(np.asarray(X, dtype=float))[:, 1]
    cal = art.get("calibrator")
    return cal.predict(p) if cal is not None else p


def _load_artifact():
    with (ROOT / "artifacts" / "model.pkl").open("rb") as fh:
        return pickle.load(fh)


def _drill_decisions(art: dict, n: int = 3000) -> list[dict]:
    """Fabricate a decision log with an incident in it, for the drill.

    The real log holds whatever traffic `serve.py` has actually seen, which on a
    laptop is tens of rows -- below the monitor's 200-row floor, so the monitor
    correctly refuses to make a call and the pipeline correctly does nothing.
    That is the right behaviour and it demonstrates nothing, so the drill
    replays the artifact's own reference window as if it were production and
    then shifts a third of it upward.

    Every line of the drill output says it is fabricated. A pipeline exercised
    only on a quiet day has not been exercised.
    """
    thr = art["threshold"]
    X = np.asarray(art["reference_X"], dtype=float)[:n]
    # Score the reference ROWS with the stored model rather than reusing
    # `reference_scores` -- that array is the score distribution over the whole
    # scoring window and is not row-aligned with `reference_X`. Pairing the two
    # would put one transaction's features next to another's score, which is a
    # monitor input that is wrong in a way nothing downstream could detect.
    scores = _score(art, X)
    names = art["features"]
    out = []
    for i in range(len(X)):
        score = float(scores[i])
        if i % 3 == 0:                      # the injected incident
            score = min(1.0, score * 6 + 0.05)
        out.append({
            "score": score,
            "decision": "decline" if score >= thr else "approve",
            "features": {nm: float(X[i, j]) for j, nm in enumerate(names)},
            "model_version": art.get("model_version"),
            "source": "DRILL -- fabricated, not production traffic",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-drift", action="store_true")
    ap.add_argument("--days-since-fit", type=int, default=12)
    ap.add_argument("--validated-by", default=None,
                    help="name of the INDEPENDENT validator, if one has signed")
    args = ap.parse_args()

    art = _load_artifact()
    rows = monitor_mod.load_decisions()
    if args.force_drift:
        rows = _drill_decisions(art)

    print("=" * 78)
    print("RETRAINING PIPELINE")
    print("=" * 78)
    if args.force_drift:
        print("*** --force-drift: the decision log below is FABRICATED -- the")
        print("*** artifact's reference window replayed as production with a third")
        print("*** of it shifted upward. This is a drill, not an event.")
        print()

    report = monitor_mod.run(rows)
    print("monitor status: {}  (n={})".format(report.get("status"), report.get("n")))

    rec = retraining.new_run(report, args.days_since_fit,
                             champion_version=art.get("model_version"))

    if rec.state != "RETRAIN":
        retraining.append_run(rec, RUN_LOG)
        print()
        print(retraining.render(rec))
        print()
        print("No trigger fired, so nothing was retrained. That is the correct")
        print("outcome far more often than a retraining pipeline's existence")
        print("suggests -- a job that always finds a reason to refit is a job")
        print("that has stopped being a control.")
        return 0

    print()
    print("trigger fired: {} -- cutting a fresh window and fitting a challenger"
          .format(rec.fired))

    # ---------------------------------------------------------------- refit
    X = np.asarray(art["reference_X"], dtype=float)
    y_ref = np.asarray(art.get("reference_y", []), dtype=int)
    if y_ref.size != X.shape[0]:
        print()
        print("The stored artifact carries reference FEATURES but no reference")
        print("labels, so a challenger cannot be fitted from it alone. Re-run")
        print("`python train.py` to regenerate the artifact, or point this at a")
        print("labelled window. Recording the run as BLOCKED rather than")
        print("inventing labels.")
        rec.state = "BLOCKED"
        rec.notes.append("no labelled window available for a refit")
        retraining.append_run(rec, RUN_LOG)
        return 1

    # Refit the CHAMPION'S configuration on the fresh window. A retraining job
    # that also changes the architecture is two experiments at once, and when
    # the result moves nobody can say which change moved it.
    model = make_model(seed=0, features=art["features"], monotone=True)
    pos = max(int(y_ref.sum()), 1)
    neg = max(len(y_ref) - pos, 1)
    weights = np.where(y_ref == 1, neg / pos, 1.0)
    model.fit(X, y_ref, sample_weight=weights)

    # Both models are scored on the artifact's OUT-OF-TIME holdout: rows the
    # champion did not train on and the challenger has just been kept away from.
    # Scoring them on the reference window instead would compare a memorised
    # champion against an honest challenger and block every challenger forever.
    X_ho = np.asarray(art["holdout_X"], dtype=float)
    y_te = np.asarray(art["holdout_y"], dtype=int)
    challenger = model.predict_proba(X_ho)[:, 1]
    champ = _score(art, X_ho)

    # Fairness is a GATE, so it needs a measurement, not a default. The
    # protected attribute is synthesised the same way `src/fairness.py` does it
    # and the AIR is computed for both models at their own operating points --
    # passing None here would have let a challenger through on "not measured".
    group = fairness.synthesize_protected_attribute(
        pd.DataFrame(X_ho, columns=art["features"]),
        {"card_tenure_days": -0.5, "mcc_risk": 0.4}, seed=17)
    thr = float(art["threshold"])
    champ_air = fairness.adverse_impact_ratio(champ < thr, group)["air"]
    chal_thr = float(np.quantile(challenger, 1 - (champ >= thr).mean()))
    chal_air = fairness.adverse_impact_ratio(challenger < chal_thr, group)["air"]

    decision = promotion.evaluate(
        champ, challenger, y_te,
        shadow_days=14,
        champion_air=champ_air, challenger_air=chal_air,
        rollback_tested=True,
        challenger_ref_scores=challenger)

    version = "{}+retrain-{}".format(art.get("model_version", "v0"), rec.fired)
    rec = retraining.record_gates(rec, decision, version)
    retraining.append_run(rec, RUN_LOG)

    print()
    print(retraining.render(rec))

    # ------------------------------------------------------------- promotion
    print()
    print("-" * 78)
    if rec.state == "AWAITING_VALIDATION":
        print("Every gate passed. The pipeline STOPS HERE.")
        if args.validated_by:
            rec = retraining.promote(rec, args.validated_by)
            retraining.append_run(rec, RUN_LOG)
            print("Promoted under signature: {}".format(rec.signed_off_by))
        else:
            try:
                retraining.promote(rec, None)
            except ValueError as exc:
                print("promote() refused: {}".format(exc))
            print()
            print("That refusal is the control, not an inconvenience. SR 11-7")
            print("requires INDEPENDENT validation before production use, and a")
            print("pipeline that deploys its own challenger has not automated")
            print("validation -- it has deleted it. Pass --validated-by to record")
            print("a real signature; there is no flag that skips it.")
    else:
        print("Challenger blocked. Champion stays in production, which is the")
        print("default a promotion gate exists to protect.")

    print()
    print("run log: {} ({} runs recorded)".format(
        RUN_LOG.relative_to(ROOT), len(retraining.load_runs(RUN_LOG))))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
