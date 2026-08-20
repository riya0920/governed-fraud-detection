"""Shadow a challenger against the champion and run the promotion gates.

Demonstrates the case that matters: a challenger that WINS on AUC and is still
blocked, because the gate it fails is the one the deployment actually depends on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import fairness, imbalance, metrics, promotion
from train import FEATURES, TARGET_FPR, TRAIN_END_DAY, engineer, load


def main() -> int:
    df = engineer(load())
    tr = df[df.day < TRAIN_END_DAY]
    te = df[df.day >= TRAIN_END_DAY]
    X_tr, y_tr = tr[FEATURES].to_numpy(float), tr.is_fraud.to_numpy()
    X_te, y_te = te[FEATURES].to_numpy(float), te.is_fraud.to_numpy()

    # Split the out-of-time window: first half is the shadow period the gates
    # are evaluated on, second half is held out for calibration. Calibrating on
    # the shadow window itself would produce a curve that looks perfect and
    # means nothing.
    mid = len(te) // 2
    shadow = slice(0, mid)
    calib = slice(mid, len(te))

    champion, _ = imbalance.fit_strategy("baseline", X_tr, y_tr)
    challenger, note = imbalance.fit_strategy("class_weight", X_tr, y_tr)

    ch_scores = champion.predict_proba(X_te)[:, 1]
    cl_scores = challenger.predict_proba(X_te)[:, 1]

    print("=" * 92)
    print("SHADOW PERIOD: challenger '{}' vs champion 'baseline'".format(note))
    print("-" * 92)
    print("shadow rows     : {:,}  (decisions logged, not acted on)".format(mid))
    print("calibration rows: {:,}  (held out)".format(len(te) - mid))

    grp = fairness.synthesize_protected_attribute(
        te, correlates={"card_tenure_days": -0.85, "mcc_risk": 0.55})
    thr_ch = float(np.quantile(ch_scores, 0.99))
    thr_cl = float(np.quantile(cl_scores, 0.99))
    air_ch = fairness.adverse_impact_ratio(ch_scores < thr_ch, grp)["air"]
    air_cl = fairness.adverse_impact_ratio(cl_scores < thr_cl, grp)["air"]

    decision = promotion.evaluate(
        ch_scores[shadow], cl_scores[shadow], y_te[shadow],
        target_fpr=TARGET_FPR, shadow_days=15,
        champion_air=air_ch, challenger_air=air_cl,
        challenger_ref_scores=challenger.predict_proba(X_tr)[:, 1])

    print("\n" + "=" * 92)
    print("PROMOTION GATES")
    print("=" * 92)
    print(decision.render())

    # ---- recalibrate and re-run -------------------------------------------
    failed = [g.name for g in decision.failed()]
    if "calibration" in failed:
        print("\n" + "=" * 92)
        print("RECALIBRATION")
        print("=" * 92)
        print("The challenger ranks comparably but its probabilities are wrong --")
        print("class weighting shifts the base rate it was fit on, so its output is")
        print("a probability of fraud in a reweighted world that does not exist.")
        print("Discarding it would throw away the hard part; the fix is isotonic")
        print("regression fitted on the HELD-OUT window.\n")

        iso = promotion.recalibrate(cl_scores[calib], y_te[calib])
        cl_cal = iso.predict(cl_scores)

        print("{:<26}{:>14}{:>14}".format("", "before", "after"))
        print("{:<26}{:>14.5f}{:>14.5f}".format(
            "Brier (challenger)", metrics.brier(y_te[shadow], cl_scores[shadow]),
            metrics.brier(y_te[shadow], cl_cal[shadow])))
        print("{:<26}{:>14.5f}{:>14.5f}".format(
            "Brier (champion)", metrics.brier(y_te[shadow], ch_scores[shadow]),
            metrics.brier(y_te[shadow], ch_scores[shadow])))
        print("{:<26}{:>14.4f}{:>14.4f}".format(
            "AUC (challenger)",
            metrics.full_report(y_te[shadow], cl_scores[shadow])["auc"],
            metrics.full_report(y_te[shadow], cl_cal[shadow])["auc"]))
        print("\nIsotonic regression is monotone, so AUC is unchanged by")
        print("construction -- it re-labels the scores without re-ordering them.")
        print("That is exactly why it is the right tool here: the ranking was")
        print("never the problem.")

        after = promotion.evaluate(
            ch_scores[shadow], cl_cal[shadow], y_te[shadow],
            target_fpr=TARGET_FPR, shadow_days=15,
            champion_air=air_ch, challenger_air=air_cl,
            challenger_ref_scores=iso.predict(
                challenger.predict_proba(X_tr)[:, 1]))
        print("\n" + "-" * 92)
        print(after.render())

    print("\n" + "=" * 92)
    print("A single failed gate blocks promotion. There is no aggregate score,")
    print("because an aggregate lets a strong result on one gate buy a weak one on")
    print("another -- which is how a miscalibrated model reaches production with a")
    print("good headline number.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
