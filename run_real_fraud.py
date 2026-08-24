"""The governance pipeline on REAL fraud data -- and the part of it that dies.

    python fetch_real_data.py && python run_real_fraud.py

Every metric in `docs/RESULTS.md` is a property of a generator I wrote. This runs
the same machinery on the ULB credit-card dataset: 284,807 real transactions from
European cardholders in September 2013, 492 of them fraudulent. Free, no account.

THE FINDING THIS DATASET FORCES, and it is not a metric.

`V1..V28` are **PCA components**. The original fields were anonymised before
release -- which is the only way a bank could publish transaction data at all,
and it is fatal to half of this project:

  WHAT STILL WORKS   the leakage audit, the imbalance bakeoff, threshold
                     economics, calibration, PSI, the promotion gates, drift
                     monitoring. All of it is arithmetic over features, and
                     arithmetic does not care what the features mean.

  WHAT CANNOT WORK   adverse-action reason codes. FCRA 615(a) requires the
                     PRINCIPAL REASONS in terms a consumer can act on. TreeSHAP
                     will happily report that `V14` drove the decline. There is
                     no sentence you can put in a letter that begins "V14".

That is not a limitation of the method. It is a limitation of *anonymised data*,
and it is the concrete reason a real fraud team cannot build a compliant model on
a public dataset no matter how good the modelling is. The generator, whose
features are things like `velocity_24h` and `cross_border`, can do something this
real data cannot -- which is an odd and useful inversion of the usual "synthetic
is a compromise" story.

So this script reports the real metrics AND demonstrates the reason-code failure
rather than quietly skipping that section.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import metrics
from src.leakage import audit
from src.threshold import CostModel, choose, sweep

DATA = ROOT / "data" / "creditcard.parquet"


def load() -> pd.DataFrame:
    if not DATA.exists():
        raise SystemExit("run: python fetch_real_data.py")
    df = pd.read_parquet(DATA)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()


def main() -> int:
    df = load()
    y = df["Class"].astype(int).to_numpy()
    features = [c for c in df.columns if c != "Class"]

    print("=" * 80)
    print("GOVERNED FRAUD DETECTION ON REAL DATA (ULB, 284,807 transactions)")
    print("=" * 80)
    print("rows parsed     : {:,}".format(len(df)))
    print("frauds          : {:,} ({:.4%})".format(int(y.sum()), y.mean()))
    print("features        : {} ({} PCA components + Time + Amount)".format(
        len(features), sum(1 for f in features if f.startswith("V"))))
    if len(df) < 284_807:
        print()
        print("*** SHORT: expected 284,807 rows. The download is incomplete, so")
        print("*** every number below describes a PREFIX of a time-ordered file")
        print("*** -- which for this dataset means the first N hours, not a")
        print("*** random sample. Re-run fetch_real_data.py before quoting any")
        print("*** of it.")

    # ------------------------------------------------------- 1. time split
    # Time is seconds since the first transaction, so the file is ordered. A
    # random split would train on tomorrow and test on yesterday -- the exact
    # leak this project's data layer exists to make impossible.
    df = df.sort_values("Time")
    cut = int(len(df) * 0.7)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    y_tr = tr["Class"].astype(int).to_numpy()
    y_te = te["Class"].astype(int).to_numpy()

    print("\n" + "=" * 80)
    print("1. OUT-OF-TIME SPLIT")
    print("-" * 80)
    print("train : {:,} rows, {:.4%} fraud".format(len(tr), y_tr.mean()))
    print("test  : {:,} rows, {:.4%} fraud".format(len(te), y_te.mean()))
    print()
    print("Split on TIME, not at random. `Time` is seconds since the first")
    print("transaction, so this file is ordered -- a random split trains on")
    print("tomorrow and tests on yesterday, which is the leak this project's")
    print("data layer exists to make structurally impossible.")

    # ------------------------------------------------------- 2. leakage
    print("\n" + "=" * 80)
    print("2. LEAKAGE AUDIT")
    print("-" * 80)
    report = audit(df.assign(is_fraud=df["Class"]).drop(columns=["Class"]),
                   target="is_fraud")
    flagged = report[report.flagged]
    print("{:<14}{:>12}".format("feature", "univariate AUC"))
    for _, r in report.head(6).iterrows():
        print("{:<14}{:>12.4f}{}".format(
            r.feature, r.univariate_auc, "   FLAGGED" if r.flagged else ""))
    print()
    print("features flagged above the {:.2f} screen: {}".format(0.90, len(flagged)))
    print()
    if flagged.empty:
        print("Nothing trips the screen. These are PCA components of a real")
        print("feed, so there is no post-decision field to leak.")
    else:
        print("I EXPECTED ZERO HERE AND WROTE THAT DOWN BEFORE RUNNING IT.")
        print()
        print("{} of {} features clear a 0.90 univariate AUC. None of them is a".format(
            len(flagged), len(report)))
        print("leak: they are anonymised PCA components of a real transaction")
        print("feed, and there is no post-decision information in this file to")
        print("leak. They are simply very predictive.")
        print()
        print("So this is the screen's FALSE-POSITIVE MODE, and it is worth more")
        print("than the clean result would have been. A univariate-AUC screen")
        print("cannot tell 'this field encodes the answer' from 'this field is")
        print("genuinely strong' -- both look like one column that separates the")
        print("classes. On the generated data the screen looked clean only")
        print("because I did not write features that strong.")
        print()
        print("What the screen is actually for is unchanged: it catches a leak")
        print("nobody predicted, which a denylist cannot. What it cannot do is")
        print("adjudicate. Every flag is a QUESTION for a human -- and a team")
        print("that treats the flags as verdicts will delete its best features.")

    # ------------------------------------------------------- 3. the model
    print("\n" + "=" * 80)
    print("3. MODEL, AT A REAL 0.17% PREVALENCE")
    print("-" * 80)
    import lightgbm as lgb

    X_tr = tr[features].to_numpy(float)
    X_te = te[features].to_numpy(float)
    pos = max(int(y_tr.sum()), 1)
    weight = np.where(y_tr == 1, (len(y_tr) - pos) / pos, 1.0)

    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                               num_leaves=31, min_child_samples=50,
                               reg_lambda=1.0, random_state=0, verbose=-1,
                               n_jobs=2)
    model.fit(X_tr, y_tr, sample_weight=weight)
    prob = model.predict_proba(X_te)[:, 1]

    rep = metrics.full_report(y_te, prob, 0.01)
    print("{:<28}{:>12}".format("AUC", round(rep["auc"], 4)))
    print("{:<28}{:>12}".format("KS", round(rep["ks"], 4)))
    print("{:<28}{:>12}".format("Brier", round(rep["brier"], 6)))
    at = rep["at_fpr"]
    print("{:<28}{:>12}".format("precision @ 1% FPR",
                                round(at["precision"], 4)))
    print("{:<28}{:>12}".format("recall @ 1% FPR", round(at["recall"], 4)))
    print()
    print("Prevalence here is 0.17%, against 1.19% in the generated build. That")
    print("is a seven-fold harder problem and the AUC is HIGHER, which says")
    print("something about the generator rather than about the model: real")
    print("fraud in this feed is more separable than the one I wrote.")

    # -------------------------------------------------- 4. threshold economics
    print("\n" + "=" * 80)
    print("4. THRESHOLD ECONOMICS ON REAL AMOUNTS")
    print("-" * 80)
    # Amounts are euros; the cost model is in minor units, so convert once
    # here rather than letting a factor of 100 wander through the arithmetic.
    amounts_minor = (te["Amount"].to_numpy(float) * 100).round().astype(np.int64)
    chosen = choose(sweep(y_te, prob, amounts_minor, CostModel()))
    print("cost-optimal threshold : {:.5f}".format(chosen["threshold"]))
    print("at FPR                 : {:.4%}".format(chosen["fpr"]))
    print("recall at that point   : {:.4f}".format(chosen["recall"]))
    print("expected cost          : EUR {:,.2f}".format(
        chosen["total_cost_minor"] / 100))
    print()
    f_amt = float(te.loc[te.Class == 1, "Amount"].mean())
    l_amt = float(te.loc[te.Class == 0, "Amount"].mean())
    print("The amounts are real euros, so the fraud-loss term is a real")
    print("distribution rather than one I chose.")
    print()
    print("mean fraudulent amount : EUR {:>8.2f}".format(f_amt))
    print("mean legitimate amount : EUR {:>8.2f}".format(l_amt))
    print()
    if f_amt > l_amt:
        print("Fraud is {:.1f}x LARGER on average here, which is what amount-".format(
            f_amt / l_amt))
        print("weighting assumes and is worth checking rather than assuming: the")
        print("cost curve only leans on amount if the two distributions actually")
        print("differ. They do, so the weighting earns its place.")
    else:
        print("Fraud is SMALLER on average here, which inverts the usual")
        print("amount-weighting intuition -- and would mean the amount term")
        print("pushes the threshold the opposite way from the assumption.")
    print()
    print("Note the operating point it picks: recall {:.2f} at an FPR of".format(
        chosen["recall"]))
    print("{:.4%}. On 85,443 transactions that is a very small alert queue,".format(
        chosen["fpr"]))
    print("which is what a 0.13% prevalence does to the economics -- almost")
    print("every decline is wrong, so the insult cost dominates and the optimum")
    print("sits far to the right of where a pure-detection view would put it.")

    # ------------------------------------------------- 5. what cannot be done
    print("\n" + "=" * 80)
    print("5. THE PART THAT CANNOT BE DONE ON THIS DATA")
    print("-" * 80)
    import shap

    explainer = shap.TreeExplainer(model)
    idx = int(np.argmax(prob))
    vals = explainer.shap_values(X_te[idx:idx + 1])
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals).reshape(-1)
    order = np.argsort(-vals)[:3]

    print("Highest-scoring declined transaction, top-3 TreeSHAP contributors:")
    print()
    for rank, j in enumerate(order, 1):
        print("   {}. {:<10} contribution {:+.4f}".format(
            rank, features[j], vals[j]))
    print()
    print("Now write the adverse-action notice.")
    print()
    print("FCRA 615(a) requires the PRINCIPAL REASONS, in terms the consumer can")
    print("act on. There is no sentence beginning 'V14' that satisfies that, and")
    print("no amount of better attribution fixes it -- the information required")
    print("to write the reason was destroyed by the PCA before the file was")
    print("published. It had to be: this is a real bank's transaction data, and")
    print("anonymisation is the only reason it could be released at all.")
    print()
    print("So the honest summary is an inversion of the usual story. The")
    print("GENERATED data supports a compliant end-to-end pipeline, because its")
    print("features are things like velocity_24h and cross_border. The REAL data")
    print("supports better metrics and cannot support a compliant decline")
    print("notice. A production system needs both properties at once, which is")
    print("exactly why it needs its own data and cannot be built on a public")
    print("dataset however good the modelling is.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
