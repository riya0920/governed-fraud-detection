"""Disparate-impact testing on a SYNTHETIC protected-attribute overlay.

Read this paragraph before any number below is quoted anywhere.

The protected attribute here does not exist in the data. It is constructed by
this module, correlated with features that already exist, and then used to test
the MACHINERY of a fair-lending review. Every ratio produced is a property of
that construction. No statement about real-world disparate impact follows from
any of it, and a validation report that presented these as findings would be
committing the exact fraud this repo is built to avoid.

What the machinery consists of, and why each piece is here:

  adverse_impact_ratio   The 80% rule (EEOC Uniform Guidelines 29 CFR 1607.4(D),
                         borrowed into fair-lending practice). Selection rate of
                         the disadvantaged group over the advantaged group. Below
                         0.80 is a SCREEN that triggers investigation -- it is
                         neither proof of discrimination nor, above 0.80, a safe
                         harbour.
  score_distribution     Two models can share an AIR and distribute risk very
                         differently across groups. Mean/median/percentile
                         separation catches what a single threshold ratio hides.
  proxy_ablation         The one that matters. Dropping the protected attribute
                         achieves nothing if a correlated feature reconstructs
                         it -- that is disparate impact via proxy, and it is the
                         normal case, not the exotic one. This measures how much
                         of the disparity survives removing the suspected proxies,
                         and how well the proxies predict the attribute in the
                         first place.
  threshold_sweep        AIR at one operating point is a single sample from a
                         curve. If the disparity appears only at the threshold
                         you happened to pick, you want to know that.

Disparate impact vs disparate treatment, stated because misusing the terms is
worse than not using them: TREATMENT is using the protected attribute (or a
deliberate stand-in) as an input -- a facially discriminatory practice. IMPACT is
a neutral practice producing disproportionate outcomes. Neither model in this
repo uses the attribute as an input, so treatment is not the question; impact is.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

AIR_THRESHOLD = 0.80


def synthesize_protected_attribute(df, correlates: dict[str, float],
                                   base_rate: float = 0.35,
                                   seed: int = 99) -> np.ndarray:
    """Build a synthetic group membership correlated with real features.

    `correlates` maps column -> weight on the standardised column. This is how
    the overlay gets a realistic proxy structure: membership is not random, so
    features that track those correlates will partially reconstruct it, which is
    precisely the situation `proxy_ablation` is built to measure.
    """
    rng = np.random.default_rng(seed)
    z = np.zeros(len(df), dtype=float)
    for col, w in correlates.items():
        v = df[col].to_numpy(dtype=float)
        sd = v.std() or 1.0
        z += w * (v - v.mean()) / sd
    logit = z + np.log(base_rate / (1 - base_rate))
    p = 1 / (1 + np.exp(-logit))
    p *= base_rate / p.mean()
    return (rng.random(len(df)) < np.clip(p, 0, 1)).astype(int)


def adverse_impact_ratio(selected: np.ndarray, group: np.ndarray) -> dict:
    """`selected` = True where the person got the FAVOURABLE outcome.

    For a fraud model the favourable outcome is approval, so callers must pass
    ~declined. Getting this inversion wrong silently reports the reciprocal.
    """
    r1 = float(selected[group == 1].mean()) if (group == 1).any() else float("nan")
    r0 = float(selected[group == 0].mean()) if (group == 0).any() else float("nan")
    lo, hi = min(r1, r0), max(r1, r0)
    air = lo / hi if hi else float("nan")
    return {
        "rate_group_1": r1,
        "rate_group_0": r0,
        "disadvantaged": 1 if r1 < r0 else 0,
        "air": air,
        "flags": bool(air < AIR_THRESHOLD),
        "pp_gap": abs(r1 - r0),
    }


def score_distribution(score: np.ndarray, group: np.ndarray) -> dict:
    a, b = score[group == 1], score[group == 0]
    out = {"n_group_1": int(len(a)), "n_group_0": int(len(b))}
    for label, arr in (("group_1", a), ("group_0", b)):
        out[label] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.quantile(arr, 0.90)),
            "p99": float(np.quantile(arr, 0.99)),
        }
    # Probability a randomly chosen group-1 member scores above a group-0 member.
    # 0.5 means the score distributions are interchangeable across groups.
    out["auc_group_separation"] = float(roc_auc_score(group, score))
    out["mean_gap"] = out["group_1"]["mean"] - out["group_0"]["mean"]
    return out


def threshold_sweep(score: np.ndarray, group: np.ndarray,
                    quantiles=(0.90, 0.95, 0.975, 0.99, 0.995)) -> list[dict]:
    rows = []
    for q in quantiles:
        thr = float(np.quantile(score, q))
        approved = score < thr
        r = adverse_impact_ratio(approved, group)
        rows.append({"decline_rate_target": 1 - q, "threshold": thr,
                     "air": r["air"], "flags": r["flags"],
                     "approval_1": r["rate_group_1"], "approval_0": r["rate_group_0"]})
    return rows


def proxy_ablation(X: np.ndarray, feature_names: list[str], y: np.ndarray,
                   group: np.ndarray, suspected_proxies: list[str],
                   model_factory, threshold_q: float = 0.98) -> dict:
    """Retrain without the suspected proxies and re-measure the disparity.

    Three numbers come out, and all three are needed:

      proxy_auc_full      how well ALL features reconstruct group membership
      proxy_auc_ablated   how well the remaining features still reconstruct it
      air_full/air_ablated  whether the outcome disparity actually moved

    The interesting failure is `air_ablated` staying put while `proxy_auc`
    barely drops: the information did not live in the dropped columns, it lives
    distributed across the rest, and dropping columns is theatre. That result is
    common and it is the reason "we removed the zip code" is not a defence.
    """
    keep = [i for i, f in enumerate(feature_names) if f not in suspected_proxies]
    dropped = [f for f in feature_names if f in suspected_proxies]

    def fit_and_air(cols):
        m = model_factory()
        m.fit(X[:, cols], y)
        s = m.predict_proba(X[:, cols])[:, 1]
        thr = float(np.quantile(s, threshold_q))
        return s, adverse_impact_ratio(s < thr, group)

    s_full, air_full = fit_and_air(list(range(len(feature_names))))
    s_abl, air_abl = fit_and_air(keep)

    def reconstruct_auc(cols):
        """Can the features predict group membership at all? A logistic probe."""
        probe = LogisticRegression(max_iter=500)
        Xs = X[:, cols]
        Xs = (Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-9)
        probe.fit(Xs, group)
        return float(roc_auc_score(group, probe.predict_proba(Xs)[:, 1]))

    return {
        "dropped": dropped,
        "kept": [feature_names[i] for i in keep],
        "air_full": air_full["air"],
        "air_ablated": air_abl["air"],
        "air_change": air_abl["air"] - air_full["air"],
        "proxy_auc_full": reconstruct_auc(list(range(len(feature_names)))),
        "proxy_auc_ablated": reconstruct_auc(keep),
        "scores_full": s_full,
        "scores_ablated": s_abl,
    }
