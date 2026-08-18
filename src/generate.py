"""Synthetic card-transaction generator with a fraud process, an out-of-time
drift regime, and three deliberately planted leakage traps.

Why synthetic rather than IEEE-CIS: this repo runs offline end to end, and a
generator gives me ground truth for the drift and leakage sections that a Kaggle
file cannot. The feature semantics mirror IEEE-CIS/Sparkov-style tabular fraud
data; swapping in the real file means replacing load() and re-running the same
pipeline (see README 'remaining work').

The planted traps -- chargeback_filed, review_queue_flag, card_blocked_24h --
are all populated only AFTER the authorization decision exists. Any model that
uses them scores near-perfectly and is worthless in production. docs/LEAKAGE_AUDIT.md
is the write-up; src/leakage.py is the check that fails the build if they leak in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEAKY_COLUMNS = ["chargeback_filed", "review_queue_flag", "card_blocked_24h"]

MCC_GROUPS = ["grocery", "fuel", "electronics", "travel", "gambling", "digital_goods"]
MCC_RISK = {"grocery": 0.2, "fuel": 0.5, "electronics": 1.4,
            "travel": 1.1, "gambling": 2.2, "digital_goods": 1.8}


def generate(n_rows: int = 250_000, n_cards: int = 5_000, n_days: int = 90,
             base_fraud_rate: float = 0.012, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    card_id = rng.integers(0, n_cards, n_rows)
    # Card tenure in days at the time of the portfolio snapshot.
    card_age = rng.gamma(4.0, 120.0, n_cards).clip(1, 3000)
    ts = np.sort(rng.uniform(0, n_days, n_rows))
    hour = ((ts % 1) * 24).astype(int)

    mcc = rng.choice(MCC_GROUPS, n_rows, p=[0.30, 0.22, 0.16, 0.14, 0.06, 0.12])
    mcc_risk = np.array([MCC_RISK[m] for m in mcc])

    # Amount: lognormal per merchant category, in integer minor units.
    scale = {"grocery": 3.6, "fuel": 3.5, "electronics": 5.1,
             "travel": 5.6, "gambling": 4.6, "digital_goods": 3.2}
    mu = np.array([scale[m] for m in mcc])
    amount_minor = (np.exp(rng.normal(mu, 0.9)) * 100).astype(np.int64).clip(100, 5_000_000)

    # Velocity: transactions on the same card in the prior 24h (behavioural).
    order = np.argsort(card_id, kind="stable")
    velocity = np.zeros(n_rows, dtype=int)
    last_seen: dict[int, list[float]] = {}
    for i in order:
        hist = last_seen.setdefault(int(card_id[i]), [])
        while hist and ts[i] - hist[0] > 1.0:
            hist.pop(0)
        velocity[i] = len(hist)
        hist.append(ts[i])

    cross_border = (rng.random(n_rows) < 0.07).astype(int)
    device_change = (rng.random(n_rows) < 0.11).astype(int)
    card_tenure = card_age[card_id]
    amount_z = (np.log1p(amount_minor) - np.log1p(amount_minor).mean()) / np.log1p(amount_minor).std()

    # --- fraud process -----------------------------------------------------
    # Regime shift at day 60: fraudsters move to lower amounts and to
    # digital_goods, which is what the PSI/stability section is meant to catch.
    late = (ts >= 60).astype(float)
    logit = (-6.4
             + 0.75 * amount_z * (1 - 0.30 * late)
             + 0.60 * mcc_risk
             + 0.34 * np.minimum(velocity, 12)
             + 1.00 * cross_border
             + 0.90 * device_change
             + 0.45 * ((hour >= 1) & (hour <= 5)).astype(float)
             - 0.0006 * card_tenure
             + 0.55 * late * (mcc == "digital_goods").astype(float)
             + rng.normal(0, 0.35, n_rows))
    p = 1 / (1 + np.exp(-logit))
    p = p * (base_fraud_rate / p.mean())          # calibrate prevalence
    is_fraud = (rng.random(n_rows) < p).astype(int)

    # --- planted leakage traps (post-decision information) ------------------
    chargeback = np.where(is_fraud == 1, (rng.random(n_rows) < 0.82).astype(int),
                          (rng.random(n_rows) < 0.004).astype(int))
    review_q = np.where(is_fraud == 1, (rng.random(n_rows) < 0.63).astype(int),
                        (rng.random(n_rows) < 0.02).astype(int))
    blocked = np.where(is_fraud == 1, (rng.random(n_rows) < 0.71).astype(int),
                       (rng.random(n_rows) < 0.003).astype(int))

    df = pd.DataFrame({
        "txn_id": np.arange(n_rows),
        "day": ts,
        "card_id": card_id,
        "hour": hour,
        "amount_minor": amount_minor,
        "mcc_group": mcc,
        "mcc_risk": mcc_risk,
        "velocity_24h": velocity,
        "cross_border": cross_border,
        "device_change": device_change,
        "card_tenure_days": card_tenure.round(1),
        "is_fraud": is_fraud,
        # post-decision fields -- present in the raw feed, excluded by the audit
        "chargeback_filed": chargeback,
        "review_queue_flag": review_q,
        "card_blocked_24h": blocked,
    })
    return df


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parents[1] / "data" / "transactions.parquet"
    df = generate()
    try:
        df.to_parquet(out)
    except Exception:
        out = out.with_suffix(".csv")
        df.to_csv(out, index=False)
    print("wrote {}  rows={}  fraud_rate={:.4%}".format(out, len(df), df.is_fraud.mean()))
