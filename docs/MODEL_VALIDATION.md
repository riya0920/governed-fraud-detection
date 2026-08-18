# Model Validation Report — `fraud-gbm-0.1.0`

**Document status: PARTIAL (sections 1–5 written; 6–10 not written).** This is a
20% slice of the artifact the spec asks for. Sections that do not exist are
listed as missing rather than filled with plausible text, because a validation
report whose gaps are invisible is worse than no report — it is a control
failure, and it is exactly what an MRM reviewer is trained to find.

Structure follows the SR 11-7 model documentation expectations (Federal Reserve
SR 11-7 / OCC 2011-12, *Supervisory Guidance on Model Risk Management*),
specifically the requirement that documentation be detailed enough that a
knowledgeable third party can evaluate the model without the developer present.
All quantitative results in this document are produced by `train.py` and land in
[RESULTS.md](RESULTS.md); nothing here is retyped by hand.

---

## 1. Purpose, scope, and intended use

**Purpose.** Score card-present and card-not-present authorizations for fraud
risk at authorization time, producing a decline/approve recommendation and, on a
decline, the principal reasons for it.

**Intended use.** One input to an authorization decision, at a threshold set by
the fraud policy owner from the cost curve in RESULTS.md §5. The model outputs a
calibrated probability; it does not output a decision. The threshold is a
business parameter and lives outside the model.

**Explicitly out of scope** (using the model for any of these is a misuse and
would require separate validation):
- Credit decisioning of any kind. This model is not a creditworthiness model, is
  not built on credit data, and has had no fair-lending analysis under ECOA/Reg B.
- Account closure, or any action against a customer beyond declining the
  individual transaction.
- Merchant risk rating or underwriting.
- Any use where the score is consumed as a *rate* without the calibration checks
  in §4 being re-run on the consuming population.

**Model owner / developer / validator.** Developer: this repository. Owner and
validator: unassigned — this is a portfolio artifact, not a production model, and
the absence of independent validation is itself a limitation (SR 11-7 requires
validation independent of development).

## 2. Known limitations

Stated first rather than last, on purpose.

1. **Synthetic data.** The model is trained on generated data (`src/generate.py`),
   not on production or IEEE-CIS transactions. Every performance number is a
   property of the generator's fraud process. None of it transfers.
2. **No independent validation.** Development and "validation" are the same
   party. Under SR 11-7 this alone blocks production use.
3. **Label delay is not modelled.** In production, fraud labels arrive weeks to
   months later via chargebacks. The evaluation here assumes labels are available
   at the end of the test window, which is optimistic in a way that matters most
   for the monitoring plan (§6, not written).
4. **Adversarial drift is only partly observable.** See RESULTS.md §6: population
   PSI is near zero while the fraud subpopulation shifts. The monitoring
   implication — alert-rate drift at a frozen threshold is the only label-free
   early signal — is documented but not implemented.
5. **Attribution is occlusion-based, not SHAP.** `src/reason_codes.py` computes a
   one-at-a-time counterfactual against the training median. It is exactly
   reproducible and cheap, and it is blind to interactions: interaction credit is
   assigned to whichever feature is occluded. TreeSHAP would split that credit
   fairly. The reason codes are therefore directionally right and not
   game-theoretically fair, and that distinction has to be disclosed to anyone
   relying on them for adverse-action purposes.
6. **No fairness testing has been performed.** See §7 below.
7. **Prevalence is a generator parameter** (1.19% in the current build), higher
   than many real portfolios. Precision at any FPR scales with prevalence, so the
   precision figures are not comparable to a portfolio at 0.1%.

## 3. Data lineage and feature dictionary

Source: `src/generate.py` → `data/transactions.parquet` (250,000 rows, 90 days).
Split: out-of-time, days 0–59 train / 60–89 test. **Not random** — a random split
puts the same card's later behaviour into training and inflates every metric.

| Feature | Source | Logic | Available at auth time? | Leakage assessment |
|---|---|---|---|---|
| `amount_minor` | auth message | Transaction amount, integer minor units | Yes | Clean |
| `velocity_24h` | streaming counter | Count of prior auths on this card in trailing 24h | Yes, if the counter is point-in-time | **Watch**: computing this from a batch table rebuilt nightly would include same-day future transactions. Must be a streaming/as-of read. |
| `cross_border` | auth message | Acquirer country ≠ issuer country | Yes | Clean |
| `device_change` | device fingerprint svc | Device unseen on this account | Yes | Clean |
| `mcc_risk` | reference table | Category risk weight, versioned | Yes | **Watch**: the weight table must be the version in force at scoring time, not today's. A refreshed table backfilled over history is leakage. |
| `hour`, `is_night` | auth timestamp | Local hour; night = 01:00–05:00 | Yes | Clean |
| `card_tenure_days` | account master | Days since account open | Yes | Clean |
| `amount_per_velocity` | derived | `amount_minor / (1 + velocity_24h)` | Inherits `velocity_24h` | Inherits the watch above |

### Excluded — post-decision fields (the leakage audit)

| Field | Why it is unusable | Univariate AUC |
|---|---|---|
| `chargeback_filed` | Filed by the issuer days-to-months after settlement | see RESULTS.md §1 |
| `review_queue_flag` | Set by the queue this model feeds — circular | see RESULTS.md §1 |
| `card_blocked_24h` | A consequence of the decline, not an input to it | see RESULTS.md §1 |

Enforcement is not a convention: `leakage.assert_clean()` raises before every
fit, and `leakage.audit()` independently flags any feature whose single-feature
AUC exceeds 0.80, which is how the *unknown* leaks get caught when an upstream
team adds a column.

## 4. Performance and calibration

See [RESULTS.md](RESULTS.md) §3–§5 for the generated tables. Summary of the
current build, out-of-time:

- **AUC 0.7251 / Gini 0.4501 / KS 0.3219**, in-sample AUC 0.8032. The gap is
  reported deliberately.
- **Brier 0.01129**, with the reliability table in §4 of RESULTS.md.
- **Precision at 1% FPR: 0.090** (recall 0.083).
- Imbalance handling was compared, not assumed: baseline vs class weights vs
  SMOTE, identical model config. Selected on precision at the operating FPR.
- Threshold chosen by minimising $-weighted expected cost, not F1: the
  cost-optimal point sits at FPR 1.94%, saving **$6,013.58** against the
  rules-only incumbent over the 30-day test window.

Accuracy is deliberately absent from this document. At 1.19% prevalence,
approving every transaction is 98.8% accurate and is the most expensive policy in
the cost table.

## 5. Champion / challenger

Champion is the selected strategy from RESULTS.md §3. The other two strategies
are retained as challengers with their metrics on the same out-of-time window.
**Promotion criteria are not yet defined** — see §9.

---

## Sections NOT written (the remaining 80% of this document)

6. **Monitoring plan.** What is tracked in production, alert thresholds,
   retraining triggers, ownership, and paging. The *analysis* that should drive
   it exists (RESULTS.md §6 ranks four drift signals by sensitivity and explains
   why feature PSI is last); the plan itself does not.
7. **Fairness / disparate impact.** No synthetic protected-attribute overlay, no
   adverse impact ratio, no score-distribution comparison, no proxy-feature test.
   Nothing about fairness has been done, and the empty section is left visible.
8. **Stability thresholds and action.** PSI is computed and banded; what happens
   at each band, and who does it, is not written.
9. **Champion/challenger promotion criteria**: shadow period, minimum lift,
   calibration gate, rollback trigger.
10. **Ongoing validation schedule, change log, sign-offs.**

Also missing outside this document: the scoring API (FastAPI/Docker) returning
score + reason codes + model version, and the Evidently drift job.
