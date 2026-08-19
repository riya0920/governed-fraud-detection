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

### 5.1 Promotion criteria

A challenger is promoted only if **all** of the following hold. Any single
failure blocks promotion; there is no aggregate score that lets a strong result
on one criterion buy a weak one on another.

| Gate | Requirement | Rationale |
|---|---|---|
| Shadow period | ≥ 14 days scoring live traffic with decisions logged but not acted on | Two weeks covers a full weekly seasonality cycle; anything shorter measures a Tuesday |
| Discrimination | precision at the incumbent's operating FPR ≥ champion + 0.01 absolute | The operating point is the only place lift is worth anything |
| Calibration | Brier ≤ champion × 1.05, reliability curve monotone | The threshold is chosen by $-weighted cost, which consumes calibrated probabilities |
| Stability | score PSI < 0.10 across the shadow window | A challenger that is already drifting in shadow will not improve in production |
| Fairness | AIR at the operating threshold not worse than champion by > 0.02 | A challenger that buys accuracy with disparity is not an improvement |
| Explainability | reason codes generated for 100% of shadow declines | A decline we cannot explain is not shippable regardless of AUC |
| Rollback | one-command revert to the prior artifact, tested during shadow | Untested rollback is not rollback |

### 5.2 Rollback triggers (post-promotion)

Automatic revert to the prior artifact on any of: decline rate moving > 50%
relative to the shadow baseline within a 1-hour window; score PSI > 0.25 over
24 hours; or reason-code generation failure rate > 1%. Rollback is a mechanical
action, not a meeting — the discussion happens after the revert.

## 6. Monitoring plan

Implemented in `monitor.py`, which reads the **decision log written by the API**
(`artifacts/decisions.jsonl`), not a data warehouse snapshot. That distinction
matters: drift must be measured on what was actually scored, including any
feature-pipeline breakage between the warehouse and the model.

Signals in priority order, which is the ordering derived in RESULTS.md §6:

| # | Signal | Needs labels? | Threshold | Action |
|---|---|---|---|---|
| 1 | Decline rate at the frozen threshold | **No** | ±50% relative | Page. Fastest available signal |
| 2 | Score distribution PSI | **No** | ≥0.10 warn, ≥0.25 page | Investigate upstream features |
| 3 | Feature PSI | **No** | ≥0.25 warn | Locate which input moved |
| 4 | Precision / recall decay | Yes | −10% relative | Retrospective confirmation only |
| 5 | Reason-code failure rate | No | >1% | Page — an unexplainable decline is a compliance issue |

**Why this ordering.** Signal 4 is the one most portfolios lead with, and it is
last on this list because fraud labels arrive weeks-to-months late via chargeback.
By the time precision decay is visible, the loss is booked. Signals 1–3 are
label-free, which is what makes them able to page someone in time.

**Insufficient-data handling.** Below 200 logged decisions, `monitor.run()`
returns `insufficient_data`, never `healthy`. A monitor that shows green on an
empty window is worse than one that shows nothing, because it is trusted.

**Retraining triggers.** Scheduled: quarterly. Event-driven: any signal-1 or
signal-2 page sustained over 48 hours, or a confirmed precision decay > 10%
relative. Retraining is proposed by the monitoring job and executed by a human —
automatic retraining on a drifting feed trains on the drift.

**Ownership.** Model owner is accountable for signals 1–5; the fraud policy owner
owns the threshold and the degradation posture. Neither role is filled — this is
a portfolio artifact — and that gap is itself a blocker to production use.

## 7. Fairness / disparate impact

Performed on a **synthetic** protected-attribute overlay. Full results in
RESULTS.md §8; the mechanics are in `src/fairness.py`. Summary of what is tested:

1. **Adverse impact ratio** at the operating threshold, against the 80% rule
   (EEOC Uniform Guidelines 29 CFR 1607.4(D), as borrowed into fair-lending
   practice). Screen, not proof; and not a safe harbour above the line either.
2. **Score-distribution comparison** across groups, including a group-separation
   AUC of the score itself — two models can share an AIR and distribute risk
   very differently.
3. **AIR across five operating points**, because AIR at one threshold is a single
   sample from a curve.
4. **Proxy ablation** — the substantive test. Features suspected of encoding the
   attribute are dropped, the model is refit, and both the disparity *and* the
   reconstructability of group membership are re-measured. The report quantifies
   what fraction of the recoverable group signal survives ablation, rather than
   reporting a binary.

**Disparate treatment vs disparate impact**: neither model uses the protected
attribute as an input, so treatment is not at issue; impact is. These are
distinct legal concepts and the report does not blur them.

**Hard limitation, restated because it is the one that matters**: the attribute
is synthesised by this repository. Every ratio is a property of that
construction. Presenting these as findings about a real population would be a
control failure, and no such claim is made anywhere. Also absent: business-
necessity analysis, less-discriminatory-alternative search, and any
intersectional cut — the analysis uses one binary attribute, the crudest
possible.

## 8. Stability: bands and actions

Bands are the standard PSI ones, stated in `src/metrics.py`: <0.10 stable,
0.10–0.25 monitor, >0.25 investigate.

| Band | Action | Owner |
|---|---|---|
| < 0.10 | None. Recorded in the weekly report | Automated |
| 0.10–0.25 | Ticket. Identify which feature moved and whether an upstream change explains it | Model owner |
| > 0.25 on a feature | Investigate within 48h. Do not retrain reflexively — establish the cause first | Model owner |
| > 0.25 on the score | Page. Score-level drift means the decision surface moved | On-call |

RESULTS.md §6 contains the caveat that governs all of the above: population-level
PSI is near zero even while the model decays out-of-time, because the adversary
moves inside the ~1% fraud subpopulation. PSI is a necessary check, not a
sufficient one, and it is ranked third for that reason.

## 9. Ongoing validation schedule

| Activity | Frequency |
|---|---|
| Monitoring report review | Weekly |
| Full revalidation (this document) | Annually, or on any material model change |
| Fairness re-test | On promotion, and on any threshold change |
| Threshold review against realised costs | Quarterly |
| Leakage audit re-run | On any feature addition |

## 10. Change log and sign-offs

| Version | Change | Date | Sign-off |
|---|---|---|---|
| 0.1.0 | Initial model, threshold economics, leakage audit | 2026-08 | **unsigned** |
| 0.2.0 | Fairness §7, monitoring §6, promotion criteria §5.1, scoring API | 2026-08 | **unsigned** |

**No sign-offs exist.** Under SR 11-7 this document requires review by a
validation function independent of development, and there isn't one — developer
and validator are the same party. That is limitation #2 in §2 and it remains the
single largest blocker to any production use of this model.

---

## Still missing from this document

- **Independent validation.** Structural, not fixable by writing more here.
- **Real-data revalidation.** Every number is generator-dependent; the IEEE-CIS
  swap-in changes all of them.
- **Business-necessity and less-discriminatory-alternative analysis** in §7.
- **Benchmarking against a vendor or bureau score**, which is what a real MRM
  review would demand as an external reference point.
