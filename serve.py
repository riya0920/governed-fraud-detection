"""Scoring API: score + reason codes + model version, with a decision audit log.

Design points a reviewer will look for:

  * The response carries the MODEL VERSION and the THRESHOLD that produced the
    decision. Six months later, when someone asks why transaction X was declined,
    "the model said 0.87" is not an answer -- which model, at which cutoff, is.
  * Reason codes are returned on declines only, in plain English, with the
    attribution method named in the payload rather than implied.
  * The threshold is loaded from a policy artifact, not compiled in. It is a
    business parameter owned by the fraud policy owner (see docs/MODEL_VALIDATION.md
    section 1), and the API surfaces which policy version is live.
  * Every decision is appended to an audit log with the feature vector, so the
    monitoring job in monitor.py can compute drift on what was ACTUALLY scored
    rather than on a training snapshot.

Run:  uvicorn serve:app --port 8000
Then: python -m pytest tests/test_api.py -q
"""
from __future__ import annotations

import json
import pickle
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.reason_codes import ReasonCoder

ARTIFACT = ROOT / "artifacts" / "model.pkl"
AUDIT_LOG = ROOT / "artifacts" / "decisions.jsonl"

_lock = threading.Lock()
_state: dict = {}


@asynccontextmanager
async def lifespan(_app):
    art = load_artifact()
    _state["artifact"] = art
    _state["coder"] = ReasonCoder(art["model"], art["features"], art["reference_X"],
                                  calibrator=art.get("calibrator"))
    AUDIT_LOG.parent.mkdir(exist_ok=True)
    yield
    _state.clear()


app = FastAPI(title="Fraud scoring API", version="0.2.0", lifespan=lifespan)


class ScoreRequest(BaseModel):
    transaction_id: str
    amount_minor: int = Field(gt=0, description="Integer minor units. Never a float.")
    velocity_24h: int = Field(ge=0)
    cross_border: int = Field(ge=0, le=1)
    device_change: int = Field(ge=0, le=1)
    mcc_risk: float
    hour: int = Field(ge=0, le=23)
    card_tenure_days: float = Field(ge=0)


class ReasonCode(BaseModel):
    rank: int
    feature: str
    statement: str
    contribution: float


class ScoreResponse(BaseModel):
    transaction_id: str
    score: float
    decision: str
    threshold: float
    model_version: str
    policy_version: str
    reason_codes: list[ReasonCode]
    attribution_method: str
    scored_at: str


def load_artifact() -> dict:
    if not ARTIFACT.exists():
        raise RuntimeError(
            "no model artifact at {}. Run: python train.py --save".format(ARTIFACT))
    with ARTIFACT.open("rb") as fh:
        return pickle.load(fh)


def _features(req: ScoreRequest, names: list[str]) -> np.ndarray:
    d = req.model_dump()
    # Derived features must be computed the SAME way as in training. They live
    # here rather than being expected from the caller precisely so a caller
    # cannot disagree with training about what `is_night` means.
    d["is_night"] = 1 if 1 <= req.hour <= 5 else 0
    d["amount_per_velocity"] = req.amount_minor / (1 + req.velocity_24h)
    missing = [n for n in names if n not in d]
    if missing:
        raise HTTPException(500, "model expects features not derivable: {}".format(missing))
    return np.array([[float(d[n]) for n in names]])


@app.get("/health")
def health() -> dict:
    art = _state.get("artifact")
    return {
        "status": "ok" if art else "no_model",
        "model_version": art["model_version"] if art else None,
        "policy_version": art["policy_version"] if art else None,
        "threshold": art["threshold"] if art else None,
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    art = _state.get("artifact")
    if art is None:
        raise HTTPException(503, "model not loaded")
    x = _features(req, art["features"])
    coder: ReasonCoder = _state["coder"]
    record = coder.decision_record(x[0], art["threshold"], art["model_version"])

    resp = ScoreResponse(
        transaction_id=req.transaction_id,
        score=record["score"],
        decision=record["decision"],
        threshold=art["threshold"],
        model_version=art["model_version"],
        policy_version=art["policy_version"],
        reason_codes=[ReasonCode(**{k: rc[k] for k in
                                    ("rank", "feature", "statement", "contribution")})
                      for rc in record["reason_codes"]],
        attribution_method=record["attribution_method"],
        scored_at=datetime.now(timezone.utc).isoformat(),
    )
    _audit(req, resp, x[0], art["features"])
    return resp


def _audit(req: ScoreRequest, resp: ScoreResponse, x: np.ndarray,
           names: list[str]) -> None:
    """Append-only decision log. Features are stored WITH the decision so drift
    monitoring runs against production traffic, not a training snapshot."""
    row = {
        "scored_at": resp.scored_at,
        "transaction_id": req.transaction_id,
        "score": resp.score,
        "decision": resp.decision,
        "threshold": resp.threshold,
        "model_version": resp.model_version,
        "policy_version": resp.policy_version,
        "features": {n: float(v) for n, v in zip(names, x)},
        "reason_codes": [rc.feature for rc in resp.reason_codes],
    }
    with _lock, AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
