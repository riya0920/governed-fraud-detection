"""API + monitoring tests. Run train.py once first to produce the artifact."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

ARTIFACT = ROOT / "artifacts" / "model.pkl"
pytestmark = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="run `python train.py` first to build the artifact")

import monitor  # noqa: E402
import serve  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(serve.app) as c:
        yield c


def _txn(**over):
    base = {"transaction_id": "t1", "amount_minor": 12_000, "velocity_24h": 1,
            "cross_border": 0, "device_change": 0, "mcc_risk": 0.5, "hour": 14,
            "card_tenure_days": 800.0}
    base.update(over)
    return base


def test_health_exposes_model_and_policy_version(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] and body["policy_version"]
    assert 0 < body["threshold"] < 1


def test_response_carries_versions_and_threshold(client):
    """Six months later, 'the model said 0.87' is not an answer. Which model, at
    which cutoff, is."""
    r = client.post("/score", json=_txn())
    assert r.status_code == 200
    b = r.json()
    for field in ("model_version", "policy_version", "threshold", "scored_at",
                  "attribution_method"):
        assert b[field] is not None, field
    assert b["decision"] in ("approve", "decline")


def test_decline_carries_reason_codes_and_approve_does_not(client):
    risky = client.post("/score", json=_txn(
        transaction_id="risky", amount_minor=900_000, velocity_24h=11,
        cross_border=1, device_change=1, mcc_risk=2.2, hour=3,
        card_tenure_days=15.0)).json()
    safe = client.post("/score", json=_txn(
        transaction_id="safe", amount_minor=1_500, velocity_24h=0, mcc_risk=0.2,
        hour=13, card_tenure_days=2000.0)).json()

    assert risky["score"] > safe["score"], "risk ordering is inverted"
    if risky["decision"] == "decline":
        assert 1 <= len(risky["reason_codes"]) <= 3
        assert all(rc["statement"] for rc in risky["reason_codes"])
    assert safe["reason_codes"] == [] or safe["decision"] == "decline"


def test_float_amount_is_rejected_at_the_schema_boundary(client):
    """Money is integer minor units. The API must refuse a float rather than
    silently truncating it."""
    r = client.post("/score", json=_txn(amount_minor=120.55))
    assert r.status_code == 422


def test_invalid_values_are_rejected(client):
    assert client.post("/score", json=_txn(amount_minor=0)).status_code == 422
    assert client.post("/score", json=_txn(hour=25)).status_code == 422
    assert client.post("/score", json=_txn(cross_border=7)).status_code == 422


def test_derived_features_are_computed_server_side(client):
    """`is_night` and `amount_per_velocity` are NOT accepted from the caller --
    a caller must not be able to disagree with training about what they mean."""
    r = client.post("/score", json={**_txn(), "is_night": 1,
                                    "amount_per_velocity": 999999})
    assert r.status_code in (200, 422)
    if r.status_code == 200:
        night = client.post("/score", json=_txn(hour=3, transaction_id="n")).json()
        day = client.post("/score", json=_txn(hour=14, transaction_id="d")).json()
        assert night["score"] != day["score"], "hour is not reaching the model"


def test_every_decision_is_audited_with_its_features(client):
    before = len(monitor.load_decisions())
    client.post("/score", json=_txn(transaction_id="audit-me"))
    rows = monitor.load_decisions()
    assert len(rows) == before + 1
    last = rows[-1]
    assert last["transaction_id"] == "audit-me"
    assert last["features"] and "amount_minor" in last["features"]
    assert last["model_version"] and last["threshold"]


def test_monitor_refuses_to_report_healthy_on_thin_data():
    """An empty monitor showing green is worse than one showing nothing."""
    report = monitor.run(decisions=[])
    assert report["status"] == "insufficient_data"


def test_monitor_pages_on_alert_rate_shift():
    """The label-free early signal: decline rate moves at a frozen threshold."""
    import pickle
    with ARTIFACT.open("rb") as fh:
        art = pickle.load(fh)
    thr = art["threshold"]
    feats = {n: 0.0 for n in art["features"]}
    # Every request declines -- a drastic shift from the reference decline rate.
    rows = [{"score": thr + 0.5, "decision": "decline", "features": dict(feats)}
            for _ in range(monitor.MIN_SAMPLE + 50)]
    report = monitor.run(decisions=rows)
    assert report["status"] == "page"
    assert any(f["signal"] == "alert_rate" for f in report["findings"])
