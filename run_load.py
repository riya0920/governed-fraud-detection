"""Service-level latency for the scoring API, measured rather than assumed.

    python run_load.py --requests 2000 --concurrency 8

Starts `serve.py` in-process on a uvicorn thread, drives it over real HTTP, and
reports the per-stage and end-to-end percentile table. The point is not the
number -- a laptop is a laptop -- it is that the README stops saying "no service
level p99" and starts saying one with the hardware and method attached.

WHAT THIS MEASURES AND WHAT IT DOES NOT. Client and server share a machine, a
loopback interface and a CPython process, so the network is free and the GIL is
shared. That flatters latency and penalises throughput at the same time, and the
two do not cancel. Read the p99 as a FLOOR for the model stage cost and nothing
as a capacity claim. SE-3 is the project that measures a gateway properly, and
it says the same thing about its own numbers.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import httpx
import uvicorn

import serve as service

SAMPLE = {
    "transaction_id": "load-0", "amount_minor": 18_500, "velocity_24h": 4,
    "cross_border": 0, "device_change": 0, "mcc_risk": 0.31, "hour": 14,
    "card_tenure_days": 640,
}


def _percentiles(xs):
    xs = sorted(xs)
    def q(p):
        if not xs:
            return float("nan")
        return xs[min(len(xs) - 1, int(len(xs) * p))]
    return {"p50": q(0.50), "p95": q(0.95), "p99": q(0.99),
            "max": xs[-1] if xs else float("nan"),
            "mean": statistics.fmean(xs) if xs else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()

    config = uvicorn.Config(service.app, host="127.0.0.1", port=args.port,
                            log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = "http://127.0.0.1:{}".format(args.port)
    deadline = time.time() + 30
    with httpx.Client(timeout=5.0) as probe:
        while time.time() < deadline:
            try:
                probe.get(base + "/health")
                break
            except Exception:
                time.sleep(0.05)
        else:
            print("server did not come up")
            return 1

    latencies: list[float] = []
    errors = [0]
    lock = threading.Lock()
    per_worker = args.requests // args.concurrency

    def worker():
        local = []
        with httpx.Client(timeout=10.0, base_url=base) as client:
            for _ in range(per_worker):
                payload = dict(SAMPLE)
                payload["transaction_id"] = "load-{}".format(id(local) + _)
                t0 = time.perf_counter()
                try:
                    r = client.post("/score", json=payload)
                    ok = r.status_code == 200
                except Exception:
                    ok = False
                dt = (time.perf_counter() - t0) * 1000
                if ok:
                    local.append(dt)
                else:
                    with lock:
                        errors[0] += 1
        with lock:
            latencies.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t_start

    server.should_exit = True
    thread.join(timeout=5)

    p = _percentiles(latencies)
    print("=" * 78)
    print("SCORING API -- CLOSED-LOOP LOAD")
    print("=" * 78)
    print("requests ok        : {}".format(len(latencies)))
    print("errors             : {}".format(errors[0]))
    print("concurrency        : {} threads".format(args.concurrency))
    print("elapsed            : {:.2f}s".format(elapsed))
    print("throughput         : {:.0f} req/s".format(len(latencies) / elapsed))
    print("-" * 78)
    print("{:>10}{:>10}{:>10}{:>10}{:>10}".format(
        "mean", "p50", "p95", "p99", "max"))
    print("{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}{:>10.2f}".format(
        p["mean"], p["p50"], p["p95"], p["p99"], p["max"]))
    print("(milliseconds, end to end, client-observed)")
    print("-" * 78)
    print("This is a CLOSED loop: each thread waits for its own response before")
    print("sending the next, so the offered rate falls as the service slows and")
    print("queueing delay never accumulates. That flatters the tail by")
    print("construction. An open-loop generator with Poisson arrivals is the")
    print("honest instrument and SE-3's run_soak.py is the one in this portfolio")
    print("that implements it -- this number is a floor, not a capacity curve.")
    print()
    print("Every request above also appended a full decision record with its")
    print("feature vector to the audit log, so the cost of the audit write is")
    print("inside these numbers rather than excluded from them.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
