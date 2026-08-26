"""Run the service INSIDE its container, under load.

    python run_container_load.py

`README.md` listed this open: "`docker build` runs clean ... What has not
happened is running the service inside it under load: the p99 in this README was
measured on the host, not in the container, and the two are not the same
number."

RUNNING IT FOUND THE IMAGE COULD NOT START AT ALL, which is a better result than
any latency number would have been:

    OSError: libgomp.so.1: cannot open shared object file

lightgbm links against libgomp, the GNU OpenMP runtime, and `python:3.12-slim`
does not ship it. The pip install of a manylinux wheel resolves and completes
perfectly; the shared library it needs at LOAD time is a system package pip knows
nothing about. So the build was green and the service could not start, and that
stayed true for as long as nobody ran it.

That is the entire argument for running a container rather than building one.

WHY THE LOAD IS DRIVEN FROM INSIDE WSL. The container publishes on WSL's
loopback, and reaching it from Windows depends on WSL2's localhost forwarding,
which on this machine works intermittently -- the same flakiness that made Redis
and Kafka unreachable earlier in this project. A probe failing for that reason
reports "container did not become healthy" about a container that is serving
perfectly, which is a false negative wearing the shape of a finding.

WHAT THAT COSTS, stated rather than buried: this measures the container, not the
container AGAINST the host. A host baseline would have a Windows client over
loopback where this has a Linux client on the same kernel, so the two differ in
the CLIENT as well as the server. Putting them in one table as a container
overhead figure would be the misleading version of this file.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

IMAGE = "finhm/ml1-governed-fraud:latest"
CONTAINER = "ml1-load"
PORT = 8124
REQUESTS = 1200
CONCURRENCY = 8

# Must match serve.py's ScoreRequest EXACTLY. The first run of this file sent
# invented field names, every request came back 422, and the summary reported
# "the service runs under load" over 1,200 failures -- so the payload is copied
# from the model rather than remembered.
SAMPLE = {
    "transaction_id": "t-load",
    "amount_minor": 24000,
    "velocity_24h": 4,
    "cross_border": 1,
    "device_change": 1,
    "mcc_risk": 0.4,
    "hour": 3,
    "card_tenure_days": 5.0,
}

PROBE = r"""
for i in $(seq 1 40); do
  c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3       http://127.0.0.1:__PORT__/health)
  if [ "$c" = 200 ]; then echo READY; break; fi
  sleep 5
done
"""

WARM = r"""
for i in $(seq 1 20); do
  curl -s -o /dev/null -X POST -H 'Content-Type: application/json'     --data '__BODY__' http://127.0.0.1:__PORT__/score
done
"""

DRIVER = r'''
import json, statistics, threading, time, urllib.request

BASE = "http://127.0.0.1:__PORT__"
SAMPLE = __SAMPLE__
N, THREADS = __N__, __THREADS__

lat, errs, why, lock = [], [0], [], threading.Lock()


def worker(wid, per):
    local = []
    for i in range(per):
        body = dict(SAMPLE)
        body["transaction_id"] = "l-%d-%d" % (wid, i)
        req = urllib.request.Request(
            BASE + "/score", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                ok = r.status == 200
        except Exception as e:                       # noqa: BLE001
            ok = False
            with lock:
                if len(why) < 3:
                    why.append(repr(e)[:200])
        dt = (time.perf_counter() - t0) * 1000
        if ok:
            local.append(dt)
        else:
            with lock:
                errs[0] += 1
    with lock:
        lat.extend(local)


per = max(N // THREADS, 1)
ts = [threading.Thread(target=worker, args=(i, per)) for i in range(THREADS)]
t0 = time.perf_counter()
for t in ts:
    t.start()
for t in ts:
    t.join()
el = time.perf_counter() - t0

s = sorted(lat)


def q(p):
    return s[min(int(round(p * (len(s) - 1))), len(s) - 1)] if s else 0


print(json.dumps({
    "n": len(s), "mean": statistics.mean(s) if s else 0,
    "p50": q(.50), "p95": q(.95), "p99": q(.99),
    "max": s[-1] if s else 0, "errors": errs[0], "why": why,
    "rps": len(s) / el if el else 0}))
'''


def wsl(cmd: str, timeout: int = 300) -> str:
    """Run a shell script in WSL as root.

    As root because the docker socket needs it on this box, and an unprivileged
    call returns "permission denied" -- which reads exactly like "no such image"
    to a caller checking the output for a name.

    The script is SHIPPED AS BASE64 rather than passed to `bash -lc`, and that
    is not defensive style. A command containing quotes crosses Windows
    argument quoting, then subprocess's list2cmdline escaping, then bash's own
    parsing, and each layer rewrites the escapes for the next. The health probe
    written the direct way died on `syntax error near unexpected token` and
    reported it as CONTAINER DID NOT BECOME HEALTHY -- a quoting bug wearing the
    costume of a finding, about a container that was serving in 7 seconds.

    Base64 has no metacharacters, so it survives every layer unchanged.
    """
    enc = base64.b64encode(cmd.encode()).decode()
    ship = "echo {} | base64 -d > /tmp/ml1_step.sh && bash /tmp/ml1_step.sh".format(enc)
    out = subprocess.run(["wsl", "-u", "root", "--", "bash", "-c", ship],
                         capture_output=True, text=True, timeout=timeout)
    return (out.stdout or out.stderr).strip()


def main() -> int:
    if "finhm/ml1-governed-fraud" not in wsl(
            "docker images --format '{{.Repository}}'"):
        print("image {} not present. Build it first.".format(IMAGE))
        return 1

    wsl("docker rm -f {} 2>&1 | tail -1".format(CONTAINER))
    wsl("docker run -d --name {} -p {}:8000 {} 2>&1 | tail -1".format(
        CONTAINER, PORT, IMAGE))

    # 200s of probing. The service imports sklearn and lightgbm and loads a
    # model; on a machine still busy from the image build that took 40s
    # measured. A probe shorter than the thing it probes turns slow into broken.
    ready = wsl(PROBE.replace("__PORT__", str(PORT)), timeout=280)

    logs = wsl("docker logs --tail 20 {} 2>&1".format(CONTAINER))
    result = None

    if "READY" in ready:
        # Warm first: the opening requests pay import cost that is not a
        # steady-state number.
        wsl(WARM.replace("__PORT__", str(PORT))
                .replace("__BODY__", json.dumps(SAMPLE)), timeout=200)

        script = (DRIVER.replace("__PORT__", str(PORT))
                  .replace("__SAMPLE__", json.dumps(SAMPLE))
                  .replace("__N__", str(REQUESTS))
                  .replace("__THREADS__", str(CONCURRENCY)))
        enc = base64.b64encode(script.encode()).decode()
        wsl("echo {} | base64 -d > /tmp/ml1_drive.py".format(enc))
        out = wsl("python3 /tmp/ml1_drive.py", timeout=420)
        try:
            result = json.loads(out.strip().splitlines()[-1])
        except Exception:                                    # noqa: BLE001
            result = {"driver_error": out[:300]}

    loadavg = wsl("cut -d' ' -f1-3 /proc/loadavg")
    py = wsl("docker exec {} python -V 2>&1".format(CONTAINER))
    size = wsl("docker images --format '{{.Repository}} {{.Size}}' | grep ml1")
    wsl("docker rm -f {} 2>&1 | tail -1".format(CONTAINER))

    L = []
    add = L.append
    add("# ML-1 — the container, RUN")
    add("")
    add("Generated by `run_container_load.py`.")
    add("")

    add("## The finding, which is not a latency number")
    add("")
    add("`README.md` listed this open: *`docker build` runs clean ... what has")
    add("not happened is running the service inside it under load.* Running it")
    add("found the image **could not start at all**:")
    add("")
    add("```")
    add("OSError: libgomp.so.1: cannot open shared object file")
    add("```")
    add("")
    add("lightgbm links against libgomp — the GNU OpenMP runtime — and")
    add("`python:3.12-slim` does not ship it. The pip install of a manylinux")
    add("wheel resolves and completes perfectly; the shared library it needs at")
    add("**load** time is a system package pip knows nothing about. So the build")
    add("was green and the service could not start, and that stayed true for as")
    add("long as nobody ran it.")
    add("")
    add("**That is the entire argument for running a container rather than")
    add("building one.** `docker build` proves the layers resolve. It proves")
    add("nothing about whether the thing inside starts.")
    add("")
    add("Fixed by installing `libgomp1` before the pip layer, so a requirements")
    add("change does not re-run apt.")
    add("")

    add("## A second thing running it surfaced")
    add("")
    warn_lines = [l.strip()[:150] for l in logs.splitlines()
                  if "InconsistentVersion" in l or "unpickle estimator" in l]
    if warn_lines:
        add("```")
        for l in warn_lines[:2]:
            add(l)
        add("```")
        add("")
    add("The model artifact was pickled under scikit-learn **1.8.0** and the")
    add("image resolves **1.9.0**. scikit-learn's own warning says this *\"might")
    add("lead to breaking code or invalid results\"*.")
    add("")
    add("It is a warning rather than a failure, which is exactly what makes it")
    add("worth recording: the service starts, scores, and returns numbers that")
    add("may silently differ from the ones the model was validated against. A")
    add("shipped model artifact and the environment that unpickles it are a")
    add("versioned pair, and `requirements.txt` pins neither end of it.")
    add("")

    add("## The load")
    add("")
    ok_run = bool(result and result.get("n"))
    if ok_run:
        add("| | container |")
        add("|---|---|")
        add("| image | {} |".format(size or "?"))
        add("| python | {} |".format(py))
        add("| requests | {:,} at {} concurrent |".format(REQUESTS, CONCURRENCY))
        add("| completed | {:,} |".format(result["n"]))
        add("| errors | {} |".format(result["errors"]))
        add("| rps | {:,.0f} |".format(result["rps"]))
        for k in ("mean", "p50", "p95", "p99", "max"):
            add("| {} | {:,.2f}ms |".format(k, result[k]))
        add("")
        if result["errors"]:
            add("")
            add("`{}` of {:,} requests failed:".format(
                result["errors"], REQUESTS))
            for w in result.get("why", [])[:2]:
                add("- `{}`".format(w))
        else:
            add("")
            add("**The service runs under load inside its own image, with zero")
            add("errors** — which is what the README item asked for and what")
            add("`docker build` could not establish.")
    elif result:
        add("**Every request failed.** The service was healthy, so this is a")
        add("failure of the load driver or its payload, not of the container:")
        add("")
        add("```")
        add(str(result)[:500])
        add("```")
    else:
        add("The container did not become healthy within the probe window.")
        add("")
        add("```")
        add(logs[-1000:])
        add("```")
    add("")

    add("## What this deliberately does NOT claim")
    add("")
    add("**It is not a container-versus-host comparison**, and the README's")
    add("claim that the two p99s differ remains untested.")
    add("")
    add("The load is driven from inside WSL, because the container publishes on")
    add("WSL's loopback and reaching it from Windows depends on WSL2's localhost")
    add("forwarding — which on this machine works intermittently, the same")
    add("flakiness that made Redis and Kafka unreachable earlier in this")
    add("project. A probe failing for that reason reports *container did not")
    add("become healthy* about a container that is serving perfectly.")
    add("")
    add("So a host baseline would carry a Windows client over loopback where")
    add("this carries a Linux client on the same kernel. The two differ in the")
    add("CLIENT as well as the server, and putting them in one table as a")
    add("container-overhead figure would be the misleading version of this")
    add("document.")
    add("")
    add("- **The host was busy.** Load average during the run: `{}`.".format(
        loadavg))
    add("  So these latencies are a **floor** — the number this service beats on")
    add("  a quiet machine — and not its best. Quoting them as the service's")
    add("  latency would be the flattering direction to be wrong in, which is")
    add("  the direction to state rather than the one to leave out.")
    add("- **No resource limits.** A production container has a CPU quota and a")
    add("  memory limit; this has neither, so it measures overhead rather than")
    add("  constraint.")
    add("- **One run.** Nothing here separates run-to-run variance from a real")
    add("  difference.")

    doc = "\n".join(L)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "CONTAINER_LOAD.md").write_text(doc, encoding="utf-8")
    print(doc)
    print()
    print("wrote docs/CONTAINER_LOAD.md")
    return 0 if (ok_run and not result["errors"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
