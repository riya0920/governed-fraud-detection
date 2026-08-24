"""Fetch the ULB credit-card fraud dataset -- resumably.

    python fetch_real_data.py

WHY THIS IS NOT A ONE-LINE urlretrieve. The file is 156 MB from OpenML and the
connection here drops it partway: two straight attempts returned 64 MB and 26 MB
respectively, both with a clean exit code and a file that *looked* fine. The
second one ended mid-number, so the last row parsed as garbage rather than
failing loudly.

That is the failure mode worth engineering against: **a truncated download does
not announce itself.** It produces a smaller dataset that trains, scores, and
reports metrics -- on however much of the data arrived. A fraud model fitted on
the first 40% of a time-ordered file is a model fitted on the first two days.

So this does three things a one-liner does not:

  RESUMES     HTTP Range requests, so a drop costs the remaining bytes rather
              than all of them.
  VERIFIES    against Content-Length, and refuses to report success on a short
              file.
  IS IDEMPOTENT  re-running continues rather than restarting.

THE DATA. Credit-card transactions from September 2013, European cardholders,
released by the Machine Learning Group at ULB. 284,807 transactions, 492 frauds
-- 0.172%. Features V1..V28 are PCA components: the original fields were
anonymised before release, which matters enormously for this project and is
discussed in `run_real_fraud.py`.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

# The ARFF endpoint is 156 MB and its host IGNORES Range requests -- it answers
# 200 with the whole file, so a drop at 22 MB costs all 22 MB and the next
# attempt starts again from zero. Measured three times; it never finished.
#
# The parquet mirror is 73 MB (same data, columnar and compressed) and answers
# 206 with a Content-Range, so a drop costs only what is left. That difference
# is the whole reason this file exists rather than a urlretrieve call.
URL = "https://data.openml.org/datasets/0004/42175/dataset_42175.pq"
DEST = Path(__file__).resolve().parent / "data" / "creditcard.parquet"
UA = {"User-Agent": "Mozilla/5.0 (research; portfolio)"}


def total_size(url: str = URL) -> int | None:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:                                        # noqa: BLE001
        return None


def fetch(url: str = URL, dest: Path = DEST, attempts: int = 40) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = total_size(url)
    if expected is None:
        print("server did not report a size; cannot verify completeness")

    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected and have >= expected:
            break

        headers = dict(UA)
        if have:
            headers["Range"] = "bytes={}-".format(have)

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r:
                # 206 = the server honoured the range. 200 means it ignored it
                # and is sending the whole file again, so the local file has to
                # be truncated or the two halves interleave into nonsense.
                mode = "ab"
                if r.status == 200 and have:
                    print("   server ignored Range; restarting from zero")
                    mode = "wb"
                with dest.open(mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
        except Exception as exc:                             # noqa: BLE001
            print("   attempt {}: {} after {:,} bytes".format(
                attempt, type(exc).__name__, dest.stat().st_size
                if dest.exists() else 0))
            time.sleep(2)
            continue

        got = dest.stat().st_size
        pct = (100.0 * got / expected) if expected else 0.0
        print("   attempt {}: {:,} bytes ({:.1f}%)".format(attempt, got, pct))
        if expected and got >= expected:
            break

    got = dest.stat().st_size if dest.exists() else 0
    if expected and got < expected:
        print("INCOMPLETE: {:,} of {:,} bytes".format(got, expected))
        return False
    print("COMPLETE: {:,} bytes".format(got))
    return True


def verify(dest: Path = DEST) -> bool:
    """A size check is necessary and not sufficient -- open the file too.

    Parquet keeps its footer at the END, so a truncated file refuses to open
    rather than reading short. That is a strictly better failure than the
    ARFF's, where a cut file parsed cleanly and simply contained fewer rows --
    a fraud model fitted on the first 40% of a time-ordered file is a model
    fitted on the first two days, and nothing about it looks wrong.
    """
    if not dest.exists():
        return False
    try:
        import pandas as pd

        df = pd.read_parquet(dest)
    except Exception as exc:                                 # noqa: BLE001
        print("cannot open: {} -- file is incomplete".format(type(exc).__name__))
        return False
    ok = len(df) == 284_807
    print("{:,} rows, {} columns, {:.4%} fraud -- {}".format(
        len(df), len(df.columns), df["Class"].astype(int).mean(),
        "OK" if ok else "SHORT (expected 284,807)"))
    return ok


if __name__ == "__main__":
    good = fetch()
    good = verify() and good
    sys.exit(0 if good else 1)
