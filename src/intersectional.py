"""Intersectional disparate impact, and the reason one-attribute-at-a-time misses it.

The standard fairness section computes an adverse impact ratio for each protected
attribute on its own: an AIR by group A, an AIR by group B, both above 0.80, file
closed. That procedure has a documented failure mode -- **each marginal AIR can
pass while a CELL of the cross fails**, because the two attributes are correlated
with each other and with the score, and averaging over one of them hides the
other. It is the same structure as Simpson's paradox and it is not exotic; it is
what happens whenever the disadvantage concentrates in an intersection.

The regulatory framing is not settled and this module does not pretend it is.
US fair-lending enforcement is organised around single prohibited bases, and
"intersectional discrimination" has no clean statutory home. That is a reason to
MEASURE it and say so, not a reason to skip it: a cell of the book taking a
materially worse outcome is a business problem before it is a legal one.

REFERENCE CELL. Each cell's ratio is computed against the **most-favoured cell**,
not against a chosen reference group. Picking the reference is picking the answer,
and the most-favoured cell is the only choice that does not require a defence --
it is the treatment the model demonstrably CAN give, so every gap below it is a
gap the model itself created.

SMALL CELLS. The 2x2 cross quarters the sample and the smallest cell drives the
verdict, so every cell is reported with its n and a bootstrap interval, and cells
below `min_n` are marked `insufficient` rather than being given a point estimate.
An AIR of 0.62 on 40 applicants is not a finding, and printing it as one is how
an intersectional section produces false alarms and loses its audience.
"""
from __future__ import annotations

import numpy as np

MIN_CELL_N = 200


def _rate(selected, mask):
    return float(selected[mask].mean()) if mask.any() else float("nan")


def intersectional_table(selected: np.ndarray, attr_a: np.ndarray,
                         attr_b: np.ndarray, name_a: str = "A",
                         name_b: str = "B", min_n: int = MIN_CELL_N,
                         n_boot: int = 400, seed: int = 11) -> dict:
    """Favourable-outcome rate per cell of the A x B cross, vs the best cell."""
    selected = np.asarray(selected).astype(float)
    a = np.asarray(attr_a).astype(int)
    b = np.asarray(attr_b).astype(int)
    rng = np.random.default_rng(seed)

    cells = []
    for va in (0, 1):
        for vb in (0, 1):
            mask = (a == va) & (b == vb)
            cells.append({"cell": "{}={} , {}={}".format(name_a, va, name_b, vb),
                          "a": va, "b": vb, "n": int(mask.sum()),
                          "rate": _rate(selected, mask), "mask": mask})

    usable = [c for c in cells if c["n"] >= min_n and np.isfinite(c["rate"])]
    best = max(usable, key=lambda c: c["rate"])["rate"] if usable else float("nan")

    for c in cells:
        if c["n"] < min_n or not np.isfinite(c["rate"]):
            c["ratio"], c["ci"], c["status"] = float("nan"), None, "insufficient"
            continue
        c["ratio"] = c["rate"] / best if best else float("nan")
        idx = np.flatnonzero(c["mask"])
        draws = [selected[rng.choice(idx, len(idx), replace=True)].mean() / best
                 for _ in range(n_boot)]
        c["ci"] = (float(np.percentile(draws, 2.5)),
                   float(np.percentile(draws, 97.5)))
        c["status"] = "below 0.80" if c["ratio"] < 0.80 else "ok"
        c.pop("mask")

    for c in cells:
        c.pop("mask", None)

    # Marginals: exactly the one-attribute-at-a-time analysis this module exists
    # to compare against.
    def marginal(attr, name):
        r1, r0 = _rate(selected, attr == 1), _rate(selected, attr == 0)
        lo, hi = (r1, r0) if r1 <= r0 else (r0, r1)
        return {"attribute": name, "rate_1": r1, "rate_0": r0,
                "air": lo / hi if hi else float("nan")}

    marginals = [marginal(a, name_a), marginal(b, name_b)]
    failing_cells = [c for c in cells if c["status"] == "below 0.80"]
    marginals_pass = all(m["air"] >= 0.80 for m in marginals
                         if np.isfinite(m["air"]))

    return {
        "cells": cells,
        "best_cell_rate": best,
        "marginals": marginals,
        "min_cell_n": min_n,
        "masked_by_marginals": bool(failing_cells and marginals_pass),
        "verdict": (
            "a CELL fails the 80% rule while every marginal AIR passes -- the "
            "single-attribute analysis would have closed this file"
            if failing_cells and marginals_pass else
            "cell(s) fail and the marginals flag it too"
            if failing_cells else
            "no cell below 0.80 at this operating point"),
    }


def format_table(res: dict) -> str:
    lines = ["{:<22}{:>8}{:>10}{:>10}{:>22}{:>14}".format(
        "cell", "n", "rate", "ratio", "95% CI", "status")]
    for c in res["cells"]:
        ci = ("[{:.3f}, {:.3f}]".format(*c["ci"]) if c["ci"] else "--")
        ratio = "{:.3f}".format(c["ratio"]) if np.isfinite(c["ratio"]) else "--"
        lines.append("{:<22}{:>8}{:>10.4f}{:>10}{:>22}{:>14}".format(
            c["cell"], c["n"], c["rate"], ratio, ci, c["status"]))
    lines.append("")
    lines.append("{:<22}{:>10}{:>10}{:>10}".format(
        "marginal (one at a time)", "rate=1", "rate=0", "AIR"))
    for m in res["marginals"]:
        lines.append("{:<22}{:>10.4f}{:>10.4f}{:>10.3f}".format(
            m["attribute"], m["rate_1"], m["rate_0"], m["air"]))
    return "\n".join(lines)
