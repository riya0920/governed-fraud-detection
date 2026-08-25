"""The intersectional analysis, on protected attributes this repo did not invent."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.intersectional import intersectional_table

HMDA = ROOT.parent / "ml3-credit-scorecard" / "data" / "hmda_de_2023.csv"

real_only = pytest.mark.skipif(
    not HMDA.exists(),
    reason="HMDA extract not downloaded -- see ml3-credit-scorecard/src/hmda.py")


def _hmda():
    """Load ML-3's module BY PATH.

    Both projects have a package called `src` and ML-1's is already imported,
    so `from src.hmda import ...` resolves against this repo and fails. Putting
    ML-3 on sys.path would be worse than failing: it would make which module
    wins depend on import order.
    """
    spec = importlib.util.spec_from_file_location(
        "hmda_ml3_test", HMDA.parents[1] / "src" / "hmda.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hmda_ml3_test"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------- the constraint, on synthetic data
def test_the_table_requires_binary_attributes():
    """A real constraint of the module, not a convenience of the runner:
    `intersectional_table` iterates over (0, 1) for each axis. Naming what that
    costs is the first finding of running it on real data."""
    import inspect

    import src.intersectional as inter

    src = inspect.getsource(inter.intersectional_table)
    assert "for va in (0, 1)" in src
    assert "for vb in (0, 1)" in src


def test_a_cell_can_fail_while_both_marginals_pass():
    """The whole reason the module exists, on constructed data where the effect
    is known -- so the real run can be read against a working detector rather
    than a hopeful one."""
    rng = np.random.default_rng(0)
    n = 8000
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    both = (a == 1) & (b == 1)
    # Calibrated deliberately. At 0.45 the effect is so strong it drags BOTH
    # marginals under 0.80 as well, and the test then proves nothing -- the
    # single-attribute analysis would have caught it. 0.58 fails the cell
    # (0.58/0.80 = 0.725) while each marginal averages to ~0.69 against 0.80,
    # an AIR of ~0.86 that passes. That gap IS the phenomenon.
    p = np.where(both, 0.58, 0.80)
    selected = (rng.random(n) < p).astype(int)

    res = intersectional_table(selected, a, b, min_n=100)
    assert res["masked_by_marginals"] is True
    assert "closed this file" in res["verdict"]


def test_a_clean_population_reports_no_finding():
    """A fairness tool that only ever reports findings is one nobody should
    trust."""
    rng = np.random.default_rng(1)
    n = 8000
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    selected = (rng.random(n) < 0.75).astype(int)
    res = intersectional_table(selected, a, b, min_n=100)
    assert not [c for c in res["cells"] if c["status"] == "below 0.80"]


def test_a_thin_cell_is_marked_insufficient_rather_than_ranked():
    """A cell of nine applicants has a rate that is noise, and ranking on it
    puts noise at the top of the report -- which is where the reader looks."""
    rng = np.random.default_rng(2)
    n = 4000
    a = rng.integers(0, 2, n)
    b = np.zeros(n, dtype=int)
    b[:20] = 1                       # a deliberately tiny b=1 population
    selected = (rng.random(n) < 0.7).astype(int)
    res = intersectional_table(selected, a, b, min_n=100)
    thin = [c for c in res["cells"] if c["n"] < 100]
    assert thin and all(c["status"] == "insufficient" for c in thin)


# ------------------------------------------------------ on the real thing
@real_only
def test_the_analysis_runs_on_two_REAL_protected_attributes():
    """The README's item: "both attributes ... are synthetic overlays
    constructed by this repo, so no fairness number here describes a real
    population." HMDA carries race AND sex, self-reported, on the same
    applicants."""
    hmda = _hmda()
    df = hmda.protected(hmda.load())
    df = df[df.race_reported & df.derived_sex.isin(["Male", "Female"])]

    selected = (df.denied == 0).to_numpy().astype(int)
    res = intersectional_table(selected,
                               df.is_minority.to_numpy().astype(int),
                               (df.derived_sex == "Female").to_numpy().astype(int),
                               min_n=100)

    assert len(df) > 10_000
    assert all(c["n"] >= 100 for c in res["cells"]), (
        "every cell should be populated on a state-sized book")
    for m in res["marginals"]:
        assert 0 < m["air"] <= 1.0


@real_only
def test_the_intersection_is_worse_than_either_marginal_on_this_data():
    """The direction, which a threshold verdict discards.

    The worst cell sits at 0.8443 against marginals of 0.8569 and 0.9878 -- so
    the effect the module was written to find IS present; it simply does not
    cross 0.80. Reporting that as "no finding" would throw the direction away.
    """
    hmda = _hmda()
    df = hmda.protected(hmda.load())
    df = df[df.race_reported & df.derived_sex.isin(["Male", "Female"])]

    res = intersectional_table(
        (df.denied == 0).to_numpy().astype(int),
        df.is_minority.to_numpy().astype(int),
        (df.derived_sex == "Female").to_numpy().astype(int), min_n=100)

    worst = min(c["ratio"] for c in res["cells"]
                if c["ratio"] == c["ratio"])
    marginals = [m["air"] for m in res["marginals"]]
    assert worst < min(marginals), (
        "the intersection should be worse than either attribute alone")
    assert worst >= 0.80, (
        "if this now fails the 80% rule the document's 'no finding' verdict is "
        "wrong and needs rewriting rather than trusting")


@real_only
def test_binarising_race_is_the_aggregation_ML3_showed_hides_a_group():
    """The limitation running it on real data exposed.

    Both axes must be binary, which forces race to minority/non-minority. On
    this same data ML-3 found the aggregate AIR passes while individual groups
    fail -- so a binary table can report "no cell below 0.80" on a population
    where a named group is being declined at a failing rate.
    """
    hmda = _hmda()
    df = hmda.protected(hmda.load())
    reported = df[df.race_reported]

    rates = hmda.approval_rates(reported, "race_group")
    big = rates[rates.applications >= 50]
    failing = big[~big.passes_80pc]
    assert len(failing) >= 1, (
        "no individual race group fails any more, so the aggregation warning "
        "in docs/INTERSECTIONAL_REAL.md needs rechecking")
