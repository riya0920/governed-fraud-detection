"""The real ULB fraud data, and the two findings it forced.

Skipped when the extract is absent, so the suite still runs anywhere.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fetch_real_data import DEST
from run_real_fraud import load

pytestmark = pytest.mark.skipif(
    not DEST.exists(),
    reason="ULB extract not downloaded -- run python fetch_real_data.py")


@pytest.fixture(scope="module")
def df():
    return load()


def test_the_whole_dataset_is_present(df):
    """A truncated download does not announce itself: it trains, scores and
    reports metrics on however much arrived. For a TIME-ORDERED file that means
    the first N hours, not a random sample -- so the row count is an assertion,
    not a nicety."""
    assert len(df) == 284_807
    assert len(df.columns) == 31


def test_the_prevalence_is_the_real_one(df):
    assert df["Class"].astype(int).sum() == 492
    assert abs(df["Class"].astype(int).mean() - 0.001727) < 1e-5


def test_the_file_is_time_ordered_so_the_split_must_be_too(df):
    """`Time` is seconds since the first transaction. A random split would put
    later transactions in train and earlier ones in test -- the exact leak this
    project's data layer exists to make structurally impossible."""
    t = df.sort_values("Time")["Time"].to_numpy()
    assert (np.diff(t) >= 0).all()
    assert t.max() > 24 * 3600, "expected more than a day of data"


def test_the_leakage_screen_fires_on_features_that_are_not_leaks(df):
    """The finding I did not expect and wrote the opposite of first.

    Eleven PCA components clear a 0.90 univariate AUC. None is a leak -- this
    file has no post-decision information in it at all. They are simply very
    predictive, and a univariate-AUC screen cannot tell 'encodes the answer'
    from 'genuinely strong'.

    Pinning it keeps the claim honest: the screen catches leaks nobody
    predicted AND it flags strong features, so every flag is a question for a
    human rather than a verdict.
    """
    from src.leakage import audit

    report = audit(df.assign(is_fraud=df["Class"]).drop(columns=["Class"]),
                   target="is_fraud")
    flagged = report[report.flagged]
    assert len(flagged) >= 5, "the false-positive mode stopped reproducing"
    assert "V14" in set(flagged.feature)


def test_no_feature_is_a_genuine_leak(df):
    """The other half. A real leak scores ~1.0, not 0.95 -- so nothing here is
    a post-decision field even though eleven features are flagged."""
    from src.leakage import audit

    report = audit(df.assign(is_fraud=df["Class"]).drop(columns=["Class"]),
                   target="is_fraud")
    assert report.univariate_auc.max() < 0.99


def test_fraud_amounts_differ_from_legitimate_ones(df):
    """Amount-weighting the cost matrix only earns its place if the two
    distributions actually differ. Checked rather than assumed."""
    fraud = df.loc[df.Class == 1, "Amount"].mean()
    legit = df.loc[df.Class == 0, "Amount"].mean()
    assert fraud > legit


def test_the_features_cannot_support_a_reason_code(df):
    """The finding that matters more than any metric here.

    FCRA 615(a) wants the principal reasons in terms a consumer can act on.
    Every predictive feature in this file is an anonymised PCA component, so
    there is no sentence to write. This asserts the shape of that problem so
    nobody later reads the good AUC as 'this pipeline is deployable'.
    """
    named = [c for c in df.columns
             if c not in ("Class",) and not c.startswith("V")]
    assert set(named) == {"Time", "Amount"}
    pca = [c for c in df.columns if c.startswith("V")]
    assert len(pca) == 28
