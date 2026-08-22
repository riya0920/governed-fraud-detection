"""Imbalance-handling bakeoff: baseline vs class weights vs SMOTE vs focal loss.

The deliverable is the comparison table, not the winner. "SMOTE is usually wrong
on tabular fraud" is a claim you have to earn by running it, so this module
implements SMOTE rather than importing it, and the training script reports all
four on the same out-of-time split.

SMOTE is implemented here (imbalanced-learn is not a dependency) so the
interpolation semantics are visible: synthetic minority points are convex
combinations of a minority sample and one of its k minority neighbours. That
construction is exactly why it tends to disappoint on fraud data -- fraud
minority points are not a smooth manifold, they are scattered modes, so the
interpolated points land in regions where no fraud actually lives and the model
learns a blurred boundary. Calibration is the first casualty.

FOCAL LOSS needs a custom objective, which is why it was absent while this
project used sklearn's HistGradientBoosting. LightGBM exposes one, so it is now
a real fourth arm rather than a note in the README.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

The (1-p_t)^gamma factor down-weights examples the model already gets right, so
gradient budget flows to the hard ones. On fraud that is a different bet from
class weighting: weighting says "every positive matters more", focal says "every
UNCERTAIN example matters more, whichever class it is in". Those are not the
same claim and they do not have to agree.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Domain monotonicity. Risk rises with amount, velocity, cross-border, device
# change and merchant risk; it falls with account tenure. An unconstrained model
# that disagrees in a thin region is fitting noise, and it produces reason codes
# that read as absurd to the agent reading them out.
MONOTONE_BY_FEATURE = {
    "amount_minor": 1,
    "velocity_24h": 1,
    "cross_border": 1,
    "device_change": 1,
    "mcc_risk": 1,
    "hour": 0,
    "card_tenure_days": -1,
    "is_night": 1,
    "amount_per_velocity": 1,
}

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


def smote(X: np.ndarray, y: np.ndarray, target_ratio: float = 0.25,
          k: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class up to `target_ratio` of the majority."""
    rng = np.random.default_rng(seed)
    minority = X[y == 1]
    n_major = int((y == 0).sum())
    n_needed = int(target_ratio * n_major) - len(minority)
    if n_needed <= 0 or len(minority) <= k:
        return X, y

    nn = NearestNeighbors(n_neighbors=k + 1).fit(minority)
    _, idx = nn.kneighbors(minority)
    base = rng.integers(0, len(minority), n_needed)
    neigh = idx[base, rng.integers(1, k + 1, n_needed)]
    lam = rng.random((n_needed, 1))
    synthetic = minority[base] + lam * (minority[neigh] - minority[base])

    X_out = np.vstack([X, synthetic])
    y_out = np.concatenate([y, np.ones(n_needed, dtype=y.dtype)])
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


class FocalObjective:
    """LightGBM custom objective: gradient and hessian of focal loss.

    A CLASS rather than a closure, because LightGBM stores the objective on the
    fitted estimator and the estimator gets pickled into the serving artifact.
    A closure is not picklable, so a closure here means the model trains fine
    and then cannot be saved -- a failure that only appears at the very end of
    the pipeline, after all the expensive work.
    """

    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA):
        self.alpha = alpha
        self.gamma = gamma

    def _grad(self, y, raw):
        """dFL/dz, derived rather than adapted from a blog post.

        With p = sigmoid(z), pt = p if y=1 else 1-p, at = alpha if y=1 else
        1-alpha, and dp/dz = p(1-p):

            d/dz (1-pt)^gamma = -gamma * pt * (1-pt)^gamma        [y=1]
            d/dz log(pt)      = (1-pt)                            [y=1]

        so  d/dz [ (1-pt)^gamma * log(pt) ] = (1-pt)^gamma * [(1-pt) - gamma*pt*log(pt)]
        and FL = -at * (1-pt)^gamma * log(pt) gives

            grad = -at * (1-pt)^gamma * [(1-pt) - gamma*pt*log(pt)]   for y=1
            grad = +at * (1-pt)^gamma * [(1-pt) - gamma*pt*log(pt)]   for y=0

        The sign flip for y=0 is because dpt/dz reverses. An earlier version of
        this function carried a spurious 1/(1-pt) on the log term; the model
        trained to AUC 0.5000 and Brier 0.25 -- a constant predictor. Worth
        noting HOW that failed: it did not error, it produced a model that ran,
        scored, and was useless, and only the bakeoff table caught it.
        """
        p = np.clip(1.0 / (1.0 + np.exp(-raw)), 1e-9, 1 - 1e-9)
        pt = np.where(y == 1, p, 1 - p)
        at = np.where(y == 1, self.alpha, 1 - self.alpha)
        g = self.gamma
        base = at * (1 - pt) ** g * ((1 - pt) - g * pt * np.log(pt))
        return np.where(y == 1, -base, base)

    def __call__(self, y_true, raw):
        y = np.asarray(y_true, dtype=np.float64)
        raw = np.asarray(raw, dtype=np.float64)
        grad = self._grad(y, raw)

        # Numerical second derivative. The analytic hessian for focal loss is
        # long and easy to get wrong, and a wrong hessian is worse than an
        # approximate one because it biases every split silently rather than
        # failing loudly.
        eps = 1e-4
        hess = (self._grad(y, raw + eps) - grad) / eps
        # LightGBM divides by the hessian when scoring splits; a hessian that
        # underflows produces enormous leaf values and +/-inf predictions.
        return grad, np.clip(hess, 1e-6, None)


def focal_loss_objective(alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA):
    return FocalObjective(alpha, gamma)


class FocalClassifier:
    """Wraps a LightGBM model trained with a custom objective.

    LightGBM refuses to produce probabilities for a custom objective and returns
    the raw margin instead -- correctly, because it cannot know the link
    function. Focal loss is defined on the sigmoid, so that is the link applied
    here.

    Worth stating plainly: a sigmoid of a focal-trained margin is NOT a
    calibrated probability. Focal loss deliberately down-weights easy examples,
    which distorts the base rate the model is fit against in the same way class
    weighting does. Section 5 chooses the operating threshold from a $-weighted
    cost curve and that arithmetic consumes calibrated probabilities, so a focal
    model would need isotonic recalibration before it could be promoted -- which
    is exactly what the calibration gate in src/promotion.py is for.
    """

    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw = np.asarray(self.model.predict_proba(X), dtype=float)
        if raw.ndim == 2 and raw.shape[1] == 2:
            return raw
        raw = raw.reshape(-1)
        p = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def booster_(self):
        return self.model.booster_

    def __getattr__(self, item):
        # Guard dunder lookups. Without this, pickle asks for __getstate__,
        # __getattr__ forwards to self.model, and during UNpickling self.model
        # does not exist yet -- so the forward recurses and the artifact cannot
        # be saved. A wrapper that forwards everything must still refuse to
        # forward the protocol methods that construct it.
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return getattr(self.model, item)

    def __getstate__(self):
        return {"model": self.model, "classes_": self.classes_}

    def __setstate__(self, state):
        self.__dict__.update(state)


def make_model(seed: int = 0, features: list[str] | None = None,
               monotone: bool = False, objective=None):
    """LightGBM, with one config shared across every strategy.

    Sharing the config is the point: otherwise the comparison measures
    hyperparameter luck rather than the imbalance treatment.
    """
    import lightgbm as lgb

    params = dict(
        n_estimators=400, learning_rate=0.05, num_leaves=15,
        min_child_samples=200, reg_lambda=5.0, subsample=0.9,
        subsample_freq=1, colsample_bytree=0.9,
        random_state=seed, verbose=-1, n_jobs=1)

    if monotone and features:
        params["monotone_constraints"] = [
            MONOTONE_BY_FEATURE.get(f, 0) for f in features]
    if objective is not None:
        params["objective"] = objective
        # THE ONE PLACE THE CONFIGS DIVERGE, and it is disclosed rather than
        # buried. reg_lambda=5.0 -- which suits logloss fine -- suppresses focal
        # training completely: LightGBM builds ONE tree, every raw score is 0.0,
        # and the arm scores AUC 0.5000. It does not error; it produces a model
        # that runs, predicts, and is a constant.
        #
        # Measured sweep on this data (50 trees, all else held):
        #     reg_lambda  5.0 -> 1 tree,  AUC 0.5000
        #     reg_lambda  1.0 -> 50 trees, AUC 0.7666
        #     reg_lambda  0.1 -> 50 trees, AUC 0.7712
        #     reg_lambda  0.0 -> 50 trees, AUC 0.7708
        #
        # HYPOTHESIS, not a proven mechanism: LightGBM's split gain is
        # G^2/(H+lambda) summed over children minus the parent, so lambda
        # competes with the summed hessian. Focal deliberately shrinks the
        # gradient contribution of the easy majority, so after the first tree
        # the aggregate G collapses and a lambda tuned for logloss dominates
        # every candidate gain. The measured sweep is the evidence; the
        # explanation is inference from it.
        #
        # The wider point is the uncomfortable one: "identical config across all
        # arms" was chosen so the bakeoff would measure the imbalance treatment
        # rather than hyperparameter luck. For a custom objective that fairness
        # rule is what broke the arm, because regularisation strength is not
        # comparable across objectives. The comparison below is therefore
        # honest-but-imperfect, and saying so is better than either silently
        # retuning every arm or shipping a constant predictor.
        params["reg_lambda"] = 0.1
    return lgb.LGBMClassifier(**params)


def fit_strategy(name: str, X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 0,
                 features: list[str] | None = None, monotone: bool = False):
    """Returns (fitted_model, note). Strategies share the model config above."""
    if name == "baseline":
        m = make_model(seed, features, monotone)
        m.fit(X_tr, y_tr)
        return m, "no imbalance handling"

    if name == "class_weight":
        m = make_model(seed, features, monotone)
        pos, neg = float((y_tr == 1).sum()), float((y_tr == 0).sum())
        w = np.where(y_tr == 1, neg / pos, 1.0)
        m.fit(X_tr, y_tr, sample_weight=w)
        return m, "sample_weight = neg/pos = {:.1f} on positives".format(neg / pos)

    if name == "smote":
        m = make_model(seed, features, monotone)
        Xs, ys = smote(X_tr, y_tr, target_ratio=0.25, seed=seed)
        m.fit(Xs, ys)
        return m, "minority resampled to 25% of majority ({} -> {} rows)".format(
            len(y_tr), len(ys))

    if name == "focal":
        m = make_model(seed, features, monotone,
                       objective=focal_loss_objective())
        m.fit(X_tr, y_tr)
        return FocalClassifier(m), (
            "focal a={} g={}; reg_lambda 0.1 not 5.0 (see imbalance.py) -- "
            "raw margin via sigmoid, NOT calibrated".format(
                FOCAL_ALPHA, FOCAL_GAMMA))

    raise ValueError(name)


STRATEGIES = ["baseline", "class_weight", "smote", "focal"]
