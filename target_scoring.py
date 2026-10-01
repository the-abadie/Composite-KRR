"""Optional fold-transformed target objectives shared by sklearn and cached search."""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import get_scorer

from target_utils import align_targets_for_scoring, as_target_matrix


class _Predictions(RegressorMixin, BaseEstimator):
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X):
        return self.prediction


class TransformedTargetScorer:
    """Score after the target transform fitted on this CV fold's training rows."""

    def __init__(self, name):
        self.name = name
        self.scorer = get_scorer(name)
        self._sign = self.scorer._sign

    def __call__(self, estimator, X, y):
        transformer = getattr(estimator, "transformer_", None)
        return self.score_predictions(y, estimator.predict(X), transformer)

    def score_predictions(self, y_true, y_pred, transformer):
        if transformer is None:
            raise ValueError("Transformed target scoring requires a fold-fitted target transformer.")
        true = transformer.transform(as_target_matrix(y_true, dtype=float))
        pred = transformer.transform(as_target_matrix(y_pred, dtype=float))
        true, pred = align_targets_for_scoring(true, pred)
        return float(self.scorer(_Predictions(pred), None, true))

    def __repr__(self):
        return f"TransformedTargetScorer({self.name!r})"


def make_target_scorer(name, target_space="original"):
    name = "neg_root_mean_squared_error" if name == "rmse" else name
    if target_space == "original":
        return name
    if target_space == "transformed":
        return TransformedTargetScorer(name)
    raise ValueError("Scoring target_space must be original or transformed.")
