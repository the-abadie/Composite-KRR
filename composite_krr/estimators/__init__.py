"""Scikit-learn estimators for exact and Nyström composite kernel ridge regression."""
from .exact import CompositeKRREstimator
from .nystrom import CompositeNystromKRREstimator

__all__ = ["CompositeKRREstimator", "CompositeNystromKRREstimator"]
