"""Historical import path for the Nyström estimator."""
from composite_krr.estimators.nystrom import (
    CompositeNystromKRREstimator,
    select_landmark_indices,
    torch_exp_scaled,
)

__all__ = ["CompositeNystromKRREstimator", "select_landmark_indices", "torch_exp_scaled"]
