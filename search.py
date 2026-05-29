import numpy as np
from scipy.stats import loguniform, uniform
from sklearn.model_selection import RandomizedSearchCV

from class_CompositeKRR import CompositeKRR, KernelComponent


def staged_random_search_cv(estimator) -> RandomizedSearchCV:
    search = RandomizedSearchCV(estimator=estimator)

    return search
