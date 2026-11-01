# CODE MAP of classifiers/lr.py
#   Classifier   make(): logistic regression, the linear baseline of every node
#   AI use       Claude Opus 5.5 (claude-opus-5-5), default setting

# %% Classifier
from sklearn.linear_model import LogisticRegression

import common


def make(C=1.0):
    """Logistic regression with L2 penalty (smaller C = stronger). Deterministic, so no seed is needed."""
    return LogisticRegression(C=C, max_iter=2000, random_state=common.SEED)
