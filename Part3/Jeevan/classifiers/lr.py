# CODE MAP of classifiers/lr.py
#   Classifier   make(): logistic regression on the 46 joint-angle features
#   AI use       Claude Opus 5.5 (claude-opus-5-5), default setting

# %% Classifier
from sklearn.linear_model import LogisticRegression

SEEDS = [0]                  # deterministic: one run is enough


def make(seed=0):
    """Logistic regression, C = 1 (default L2 penalty)."""
    return LogisticRegression(C=1.0, max_iter=2000)
