# CODE MAP of classifiers/lr.py
#   Classifier   make(): logistic regression with a strong L2 penalty, the linear baseline of every node
#   AI use       Claude Opus 5.5 (claude-opus-5-5), default setting

# %% Classifier
from sklearn.linear_model import LogisticRegression

SEEDS = [0]                  # deterministic: one run is enough


def make(seed=0):
    """Logistic regression, C = 0.1 (smaller C = stronger penalty: 384-768 features, ~90 training clips)."""
    return LogisticRegression(C=0.1, max_iter=2000)
