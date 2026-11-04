# CODE MAP of classifiers/lr.py
#   Fallback LR   FallbackLR: logistic regression on the clips whose features exist; a clip with no usable
#                 motion (NaN features) gets the majority label of the training clips (tie -> light)
#   Classifier    make(C): the classifier; C (L2 strength) is chosen on validation from {0.1, 1}
#   AI use        GPT-6.1 Sol (setting: medium); GPT-6 Astra (setting: light) (original code);; GitHub Copilot (VS Code; extension version, model/version and setting unavailable)
#                 Claude Opus 5.5 (claude-opus-5-5), default setting (restructure)

# %% Fallback LR
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

SEEDS = [0]                  # deterministic: one run is enough
C_VALUES = [0.1, 1.0]        # tried on validation


class FallbackLR(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        ok = np.isfinite(X).all(axis=1)
        self.classes_ = np.array([0, 1])
        self.fallback_ = int(np.sum(y) > len(y) / 2)     # majority of ALL training labels; a tie predicts light
        self.model_ = None
        if ok.any() and len(np.unique(y[ok])) == 2:
            self.model_ = LogisticRegression(C=self.C, max_iter=2000, random_state=0).fit(X[ok], y[ok])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        ok = np.isfinite(X).all(axis=1) & (self.model_ is not None)
        p = np.full(len(X), float(self.fallback_))
        if ok.any():
            p[ok] = self.model_.predict_proba(X[ok])[:, 1]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# %% Classifier
def make(seed=0, C=1.0):
    """Fallback logistic regression with penalty C (smaller C = stronger)."""
    return FallbackLR(C=C)
