# CODE MAP of classifiers/mlp_ensemble.py
#   Ensemble     EnsembleMLP: the 5 MLPs of mlp.py (seeds 0-4) trained on the same clips, probabilities averaged
#   Classifier   make(): the ensemble; this is what ../FinalMethod.py trains
#   AI use       Claude Opus 5.5 (claude-opus-5-5), default setting
#
# mlp.py reports the mean accuracy of 5 separate models; this is ONE model that averages them,
# which removes the luck of the starting weights (usually as good or better).

# %% Ensemble
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from . import mlp

SEEDS = [0]                  # the 5 seeds are inside the ensemble


class EnsembleMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, seeds=tuple(mlp.SEEDS)):
        self.seeds = seeds

    def fit(self, X, y):
        self.members_ = [mlp.make(s).fit(X, y) for s in self.seeds]
        self.classes_ = self.members_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members_], axis=0)

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


# %% Classifier
def make(seed=0):
    """5-MLP ensemble (seeds 0-4); the seed argument is unused."""
    return EnsembleMLP()
