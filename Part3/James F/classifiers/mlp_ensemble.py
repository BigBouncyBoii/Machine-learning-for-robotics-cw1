# CODE MAP of classifiers/mlp_ensemble.py
#   Ensemble     EnsembleMLP: 10 MLPs that differ only in their starting weights, probabilities averaged
#   Classifier   make(): the ensemble with L2 penalty alpha
#   AI use       Claude Opus 5.5 (claude-opus-5-5), GPT-6 Astra, GPT-6.1 Sol; all Medium setting
#
# With ~90 training clips one MLP depends a lot on where it starts; averaging 10 removes that luck.
# lbfgs (full-batch quasi-Newton) is the usual solver for small data sets: it converges properly.

# %% Ensemble
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier

import common

HIDDEN = (16,)               # one hidden layer of 16 neurons
N_MEMBERS = 10


class EnsembleMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=1.0, n_members=N_MEMBERS, hidden=HIDDEN):
        self.alpha, self.n_members, self.hidden = alpha, n_members, hidden

    def fit(self, X, y):
        self.members_ = [MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha, solver="lbfgs",
                                       max_iter=2000, random_state=common.SEED + s).fit(X, y)
                         for s in range(self.n_members)]
        self.classes_ = self.members_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members_], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# %% Classifier
def make(alpha=10.0):
    """10-MLP ensemble with L2 penalty alpha (seeds SEED .. SEED + 9)."""
    return EnsembleMLP(alpha=alpha)
