# CODE MAP of classifiers/mlp.py
#   Classifier   make(): one small MLP (16 hidden units, Adam), the first neural-network node
#   AI use       Claude Opus 5.5 (claude-opus-5-5), default setting
#
# With ~90 training clips per fold one MLP depends a lot on its random starting weights;
# mlp_ensemble.py was the next node because of that.

# %% Classifier
from sklearn.neural_network import MLPClassifier

import common

HIDDEN = (16,)               # one hidden layer of 16 neurons


def make(alpha=1.0):
    """One MLP with L2 penalty alpha (strong: few training clips, so overfitting is the risk)."""
    return MLPClassifier(hidden_layer_sizes=HIDDEN, alpha=alpha, max_iter=3000, random_state=common.SEED)
