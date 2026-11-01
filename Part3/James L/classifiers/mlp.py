# CODE MAP of classifiers/mlp.py
#   Classifier   make(seed): one MLP, 1 hidden layer of 64 ReLU units
#   AI use       Claude Opus 5.5 (claude-opus-5-5), default setting
#
# The result depends on the random starting weights, so main.py validates 5 seeds and reports the MEAN of
# their accuracies (each seed is one model). This is the "MLP" column of the tree.

# %% Classifier
from sklearn.neural_network import MLPClassifier

SEEDS = list(range(5))


def make(seed=0):
    """One MLP (64 hidden units, L2 penalty alpha = 0.01) with starting weights from seed."""
    return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=seed)
