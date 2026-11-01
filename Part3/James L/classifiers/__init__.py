# CODE MAP of classifiers/__init__.py
#   Registry   CLASSIFIERS: name -> module with make(seed) and SEEDS (validation is averaged over SEEDS)
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting

# %% Registry
from . import lr, mlp, mlp_ensemble

CLASSIFIERS = {"lr": lr, "mlp": mlp, "mlp_ensemble": mlp_ensemble}
