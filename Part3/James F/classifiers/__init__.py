# CODE MAP of classifiers/__init__.py
#   Registry   CLASSIFIERS: name -> module with make(**params); TAKES_ALPHA: which ones have an L2 penalty alpha
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting

# %% Registry
from . import lr, mlp, mlp_ensemble

CLASSIFIERS = {"lr": lr, "mlp": mlp, "mlp_ensemble": mlp_ensemble}
TAKES_ALPHA = {"mlp", "mlp_ensemble"}
