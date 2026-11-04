# CODE MAP of classifiers/__init__.py
#   Registry   CLASSIFIERS: name -> module with make(seed, C) and SEEDS (validation is averaged over SEEDS)
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting; GPT-6.1 Sol (setting: medium); GPT-6 Astra (setting: light)

# %% Registry
from . import lr

CLASSIFIERS = {"lr": lr}
