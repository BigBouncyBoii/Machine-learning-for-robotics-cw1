# CODE MAP of extractors/__init__.py
#   Registry   EXTRACTORS: extractor name -> module; each module has FEATURE_SETS and load(clips, names)
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting

# %% Registry
from . import joint_angles

EXTRACTORS = {"joint_angles": joint_angles}
