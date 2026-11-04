# CODE MAP of extractors/__init__.py
#   Registry   EXTRACTORS: extractor module name -> module; each module has FEATURE_SETS and load(clips)
#   AI use     Claude Opus 5.5 (claude-opus-5-5), GPT-6 Astra, GPT-6.1 Sol; all Medium setting

# %% Registry
from . import finger_patches, hand_colour

EXTRACTORS = {"hand_colour": hand_colour, "finger_patches": finger_patches}
