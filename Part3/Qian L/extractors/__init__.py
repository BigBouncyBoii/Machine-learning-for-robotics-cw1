# CODE MAP of extractors/__init__.py
#   Registry   EXTRACTORS: extractor name -> module; each module has FEATURE_SETS and load(clips, names)
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting; GPT-6.1 Sol (setting: medium); GPT-6 Astra (setting: light); GitHub Copilot (VS Code; extension version, model/version and setting unavailable)

# %% Registry
from . import both, shaking, speed

EXTRACTORS = {
    "speed": speed,       # node: finger-closing speed, early window
    "shaking": shaking,   # node: hand tremor, later window
    "both": both,         # node: the 8 features together
}
