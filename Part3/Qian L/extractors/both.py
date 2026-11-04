# CODE MAP of extractors/both.py   (tree node: do speed and shaking together beat either alone?)
#   Features        the 4 speed features (speed.py) followed by the 4 shaking features (shaking.py)
#   Feature sets    "both"; load() -> {name: X (160, 8)}, NaN if either window is unusable
#   AI use          GPT-6.1 Sol (setting: medium); GPT-6 Astra (setting: light) (original code);
#                   Claude Opus 5.5 (claude-opus-5-5), default setting (restructure)

# %% Features
from . import motion, shaking, speed

FEATURES = speed.FEATURES + shaking.FEATURES


def window_features(series):
    s, k = speed.window_features(series), shaking.window_features(series)
    return {**s, **k} if s is not None and k is not None else None


# %% Feature sets
FEATURE_SETS = ["both"]


def load(clips, names=FEATURE_SETS):
    return {"both": motion.table(clips, window_features, FEATURES)}
