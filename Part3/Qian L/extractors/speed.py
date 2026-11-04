# CODE MAP of extractors/speed.py   (tree node: does a firm grip close the fingers faster?)
#   Window          WINDOW: the early part of the clip (10-45 %), when the fingers close on the object
#   Features        FEATURES: 4 speed numbers, each the mean over the valid frame pairs of the window
#   Feature sets    "speed"; load() -> {name: X (160, 4)}, NaN where the window has too little visible skin
#   AI use          GPT-6.1 Sol (setting: medium); GPT-6 Astra (setting: light) (original code);
#                   Claude Opus 5.5 (claude-opus-5-5), default setting (restructure)
#
# local speed  = optical flow of skin pixels minus the hand's median flow (fingers moving relative to the hand)
# inward speed = the part of that local motion pointing towards the centre of the skin region (closing)
# all in ROI diagonals per second, so they do not depend on the frame size

# %% Window
import numpy as np

from . import motion

WINDOW = (0.10, 0.45)
FEATURES = ["local_speed_mean", "local_speed_p90", "inward_speed_mean", "inward_speed_p90"]


# %% Features
def window_features(series):
    """One clip's motion series -> {feature: value}, or None if the window has too little visible skin."""
    selected = motion.phase(series["samples"], series["frame_count"], WINDOW)
    if not motion.enough_valid(selected):
        return None
    valid = [s for s in selected if s["valid"]]
    return {name: float(np.mean([s[name] for s in valid])) for name in FEATURES}


# %% Feature sets
FEATURE_SETS = ["speed"]


def load(clips, names=FEATURE_SETS):
    return {"speed": motion.table(clips, window_features, FEATURES)}
