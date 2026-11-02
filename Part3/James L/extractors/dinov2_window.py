# CODE MAP of extractors/dinov2_window.py   (tree node: how many frames of the middle of the grip?)
#   Frames          frames_used(): WHICH of the 16 frames each window averages, written out explicitly
#   Feature sets    POOL: mean CLS over those frames; load() -> {name: X (160, 384)}
#   AI use          Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Follows the phase node: the middle of the grip carries the signal, the start does not.
# Question: within the middle, how wide a window? 1 frame may be noisy (blur, a bad moment), a wide window
# may pull in the reach / release frames again. Windows are centred on the middle of the 16 (between 7 and 8).
# Uses the DINOv2 encoder and cache of dinov2.py (nothing new is trained or encoded).

# %% Frames
from . import dinov2
from .dinov2 import MODEL_ID, load_encoder   # noqa: F401  (for predict_clip.py)

N_FRAMES = 16
# name -> indices of the 16 cached frames that are averaged
FRAMES = {
    "dinov2_mid1":  [8],                                  # = dinov2_1frame
    "dinov2_mid2":  [7, 8],
    "dinov2_mid4":  [6, 7, 8, 9],
    "dinov2_mid6":  [5, 6, 7, 8, 9, 10],                  # = dinov2_middle (the final method)
    "dinov2_mid8":  [4, 5, 6, 7, 8, 9, 10, 11],
    "dinov2_mid10": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
}


def frames_used(name):
    """Which of the 16 evenly spaced frames this window averages."""
    return FRAMES[name]


# %% Feature sets
POOL = {name: (N_FRAMES, dinov2.mean_cls(idx)) for name, idx in FRAMES.items()}
FEATURE_SETS = list(POOL)


def load(clips, names=FEATURE_SETS):
    """{feature set name: X (n_clips, 384)}, rows in clips order."""
    return dinov2.pooled(clips, names, POOL)
