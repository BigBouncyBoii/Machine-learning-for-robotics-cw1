# CODE MAP of extractors/dinov2_phase.py   (tree node: which phase of the grip carries the signal?)
#   Frames          frames_used(): WHICH of the 16 frames each phase uses, written out explicitly
#   Feature sets    POOL: mean CLS over those frames; load() -> {name: X (160, 384)}
#   AI use          Claude Opus 5.5 (claude-opus-5-5), default setting
#
# The 16 frames are evenly spaced over the clip: the hand reaches in, squeezes and holds, then releases.
# "start" is a control: before the grip nothing should give the label away, so it should score ~50 %.
# Uses the DINOv2 encoder and cache of dinov2.py (nothing new is trained or encoded).
# (was common.FEATURES)

# %% Frames
from . import dinov2
from .dinov2 import MODEL_ID, load_encoder   # noqa: F401  (for predict_clip.py)

N_FRAMES = 16
# name -> indices of the 16 cached frames that are averaged
FRAMES = {
    "dinov2_start":  [0, 1, 2],              # first 3 frames: hand reaching in, not gripping yet
    "dinov2_middle": [5, 6, 7, 8, 9, 10],    # middle 6 frames: squeeze / hold
}


def frames_used(name):
    """Which of the 16 evenly spaced frames this phase averages."""
    return FRAMES[name]


# %% Feature sets
POOL = {name: (N_FRAMES, dinov2.mean_cls(idx)) for name, idx in FRAMES.items()}
FEATURE_SETS = list(POOL)


def load(clips, names=FEATURE_SETS):
    """{feature set name: X (n_clips, 384)}, rows in clips order."""
    return dinov2.pooled(clips, names, POOL)
