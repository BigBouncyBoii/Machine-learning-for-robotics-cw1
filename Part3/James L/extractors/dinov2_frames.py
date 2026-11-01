# CODE MAP of extractors/dinov2_frames.py   (tree node: how many frames does DINOv2 need?)
#   Frames          frames_used(): WHICH frames each version averages, written out explicitly
#   Feature sets    POOL: mean CLS over those frames; load() -> {name: X (160, 384)}
#   AI use          Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Question: does averaging more frames help, or does it dilute the grip with the rest / release frames?
# Uses the DINOv2 encoder and caches of dinov2.py (nothing new is trained or encoded, except the 8-frame cache).
# (was common.FEATURES and frames_node.py)

# %% Frames
from . import dinov2
from .dinov2 import MODEL_ID, load_encoder   # noqa: F401  (for predict_clip.py)

# name -> (frames per clip in the cache, indices of the frames that are averaged)
FRAMES = {
    "dinov2_1frame":      (16, [8]),                          # the middle frame only
    "dinov2_4frames":     (16, [0, 4, 8, 12]),                # every 4th of the 16
    "dinov2_8frames":     (16, [0, 2, 4, 6, 8, 10, 12, 14]),  # every 2nd of the 16
    "dinov2_resampled_8": (8, [0, 1, 2, 3, 4, 5, 6, 7]),      # 8 frames re-sampled from the video (all of them):
}                                                             # spans the whole clip, compare with dinov2_cls_mean (16)


def frames_used(name):
    """How many frames each clip was sampled into, and which of them this version averages."""
    return FRAMES[name]


# %% Feature sets
POOL = {name: (n, dinov2.mean_cls(idx)) for name, (n, idx) in FRAMES.items()}
FEATURE_SETS = list(POOL)


def load(clips, names=FEATURE_SETS):
    """{feature set name: X (n_clips, 384)}, rows in clips order."""
    return dinov2.pooled(clips, names, POOL)
