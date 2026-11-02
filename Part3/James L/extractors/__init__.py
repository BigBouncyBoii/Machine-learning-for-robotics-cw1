# CODE MAP of extractors/__init__.py
#   Registry   EXTRACTORS: extractor name -> module; each module has FEATURE_SETS and load(clips, names)
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting
#
# torch / transformers are imported only when a cache has to be built, so validation runs without them.

# %% Registry
from . import dinov2, dinov2_frames, dinov2_phase, dinov2_window, videomae

EXTRACTORS = {
    "dinov2": dinov2,                 # baseline: all 16 frames
    "dinov2_frames": dinov2_frames,   # node: how many frames
    "dinov2_phase": dinov2_phase,     # node: which phase of the grip
    "dinov2_window": dinov2_window,   # node: how many middle frames
    "videomae": videomae,             # node: a video model
}
