# CODE MAP of extractors/__init__.py
#   Registry   EXTRACTORS: extractor name -> module; each module has FEATURE_SETS and load(clips, names)
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting
#
# torch / transformers are imported only when a cache has to be built, so validation runs without them.

# %% Registry
from . import dinov2, videomae

EXTRACTORS = {"dinov2": dinov2, "videomae": videomae}
