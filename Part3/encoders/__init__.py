# CODE MAP of encoders/__init__.py
#   get            import an encoder module by name ("dinov2" or "videomae")
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Each encoder module has:
#   MODEL_ID   the Hugging Face model it downloads
#   FILE       name of its cached feature file, features/<FILE>.npz
#   load()     loads the frozen model and returns encode(frames) -> dict of numpy arrays for one clip
# Modules are imported only when needed, so explore.py runs without torch.

import importlib

ENCODERS = ["dinov2", "videomae"]


def get(name):
    return importlib.import_module(f"encoders.{name}")
