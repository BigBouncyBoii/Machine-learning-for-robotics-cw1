# CODE MAP of make_split.py
#   Settings       random seed, paths
#   Choose test    hold out ALL clips of one person (25% of the data), person drawn with a fixed seed
#   Save           data/split.csv: every clip from labels.csv plus its set ("train" or "test")
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Run ONCE, before exploring, then commit data/split.csv and never change it.
# The videos are not copied or moved; split.csv is the only record of which clips are in the test set.
# Every later script (explore.py, FinalMethod.py) reads split.csv. Nothing is ever fitted on "test" rows.
#
# HOW TO RUN (from the Part3 folder)
#   python make_split.py

# %% Settings
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 144
HERE = Path(__file__).resolve().parent
LABELS = HERE / "data" / "labels.csv"
OUT = HERE / "data" / "split.csv"

if OUT.exists():
    raise SystemExit(f"{OUT} already exists. The test set is fixed; delete it only if you really mean to redo the split.")

# %% Choose test
df = pd.read_csv(LABELS)
people = sorted(df["person"].unique())
test_person = np.random.default_rng(SEED).choice(people)   # random, so nobody picks the "easy" person
df["set"] = np.where(df["person"] == test_person, "test", "train")

# %% Save
df[["file", "person", "object", "label", "set"]].to_csv(OUT, index=False)
print(f"test person: {test_person}")
print(df.groupby(["set", "person", "object", "label"]).size().unstack("label"))
