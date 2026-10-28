# CODE MAP of make_split.py
#   Settings       random seed, paths
#   Choose test    random 25% of the clips (brief: "about 25%", part of every member's recordings), fixed seed.
#                  Each person x object x label group has 10 clips, so it gives 3 or 2 of them to test.
#                  Which groups give 3 is arranged so every person, object and label has exactly 25% in test:
#                  per person 10 (5 light, 5 firm), per object 20, per label 20; 40 test / 120 train
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

SEED = 42
TEST_FRAC = 0.25      # 75/25 train/test
HERE = Path(__file__).resolve().parent
LABELS = HERE / "data" / "labels.csv"
OUT = HERE / "data" / "split.csv"

if OUT.exists():
    raise SystemExit(f"{OUT} already exists. The test set is fixed; delete it only if you really mean to redo the split.")

# %% Choose test
df = pd.read_csv(LABELS)
rng = np.random.default_rng(SEED)
people = sorted(df["person"].unique())
objects = sorted(df["object"].unique())          # ["hard", "soft"]
labels = sorted(df["label"].unique())            # ["firm", "light"]

# 25% of 10 clips is 2.5, so half the groups give 3 and half give 2. Half the people (drawn at random)
# give 3 hard-firm + 3 soft-light, the other half 3 hard-light + 3 soft-firm; all other groups give 2.
flip = set(rng.choice(people, len(people) // 2, replace=False))
df["set"] = "train"
for (p, o, l), group in df.groupby(["person", "object", "label"]):   # whole clips are drawn, so frames of a clip stay together
    extra = (objects.index(o) == labels.index(l)) != (p in flip)      # True for the groups that give the 3rd clip
    n_test = int(TEST_FRAC * len(group)) + extra
    df.loc[rng.choice(group.index, n_test, replace=False), "set"] = "test"

# %% Save
df[["file", "person", "object", "label", "set"]].to_csv(OUT, index=False)
print(df["set"].value_counts().to_string())
print(df.groupby(["set", "person", "object", "label"]).size().unstack("label"))
test = df[df["set"] == "test"]
print("\ntest clips per person:", test["person"].value_counts().sort_index().to_dict())
print("test clips per object:", test["object"].value_counts().sort_index().to_dict())
print("test clips per label: ", test["label"].value_counts().sort_index().to_dict())
