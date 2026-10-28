# CODE MAP of make_split.py
#   Settings       random seed, paths
#   Choose test    random 20% of the clips, stratified: 2 of the 10 clips in every person x object x label
#                  group (fixed seed), so test is balanced and every person/object/label is in both sets
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
TEST_FRAC = 0.2       # 80/20 train/test
HERE = Path(__file__).resolve().parent
LABELS = HERE / "data" / "labels.csv"
OUT = HERE / "data" / "split.csv"

if OUT.exists():
    raise SystemExit(f"{OUT} already exists. The test set is fixed; delete it only if you really mean to redo the split.")

# %% Choose test
df = pd.read_csv(LABELS)
rng = np.random.default_rng(SEED)
df["set"] = "train"
for _, group in df.groupby(["person", "object", "label"]):   # whole clips are drawn, so all frames of a clip stay together
    n_test = round(TEST_FRAC * len(group))
    df.loc[rng.choice(group.index, n_test, replace=False), "set"] = "test"

# %% Save
df[["file", "person", "object", "label", "set"]].to_csv(OUT, index=False)
print(df["set"].value_counts().to_string())
print(df.groupby(["set", "person", "object", "label"]).size().unstack("label"))
