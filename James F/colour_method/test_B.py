# CODE MAP of test_B.py   (colour method, case B: person B is the test set)
#   Settings   which person is held out
#   Run        colour_method.run(): validation on people A, C, D only, choose input + model,
#              train on A, C, D, test once on person B, save one image per test grip in figures/test_B/
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting
#
# HOW TO RUN (from the repo root)
#   .venv\Scripts\python "James F\colour_method\test_B.py"

# %% Settings
TEST_PERSON = "B"           # all 40 clips of person B are the test set; never used for training

# %% Run
from colour_method import run

run(TEST_PERSON)
