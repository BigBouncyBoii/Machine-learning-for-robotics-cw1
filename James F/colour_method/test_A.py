# CODE MAP of test_A.py   (colour method, case A: person A is the test set)
#   Settings   which person is held out
#   Run        colour_method.run(): validation on people B, C, D only, choose input + model,
#              train on B, C, D, test once on person A, save one image per test grip in figures/test_A/
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting
#
# HOW TO RUN (from the repo root)
#   .venv\Scripts\python "James F\colour_method\test_A.py"

# %% Settings
TEST_PERSON = "A"           # all 40 clips of person A are the test set; never used for training

# %% Run
from colour_method import run

run(TEST_PERSON)
