# CODE MAP of test_D.py   (colour method, case D: person D is the test set)
#   Settings   which person is held out
#   Run        colour_method.run(): validation on people A, B, C only, choose input + model,
#              train on A, B, C, test once on person D, save one image per test grip in figures/test_D/
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting
#
# HOW TO RUN (from the repo root)
#   .venv\Scripts\python "James F\colour_method\test_D.py"

# %% Settings
TEST_PERSON = "D"           # all 40 clips of person D are the test set; never used for training

# %% Run
from colour_method import run

run(TEST_PERSON)
