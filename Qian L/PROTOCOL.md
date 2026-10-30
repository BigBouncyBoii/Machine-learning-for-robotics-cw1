# Shared Part 3 protocol — Qian L speed / shaking / both

This document records the user's group protocol. It overrides all older 95/25/40, D-only-test, split_manifest-driven and redraw-split workflows. It does not change the group's split or other members' implementations.

## Data and immutable split

- Read `Part3/data/labels.csv` (file, person, object, label; an extra original-name column may exist).
- 160 clips: A/B/C/D × hard/soft × light/firm × 10.
- Original videos are expected under `Part3/data/videos/`; same-name mp4 copies under `Part3/data/videos_480p/`. This implementation uses the mp4 copies and does not upload videos. Original MOVs, if needed, remain in the shared local data directory.
- Always read `Part3/data/split.csv`: 120 train, 40 test. Never redraw, reshuffle, re-seed, edit it or run `make_split.py`.
- Test: 10 clips/person, 5 light and 5 firm/person, 20/object, 20/label.
- Every executable entry point prints and checks:

```python
test_files = split.loc[split['set'] == 'test', 'file'].tolist()
fingerprint = hashlib.md5('\n'.join(sorted(test_files)).encode()).hexdigest()[:8]
if fingerprint != '3696c80d':
    raise RuntimeError('Wrong split')
```

Use original `file` values, not mp4 paths or replacement IDs. Imported helper modules are not standalone entry points.

## Three current trials

1. Speed: early visible-skin-colour local/inward movement speed, a grip-closing proxy; not measured fingertip speed.
2. Shaking: later signed regional motion variability, acceleration and reversals; not mean speed renamed as shaking.
3. Both: concatenate the first two feature sets.

The windows/formulas/quality thresholds are fixed before this run's validation, label-free and independently calculated per clip. No automatic physical contact/holding claim is made. See README for formulas, colour-mask false positives, approximate timing and historical limitations.

## Train-only validation and selection

- LeaveOneGroupOut(groups=person) only on 120 train clips: four folds of 90 training and 30 validation clips.
- All frames of one clip remain together; check original source identities for duplicates before evaluation.
- Fit every learned step inside the fold's training partition, using sklearn Pipeline. No cross-video scaler, PCA, imputer, selection or normalisation sees current validation or test clips.
- Fixed-form single-video detrending and size/time conversion use only that clip and are not cross-video fitted normalisation.
- Report each fold's accuracy, mean validation accuracy, per-object accuracy, counts, failures and model coverage. Guessing baseline is 50%.
- Use mean all-clip validation accuracy to select C within each trial and then select one final method; never training/test accuracy. Current logistic-regression C choices: 0.1 and 1.0, random_state=0, one seed. State tie-breaking.
- Preserve every clip in the denominator. Use the predefined fold-training all-label majority fallback for failed features (ties light); report fallback counts and model coverage separately. Apply the same policy at final refit/test.

## Test protection

- Only the final script uses test clips in this branch; current development and visual checks use train only.
- Never fit scalers, PCA, imputers, thresholds, features, classifiers, hyperparameters or early stopping on test; never inspect test plots to choose the method.
- Label-free independent extraction could be cached on all clips under the group protocol, but this branch has deliberately processed only the 120 train clips so far.
- Freeze the selected features/model/hyperparameters before the formal run. No final test has currently been run.
- The real freeze and test entries refuse to overwrite a prior freeze/test marker. Verification uses temporary train-only artifacts, not the test set.

## Final evaluation

- Refit the chosen full Pipeline on all 120 train clips under the fixed failure policy.
- Predict the fixed 40 test clips once, in `Qian L/FinalMethod.py`.
- Report n correct/40 and accuracy, 95% binomial CI with method stated (Clopper–Pearson exact), one-sided binomial p-value versus 50%, per-person and per-object accuracy/coverage, confusion matrix and fingerprint.
- Keep seeds fixed; state the number of seeds if averaging (current method: one).
- Do not compare methods or choose ablations after observing the test number. Disclose any post-test method change. Retain an interrupted-test marker and disclose failure rather than silently restarting.

## Reproducibility and historical disclosure

- Label-free per-clip cache includes source path/content, extraction config and version/code hashes; save every clip and support resume.
- Headers state purpose, running instructions and known AI provenance, with CODE MAP and matching `# %%` sections. Do not invent unknown model settings.
- Current test exposure: 27 old train, 3 old validation, 10 old D test, based on original-name identity matching. The first 30 have historical artifacts. Human-viewing history is incomplete. Do not claim a pristine unseen test.
- Old preprocessing choices may also have seen current LOPO people/clips; disclose this limitation despite correct current fold-local fitting.
- A separate whole-person hold-out experiment may be reported as an extra experiment. The headline test must still come from the fixed split and must not be replaced by that experiment.
- If a future request conflicts with these rules, explain the conflict before implementing it.
