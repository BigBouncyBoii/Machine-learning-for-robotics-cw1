# CODE MAP of verify.py
#   Inputs         fixed protocol and training caches only
#   Validation     recompute every out-of-fold prediction and model reload
#   Failure checks fingerprint rejection and majority fallback
#   Diagnostics    eight fixed training representatives, no test images
#   Freeze checks  temporary freeze/refit and repeated-test guard
#   Evidence       save verification results
#   AI use         GPT-6 (Codex); exact version/settings unavailable
# Run before formal test: .venv\Scripts\python.exe "Qian L/verify.py"
# %% Inputs
from pathlib import Path
import sys, io, tempfile, shutil
import numpy as np
import pandas as pd
import joblib
import cv2
from sklearn.model_selection import LeaveOneGroupOut

root = Path(__file__).resolve().parent
scratch = root / 'work'
scratch.mkdir(exist_ok=True)
sys.path.insert(0, str(root))
import protocol
from features import FEATURES, load_clip, ROI, CONFIG, shaking_features

split = protocol.load_protocol()
if (root/'results/formal_test_started.json').exists():
    raise RuntimeError('This verifier is for pre-test development only')
meta = split.loc[split['set'] == 'train'].reset_index(drop=True)
rows, hits = [], []
for record in meta.itertuples():
    row, hit = load_clip(Path(record.video_path), protocol.CACHE)
    rows.append(row)
    hits.append(hit)
assert all(hits), 'Expected all train clips cached'
saved = pd.read_csv(root / 'results/validation_predictions.csv')
y = (meta.label == 'firm').to_numpy(dtype=int)
# %% Validation
n_verified = 0
for group, names in FEATURES.items():
    window = group
    X = protocol.matrix(rows, window, names)
    for C in [0.1, 1.0]:
        method = f'{group}_C{C:g}'
        expected = saved.loc[saved.method == method].set_index('file')
        assert len(expected) == 120
        for train, val in LeaveOneGroupOut().split(X, y, groups=meta.person):
            assert len(train) == 90 and len(val) == 30
            assert not set(meta.iloc[train].person) & set(meta.iloc[val].person)
            model, fallback = protocol.fit_model(X[train], y[train], C)
            available = np.isfinite(X[train]).all(axis=1)
            if model is not None:
                np.testing.assert_allclose(model[0].mean_, X[train][available].mean(axis=0))
            buffer = io.BytesIO()
            joblib.dump((model, fallback), buffer)
            buffer.seek(0)
            restored = joblib.load(buffer)
            pred, probability, used = protocol.predict(*restored, X[val])
            direct = protocol.predict(model, fallback, X[val])
            for a, b in zip((pred, probability, used), direct):
                np.testing.assert_array_equal(a, b)
            for j, index in enumerate(val):
                actual = expected.loc[meta.iloc[index].file]
                assert actual.prediction == ('firm' if pred[j] else 'light')
                np.testing.assert_allclose(actual.p_firm, probability[j], rtol=1e-12, atol=1e-12)
                assert bool(actual.model_used) == bool(used[j])
                n_verified += 1
        print('Verified', method, flush=True)

# %% Failure checks
# Missing features retain their place and use the fold-training majority.
Xm = np.array([[np.nan], [np.nan]])
for yy, answer in [(np.array([0, 1]), 0), (np.array([1, 1]), 1)]:
    model, fallback = protocol.fit_model(Xm, yy, 0.1)
    pred, _, used = protocol.predict(model, fallback, Xm)
    assert list(pred) == [answer, answer] and not used.any()

# Validate fail-closed behavior without touching the fixed source files.
original_root = protocol.ROOT
with tempfile.TemporaryDirectory(dir=scratch) as directory:
    fake = Path(directory)
    (fake / 'data').mkdir()
    shutil.copy(original_root / 'data/labels.csv', fake / 'data/labels.csv')
    damaged = pd.read_csv(original_root / 'data/split.csv')
    index = damaged.index[damaged['set'] == 'test'][0]
    damaged.loc[index, 'file'] += '.wrong'
    damaged.to_csv(fake / 'data/split.csv', index=False)
    protocol.ROOT = fake
    try:
        protocol.load_protocol()
    except RuntimeError as error:
        assert 'fingerprint' in str(error)
    else:
        raise AssertionError('Incorrect fingerprint accepted')
    finally:
        protocol.ROOT = original_root

# Same-speed rigid translation must not be called shaking; oscillation must be detectable.
fps = 30.0
t = np.arange(60) / fps
constant = np.column_stack([np.full(60, .03), np.zeros(60)])
oscillating = np.column_stack([.03*np.sin(2*np.pi*3*t), np.zeros(60)])
calm = shaking_features(constant, fps)
shaky = shaking_features(oscillating, fps)
assert calm['position_jitter_rms'] < 1e-12 and calm['velocity_jitter_rms'] < 1e-12
assert shaky['position_jitter_rms'] > calm['position_jitter_rms'] + 1e-5
assert shaky['velocity_jitter_rms'] > .001 and shaky['reversal_rate'] > 1

# %% Diagnostics
# Only fixed train representatives are rendered; no test video is decoded here.
tile_w, tile_h = 160, 310
canvas = np.full((8 * tile_h, 6 * tile_w, 3), 25, np.uint8)
representatives = meta.groupby(['person', 'object'], sort=True).head(1)
for rowindex, record in enumerate(representatives.itertuples()):
    cap = cv2.VideoCapture(record.video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    target = [round(f*(len(frames)-1)) for f in [.1, .275, .45, .45, .65, .85]]
    for col, index in enumerate(target):
        frame = frames[index].copy()
        h, w = frame.shape[:2]
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin = ((ycc[:, :, 0] > CONFIG['skin_y_min']) &
                (ycc[:, :, 1] >= CONFIG['skin_cr'][0]) & (ycc[:, :, 1] <= CONFIG['skin_cr'][1]) &
                (ycc[:, :, 2] >= CONFIG['skin_cb'][0]) & (ycc[:, :, 2] <= CONFIG['skin_cb'][1]))
        tint = frame.copy()
        tint[skin] = (0, 255, 0)
        frame = cv2.addWeighted(frame, 0.7, tint, 0.3, 0)
        x0, y0, x1, y1 = [round(v * n) for v, n in zip(ROI, (w, h, w, h))]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
        scale = min(tile_w/w, (tile_h-40)/h)
        frame = cv2.resize(frame, (round(w*scale), round(h*scale)))
        x = col*tile_w+(tile_w-frame.shape[1])//2
        top = rowindex*tile_h+30
        canvas[top:top+frame.shape[0], x:x+frame.shape[1]] = frame
        cv2.putText(canvas, f'{record.person} {record.object} f{index}', (col*tile_w+2,rowindex*tile_h+16), cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
        cv2.putText(canvas, 'SPEED' if col<3 else 'SHAKING', (col*tile_w+2,rowindex*tile_h+30),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,240,240),1)
cv2.imwrite(str(root/'results/train_representatives.jpg'), canvas)
# %% Freeze checks
import FinalMethod as final
import contextlib
with tempfile.TemporaryDirectory(dir=scratch) as directory:
    temporary = Path(directory)
    shutil.copy(root/'results/proposed_method.json', temporary/'proposed_method.json')
    shutil.copy(root/'results/validation_predictions.csv', temporary/'validation_predictions.csv')
    final.OUT = temporary
    final.FROZEN = temporary/'frozen_method.json'
    final.MODEL = temporary/'frozen_model.joblib'
    final.STARTED = temporary/'formal_test_started.json'
    with contextlib.redirect_stdout(io.StringIO()):
        final.freeze(split)
    assert final.FROZEN.exists() and final.MODEL.exists()
    final.STARTED.write_text('{}')
    try:
        final.run_test(split)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Repeated formal test was not blocked')
# %% Evidence
assert not (root/'results/formal_test_started.json').exists()
assert len(list(protocol.CACHE.glob('*.json'))) == 120
protocol.save_json(root/'results/verification.json', dict(train_cache_hits=sum(hits),
    predictions_recomputed_and_serialization_checked=n_verified, folds_checked=24,
    fold_scaler_means_match_training_only=True, bad_fingerprint_rejected=True,
    missing_feature_fallback_checked=True,
    constant_translation_vs_oscillation_checked=True, test_cache_count=0, formal_test_run=False,
    freeze_refit_and_model_reload_checked_in_temporary_directory=True,
    existing_formal_test_marker_blocks_test_before_extraction=True,
    split_sha256=protocol.digest(original_root/'data/split.csv')))
print('Verified', n_verified, 'per-video predictions across 24 folds; all cache hits; no test execution.')
