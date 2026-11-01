# CODE MAP of features.py
#   Settings       three trials and fixed per-clip image/motion formulas
#   Decode         every decoded frame, aspect-preserving skin and background flow
#   Features       early nonrigid closing-speed proxy, later temporal shaking proxy
#   Cache          label-free per-clip content/config/code hashes and atomic writes
#   AI use         GPT-6 (Codex); exact version/settings unavailable
# Imported by explore.py / FinalMethod.py. Run those entry points, not this module.
# %% Settings
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np

PIPELINE_VERSION = 'speed-shaking-v2'
LONG_EDGE = 320
ROI = (0.20, 0.30, 0.80, 0.78)
BACKGROUND = (0.05, 0.08, 0.95, 0.23)
SPEED_WINDOW = (0.10, 0.45)
SHAKING_WINDOW = (0.45, 0.85)
SPEED = ('local_speed_mean', 'local_speed_p90', 'inward_speed_mean', 'inward_speed_p90')
SHAKING = ('position_jitter_rms', 'velocity_jitter_rms', 'acceleration_rms', 'reversal_rate')
FEATURES = {'speed': SPEED, 'shaking': SHAKING, 'both': SPEED + SHAKING}
CONFIG = dict(version=PIPELINE_VERSION, long_edge=LONG_EDGE, roi=list(ROI), background=list(BACKGROUND),
    speed_window=list(SPEED_WINDOW), shaking_window=list(SHAKING_WINDOW), sample_step=1,
    skin_y_min=35, skin_cr=[133, 180], skin_cb=[70, 135], min_skin_pixels=30,
    min_phase_pairs=8, min_phase_fraction=0.7, trend_seconds=0.33, reversal_min_speed=0.002)


def box(shape, bounds):
    h, w = shape[:2]
    x0, y0, x1, y1 = bounds
    return round(x0*w), round(y0*h), round(x1*w), round(y1*h)


# %% Decode
def decode_motion(path):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise OSError('Cannot decode video or invalid FPS')
    samples = []
    previous = previous_skin = None
    n_frames = 0
    shape = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = LONG_EDGE / max(h, w)
            image = cv2.resize(frame, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            skin = ((ycc[:, :, 0] > CONFIG['skin_y_min']) &
                    (ycc[:, :, 1] >= CONFIG['skin_cr'][0]) & (ycc[:, :, 1] <= CONFIG['skin_cr'][1]) &
                    (ycc[:, :, 2] >= CONFIG['skin_cb'][0]) & (ycc[:, :, 2] <= CONFIG['skin_cb'][1]))
            shape = list(gray.shape)
            if previous is not None:
                x0, y0, x1, y1 = box(gray.shape, ROI)
                bx0, by0, bx1, by1 = box(gray.shape, BACKGROUND)
                flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
                mask = previous_skin[y0:y1, x0:x1]
                sample = dict(frame=n_frames, skin_pixels=int(mask.sum()), valid=False)
                if mask.sum() >= CONFIG['min_skin_pixels']:
                    vectors = flow[y0:y1, x0:x1][mask].astype(float)
                    bg = np.median(flow[by0:by1, bx0:bx1].reshape(-1, 2), axis=0)
                    hand = np.median(vectors, axis=0)
                    # Local motion relative to whole-hand translation: a curl-speed proxy.
                    local = vectors - hand
                    diagonal = float(np.hypot(y1-y0, x1-x0))
                    local_speed = np.linalg.norm(local, axis=1) * fps / diagonal
                    xy = np.argwhere(mask)[:, ::-1].astype(float)
                    inward_direction = xy.mean(axis=0) - xy
                    inward_direction /= np.maximum(np.linalg.norm(inward_direction, axis=1, keepdims=True), 1.0)
                    inward = np.maximum(np.sum(local * inward_direction, axis=1), 0) * fps / diagonal
                    sample.update(valid=True, local_speed_mean=float(local_speed.mean()),
                        local_speed_p90=float(np.percentile(local_speed, 90)),
                        inward_speed_mean=float(inward.mean()), inward_speed_p90=float(np.percentile(inward, 90)),
                        hand_velocity=((hand - bg) * fps / diagonal).tolist())
                samples.append(sample)
            previous, previous_skin = gray, skin
            n_frames += 1
    finally:
        cap.release()
    if n_frames < 12:
        raise ValueError('Too few decoded frames')
    return samples, fps, n_frames, shape


# %% Features
def phase(samples, frame_count, bounds):
    # BOTH endpoints of a pair must be in the same fixed phase.
    lo, hi = bounds
    return [s for s in samples if lo <= (s['frame']-1)/(frame_count-1) and s['frame']/(frame_count-1) <= hi]


def shaking_features(velocity, fps):
    velocity = np.asarray(velocity, dtype=float)
    width = min(int(round(CONFIG['trend_seconds'] * fps)) | 1, (len(velocity)-1) | 1)
    width = max(3, width)
    pad = width // 2
    smooth = np.column_stack([np.convolve(np.pad(velocity[:, j], pad, mode='reflect'), np.ones(width)/width, mode='valid') for j in range(2)])
    residual = velocity - smooth
    position = np.cumsum(velocity, axis=0) / fps
    # Per-video fixed-form linear detrending, not a fitted cross-video normaliser.
    t = np.linspace(-1, 1, len(position))
    detrended = position - position.mean(axis=0) - t[:, None] * (t @ position / (t @ t))[None, :]
    acceleration = np.diff(velocity, axis=0) * fps
    magnitude = np.linalg.norm(residual, axis=1)
    reversals = ((np.sum(residual[1:] * residual[:-1], axis=1) < 0) &
                 (magnitude[1:] > CONFIG['reversal_min_speed']) & (magnitude[:-1] > CONFIG['reversal_min_speed']))
    return dict(position_jitter_rms=float(np.sqrt(np.mean(np.sum(detrended**2, axis=1)))),
        velocity_jitter_rms=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        acceleration_rms=float(np.sqrt(np.mean(np.sum(acceleration**2, axis=1)))),
        reversal_rate=float(reversals.sum() / ((len(velocity)-1)/fps)))


def summarize_phase(selected, names, fps, shaking=False):
    valid = [s for s in selected if s['valid']]
    fraction = len(valid) / len(selected) if selected else 0
    if len(valid) < CONFIG['min_phase_pairs'] or fraction < CONFIG['min_phase_fraction']:
        return None
    if shaking:
        # Only fill short holes for a uniformly sampled series; never bridge long failures.
        indices = np.flatnonzero([s['valid'] for s in selected])
        run = 0
        for s in selected[indices[0]:indices[-1]+1]:
            run = 0 if s['valid'] else run + 1
            if run > 2:
                return None
        times = np.arange(indices[0], indices[-1]+1)
        velocity = np.asarray([s['hand_velocity'] for s in valid])
        uniform = np.column_stack([np.interp(times, indices, velocity[:, j]) for j in range(2)])
        return shaking_features(uniform, fps)
    return {name: float(np.mean([s[name] for s in valid])) for name in names}


def extract(path):
    samples, fps, n_frames, shape = decode_motion(path)
    early = phase(samples, n_frames, SPEED_WINDOW)
    late = phase(samples, n_frames, SHAKING_WINDOW)
    speed = summarize_phase(early, SPEED, fps)
    shaking = summarize_phase(late, SHAKING, fps, shaking=True)
    both = {**speed, **shaking} if speed is not None and shaking is not None else None
    windows = dict(speed=speed, shaking=shaking, both=both)
    return dict(status='success' if both is not None else 'partial_or_failed', windows=windows,
        fps=fps, frame_count=n_frames, resized_shape=shape,
        phase_pairs=dict(speed=len(early), shaking=len(late)),
        phase_valid_pairs=dict(speed=sum(s['valid'] for s in early), shaking=sum(s['valid'] for s in late)),
        failure_reason=[name + ': insufficient visible-skin motion pairs or long gaps' for name, value in windows.items() if value is None],
        # Scalar/vector cues allow temporal diagnostics without decoding test images.
        motion_series=samples)


# %% Cache
def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_clip(path, cache_dir):
    path = path.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (hashlib.sha256(str(path).encode()).hexdigest() + '.json')
    signature = dict(source=str(path), source_sha256=sha256(path) if path.is_file() else 'missing',
        config=CONFIG, extraction_code_sha256=sha256(Path(__file__)))
    if cache.exists():
        try:
            saved = json.loads(cache.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            saved = {}
        if saved.get('signature') == signature:
            return saved['result'], True
    begin = time.perf_counter()
    try:
        result = extract(path)
    except (OSError, ValueError, cv2.error) as error:
        result = dict(status='failed', windows={name: None for name in FEATURES}, failure_reason=[str(error)])
    result['seconds'] = round(time.perf_counter()-begin, 3)
    temporary = cache.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(signature=signature, result=result), allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(cache)
    return result, False


if __name__ == '__main__':
    from protocol import load_protocol
    load_protocol()
