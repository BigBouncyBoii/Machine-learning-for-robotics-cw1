# CODE MAP of extractors/shaking.py   (tree node: does the hand shake more under a firm grip?)
#   Window          WINDOW: the later part of the clip (45-85 %), when the object is held
#   Velocity        the hand's velocity relative to the background, gaps of <= 2 frames filled, longer gaps rejected
#   Features        FEATURES: 4 tremor numbers from that velocity (jitter around its smooth trend, reversals)
#   Feature sets    "shaking"; load() -> {name: X (160, 4)}, NaN where the window is unusable
#   AI use          GPT-6.1 Sol (setting: medium); GPT-6 Astra (setting: light) (original code);
#                   Claude Opus 5.5 (claude-opus-5-5), default setting (restructure)

# %% Window
import numpy as np

from . import motion

WINDOW = (0.45, 0.85)
FEATURES = ["position_jitter_rms", "velocity_jitter_rms", "acceleration_rms", "reversal_rate"]
TREND_SECONDS = 0.33          # moving-average width that defines the smooth (intended) motion
REVERSAL_MIN_SPEED = 0.002    # a direction reversal only counts above this speed (ignores noise at rest)


# %% Velocity
def window_velocity(series):
    """Uniformly sampled hand velocity (n, 2) over the window, or None if it is unusable."""
    selected = motion.phase(series["samples"], series["frame_count"], WINDOW)
    if not motion.enough_valid(selected):
        return None
    # Only fill short holes for a uniformly sampled series; never bridge long failures.
    indices = np.flatnonzero([s["valid"] for s in selected])
    run = 0
    for s in selected[indices[0]:indices[-1] + 1]:
        run = 0 if s["valid"] else run + 1
        if run > 2:
            return None
    times = np.arange(indices[0], indices[-1] + 1)
    velocity = np.asarray([s["hand_velocity"] for s in selected if s["valid"]])
    return np.column_stack([np.interp(times, indices, velocity[:, j]) for j in range(2)])


# %% Features
def shaking_features(velocity, fps):
    velocity = np.asarray(velocity, dtype=float)
    width = min(int(round(TREND_SECONDS * fps)) | 1, (len(velocity) - 1) | 1)
    width = max(3, width)
    pad = width // 2
    smooth = np.column_stack([np.convolve(np.pad(velocity[:, j], pad, mode="reflect"), np.ones(width) / width,
                                          mode="valid") for j in range(2)])
    residual = velocity - smooth                                   # shaking = motion minus its smooth trend
    position = np.cumsum(velocity, axis=0) / fps
    # Per-video fixed-form linear detrending, not a fitted cross-video normaliser.
    t = np.linspace(-1, 1, len(position))
    detrended = position - position.mean(axis=0) - t[:, None] * (t @ position / (t @ t))[None, :]
    acceleration = np.diff(velocity, axis=0) * fps
    magnitude = np.linalg.norm(residual, axis=1)
    reversals = ((np.sum(residual[1:] * residual[:-1], axis=1) < 0) &
                 (magnitude[1:] > REVERSAL_MIN_SPEED) & (magnitude[:-1] > REVERSAL_MIN_SPEED))
    return dict(position_jitter_rms=float(np.sqrt(np.mean(np.sum(detrended ** 2, axis=1)))),
                velocity_jitter_rms=float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
                acceleration_rms=float(np.sqrt(np.mean(np.sum(acceleration ** 2, axis=1)))),
                reversal_rate=float(reversals.sum() / ((len(velocity) - 1) / fps)))


def window_features(series):
    """One clip's motion series -> {feature: value}, or None if the window is unusable."""
    velocity = window_velocity(series)
    return None if velocity is None else shaking_features(velocity, series["fps"])


# %% Feature sets
FEATURE_SETS = ["shaking"]


def load(clips, names=FEATURE_SETS):
    return {"shaking": motion.table(clips, window_features, FEATURES)}
