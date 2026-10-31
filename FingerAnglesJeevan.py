# CODE MAP of FingerAngles.py
# Settings        paths, frame step, minimum hand-detection rate, random seed
# Joint triples   which 3 landmarks define each of the 15 finger joint angles
# Angle features  joint-angle mean/std/min and hand-detection rate per video
# Extract         process every video in labels.csv and cache features
# Validate        test each person A-D using the other three people for training
# AI use          <name your AI model, version and setting here>
#
# Install:  pip install "mediapipe>=0.10.35,<1" scikit-learn pandas numpy matplotlib
# Run:      python FingerAngles.py
#
# The Part3 dataset uses data/labels.csv with file,person,object,label,original
# columns. Its .MOV filenames map to .mp4 clips under data/videos_480p/.
# MediaPipe's hand-landmarker model is downloaded automatically on first run.
# The original video,person,object,label,split format is also supported.

# %% Settings
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_CANDIDATES = (
    SCRIPT_DIR.parent,
    Path.cwd(),
    Path.home() / "OneDrive" / "Documents" / "Machine-learning-for-robotics-cw1-main",
)
PROJECT_DIR = next(
    (
        candidate
        for candidate in PROJECT_CANDIDATES
        if (candidate / "Part3" / "data" / "labels.csv").is_file()
    ),
    PROJECT_CANDIDATES[-1],
)
DATA_DIR = PROJECT_DIR / "Part3" / "data"
VIDEO_DIR = DATA_DIR / "videos_480p"
LABELS_CSV = DATA_DIR / "labels.csv"
CACHE_NAME = "angle_features_all_videos_image_mode.npz"
CACHE_CANDIDATES = (
    SCRIPT_DIR / CACHE_NAME,
    PROJECT_DIR / "Jeevan's work" / CACHE_NAME,
)
CACHE = next((path for path in CACHE_CANDIDATES if path.is_file()), CACHE_CANDIDATES[0])
MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FRAME_STEP = 2          # use every 2nd frame
SEED = 1
PERSONS = ("A", "B", "C", "D")
_HAND_LANDMARKER = None
np.random.seed(SEED)

# %% Joint triples
# MediaPipe landmarks: 0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20.
# Each triple (a, b, c) gives the angle at b between segments b->a and b->c.
# 180 degrees = straight finger, smaller = more bent.
TRIPLES = []
for base in (1, 5, 9, 13, 17):                      # thumb + 4 fingers
    TRIPLES += [(0, base, base + 1),                # knuckle (MCP / CMC)
                (base, base + 1, base + 2),         # middle joint (PIP / MCP)
                (base + 1, base + 2, base + 3)]     # upper joint (DIP / IP)
N_ANG = len(TRIPLES)  # 15


# %% Angle features
def frame_angles(pts):
    """pts: (21,3) landmarks -> (15,) joint angles in degrees."""
    out = np.empty(N_ANG)
    for i, (a, b, c) in enumerate(TRIPLES):
        v1, v2 = pts[a] - pts[b], pts[c] - pts[b]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        out[i] = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    return out


def summarize_video_angles(angles, detection_rate):
    """Return angle summary features plus detection rate, including no-hand clips."""
    if angles:
        values = np.asarray(angles)
        summary = np.concatenate([values.mean(0), values.std(0), values.min(0)])
    else:
        summary = np.zeros(N_ANG * 3)
    return np.concatenate([summary, [detection_rate]])


def ensure_hand_model():
    """Download the official MediaPipe Tasks model once if it is not cached."""
    if MODEL_PATH.is_file():
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_suffix(".task.download")
    try:
        with urlopen(MODEL_URL, timeout=60) as response, temporary_path.open("wb") as model:
            while chunk := response.read(1024 * 1024):
                model.write(chunk)
        temporary_path.replace(MODEL_PATH)
    except (OSError, URLError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the MediaPipe hand-landmarker model. "
            f"Check your internet connection and try again: {MODEL_URL}"
        ) from exc


def get_hand_landmarker():
    """Create one image-mode detector and reuse it across all clips."""
    global _HAND_LANDMARKER
    if _HAND_LANDMARKER is None:
        ensure_hand_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
        )
        _HAND_LANDMARKER = mp_vision.HandLandmarker.create_from_options(options)
    return _HAND_LANDMARKER


def video_features(path):
    """Return 46 features and detection rate for one video, regardless of detections."""
    hands = get_hand_landmarker()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    angs, n_seen = [], 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % FRAME_STEP == 0:
            n_seen += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = hands.detect(image)
            if res.hand_landmarks:
                lm = res.hand_landmarks[0]
                pts = np.array([[p.x, p.y, p.z] for p in lm])
                angs.append(frame_angles(pts))
        i += 1
    cap.release()
    rate = len(angs) / max(n_seen, 1)
    return summarize_video_angles(angs, rate), rate


# %% Extract
def build_dataset():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return pd.DataFrame(d["meta"].tolist()), d["X"], d["y"]
    if not LABELS_CSV.is_file():
        raise FileNotFoundError(
            f"Could not find {LABELS_CSV}. Checked the script folder, current "
            "working directory, and the default OneDrive project folder. Move "
            "FingerAngles.py into the project or update PROJECT_CANDIDATES."
        )
    df = pd.read_csv(LABELS_CSV)
    if "label" not in df.columns or not ({"file", "video"} & set(df.columns)):
        raise ValueError(
            f"{LABELS_CSV} must contain a 'label' column and either 'file' or 'video'."
        )

    rows, X, y = [], [], []
    for _, r in df.iterrows():
        if "file" in df.columns:
            relative_path = Path(str(r["file"]))
            video = (VIDEO_DIR / relative_path).with_suffix(".mp4")
        else:
            video = Path(str(r["video"]))
            if not video.is_absolute():
                video = LABELS_CSV.parent / video
        if not video.is_file():
            raise FileNotFoundError(f"Video listed in {LABELS_CSV} was not found: {video}")

        f, rate = video_features(video)
        row = r.to_dict()
        row["video"] = str(video)
        row["hand_detection_rate"] = rate
        rows.append(row)
        X.append(f)
        y.append(1 if r["label"] == "firm" else 0)
    meta = pd.DataFrame(rows)
    X, y = np.array(X), np.array(y)
    np.savez(CACHE, meta=np.array(rows, dtype=object), X=X, y=y)
    print(f"processed {len(meta)} of {len(df)} videos")
    no_hand = int((meta["hand_detection_rate"] == 0).sum())
    if no_hand:
        print(f"WARNING: no hand was detected in {no_hand} videos; those use zero angle features.")
    return meta, X, y


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def evaluate_person_holdouts(meta, X, y) -> float:
    """Train on three people per fold and return mean accuracy as a fraction."""
    if "person" not in meta.columns:
        raise ValueError(f"{LABELS_CSV} must contain a 'person' column.")

    people = meta["person"].astype(str).to_numpy()
    missing_people = set(PERSONS) - set(people)
    if missing_people:
        raise ValueError(
            f"No usable videos found for: {', '.join(sorted(missing_people))}. "
            "Check the video files and hand-detection threshold."
        )

    accuracies = []
    predictions = np.empty_like(y)
    for test_person in PERSONS:
        test_mask = people == test_person
        train_mask = np.isin(people, [p for p in PERSONS if p != test_person])
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"Training on the other three people for {test_person} "
                "must include both grip labels."
            )

        model = make_model().fit(X_train, y_train)
        predictions[test_mask] = model.predict(X_test)
        accuracy = (predictions[test_mask] == y_test).mean()
        accuracies.append(accuracy)
        train_people = ", ".join(p for p in PERSONS if p != test_person)
        print(
            f"Test {test_person} (train {train_people}): "
            f"{accuracy:.1%} accuracy "
            f"({len(X_test)} test videos, {len(X_train)} training videos)"
        )

    success_rate = float(np.mean(accuracies))
    print(f"Mean accuracy across A-D: {success_rate:.1%}")
    save_evaluation_figures(y, predictions, accuracies, success_rate)
    return success_rate


def save_evaluation_figures(y, predictions, accuracies, success_rate):
    """Save held-out-person accuracy and pooled confusion-matrix figures."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    bars = ax.bar(PERSONS, np.asarray(accuracies) * 100, color="#4c78a8")
    guessing_baseline = max(np.mean(y == 0), np.mean(y == 1))
    ax.axhline(
        guessing_baseline * 100,
        color="#e45756",
        linestyle="--",
        linewidth=1.8,
        label=f"Majority-class baseline: {guessing_baseline:.1%}",
    )
    ax.axhline(
        success_rate * 100,
        color="#72b7b2",
        linestyle=":",
        linewidth=1.8,
        label=f"Mean: {success_rate:.1%}",
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Correct predictions (%)")
    ax.set_xlabel("Held-out person (trained on the other three)")
    ax.set_title("Grip classification success rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for bar, accuracy in zip(bars, accuracies):
        ax.annotate(
            f"{accuracy:.1%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    accuracy_path = SCRIPT_DIR / "grip_success_rates.png"
    fig.savefig(accuracy_path, dpi=180)
    plt.close(fig)

    confusion = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y, predictions):
        confusion[int(actual), int(predicted)] += 1

    fig, ax = plt.subplots(figsize=(5.8, 5), layout="constrained")
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Light", "Firm"])
    ax.set_yticks([0, 1], labels=["Light", "Firm"])
    ax.set_xlabel("Predicted grip")
    ax.set_ylabel("Actual grip")
    pooled_accuracy = float(np.mean(y == predictions))
    ax.set_title(f"Pooled held-out-person predictions ({pooled_accuracy:.1%} accuracy)")
    for row in range(2):
        for column in range(2):
            color = "white" if confusion[row, column] > confusion.max() * 0.9 else "black"
            ax.text(
                column,
                row,
                str(confusion[row, column]),
                ha="center",
                va="center",
                color=color,
                fontsize=13,
            )
    fig.colorbar(image, ax=ax, label="Number of clips")
    confusion_path = SCRIPT_DIR / "grip_confusion_matrix.png"
    fig.savefig(confusion_path, dpi=180)
    plt.close(fig)
    print(f"Saved figures: {accuracy_path} and {confusion_path}")


# %% Validate
if __name__ == "__main__":
    meta, X, y = build_dataset()
    evaluate_person_holdouts(meta, X, y)
