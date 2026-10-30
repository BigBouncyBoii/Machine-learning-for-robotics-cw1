# CODE MAP of protocol.py
#   Protocol       validate immutable split, labels, counts, fingerprint and paths
#   Evaluation     fold-local scaler/classifier and predeclared majority fallback
#   Files          atomic JSON output and reproducibility hashes
#   AI use         GPT-6 (Codex); exact version/settings unavailable
# Imported by explore.py and FinalMethod.py; no standalone command.
# %% Protocol
import hashlib
import json
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'Part3'
OUT = HERE / 'results'
CACHE = HERE / 'cache_speed_shaking_v2'
EXPECTED = '3696c80d'


def load_protocol():
    split = pd.read_csv(ROOT / 'data/split.csv', dtype=str, keep_default_na=False)
    required = ['file', 'person', 'object', 'label', 'set']
    if not set(required).issubset(split.columns):
        raise RuntimeError('Missing split columns')
    test_files = split.loc[split['set'] == 'test', 'file'].tolist()
    fingerprint = hashlib.md5('\n'.join(sorted(test_files)).encode()).hexdigest()[:8]
    print(f'Test fingerprint: {fingerprint} (expected {EXPECTED})', flush=True)
    if fingerprint != EXPECTED:
        raise RuntimeError('Test fingerprint mismatch; stop')
    if len(split) != 160 or split.file.nunique() != 160 or split['set'].value_counts().to_dict() != {'train': 120, 'test': 40}:
        raise RuntimeError('Expected 160 unique files and 120/40 split')
    expected = {'person': {'A', 'B', 'C', 'D'}, 'object': {'hard', 'soft'}, 'label': {'light', 'firm'}}
    for column, values in expected.items():
        if set(split[column]) != values:
            raise RuntimeError(f'Unexpected {column} values')
    if not (split.groupby(['person', 'object', 'label']).size() == 10).all():
        raise RuntimeError('Expected 10 clips per person/object/label')
    for name, count, by_object in [('train', 15, 30), ('test', 5, 10)]:
        subset = split.loc[split['set'] == name]
        if not (subset.groupby(['person', 'label']).size() == count).all() or not (subset.groupby(['object', 'label']).size() == by_object).all():
            raise RuntimeError('Unbalanced fixed split')
    labels = pd.read_csv(ROOT / 'data/labels.csv', dtype=str, keep_default_na=False)
    cols = ['file', 'person', 'object', 'label']
    if len(labels) != 160 or labels.file.nunique() != 160 or not labels[cols].sort_values('file').reset_index(drop=True).equals(split[cols].sort_values('file').reset_index(drop=True)):
        raise RuntimeError('labels.csv and split.csv disagree')
    paths = [PurePosixPath(f) for f in split.file]
    if any(p.is_absolute() or '..' in p.parts or '\\' in str(p) for p in paths):
        raise RuntimeError('Unsafe relative file path')
    split['video_path'] = [str(ROOT / 'data/videos_480p' / Path(p).with_suffix('.mp4')) for p in paths]
    # A source identity cannot occur in both sets, even under different clip names.
    if 'original' in labels:
        identity = labels.person + '/' + labels.original
        if identity.duplicated().any():
            raise RuntimeError('Duplicate original source identity: inspect before evaluation')
    return split


# %% Evaluation
def fit_model(X, y, C):
    available = np.isfinite(X).all(axis=1)
    # Majority uses ALL fold-training labels. Ties always predict light.
    fallback = int(np.sum(y) > len(y) / 2)
    model = None
    if available.any() and len(np.unique(y[available])) == 2:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=0))
        model.fit(X[available], y[available])
    return model, fallback


def predict(model, fallback, X):
    available = np.isfinite(X).all(axis=1)
    used = available & (model is not None)
    prediction = np.full(len(X), fallback, dtype=int)
    probability = np.full(len(X), float(fallback))
    if used.any():
        probability[used] = model.predict_proba(X[used])[:, 1]
        prediction[used] = (probability[used] >= 0.5).astype(int)
    return prediction, probability, used


def matrix(rows, window, names):
    return np.asarray([[row['windows'].get(window, {}).get(n, np.nan) if row['windows'].get(window) is not None else np.nan for n in names] for row in rows])


# %% Files
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes():
    return {name: digest(HERE / name) for name in ['features.py', 'protocol.py', 'explore.py', 'FinalMethod.py']}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


if __name__ == '__main__':
    load_protocol()
