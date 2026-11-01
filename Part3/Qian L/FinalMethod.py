# CODE MAP of FinalMethod.py
#   Freeze         bind train-selected method to code/data, refit 120 train clips
#   Test           one formal run on 40 fixed clips, guarded against repetition
#   Report         predictions, coverage, exact binomial interval and test
#   AI use         GPT-6 (Codex); exact version/settings unavailable
# Run from repository with its .venv\Scripts\python.exe:
#   "Qian L/FinalMethod.py" --freeze     (no test extraction/prediction)
#   "Qian L/FinalMethod.py" --run-test   (only AFTER method freeze)
# This independent entry preserves the team's existing Part3/FinalMethod.py.
# %% Freeze
import argparse
import importlib.metadata
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix
from features import sha256
from protocol import ROOT, OUT, load_protocol, code_hashes, digest, matrix, fit_model, predict, save_json
from explore import collect

FROZEN = OUT / 'frozen_method.json'
MODEL = OUT / 'frozen_model.joblib'
STARTED = OUT / 'formal_test_started.json'


def train_hashes(meta):
    return {r.file: sha256(Path(r.video_path)) if Path(r.video_path).exists() else 'missing' for r in meta.itertuples()}


def versions():
    return {p: importlib.metadata.version(p) for p in ['numpy', 'opencv-python-headless', 'scikit-learn', 'scipy', 'joblib']}


def freeze(split):
    if FROZEN.exists() or STARTED.exists():
        raise RuntimeError('Already frozen or tested; refusing overwrite')
    proposal = json.loads((OUT / 'proposed_method.json').read_text(encoding='utf-8'))
    if proposal['code_hashes'] != code_hashes() or proposal['split_sha256'] != digest(ROOT / 'data/split.csv'):
        raise RuntimeError('Code or split changed after validation; validate again')
    meta = split.loc[split['set'] == 'train'].reset_index(drop=True)
    if proposal['train_hashes'] != train_hashes(meta) or proposal['labels_sha256'] != digest(ROOT / 'data/labels.csv'):
        raise RuntimeError('Training data or labels changed after validation')
    if proposal['validation_predictions_sha256'] != digest(OUT / 'validation_predictions.csv'):
        raise RuntimeError('Validation predictions changed after selection')
    rows = collect(meta)
    X = matrix(rows, proposal['window'], proposal['feature_names'])
    model, fallback = fit_model(X, (meta.label == 'firm').to_numpy(dtype=int), proposal['C'])
    joblib.dump(dict(model=model, fallback=fallback), MODEL)
    restored = joblib.load(MODEL)
    before = predict(model, fallback, X)
    after = predict(restored['model'], restored['fallback'], X)
    if not all(np.array_equal(a, b) for a, b in zip(before, after)):
        raise RuntimeError('Saved model predictions differ')
    proposal.update(status='frozen before formal test', train_hashes=train_hashes(meta), versions=versions(),
        model_sha256=digest(MODEL), labels_sha256=digest(ROOT / 'data/labels.csv'),
        validation_predictions_sha256=digest(OUT / 'validation_predictions.csv'))
    save_json(FROZEN, proposal)
    print('Frozen method and fitted model saved. No test clips processed.')


# %% Test
def run_test(split):
    frozen = json.loads(FROZEN.read_text(encoding='utf-8'))
    if frozen['code_hashes'] != code_hashes() or frozen['model_sha256'] != digest(MODEL):
        raise RuntimeError('Frozen code/model changed')
    if frozen['versions'] != versions():
        raise RuntimeError('Environment differs from freeze')
    if frozen['split_sha256'] != digest(ROOT / 'data/split.csv') or frozen['labels_sha256'] != digest(ROOT / 'data/labels.csv'):
        raise RuntimeError('Frozen metadata changed')
    if frozen['train_hashes'] != train_hashes(split.loc[split['set'] == 'train']):
        raise RuntimeError('Train video content changed')
    # Exclusive create occurs before any test extraction or predictions. No automatic rerun.
    with STARTED.open('x', encoding='utf-8') as handle:
        json.dump(dict(method=frozen['method'], frozen_sha256=digest(FROZEN)), handle)
    meta = split.loc[split['set'] == 'test'].reset_index(drop=True)
    rows = collect(meta)
    X = matrix(rows, frozen['window'], frozen['feature_names'])
    saved = joblib.load(MODEL)
    pred, prob, used = predict(saved['model'], saved['fallback'], X)
    y = (meta.label == 'firm').to_numpy(dtype=int)
    output = meta.drop(columns='video_path').copy()
    output['prediction'] = np.where(pred, 'firm', 'light')
    output['p_firm'] = prob
    output['model_used'] = used
    output['correct'] = pred == y
    output.to_csv(OUT / 'formal_test_predictions.csv', index=False)

    # %% Report
    n_correct = int((pred == y).sum())
    ci = binomtest(n_correct, 40).proportion_ci(confidence_level=0.95, method='exact')
    result = dict(method=frozen['method'], test_fingerprint=frozen['test_fingerprint'], n=40,
        n_correct=n_correct, accuracy=n_correct / 40, model_coverage=float(used.mean()),
        fallback_count=int((~used).sum()), chance_baseline=0.5,
        ci95=[float(ci.low), float(ci.high)], ci_method='two-sided Clopper-Pearson exact binomial',
        one_sided_binomial_p=float(binomtest(n_correct, 40, 0.5, alternative='greater').pvalue),
        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        confusion_order=['light', 'firm'], confusion_axes='rows true; columns predicted',
        limitation='Clip-binomial CI/p assume independent clips; repeated clips within people may be dependent. Historical exposure disclosed separately.')
    for column in ['person', 'object']:
        result['by_' + column] = {str(g): dict(n=len(group), correct=int(group.correct.sum()),
            accuracy=float(group.correct.mean()), model_coverage=float(group.model_used.mean())) for g, group in output.groupby(column)}
    save_json(OUT / 'formal_test_results.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze', action='store_true')
    action.add_argument('--run-test', action='store_true')
    args = parser.parse_args()
    split = load_protocol()
    if args.freeze:
        freeze(split)
    else:
        run_test(split)


if __name__ == '__main__':
    main()
