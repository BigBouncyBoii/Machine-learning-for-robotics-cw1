# CODE MAP of explore.py
#   Inputs         fixed split checks; train-only resumable feature extraction
#   Validation     three trials, two C values per trial; leave one person out
#   History        match original video identities to old artifacts, no test images
#   Output         per-video predictions, folds, objects and proposed method
#   AI use         GPT-6 (Codex); exact version/settings unavailable
# Run from repository: .venv\Scripts\python.exe "Qian L/explore.py"
# Optional: --history-root PATH (old TaskC_2510); no formal test is run.
# %% Inputs
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from features import FEATURES, load_clip, sha256
from protocol import ROOT, OUT, CACHE, EXPECTED, load_protocol, matrix, fit_model, predict, save_json, code_hashes, digest


def collect(meta):
    rows = []
    for i, row in enumerate(meta.itertuples(), 1):
        begin = time.perf_counter()
        print(f'[{i}/{len(meta)}] {row.file} starting', flush=True)
        result, cached = load_clip(Path(row.video_path), CACHE)
        flags = ', '.join(f'{name}={result["windows"][name] is not None}' for name in FEATURES)
        print(f'  {"cache" if cached else "extracted"} {time.perf_counter()-begin:.1f}s; {flags}', flush=True)
        rows.append(result)
    return rows


# %% Validation
def validate(meta, rows):
    y = (meta.label == 'firm').to_numpy(dtype=int)
    folds, predictions = [], []
    for feature, names in FEATURES.items():
        window = feature
        X = matrix(rows, window, names)
        for C in [0.1, 1.0]:
            method = f'{feature}_C{C:g}'
            for train, val in LeaveOneGroupOut().split(X, y, groups=meta.person):
                model, fallback = fit_model(X[train], y[train], C)
                pred, prob, used = predict(model, fallback, X[val])
                correct = pred == y[val]
                covered = int(used.sum())
                person = meta.iloc[val[0]].person
                fold = dict(method=method, window=window, features=feature, C=C, person=person,
                    n_train=len(train), n_train_features=int(np.isfinite(X[train]).all(axis=1).sum()),
                    n_validation=len(val), n_correct=int(correct.sum()), accuracy=float(correct.mean()),
                    n_features=int(np.isfinite(X[val]).all(axis=1).sum()), n_predicted_by_model=covered,
                    coverage=covered / len(val), fallback_count=int((~used).sum()),
                    selective_accuracy=float(correct[used].mean()) if covered else None,
                    feature_failure_count=int((~np.isfinite(X[val]).all(axis=1)).sum()))
                for obj in ['hard', 'soft']:
                    keep = meta.iloc[val].object.to_numpy() == obj
                    fold[obj + '_n'] = int(keep.sum())
                    fold[obj + '_correct'] = int(correct[keep].sum())
                    fold[obj + '_accuracy'] = float(correct[keep].mean())
                    fold[obj + '_coverage'] = float(used[keep].mean())
                folds.append(fold)
                for j, index in enumerate(val):
                    record = meta.iloc[index]
                    predictions.append(dict(method=method, file=record.file, person=record.person,
                        object=record.object, label=record.label, prediction='firm' if pred[j] else 'light',
                        p_firm=float(prob[j]), model_used=bool(used[j]), correct=bool(correct[j]),
                        status=rows[index]['status']))
            print(f'Validated {method}', flush=True)
    return pd.DataFrame(folds), pd.DataFrame(predictions)


# %% History
def history(meta, old_root):
    labels = pd.read_csv(ROOT / 'data/labels.csv', dtype=str)
    inventory = pd.read_csv(old_root / 'section_c_inventory.csv', dtype=str)
    inventory['original'] = inventory.source_video.str.replace('\\', '/', regex=False).str.rsplit('/').str[-1]
    inventory = inventory.rename(columns={'participant': 'person', 'object_condition': 'object'})
    mapping = labels.merge(inventory[['person', 'object', 'label', 'original', 'video_id']], on=['person', 'object', 'label', 'original'], how='left', validate='one_to_one')
    old_split = json.loads((old_root / 'split_manifest.json').read_text(encoding='utf-8'))
    old_sets = {video: name for name in ['train', 'validation', 'test'] for video in old_split[name]}
    artifacts = ['active_image_motion/cache', 'active_image_motion/training_previews', 'annotations',
                 'auto_annotations', 'validation_auto_annotations', 'auto_training_features_per_clip',
                 'auto_validation_features_per_clip', 'auto_previews', 'auto_review']
    artifact_ids = {name: {p.stem for p in (old_root / name).rglob('*') if p.is_file()} for name in artifacts}
    mapping = mapping.merge(meta[['file', 'set']], on='file', validate='one_to_one')
    mapping['old_set'] = mapping.video_id.map(old_sets).fillna('unmatched')
    mapping['artifact_evidence'] = [';'.join(name for name, ids in artifact_ids.items() if video in ids) for video in mapping.video_id]
    mapping['historical_development_possible'] = mapping.person.isin(['A', 'B', 'C'])
    mapping['human_view_confirmed'] = 'unknown; saved preview does not prove viewing'
    mapping.to_csv(OUT / 'historical_exposure.csv', index=False)
    test = mapping.loc[mapping['set'] == 'test']
    save_json(OUT / 'historical_exposure.json', dict(identity_basis='labels.original + person/object/label matched to old inventory; not binary-content verification',
        matched=int(mapping.video_id.notna().sum()), test_count=len(test),
        test_old_sets=test.old_set.value_counts().to_dict(),
        test_with_artifact_evidence=int(test.artifact_evidence.ne('').sum()),
        test_potential_historical_development=int(test.historical_development_possible.sum()),
        limitation='Actual prior human viewing cannot be reconstructed fully. Old thresholds/ROI were developed using A-C; outer LOPO scores are not a pristine evaluation of all preprocessing choices. Test must not be described as previously unseen.'))


# %% Output
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history-root', type=Path)
    args = parser.parse_args()
    split = load_protocol()
    if (OUT/'formal_test_started.json').exists() or (OUT/'frozen_method.json').exists():
        raise RuntimeError('Method already frozen/tested; do not silently resume development')
    meta = split.loc[split['set'] == 'train'].reset_index(drop=True)
    OUT.mkdir(exist_ok=True)
    if args.history_root:
        history(split, args.history_root)
    rows = collect(meta)
    folds, predictions = validate(meta, rows)
    folds.to_csv(OUT / 'folds.csv', index=False)
    predictions.to_csv(OUT / 'validation_predictions.csv', index=False)
    coverage = []
    for record, row in zip(meta.itertuples(), rows):
        coverage.append(dict(file=record.file, person=record.person, object=record.object, label=record.label,
            status=row['status'], **{name: row['windows'][name] is not None for name in FEATURES},
            speed_pairs=row.get('phase_pairs', {}).get('speed', 0),
            shaking_pairs=row.get('phase_pairs', {}).get('shaking', 0),
            speed_valid_pairs=row.get('phase_valid_pairs', {}).get('speed', 0),
            shaking_valid_pairs=row.get('phase_valid_pairs', {}).get('shaking', 0),
            reasons='; '.join(row.get('failure_reason', []))))
    pd.DataFrame(coverage).to_csv(OUT / 'extraction_status.csv', index=False)
    summary = folds.groupby(['method', 'window', 'features', 'C'], as_index=False).agg(
        mean_validation_accuracy=('accuracy', 'mean'), coverage=('coverage', 'mean'),
        fallback_count=('fallback_count', 'sum'), hard_accuracy=('hard_correct', 'sum'), soft_accuracy=('soft_correct', 'sum'))
    for obj in ['hard', 'soft']:
        summary[obj + '_accuracy'] /= int((meta.object == obj).sum())
    # Selection: all-clip LOPO mean, ties prefer fewer features, stronger regularisation, lexical method.
    summary['n_features'] = summary.features.map(lambda n: len(FEATURES[n]))
    summary = summary.sort_values(['mean_validation_accuracy', 'n_features', 'C', 'method'], ascending=[False, True, True, True])
    summary.to_csv(OUT / 'tuning.csv', index=False)
    summary.drop_duplicates('features').to_csv(OUT / 'summary.csv', index=False)
    best = summary.iloc[0]
    proposal = dict(method=best.method, window=best.window, features=best.features, C=float(best.C),
        feature_names=list(FEATURES[best.features]), test_fingerprint=EXPECTED,
        mean_validation_accuracy=float(best.mean_validation_accuracy),
        fallback='fold-training all-label majority, ties light; same rule at final refit',
        selection='mean all-clip LOPO accuracy; ties fewer features, lower C, lexical method',
        status='proposed from training validation; no formal test executed',
        code_hashes=code_hashes(), split_sha256=digest(ROOT / 'data/split.csv'),
        labels_sha256=digest(ROOT / 'data/labels.csv'),
        validation_predictions_sha256=digest(OUT / 'validation_predictions.csv'),
        train_hashes={r.file: sha256(Path(r.video_path)) if Path(r.video_path).exists() else 'missing' for r in meta.itertuples()})
    save_json(OUT / 'proposed_method.json', proposal)
    print(summary.drop_duplicates('features').to_string(index=False))
    print('Proposed:', best.method, '; formal test NOT run.')


if __name__ == '__main__':
    main()
