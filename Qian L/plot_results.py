# CODE MAP of plot_results.py
#   Inputs         fixed split and validated train-only out-of-fold predictions
#   Accuracy       three trials, held-out people and hard/soft comparisons
#   Confusion      one confusion matrix per trial, validation predictions only
#   Features       raw train feature distributions, no classifier tuning
#   Output         reproducible PNG/SVG figures and source data tables
#   AI use         GPT-6 (Codex); exact version/settings unavailable
# Run: .venv\Scripts\python.exe "Qian L/plot_results.py"
# %% Inputs
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from features import load_clip
from protocol import CACHE, OUT, load_protocol, save_json

TRIALS = ['speed', 'shaking', 'both']
NAMES = ['Speed proxy', 'Shaking', 'Both']
COLORS = ['#2878B5', '#C97723', '#478B65']


def main():
    split = load_protocol()
    train = split.loc[split['set'] == 'train'].copy()
    summary = pd.read_csv(OUT/'summary.csv').set_index('features').loc[TRIALS]
    folds = pd.read_csv(OUT/'folds.csv')
    predictions = pd.read_csv(OUT/'validation_predictions.csv')
    selected = []
    for trial in TRIALS:
        frame = predictions.loc[predictions.method == summary.loc[trial, 'method']].copy()
        if len(frame) != 120 or frame.file.nunique() != 120 or set(frame.file) != set(train.file):
            raise RuntimeError('Plots must contain exactly the 120 train identities for each trial')
        expected = frame.merge(train[['file', 'label', 'person', 'object']], on='file', suffixes=('', '_split'), validate='one_to_one')
        if any(not expected[c].equals(expected[c+'_split']) for c in ['label', 'person', 'object']):
            raise RuntimeError('Prediction metadata differs from fixed training manifest')
        correct = frame.prediction.eq(frame.label)
        if not np.isclose(correct.mean(), summary.loc[trial, 'mean_validation_accuracy']):
            raise RuntimeError('CSV accuracy mismatch')
        frame['trial'] = trial
        frame['correct'] = correct
        selected.append(frame)
    selected = pd.concat(selected, ignore_index=True)
    destination = OUT/'figures'
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.titleweight': 'medium', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

    def save(fig, name):
        fig.savefig(destination/(name+'.png'), dpi=200, bbox_inches='tight')
        fig.savefig(destination/(name+'.svg'), bbox_inches='tight')
        plt.close(fig)

    # %% Accuracy
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout='constrained')
    accuracy = summary.mean_validation_accuracy.to_numpy()*100
    bars = axes[0].bar(NAMES, accuracy, color=COLORS, width=.6)
    for bar, trial in zip(bars, TRIALS):
        count = int(selected.loc[selected.trial == trial, 'correct'].sum())
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{bar.get_height():.1f}%\n({count}/120)', ha='center', va='bottom')
    axes[0].set_title('A. Mean LOPO validation accuracy')
    for trial, name, color in zip(TRIALS, NAMES, COLORS):
        ff = folds.loc[folds.method == summary.loc[trial, 'method']].set_index('person').loc[list('ABCD')]
        axes[1].plot(list('ABCD'), ff.accuracy*100, marker='o', color=color, label=name)
    axes[1].set_title('B. Held-out person (30 clips/fold)')
    axes[1].set_xlabel('Person held out from fitting')
    axes[1].legend(frameon=False, fontsize=10, loc='lower left')
    x = np.arange(2)
    for i, (trial, name, color) in enumerate(zip(TRIALS, NAMES, COLORS)):
        values = selected.loc[selected.trial == trial].groupby('object').correct.mean().reindex(['hard', 'soft'])*100
        bars = axes[2].bar(x+(i-1)*.24, values, width=.22, color=color, label=name)
        for bar in bars:
            axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f'{bar.get_height():.1f}', ha='center', fontsize=9)
    axes[2].set_xticks(x, ['Hard (n=60)', 'Soft (n=60)'])
    axes[2].set_title('C. Accuracy by object')
    for ax in axes:
        ax.set_ylim(0, 100)
        ax.set_ylabel('Accuracy (%)')
        ax.axhline(50, color='#555555', linestyle='--', linewidth=1, zorder=0)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('Three trials | 120 TRAIN clips | four-fold leave-one-person-out', fontsize=15)
    fig.supxlabel('Dashed line = 50% guessing. All trials: 100% feature coverage, zero fallback. Formal test NOT run.\nSource: plot_results.py; summary.csv, folds.csv, validation_predictions.csv', fontsize=10)
    save(fig, 'validation_overview')

    # %% Confusion
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), layout='constrained')
    matrices = {}
    for ax, trial, name in zip(axes, TRIALS, NAMES):
        frame = selected.loc[selected.trial == trial]
        matrix = confusion_matrix(frame.label, frame.prediction, labels=['light', 'firm'])
        matrices[trial] = matrix.tolist()
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=60)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{matrix[i,j]}\n{matrix[i,j]/60:.1%}', ha='center', va='center', color='white' if matrix[i,j]>32 else '#172A3A')
        ax.set_xticks([0, 1], ['Light', 'Firm'])
        ax.set_yticks([0, 1], ['Light (n=60)', 'Firm (n=60)'])
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True intended label')
        ax.set_title(name)
    fig.suptitle('Out-of-fold confusion matrices | row percentages', fontsize=15)
    fig.supxlabel('Same 120 TRAIN clips per trial; all predictions made while that person was held out.\nSource: plot_results.py; validation_predictions.csv. These are NOT test results.', fontsize=10)
    save(fig, 'validation_confusion')

    # %% Features
    records = []
    for row in train.itertuples():
        result, cached = load_clip(Path(row.video_path), CACHE)
        if not cached:
            raise RuntimeError('Expected existing train feature caches; run explore.py first')
        both = result['windows']['both']
        if both is not None:
            records.append(dict(file=row.file, person=row.person, object=row.object, label=row.label, **both))
    features = pd.DataFrame(records)
    features.to_csv(destination/'train_features.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    groups = [('hard', 'light'), ('hard', 'firm'), ('soft', 'light'), ('soft', 'firm')]
    rng = np.random.default_rng(0)  # display-only jitter, not a data split/model seed
    for ax, key, title, ylabel in zip(axes,
        ['inward_speed_mean', 'velocity_jitter_rms'],
        ['Early inward-motion speed proxy', 'Later velocity-jitter proxy'],
        ['ROI diagonals / second', 'ROI diagonals / second (RMS)']):
        data = [features.loc[(features.object == obj)&(features.label == label), key].to_numpy() for obj,label in groups]
        bp = ax.boxplot(data, positions=np.arange(1,5), widths=.5, showfliers=False, patch_artist=True)
        for patch, (_, label) in zip(bp['boxes'], groups):
            patch.set_facecolor('#BFD8EC' if label=='light' else '#F0D0AC')
        for i, values in enumerate(data, 1):
            ax.scatter(i+rng.uniform(-.12,.12,len(values)), values, s=16, alpha=.65, color='#2878B5' if groups[i-1][1]=='light' else '#C97723')
        ax.set_xticks(np.arange(1,5), [o.title()+'\n'+l for o,l in groups])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('Raw single-video feature distributions | TRAIN only', fontsize=15)
    fig.supxlabel('Each dot = one clip; boxes show median and quartiles. Colour-mask motion proxies, not precise finger tracking or force.\nSource: plot_results.py; label-free per-clip caches. Features/methods were not changed after plotting.', fontsize=10)
    save(fig, 'train_feature_distributions')

    # %% Output
    selected.to_csv(destination/'selected_validation_predictions.csv', index=False)
    save_json(destination/'figure_manifest.json', dict(source_script='plot_results.py', test_fingerprint='3696c80d',
        evaluation='120 train clips; four-fold leave-one-person-out; no formal test',
        trials=TRIALS, selected_methods=summary.method.to_dict(), confusion_order=['light', 'firm'],
        confusion_matrices=matrices, feature_count=len(features), figures=['validation_overview', 'validation_confusion', 'train_feature_distributions']))
    print('Figures saved:', destination, flush=True)


if __name__ == '__main__':
    main()
