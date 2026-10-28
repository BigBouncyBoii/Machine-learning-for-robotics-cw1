# CODE MAP of performance.py   (performance statistics of the colour method over the four test cases)
#   Settings       output folder
#   Run cases      colour_method.run() for test person A, B, C, D (no per-grip images, no windows)
#   Per case       chosen input + model, validation vs test accuracy, hard / soft, ROC AUC
#   Pooled         all 160 test predictions: accuracy with 95 % CI, p-value vs guessing, confusion matrix,
#                  precision / recall / F1, balanced accuracy, ROC AUC, Brier score
#   Breakdowns     by object, by true label, by how often MediaPipe found the hand; most confident mistakes
#   Save           performance/report.txt, per_case.csv, predictions.csv, performance.png
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# HOW TO RUN (from the repo root)
#   .venv\Scripts\python "James F\colour_method\performance.py"
#
# NOTE  Each case is a different model (trained without its test person), so the pooled numbers describe
#       the METHOD on a new person, not one single trained model.

# %% Settings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (brier_score_loss, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from colour_method import PEOPLE, run

OUT = Path(__file__).resolve().parent / "performance"
OUT.mkdir(exist_ok=True)

# %% Run cases
results = [run(p, show=False, grip_images=False) for p in PEOPLE]
pred = pd.concat([r["pred"] for r in results], ignore_index=True)
pred["pred"] = (pred["p_firm"] > 0.5).astype(int)
pred["correct"] = pred["pred"] == pred["y"]
pred["confidence"] = np.where(pred["pred"] == 1, pred["p_firm"], 1 - pred["p_firm"])

lines = []


def out(s=""):
    lines.append(s)


def acc_ci(k, n):
    """Accuracy k/n with its 95 % Wilson interval and the one-sided binomial p-value against 50 % guessing."""
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")   # two-sided interval
    return k / n, ci.low, ci.high, binomtest(k, n, 0.5, alternative="greater").pvalue


# %% Per case
out("PERFORMANCE OF THE COLOUR METHOD (MediaPipe finger patches -> colour change over time -> classifier)")
out("Each case: one person is the test set; input and model are chosen by leave-one-person-out validation")
out("on the other three people only, then trained on those three and tested once.")
out()
out(f"{'test':<6}{'chosen input':<32}{'model':<24}{'val':>6}{'test':>7}{'95% CI':>14}{'p':>8}"
    f"{'hard':>7}{'soft':>7}{'AUC':>7}")
rows = []
for r in results:
    k = r["pred"]
    acc, lo, hi, pval = acc_ci(r["test_correct"], r["n_test"])
    auc = roc_auc_score(k["y"], k["p_firm"])
    rows.append({"test_person": r["test_person"], "input": r["input"], "model": r["model"],
                 "val_acc": r["val_acc"], "test_acc": acc, "ci_low": lo, "ci_high": hi, "p_value": pval,
                 "hard_acc": r["hard"], "soft_acc": r["soft"], "auc": auc,
                 "correct": r["test_correct"], "n_test": r["n_test"]})
    out(f"{r['test_person']:<6}{r['input']:<32}{r['model']:<24}{100 * r['val_acc']:>5.0f}%{100 * acc:>6.1f}%"
        f"{f'{100 * lo:.0f}-{100 * hi:.0f}%':>14}{pval:>8.3f}{100 * r['hard']:>6.0f}%{100 * r['soft']:>6.0f}%"
        f"{auc:>7.2f}")
per_case = pd.DataFrame(rows)
gap = (per_case["val_acc"] - per_case["test_acc"]).mean()
out(f"mean validation {100 * per_case['val_acc'].mean():.1f} %, mean test {100 * per_case['test_acc'].mean():.1f} %"
    f" (validation is {100 * gap:+.1f} points above test: the optimism of choosing on validation)")

# %% Pooled
n, k = len(pred), int(pred["correct"].sum())
acc, lo, hi, pval = acc_ci(k, n)
cm = confusion_matrix(pred["y"], pred["pred"], labels=[0, 1])
prec, rec, f1, sup = precision_recall_fscore_support(pred["y"], pred["pred"], labels=[0, 1], zero_division=0)
auc = roc_auc_score(pred["y"], pred["p_firm"])
out()
out(f"POOLED OVER ALL {n} TEST GRIPS (each grip predicted by the model that never saw its person)")
out(f"  accuracy           {100 * acc:.1f} % ({k}/{n}), 95 % CI {100 * lo:.1f}-{100 * hi:.1f} %")
out(f"  vs guessing (50 %) one-sided binomial p = {pval:.4f}"
    + ("  -> better than guessing at the 5 % level" if pval < 0.05 else "  -> NOT clearly better than guessing"))
out(f"  balanced accuracy  {100 * (rec[0] + rec[1]) / 2:.1f} %")
out(f"  ROC AUC            {auc:.3f}  (0.5 = guessing, 1 = perfect ranking of firm above light)")
out(f"  Brier score        {brier_score_loss(pred['y'], pred['p_firm']):.3f}  (0.25 = always saying 50 %)")
out()
out("  confusion matrix          predicted light   predicted firm")
out(f"    true light ({cm[0].sum():>3})         {cm[0, 0]:>8}         {cm[0, 1]:>8}")
out(f"    true firm  ({cm[1].sum():>3})         {cm[1, 0]:>8}         {cm[1, 1]:>8}")
out()
out(f"  {'class':<8}{'precision':>11}{'recall':>9}{'F1':>7}{'grips':>7}")
for name, p_, r_, f_, s_ in zip(("light", "firm"), prec, rec, f1, sup):
    out(f"  {name:<8}{100 * p_:>10.1f}%{100 * r_:>8.1f}%{f_:>7.2f}{s_:>7}")

# %% Breakdowns
out()
out("BREAKDOWNS (pooled test grips)")


def breakdown(title, groups):
    out(f"  {title}")
    for name, g in groups:
        a, l, h, p = acc_ci(int(g["correct"].sum()), len(g))
        out(f"    {name:<28}{100 * a:>6.1f}% ({int(g['correct'].sum())}/{len(g)}), CI {100 * l:.0f}-{100 * h:.0f}%,"
            f" p = {p:.3f}")


breakdown("by object", pred.groupby("object"))
breakdown("by true label", pred.groupby("label"))
breakdown("by object and label", [(f"{o} {l}", g) for (o, l), g in pred.groupby(["object", "label"])])
med = pred["hand_found"].median()
breakdown(f"by hand tracking (median {100 * med:.0f} % of frames)",
          [(f"hand found in >= {100 * med:.0f} % of frames", pred[pred["hand_found"] >= med]),
           (f"hand found in <  {100 * med:.0f} % of frames", pred[pred["hand_found"] < med])])
out()
out("  most confident mistakes (look at these in figures/test_<person>/)")
for _, r in pred[~pred["correct"]].nlargest(8, "confidence").iterrows():
    out(f"    {Path(r['file']).stem:<18} true {r['label']:<6} P(firm) = {r['p_firm']:.2f}"
        f"  hand found {100 * r['hand_found']:.0f} %")

# %% Save
report = "\n".join(lines)
print("\n" + report)
(OUT / "report.txt").write_text(report + "\n", encoding="utf-8")
per_case.to_csv(OUT / "per_case.csv", index=False, float_format="%.4f")
pred.to_csv(OUT / "predictions.csv", index=False, float_format="%.4f")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
x = np.arange(len(per_case))
axes[0].bar(x - 0.2, 100 * per_case["val_acc"], 0.4, label="validation (chosen version)", color="tab:grey")
axes[0].bar(x + 0.2, 100 * per_case["test_acc"], 0.4, label="test", color="tab:blue",
            yerr=[100 * (per_case["test_acc"] - per_case["ci_low"]), 100 * (per_case["ci_high"] - per_case["test_acc"])],
            capsize=4)
axes[0].axhline(50, color="k", ls="--", lw=1, label="guessing")
axes[0].set_xticks(x, [f"test {p}" for p in per_case["test_person"]])
axes[0].set_ylabel("accuracy (%)")
axes[0].set_ylim(0, 100)
axes[0].set_title("Accuracy per case (test bars: 95 % CI)")
axes[0].legend(fontsize=8)

axes[1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                     color="white" if cm[i, j] > (cm.min() + cm.max()) / 2 else "black")   # colour scale is min..max
axes[1].set_xticks([0, 1], ["light", "firm"])
axes[1].set_yticks([0, 1], ["light", "firm"])
axes[1].set_xlabel("predicted")
axes[1].set_ylabel("true")
axes[1].set_title(f"Confusion matrix, all {n} test grips ({100 * acc:.1f} %)")

for r in results:
    fpr, tpr, _ = roc_curve(r["pred"]["y"], r["pred"]["p_firm"])
    axes[2].plot(fpr, tpr, lw=1, alpha=0.7, label=f"test {r['test_person']}")
fpr, tpr, _ = roc_curve(pred["y"], pred["p_firm"])
axes[2].plot(fpr, tpr, "k", lw=2, label=f"pooled (AUC {auc:.2f})")
axes[2].plot([0, 1], [0, 1], "k--", lw=0.8, label="guessing")
axes[2].set_xlabel("false positive rate (light called firm)")
axes[2].set_ylabel("true positive rate (firm called firm)")
axes[2].set_title("ROC curves")
axes[2].legend(fontsize=8)
fig.suptitle("Colour method: performance on held-out people (performance.py)")
fig.tight_layout()
fig.savefig(OUT / "performance.png", dpi=150)
print(f"\nsaved report.txt, per_case.csv, predictions.csv and performance.png in {OUT}")
plt.show()
