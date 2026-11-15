"""Figures and a markdown report for the PBL-3 ensemble-learning review."""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree  # noqa: E402

from analysis.ensemble import COMBINERS, ENSEMBLE_METHODS, LEARNERS, REPORTED  # noqa: E402
from analysis.features import FEATURES  # noqa: E402
from analysis.review_plots import _positive_intervals  # noqa: E402

PBL1 = ("Logistic regression", "k-nearest neighbours", "Gradient boosting classifier")
PBL2 = ("K-means", "Gaussian mixture", "Gaussian HMM")
COLORS = {"PBL-3 ensemble": "#1b7f6b", "PBL-1 supervised": "#245d91", "PBL-2 unsupervised/HMM": "#c7812a",
          "Baseline": "#9aa5b1"}


def family(method: str) -> str:
    """Which review a compared method belongs to."""
    if method in ENSEMBLE_METHODS:
        return "PBL-3 ensemble"
    if method in PBL1:
        return "PBL-1 supervised"
    if method in PBL2:
        return "PBL-2 unsupervised/HMM"
    return "Baseline"


def best_ensemble(summary: pd.DataFrame) -> str:
    """Ensemble method with the highest mean held-out F1 (ties: PR-AUC)."""
    ens = summary[summary.method.isin(ENSEMBLE_METHODS)]
    return ens.sort_values(["f1", "pr_auc"], ascending=False).method.iloc[0]


def make_ensemble_artifacts(
    dataset: pd.DataFrame, predictions: pd.DataFrame, folds: pd.DataFrame, summary: pd.DataFrame,
    paired: pd.DataFrame, details: list[dict[str, Any]], cfg: Any,
) -> None:
    """Draw every PBL-3 figure into ``<output_dir>/figures`` and write the report."""
    output = cfg.ensemble.output_dir
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    dpi = cfg.evaluation.plot_dpi
    best = best_ensemble(summary)

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(figures / name, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # 1. Overall comparison.
    ordered = summary.sort_values("f1", ascending=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 7), sharey=True)
    colors = [COLORS[family(m)] for m in ordered.method]
    for ax, metric, title in zip(axes, ["f1", "pr_auc", "hit_at_1", "temporal_iou"],
                                 ["F1", "PR-AUC", "Hit@1", "Temporal IoU"]):
        values = ordered[metric].to_numpy()
        err = np.vstack([values - ordered[f"{metric}_lo"], ordered[f"{metric}_hi"] - values])
        ax.barh(ordered.method, values, xerr=np.maximum(err, 0), color=colors, capsize=2)
        ax.set(xlim=(0, 1), title=title)
        for y, v in enumerate(values):
            ax.text(v - .015, y, f"{v:.3f}", va="center", ha="right", fontsize=8, color="white", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    fig.legend(handles, COLORS.keys(), loc="lower center", ncol=4, bbox_to_anchor=(.5, -.04))
    fig.suptitle("Held-out-video comparison of all models (mean over 9 videos; bars: video-bootstrap 95% CI)", y=1.01)
    save(fig, "01_comparison.png")

    # 2. Ensemble size.
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in ("bagging", "random_forest", "extra_trees", "adaboost", "gradient_boosting"):
        curves = pd.DataFrame([{int(k): v for k, v in d["size_curve"].get(key, {}).items()} for d in details])
        if curves.empty:
            continue
        curves = curves[sorted(curves.columns)]
        mean, std = curves.mean(), curves.std()
        ax.plot(mean.index, mean.values, marker="o", label=LEARNERS[key])
        ax.fill_between(mean.index, mean - std, mean + std, alpha=.05)
    ax.set_xscale("log")
    ax.set(xlabel="Number of ensemble members (trees / boosting rounds)", ylabel="Held-out PR-AUC (unsmoothed)",
           title="Effect of ensemble size (mean over 9 held-out videos; band: ±1 std across videos)")
    ax.legend()
    save(fig, "02_ensemble_size.png")

    # 3. CART depth selection.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for d in details:
        scores = {int(k): v for k, v in d["cart_depth_scores"].items()}
        axes[0].plot(list(scores), list(scores.values()), marker="o", alpha=.6)
    axes[0].set(xlabel="max_depth", ylabel="Inner-CV PR-AUC (training videos)", title="CART depth selection per fold")
    chosen = pd.Series([d["cart_depth"] for d in details]).value_counts().sort_index()
    axes[1].bar(chosen.index.astype(str), chosen.values, color=COLORS["PBL-3 ensemble"])
    axes[1].set(xlabel="Selected max_depth", ylabel="Folds", title="Chosen depth across the 9 folds")
    save(fig, "03_cart_depth.png")

    # 4. An interpretable CART tree (fold 1's training videos, chosen depth; top 3 levels drawn).
    d0 = details[0]
    train = dataset[dataset.video_id != d0["held_out"]]
    tree = DecisionTreeClassifier(max_depth=d0["cart_depth"], min_samples_leaf=cfg.ensemble.min_samples_leaf * 4,
                                  random_state=cfg.ml.seed).fit(train[FEATURES].to_numpy(float), train.rel_binary)
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(tree, max_depth=3, feature_names=FEATURES, class_names=["background", "relevant"],
              filled=True, impurity=True, proportion=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title(f"CART tree trained without '{d0['held_out']}' (depth {d0['cart_depth']}; first 3 levels shown)")
    save(fig, "04_cart_tree.png")

    # 5. Diversity of base learners.
    raw_cols = {LEARNERS[k]: f"{REPORTED.get(k, LEARNERS[k])}__raw" for k in LEARNERS}
    raw = pd.DataFrame({name: predictions[col] for name, col in raw_cols.items() if col in predictions})
    corr = raw.rank().corr().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr, vmin=.5, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(raw.columns)), raw.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(raw.columns)), raw.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if corr[i, j] > .8 else "white")
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_title("Base-learner diversity: correlation of held-out probabilities")
    save(fig, "05_diversity.png")

    # 6. Stacking coefficients and BMA weights.
    members = [LEARNERS[m] for m in details[0]["stacking"]["members"]]
    coef = np.array([d["stacking"]["standardized_coef"] for d in details])
    weights = np.array([d["bma"]["weights"] for d in details])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].bar(members, coef.mean(0), yerr=coef.std(0), color=COLORS["PBL-3 ensemble"], capsize=3)
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set(ylabel="Meta-learner coefficient (standardized log-odds)", title="Stacking: logistic meta-learner weights")
    axes[1].bar(members, weights.mean(0), yerr=weights.std(0), color="#6b4fa0", capsize=3)
    axes[1].set(ylabel="Posterior model weight", ylim=(0, 1), title="Bayesian model averaging (pseudo-BMA+) weights")
    for ax in axes:
        ax.tick_params(axis="x", rotation=40)
        plt.setp(ax.get_xticklabels(), ha="right")
    fig.suptitle("Mean ± std over the 9 training folds", y=1.02)
    save(fig, "06_combiner_weights.png")

    # 7. Feature importance.
    imp = {LEARNERS[k]: np.mean([d["importances"][k] for d in details], axis=0)
           for k in ("cart", "bagging", "random_forest", "extra_trees", "adaboost") if k in details[0]["importances"]}
    imp = pd.DataFrame(imp, index=FEATURES)
    imp = imp / imp.sum()
    top = imp.mean(axis=1).sort_values(ascending=False).index[:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(imp.loc[top].to_numpy(), cmap="Greens", aspect="auto")
    ax.set_xticks(range(imp.shape[1]), imp.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(top)), top)
    for i, f in enumerate(top):
        for j, c in enumerate(imp.columns):
            ax.text(j, i, f"{imp.loc[f, c]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Share of impurity decrease")
    ax.set_title("Top-15 features by mean impurity importance (averaged over folds)")
    save(fig, "07_feature_importance.png")
    imp.to_csv(output / "feature_importance.csv")

    # 8. Pooled precision-recall curves.
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in [*ENSEMBLE_METHODS, "Logistic regression", "Fixed-weight fusion"]:
        if method not in predictions:
            continue
        precision, recall, _ = precision_recall_curve(predictions.rel_binary, predictions[method])
        style = "-" if method in ENSEMBLE_METHODS else "--"
        ax.plot(recall, precision, style, linewidth=2.2 if method == best else 1.1, label=method)
    ax.axhline(predictions.rel_binary.mean(), color="grey", linestyle=":", label="Random (prevalence)")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall, all held-out seconds pooled", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "08_pr_curves.png")

    # 9. Paired differences with logistic regression.
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for ax, metric in zip(axes, ["f1", "pr_auc", "hit_at_1", "temporal_iou"]):
        part = paired[paired.metric == metric].set_index("method").loc[list(ENSEMBLE_METHODS)]
        y = np.arange(len(part))
        ax.errorbar(part.mean_diff, y, xerr=[part.mean_diff - part.lo, part.hi - part.mean_diff], fmt="o",
                    color=COLORS["PBL-3 ensemble"], capsize=3)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_yticks(y, part.index)
        ax.set(title=f"Δ {metric} vs logistic regression", xlabel="Mean paired difference")
    axes[0].invert_yaxis()
    fig.suptitle("Per-video paired differences (ensemble − PBL-1 logistic regression); CI crossing 0 = not significant", y=1.03)
    save(fig, "09_paired_differences.png")

    # 10. Per-video F1.
    per = folds.pivot(index="video_id", columns="method", values="f1")
    shown = [m for m in [best, "Random forest", "Logistic regression", "K-means", "Fixed-weight fusion"] if m in per]
    shown = list(dict.fromkeys(shown))
    per = per.loc[per[best].sort_values().index]
    fig, ax = plt.subplots(figsize=(11, 5))
    for m in shown:
        ax.plot(per.index, per[m], marker="o", label=m, linewidth=2.2 if m == best else 1.2)
    ax.set(ylabel="F1 on held-out video", ylim=(0, 1), title="Per-video F1: variation between videos exceeds most differences between models")
    ax.tick_params(axis="x", rotation=30)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.legend(fontsize=8)
    save(fig, "10_per_video_f1.png")

    # 11. Training cost versus quality.
    costs = pd.DataFrame([{"learner": LEARNERS[k], "fit_s": v["fit_s"], "predict_s": v["predict_s"]}
                          for d in details for k, v in d["learners"].items()]).groupby("learner").mean()
    quality = summary.set_index("method").pr_auc
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, row in costs.iterrows():
        q = quality.get(name)
        if q is None:
            continue
        ax.scatter(row.fit_s, q, s=60, color=COLORS["PBL-3 ensemble"])
        ax.annotate(name, (row.fit_s, q), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set(xlabel="Mean fit time on ~25k training rows (s, log scale, 1 CPU core)", ylabel="Held-out PR-AUC",
           title="Training cost versus ranking quality")
    save(fig, "11_cost_vs_quality.png")
    costs.to_csv(output / "training_cost.csv")

    # 12. Confusion matrices.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m in zip(axes, [best, "Logistic regression"]):
        if m not in predictions:
            continue
        cm = confusion_matrix(predictions.rel_binary, predictions[f"{m}__pred"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["background", "relevant"])
        ax.set_yticks([0, 1], ["background", "relevant"])
        ax.set(xlabel="Predicted", ylabel="True", title=m)
    fig.suptitle("Pooled held-out confusion matrices (row-normalized in brackets)", y=1.02)
    save(fig, "12_confusion.png")

    # 13. Example timelines.
    queries = json.loads(cfg.ml.benchmark.read_text())
    per_query = []
    for (video, qid), group in predictions.groupby(["video_id", "query_id"]):
        truth, pred = group.rel_binary.astype(bool), group[f"{best}__pred"].astype(bool)
        union = int((truth | pred).sum())
        per_query.append({"video_id": video, "query_id": int(qid),
                          "iou": float((truth & pred).sum() / union) if union else 1.0})
    per_query = pd.DataFrame(per_query).sort_values(["iou", "query_id"])
    per_query.to_csv(output / "best_ensemble_per_query.csv", index=False)
    chosen = per_query.iloc[[0, len(per_query) // 2, len(per_query) - 1]]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, label, (_, ex) in zip(axes, ["Lowest IoU", "Median IoU", "Highest IoU"], chosen.iterrows()):
        g = predictions[(predictions.video_id == ex.video_id) & (predictions.query_id == ex.query_id)].sort_values("t")
        t = g.t.to_numpy()
        for start, end in _positive_intervals(t, g.rel_binary.to_numpy()):
            ax.axvspan(start, end, color="#cde5cd", alpha=.8)
        ax.plot(t, g["Fixed-weight fusion"], color="#9aa5b1", linewidth=1, label="Fixed-weight fusion")
        ax.plot(t, g["Logistic regression"], color=COLORS["PBL-1 supervised"], linewidth=1.1, label="Logistic regression")
        ax.plot(t, g[best], color=COLORS["PBL-3 ensemble"], linewidth=1.6, label=best)
        ax.plot(t, g[f"{best}__pred"] * 1.02, color="#b8322a", linestyle=":", label=f"{best} decision")
        ax.set(ylim=(0, 1.08), xlabel="Time (s)", ylabel="Score",
               title=f"{label}: IoU={ex.iou:.2f} | {ex.video_id} | “{queries[int(ex.query_id)]['query']}”")
    axes[0].legend(loc="upper right", fontsize=8, ncol=4)
    fig.suptitle("Green: true answer interval. Examples chosen after evaluation by IoU rank", y=1.01)
    save(fig, "13_example_timelines.png")

    _write_report(dataset, summary, paired, details, best, cfg)


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) if isinstance(v, str) else f"{v:.3f}" for v in row) + " |")
    return "\n".join(lines)


def _write_report(dataset: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame,
                  details: list[dict[str, Any]], best: str, cfg: Any) -> None:
    output = cfg.ensemble.output_dir
    ens = cfg.ensemble
    cols = ["method", "precision", "recall", "f1", "pr_auc", "roc_auc", "hit_at_1", "mrr", "temporal_iou"]
    s = summary.sort_values("f1", ascending=False)
    ens_rows = s[s.method.isin(ENSEMBLE_METHODS)]
    prev_rows = s[~s.method.isin(ENSEMBLE_METHODS)]
    by = summary.set_index("method")
    pf1 = paired[paired.metric == "f1"].set_index("method")
    ppr = paired[paired.metric == "pr_auc"].set_index("method")
    depths = pd.Series([d["cart_depth"] for d in details]).value_counts().to_dict()
    bma = np.mean([d["bma"]["weights"] for d in details], axis=0)
    members = [LEARNERS[m] for m in details[0]["stacking"]["members"]]
    windows = pd.DataFrame([{"method": m, **v} for d in details for m, v in d["methods"].items()])
    win = windows.groupby("method").smooth_window.agg(lambda x: x.mode().iloc[0])
    report = f"""# Review 3 (PBL-3): Ensemble Learning for Video Moment Retrieval

Generated by `scripts/run_ensemble.py` from `analysis/ensemble.py`. All numbers are
held-out results: each video is predicted by models that never saw it.

## 1. Problem identification and dataset selection

Task: for each (query, second of video) decide whether that second belongs to the
answer interval (`rel_binary`). The dataset is the Review 1 feature table
`results/ml/dataset.csv.gz`: {len(dataset):,} rows, {dataset.video_id.nunique()} NASA videos,
{dataset.query_id.nunique()} queries, {len(FEATURES)} engineered features from frozen CLIP / Whisper /
MiniLM similarities, and {int(dataset.rel_binary.sum()):,} positive rows
({dataset.rel_binary.mean():.1%}). Ensembles suit this task: the features interact
non-linearly (distance to peak × score level × speech presence), and Review 1 showed
that a single model's errors vary strongly between videos, which averaging can reduce.

Protocol: leave-one-video-out (9 folds), grouped {ens.inner_folds}-fold inner CV over training
videos for CART depth, stacking, BMA weights, output smoothing and the F1 threshold.
No class re-weighting is used, so every learner outputs calibrated-style probabilities;
the class imbalance is handled by the training-selected F1 threshold.

## 2. Implemented ensemble approaches

| Method | Type | Configuration |
|---|---|---|
| CART tree | single tree (reference) | Gini; depth from {list(ens.cart_depths)} by inner-CV PR-AUC (chosen: {depths}); min leaf {ens.min_samples_leaf * 4} |
| Bagging | parallel, bootstrap | {ens.n_estimators} unpruned CART trees, bootstrap samples |
| Random forest | parallel, bootstrap + feature subsampling | {ens.n_estimators} trees, √p features per split, min leaf {ens.min_samples_leaf} |
| Extra trees | parallel, randomized splits | {ens.n_estimators} trees, random thresholds, √p features |
| AdaBoost | sequential boosting (SAMME) | {ens.adaboost['n_estimators']} depth-{ens.adaboost['max_depth']} trees, learning rate {ens.adaboost['learning_rate']} |
| Hist. gradient boosting | sequential boosting (log-loss gradient) | {ens.gradient_boosting} |
| Soft voting | heterogeneous averaging | mean probability of {', '.join(LEARNERS[m] for m in ens.voting_members)} |
| Stacking | heterogeneous, learned combiner | logistic meta-learner on out-of-fold log-odds of {len(members)} base learners |
| Bayesian model averaging | heterogeneous, probabilistic | pseudo-BMA+ weights from out-of-fold log predictive density, Bayesian bootstrap over queries ({ens.bma_bootstrap} draws) |

Mean BMA weights: {', '.join(f'{m} {w:.2f}' for m, w in zip(members, bma) if w >= 0.01)}.
Most common smoothing width per method: {', '.join(f'{m} {int(w)} s' for m, w in win.items())}.

## 3. Comparative performance analysis

Macro mean over the 9 held-out videos (same metric code and folds as Review 2).

**Ensembles (PBL-3)**

{_table(ens_rows, cols)}

**Earlier reviews and baselines**

{_table(prev_rows, cols)}

Paired differences against PBL-1 logistic regression (mean over videos, 95% video-bootstrap CI):

| Ensemble | ΔF1 | 95% CI | videos better | ΔPR-AUC | 95% CI |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {m} | {pf1.loc[m, 'mean_diff']:+.3f} | [{pf1.loc[m, 'lo']:+.3f}, {pf1.loc[m, 'hi']:+.3f}] | "
        f"{int(pf1.loc[m, 'videos_better'])}/9 | {ppr.loc[m, 'mean_diff']:+.3f} | [{ppr.loc[m, 'lo']:+.3f}, {ppr.loc[m, 'hi']:+.3f}] |"
        for m in ENSEMBLE_METHODS) + f"""

## 4. Result interpretation

* Best ensemble by mean F1: **{best}** (F1 {by.loc[best, 'f1']:.3f}, PR-AUC {by.loc[best, 'pr_auc']:.3f},
  Hit@1 {by.loc[best, 'hit_at_1']:.3f}, temporal IoU {by.loc[best, 'temporal_iou']:.3f}).
* Single CART tree: F1 {by.loc['CART tree', 'f1']:.3f}, PR-AUC {by.loc['CART tree', 'pr_auc']:.3f}. Bagging/forests
  raise PR-AUC to {by.loc['Random forest', 'pr_auc']:.3f} (random forest) by averaging away tree variance.
* Figures in `results/ensemble/figures/`: comparison (01), ensemble size (02), CART depth (03),
  CART tree (04), diversity (05), combiner weights (06), feature importance (07), PR curves (08),
  paired differences (09), per-video F1 (10), cost vs quality (11), confusion (12), example timelines (13).
"""
    (output / "ENSEMBLE_REPORT.md").write_text(report)
