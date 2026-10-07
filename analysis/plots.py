"""All charts for the regression / classification analysis (matplotlib, PNG).

Colour rules: the three models of a task take fixed categorical slots (blue, orange,
aqua; validated for colour-blind separation) with a distinct marker/line style each, so
identity is never colour alone. Baselines are neutral grey. One axis per chart; where
two measures differ in scale they get their own panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from analysis.benchmark import load_benchmark
from analysis.features import FEATURES
from analysis.models import CLASSIFIERS, REGRESSORS

SURFACE = "#fcfcfb"
INK = "#141821"
INK_SOFT = "#52514e"
INK_FAINT = "#8b93a1"
GRID = "#e4e6ea"
BASELINE = "#a9afba"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
MARKERS = ["o", "s", "^"]
LINESTYLES = ["-", "--", "-."]
POSITIVE = "#2a78d6"

MODEL_STYLE = {
    "regression": {key: (SLOTS[i], MARKERS[i], LINESTYLES[i]) for i, key in enumerate(REGRESSORS)},
    "classification": {key: (SLOTS[i], MARKERS[i], LINESTYLES[i]) for i, key in enumerate(CLASSIFIERS)},
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SOFT,
            "axes.edgecolor": GRID,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def _sequential(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("seq", [SURFACE, color])


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _label(task: str, key: str) -> str:
    return (REGRESSORS if task == "regression" else CLASSIFIERS)[key]


# --------------------------------------------------------------------------- data


def fig_class_balance(dataset: pd.DataFrame, out: Path, dpi: int) -> None:
    """Share of seconds that fall inside the true moment, per video."""
    rate = dataset.groupby("video_id")["rel_binary"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(rate.index, rate.values * 100, color=POSITIVE, height=0.62)
    for bar, value in zip(bars, rate.values):
        ax.text(value * 100 + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", color=INK_SOFT)
    overall = dataset["rel_binary"].mean() * 100
    ax.axvline(overall, color=INK_FAINT, linestyle="--", linewidth=1)
    ax.text(overall + 0.2, -0.75, f"overall {overall:.1f}%", color=INK_FAINT, va="top")
    ax.set_xlabel("Seconds inside the true moment (%)")
    ax.set_title("Positive class is rare: about 1 second in 12")
    ax.grid(axis="y", visible=False)
    _save(fig, out / "01_class_balance.png", dpi)


def fig_feature_distributions(dataset: pd.DataFrame, out: Path, dpi: int) -> None:
    """Feature histograms for seconds outside vs inside the true moment."""
    columns = ["visual_norm", "speech_norm", "fused_baseline", "visual_z", "speech_z", "frame_change"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6))
    for ax, column in zip(axes.ravel(), columns):
        values = dataset[column]
        bins = np.linspace(values.quantile(0.005), values.quantile(0.995), 40)
        for label, flag, color in (("outside", 0, BASELINE), ("inside", 1, POSITIVE)):
            ax.hist(values[dataset["rel_binary"] == flag], bins=bins, density=True, color=color, alpha=0.75, label=label)
        ax.set_title(column)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0, 0].legend(title="second is", title_fontsize=9)
    fig.suptitle("Seconds inside the true moment score higher on the matching features", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "02_feature_distributions.png", dpi)


def fig_correlation(dataset: pd.DataFrame, out: Path, dpi: int) -> None:
    """Pearson correlation between features and with both targets."""
    columns = FEATURES + ["rel_graded", "rel_binary"]
    corr = dataset[columns].corr()
    cmap = LinearSegmentedColormap.from_list("div", [SLOTS[1], "#f4f4f2", SLOTS[0]])
    fig, ax = plt.subplots(figsize=(9, 7.6))
    image = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=60, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    ax.grid(False)
    for i in range(len(columns)):
        for j in range(len(columns)):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.65 else INK_SOFT)
    fig.colorbar(image, ax=ax, shrink=0.7, label="Pearson correlation")
    ax.set_title("Which features move together, and with the labels")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, out / "03_correlation_heatmap.png", dpi)


def fig_example_traces(dataset: pd.DataFrame, oof: pd.DataFrame, texts: dict[int, str], out: Path, dpi: int) -> None:
    """Two held-out queries: picture score, speech score and the classifier's probability over time."""
    wanted = ["astronauts with Elmo", "Orion parachute test dropped from an airplane"]
    ids = [qid for text in wanted for qid, t in texts.items() if t == text]
    frame = dataset.merge(oof[["query_id", "t", "cls_gradient_boosting"]], on=["query_id", "t"])
    fig, axes = plt.subplots(len(ids), 1, figsize=(10, 3.6 * len(ids)), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, qid in zip(axes, ids):
        part = frame[frame["query_id"] == qid].sort_values("t")
        inside = part["rel_binary"].to_numpy() == 1
        ax.fill_between(part["t"], 0, 1, where=inside, color=INK, alpha=0.08, step="mid", label="true moment")
        ax.plot(part["t"], part["visual_norm"], color=SLOTS[0], linewidth=1.4, label="picture score")
        ax.plot(part["t"], part["speech_norm"], color=SLOTS[1], linewidth=1.4, linestyle="--", label="speech score")
        ax.plot(part["t"], part["cls_gradient_boosting"], color=SLOTS[2], linewidth=2, linestyle="-.", label="classifier probability")
        ax.set_title(f"“{texts[qid]}”  ({part['video_id'].iat[0]})")
        ax.set_ylabel("score (0 to 1)")
        ax.set_ylim(0, 1.02)
    axes[-1].set_xlabel("Time in video (s)")
    axes[0].legend(ncol=4, loc="upper right")
    _save(fig, out / "04_example_traces.png", dpi)


# ------------------------------------------------------------------ regression


def _bar_panel(ax: plt.Axes, summary: pd.DataFrame, metric: str, task: str, lower_better: bool) -> None:
    order = ["dummy", *(REGRESSORS if task == "regression" else CLASSIFIERS)]
    rows = summary.set_index("model").loc[order]
    colors = [BASELINE] + [MODEL_STYLE[task][k][0] for k in order[1:]]
    labels = ["Mean\nbaseline" if task == "regression" else "Prior\nbaseline"] + [
        _label(task, k).replace(" ", "\n", 1) for k in order[1:]
    ]
    values, errors = rows[f"{metric}_mean"], rows[f"{metric}_std"].fillna(0)
    bars = ax.bar(range(len(order)), values, yerr=errors, color=colors, width=0.62,
                  error_kw={"ecolor": INK_FAINT, "capsize": 3, "linewidth": 1})
    span = float(np.nanmax(values + errors) - min(0.0, float(np.nanmin(values - errors))))
    for bar, value, err in zip(bars, values, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, max(value, 0) + err + span * 0.02, f"{value:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(order)), labels, fontsize=8)
    ax.set_title(f"{metric.upper() if metric in ('rmse', 'mae') else metric.replace('_', ' ').title()}"
                 f"  ({'lower' if lower_better else 'higher'} is better)", fontsize=10)
    ax.grid(axis="x", visible=False)


def fig_reg_metrics(summary: pd.DataFrame, out: Path, dpi: int) -> None:
    """Mean ± std over the nine held-out videos for each regression model."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    for ax, (metric, lower) in zip(axes, [("rmse", True), ("mae", True), ("r2", False), ("spearman", False)]):
        _bar_panel(ax, summary, metric, "regression", lower)
    fig.suptitle("Regression: predicting how relevant each second is (error bars: spread across held-out videos)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "05_reg_metrics.png", dpi)


def fig_pred_vs_actual(oof: pd.DataFrame, out: Path, dpi: int) -> None:
    """Out-of-fold predicted vs true graded relevance."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)
    for ax, key in zip(axes, REGRESSORS):
        color = MODEL_STYLE["regression"][key][0]
        hexes = ax.hexbin(oof["rel_graded"], oof[f"reg_{key}"], gridsize=32, bins="log", cmap=_sequential(color),
                          mincnt=1, extent=(0, 1, -0.1, 1.1))
        ax.plot([0, 1], [0, 1], color=INK_FAINT, linestyle="--", linewidth=1)
        ax.set_title(_label("regression", key))
        ax.set_xlabel("True relevance")
        fig.colorbar(hexes, ax=ax, label="seconds (log)", shrink=0.8)
    axes[0].set_ylabel("Predicted relevance")
    fig.suptitle("Predicted vs true relevance (dashed: perfect prediction)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "06_reg_pred_vs_actual.png", dpi)


def fig_residuals(oof: pd.DataFrame, out: Path, dpi: int) -> None:
    """Residual histograms and residual-vs-prediction plots."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, key in enumerate(REGRESSORS):
        color = MODEL_STYLE["regression"][key][0]
        pred, residual = oof[f"reg_{key}"], oof["rel_graded"] - oof[f"reg_{key}"]
        axes[0, col].hist(residual, bins=60, color=color, alpha=0.85)
        axes[0, col].axvline(0, color=INK_FAINT, linestyle="--", linewidth=1)
        axes[0, col].set_title(f"{_label('regression', key)}: residuals")
        axes[0, col].set_xlabel("True − predicted")
        axes[0, col].set_yscale("log")
        axes[1, col].hexbin(pred, residual, gridsize=32, bins="log", cmap=_sequential(color), mincnt=1)
        axes[1, col].axhline(0, color=INK_FAINT, linestyle="--", linewidth=1)
        axes[1, col].set_xlabel("Predicted relevance")
        axes[1, col].set_title("Residual vs prediction")
    axes[0, 0].set_ylabel("Seconds (log)")
    axes[1, 0].set_ylabel("True − predicted")
    fig.tight_layout()
    _save(fig, out / "07_reg_residuals.png", dpi)


def _importance_figure(importance: pd.DataFrame, task: str, title: str, path: Path, dpi: int) -> None:
    models = REGRESSORS if task == "regression" else CLASSIFIERS
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, key in zip(axes, models):
        part = importance[(importance["task"] == task) & (importance["model"] == key)]
        stats = part.groupby("feature")["importance"].agg(["mean", "std"]).sort_values("mean").tail(8)
        ax.barh(stats.index, stats["mean"], xerr=stats["std"].fillna(0), color=MODEL_STYLE[task][key][0],
                height=0.62, error_kw={"ecolor": INK_FAINT, "capsize": 2, "linewidth": 1})
        ax.set_title(_label(task, key))
        ax.set_xlabel("Drop in score when shuffled")
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, path, dpi)


def fig_learning_curves(curve: pd.DataFrame, out: Path, dpi: int) -> None:
    """Held-out score as the number of training videos grows."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, task, ylabel, title in (
        (axes[0], "regression", "Held-out RMSE (lower is better)", "Regression"),
        (axes[1], "classification", "Held-out PR-AUC (higher is better)", "Classification"),
    ):
        models = REGRESSORS if task == "regression" else CLASSIFIERS
        for key in models:
            color, marker, line = MODEL_STYLE[task][key]
            part = curve[(curve["task"] == task) & (curve["model"] == key)].groupby("n_train_videos")["score"]
            mean, std = part.mean(), part.std().fillna(0)
            ax.plot(mean.index, mean.values, color=color, marker=marker, linestyle=line, linewidth=2, label=_label(task, key))
            ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.set_xlabel("Training videos")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(sorted(curve["n_train_videos"].unique()))
        ax.legend()
    fig.suptitle("Gains flatten after a few training videos; spread across held-out videos stays wide", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "09_learning_curves.png", dpi)


# -------------------------------------------------------------- classification


def fig_confusion(oof: pd.DataFrame, out: Path, dpi: int) -> None:
    """Row-normalised confusion matrices at each model's tuned threshold."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, key in zip(axes, CLASSIFIERS):
        color = MODEL_STYLE["classification"][key][0]
        matrix = confusion_matrix(oof["rel_binary"], oof[f"cls_{key}_pred"])
        share = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=_sequential(color), vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{share[i, j]:.0%}\n({matrix[i, j]:,})", ha="center", va="center",
                        color="white" if share[i, j] > 0.55 else INK, fontsize=10)
        ax.set_xticks([0, 1], ["outside", "inside"])
        ax.set_yticks([0, 1], ["outside", "inside"])
        ax.set_xlabel("Predicted")
        ax.set_title(_label("classification", key))
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("True")
    fig.suptitle("Confusion matrices (share of each true class; counts in brackets)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "10_cls_confusion.png", dpi)


def fig_roc_pr(oof: pd.DataFrame, out: Path, dpi: int) -> None:
    """Overlaid ROC and precision-recall curves."""
    y = oof["rel_binary"].to_numpy()
    for name, filename, title in (("roc", "11_cls_roc.png", "ROC curves"), ("pr", "12_cls_pr.png", "Precision-recall curves")):
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        for key in CLASSIFIERS:
            color, _, line = MODEL_STYLE["classification"][key]
            score = oof[f"cls_{key}"].to_numpy()
            if name == "roc":
                x, yy, _ = roc_curve(y, score)
                area = auc(x, yy)
            else:
                yy, x, _ = precision_recall_curve(y, score)
                area = auc(x, yy)
            ax.plot(x, yy, color=color, linestyle=line, linewidth=2, label=f"{_label('classification', key)} (AUC {area:.2f})")
        if name == "roc":
            ax.plot([0, 1], [0, 1], color=BASELINE, linestyle=":", label="Chance")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
        else:
            ax.axhline(y.mean(), color=BASELINE, linestyle=":", label=f"No skill ({y.mean():.2f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.legend(loc="lower right" if name == "roc" else "upper right")
        _save(fig, out / filename, dpi)


def fig_cls_metrics(summary: pd.DataFrame, out: Path, dpi: int) -> None:
    """Grouped bars: each metric, one bar per model, mean over held-out videos."""
    metrics = ["precision", "recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc"]
    names = ["Precision", "Recall", "F1", "Balanced\naccuracy", "ROC-AUC", "PR-AUC"]
    order = ["dummy", *CLASSIFIERS]
    rows = summary.set_index("model").loc[order]
    width = 0.2
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for i, key in enumerate(order):
        color = BASELINE if key == "dummy" else MODEL_STYLE["classification"][key][0]
        label = "Prior baseline" if key == "dummy" else _label("classification", key)
        positions = np.arange(len(metrics)) + (i - 1.5) * width
        values = [rows.loc[key, f"{m}_mean"] for m in metrics]
        errors = [rows.loc[key, f"{m}_std"] if not np.isnan(rows.loc[key, f"{m}_std"]) else 0 for m in metrics]
        ax.bar(positions, values, width=width * 0.9, color=color, label=label, yerr=errors,
               error_kw={"ecolor": INK_FAINT, "capsize": 2, "linewidth": 1})
        for pos, value, err in zip(positions, values, errors):
            ax.text(pos, value + err + 0.012, f"{value:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(np.arange(len(metrics)), names)
    ax.set_ylim(0, 1.12)
    ax.set_title("Classification: is this second inside the true moment? (mean over held-out videos)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    _save(fig, out / "13_cls_metrics.png", dpi)


def fig_threshold(oof: pd.DataFrame, thresholds: pd.DataFrame, out: Path, dpi: int) -> None:
    """F1 as the probability threshold moves, with each model's chosen threshold."""
    from sklearn.metrics import f1_score

    y = oof["rel_binary"].to_numpy()
    grid = np.linspace(0.02, 0.98, 49)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key in CLASSIFIERS:
        color, marker, line = MODEL_STYLE["classification"][key]
        scores = [f1_score(y, (oof[f"cls_{key}"] >= t).astype(int), zero_division=0) for t in grid]
        ax.plot(grid, scores, color=color, linestyle=line, linewidth=2, label=_label("classification", key))
        chosen = thresholds[thresholds["model"] == key]["threshold"].mean()
        ax.scatter([chosen], [f1_score(y, (oof[f"cls_{key}"] >= chosen).astype(int), zero_division=0)],
                   color=color, marker=marker, s=60, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("F1 (pooled held-out predictions)")
    ax.set_title("F1 vs decision threshold (markers: average tuned threshold)")
    ax.legend()
    _save(fig, out / "15_cls_threshold_f1.png", dpi)


# --------------------------------------------------------------------- project


def fig_retrieval(retrieval: pd.DataFrame, ks: tuple[int, ...], out: Path, dpi: int) -> None:
    """How often the best-ranked seconds land inside the true moment, by method."""
    order = ["Random", "Picture only", "Speech only", "Fixed-weight fusion"] + [
        f"Regression: {v}" for v in REGRESSORS.values()] + [f"Classification: {v}" for v in CLASSIFIERS.values()]
    frame = retrieval.set_index("method").loc[order]
    colors = [BASELINE] * 4 + [SLOTS[0]] * 3 + [SLOTS[1]] * 3
    metrics = [(f"hit_at_{k}", f"Hit@{k}") for k in ks] + [("mrr", "Mean reciprocal rank")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 5), sharey=True)
    for ax, (column, title) in zip(np.atleast_1d(axes), metrics):
        values = frame[column].to_numpy()
        low, high = frame[f"{column}_lo"].to_numpy(), frame[f"{column}_hi"].to_numpy()
        bars = ax.barh(range(len(order)), values, color=colors, height=0.68,
                       xerr=[values - low, high - values], error_kw={"ecolor": INK_FAINT, "capsize": 2, "linewidth": 1})
        for bar, value, top in zip(bars, values, high):
            ax.text(top + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8.5)
        ax.set_yticks(range(len(order)), order)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Does a learned ranker beat hand-set fusion? Grey: baselines, blue: regression, orange: classification (whiskers: 95% bootstrap over queries)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "16_retrieval_comparison.png", dpi)


def fig_alpha_sweep(sweep: pd.DataFrame, retrieval: pd.DataFrame, baseline_alpha: float, out: Path, dpi: int) -> None:
    """Hand-set weight sweep, with the best learned models as reference lines."""
    reg = retrieval[retrieval["method"].str.startswith("Regression")].sort_values("hit_at_1").iloc[-1]
    cls = retrieval[retrieval["method"].str.startswith("Classification")].sort_values("hit_at_1").iloc[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (column, title) in zip(axes, [("hit_at_1", "Hit@1"), ("mrr", "Mean reciprocal rank")]):
        ax.plot(sweep["alpha"], sweep[column], color=INK, marker="o", linewidth=2, label="Hand-set fusion")
        ax.axhline(reg[column], color=SLOTS[0], linestyle="--", linewidth=1.8, label=f"Best regression ({reg['method'].split(': ')[1]})")
        ax.axhline(cls[column], color=SLOTS[1], linestyle="-.", linewidth=1.8, label=f"Best classifier ({cls['method'].split(': ')[1]})")
        ax.axvline(baseline_alpha, color=INK_FAINT, linestyle=":", linewidth=1)
        ax.set_xlabel("α: weight on picture (1 − α on speech)")
        ax.set_title(title)
        ax.set_ylim(0.5, 1.0)
    axes[0].legend(loc="lower left")
    fig.suptitle("Fusion weight sweep vs learned rankers (dotted line: default α)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "17_alpha_sweep.png", dpi)


def fig_per_video(per_video: pd.DataFrame, out: Path, dpi: int) -> None:
    """Hit@1 per video and method as an annotated heatmap."""
    table = per_video.pivot(index="method", columns="video_id", values="hit_at_1")
    table = table.loc[["Picture only", "Speech only", "Fixed-weight fusion",
                       "Regression: Gradient boosting", "Classification: Gradient boosting"]]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.imshow(table.values, cmap=_sequential(SLOTS[0]), vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.6 else INK, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Hit@1 by video: where each method works")
    _save(fig, out / "18_per_video_hit1.png", dpi)


# ------------------------------------------------------------------------ entry


def make_all_plots(cfg: object) -> None:
    """Render every figure from the saved tables."""
    _style()
    tables, figures, dpi = Path(cfg.ml.tables_dir), Path(cfg.ml.figures_dir), cfg.evaluation.plot_dpi
    dataset = pd.read_csv(tables / "dataset.csv.gz")
    oof = pd.read_csv(tables / "oof_predictions.csv.gz")
    texts = {q.query_id: q.query for q in load_benchmark(cfg.ml.benchmark)}
    importance = pd.read_csv(tables / "feature_importance_folds.csv")
    ks = cfg.ml.retrieval_ks
    retrieval = pd.read_csv(tables / "retrieval.csv")

    fig_class_balance(dataset, figures, dpi)
    fig_feature_distributions(dataset, figures, dpi)
    fig_correlation(dataset, figures, dpi)
    fig_example_traces(dataset, oof, texts, figures, dpi)
    fig_reg_metrics(pd.read_csv(tables / "regression_summary.csv"), figures, dpi)
    fig_pred_vs_actual(oof, figures, dpi)
    fig_residuals(oof, figures, dpi)
    _importance_figure(importance, "regression", "Regression: which features the models rely on (permutation importance, held-out videos)",
                       figures / "08_reg_importance.png", dpi)
    fig_learning_curves(pd.read_csv(tables / "learning_curve.csv"), figures, dpi)
    fig_confusion(oof, figures, dpi)
    fig_roc_pr(oof, figures, dpi)
    fig_cls_metrics(pd.read_csv(tables / "classification_summary.csv"), figures, dpi)
    _importance_figure(importance, "classification", "Classification: which features the models rely on (permutation importance, held-out videos)",
                       figures / "14_cls_importance.png", dpi)
    fig_threshold(oof, pd.read_csv(tables / "thresholds.csv"), figures, dpi)
    fig_retrieval(retrieval, ks, figures, dpi)
    fig_alpha_sweep(pd.read_csv(tables / "alpha_sweep.csv"), retrieval, cfg.ml.baseline_alpha, figures, dpi)
    fig_per_video(pd.read_csv(tables / "retrieval_per_video.csv"), figures, dpi)
    print(f"Saved figures to {figures}")
