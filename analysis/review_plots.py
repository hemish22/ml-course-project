"""Standalone figures and a readable review report from measured predictions."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from analysis.review import NEW_METHODS


def make_review_artifacts(
    dataset: pd.DataFrame, predictions: pd.DataFrame, summary: pd.DataFrame,
    details: list[dict[str, Any]], cfg: Any,
) -> None:
    """Draw dataset, clustering, HMM, comparative, and example-timeline figures."""
    output = cfg.review.output_dir
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(figures / name, dpi=cfg.evaluation.plot_dpi, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = dataset.groupby("video_id").size().sort_values()
    axes[0].barh(counts.index, counts.values, color="#245d91")
    axes[0].set(xlabel="Query × second rows", title="Benchmark coverage (repeated video frames across queries)")
    positive = int(dataset.rel_binary.sum())
    axes[1].bar(["Background", "Relevant"], [len(dataset) - positive, positive], color=["#8392a6", "#e6a23b"])
    axes[1].set(ylabel="Rows", title=f"Relevant: {positive / len(dataset):.1%}; accuracy alone is misleading")
    save(fig, "01_dataset.png")

    candidates = pd.read_csv(output / "model_selection.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, method in zip(axes, NEW_METHODS):
        metric = "silhouette" if method == "K-means" else "bic"
        for video, group in candidates[candidates.method == method].groupby("held_out"):
            ax.plot(group.k, group[metric], marker="o", alpha=0.65)
        ax.set(title=f"{method}: training {metric}", xlabel="Clusters / hidden states", ylabel=metric)
        ax.set_xticks(sorted(candidates[candidates.method == method].k.unique()))
    fig.suptitle("Model selection uses only training videos; each line is one held-out fold", y=1.04)
    save(fig, "02_model_selection.png")

    detail = details[0]
    features = list(cfg.review.features)
    train = dataset[dataset.video_id != detail["held_out"]]
    test = dataset[dataset.video_id == detail["held_out"]]
    mean, scale = np.array(detail["scaler_mean"]), np.array(detail["scaler_scale"])
    X, Xt = (train[features].to_numpy() - mean) / scale, (test[features].to_numpy() - mean) / scale
    with threadpool_limits(limits=1):
        pca = PCA(n_components=2, random_state=cfg.ml.seed).fit(X)
        projected = pca.transform(Xt)
    centers = np.array(detail["models"]["K-means"]["means"])
    labels = ((Xt[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, colors, title in zip(axes, [labels, test.rel_binary.to_numpy()], ["Predicted K-means cluster", "Ground truth (evaluation only)"]):
        scatter = ax.scatter(projected[:, 0], projected[:, 1], c=colors, s=9, alpha=0.45, cmap="viridis")
        fig.colorbar(scatter, ax=ax)
        ax.set(xlabel="Principal component 1", ylabel="Principal component 2", title=title)
    fig.suptitle(f"Held-out video: {detail['held_out']}; PCA fitted on training videos only", y=1.04)
    save(fig, "03_cluster_projection.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    km = detail["models"]["K-means"]
    image = axes[0].imshow(centers, cmap="RdBu_r", aspect="auto")
    axes[0].set_xticks(range(len(features)), features, rotation=65, ha="right")
    axes[0].set(ylabel="Cluster", title="K-means centers (standardized features)")
    fig.colorbar(image, ax=axes[0])
    axes[1].bar(range(km["k"]), km["state_relevance"], color="#245d91")
    axes[1].set(xlabel="Cluster", ylabel="Training relevant fraction", title="Training-only interpretation of clusters", ylim=(0, 1))
    save(fig, "04_cluster_interpretation.png")

    hmm = detail["models"]["Gaussian HMM"]
    order = np.argsort(hmm["state_relevance"])
    trans = np.array(hmm["transmat"])[np.ix_(order, order)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    image = axes[0].imshow(trans, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(trans)):
        for j in range(len(trans)):
            axes[0].text(j, i, f"{trans[i, j]:.3f}", ha="center", color="white" if trans[i, j] > .6 else "black")
    axes[0].set_xticks(range(len(order)), order)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set(xlabel="Next state", ylabel="Current state", title="Learned transitions")
    fig.colorbar(image, ax=axes[0])
    image = axes[1].imshow(np.array(hmm["means"])[order], aspect="auto", cmap="RdBu_r")
    axes[1].set_xticks(range(len(features)), features, rotation=65, ha="right")
    axes[1].set_yticks(range(len(order)), order)
    axes[1].set_title("Gaussian emission means (standardized)")
    fig.colorbar(image, ax=axes[1])
    axes[2].bar(range(len(order)), np.array(hmm["state_relevance"])[order], color="#b76a16")
    axes[2].set_xticks(range(len(order)), order)
    axes[2].set(title="State relevance from training labels", xlabel="State", ylim=(0, 1))
    save(fig, "05_hmm_parameters.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    for d in details:
        history = d["models"]["Gaussian HMM"]["log_likelihood_history"]
        ax.plot(np.arange(1, len(history) + 1), np.array(history) / d["training_rows"], label=d["held_out"])
    ax.set(xlabel="Baum-Welch iteration", ylabel="Training log likelihood per row", title="HMM fitting history (selected model per fold)")
    ax.legend(fontsize=7, loc="best")
    save(fig, "06_hmm_convergence.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = ["#245d91" if method in NEW_METHODS else "#99a5b3" for method in summary.method]
    for ax, metric, title in zip(axes, ["f1", "hit_at_1", "temporal_iou"], ["F1", "Hit@1", "Temporal IoU"]):
        values = summary[metric].to_numpy()
        errors = np.vstack([values - summary[f"{metric}_lo"], summary[f"{metric}_hi"] - values])
        ax.barh(summary.method, values, xerr=np.maximum(errors, 0), color=colors, capsize=3)
        ax.set(xlim=(0, 1), xlabel=title, title=title)
        ax.invert_yaxis() if ax is axes[0] else None
    fig.suptitle("Mean over held-out videos; error bars: video-bootstrap 95% intervals", y=1.02)
    save(fig, "07_comparison.png")

    examples = []
    for (video, qid), group in predictions.groupby(["video_id", "query_id"]):
        truth, predicted = group.rel_binary.astype(bool), group["Gaussian HMM__pred"].astype(bool)
        union = int((truth | predicted).sum())
        examples.append({"video_id": video, "query_id": int(qid), "temporal_iou": float((truth & predicted).sum() / union) if union else 1.0})
    examples = pd.DataFrame(examples).sort_values(["temporal_iou", "query_id"])
    examples.to_csv(output / "hmm_per_query.csv", index=False)
    chosen = examples.iloc[[0, len(examples) // 2, len(examples) - 1]].copy()
    chosen["selection"] = ["Lowest IoU", "Middle IoU", "Highest IoU"]
    chosen.to_csv(output / "examples.csv", index=False)
    queries = json.loads(cfg.ml.benchmark.read_text())
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, (_, example) in zip(axes, chosen.iterrows()):
        group = predictions[(predictions.video_id == example.video_id) & (predictions.query_id == example.query_id)].sort_values("t")
        t = group.t.to_numpy()
        ax.fill_between(t, 0, 1, where=group.rel_binary.astype(bool), color="#cde5cd", alpha=.7, step="post", label="True interval")
        for method in ["Fixed-weight fusion", "K-means", "Gaussian HMM"]:
            ax.plot(t, group[method], label=method, linewidth=1.2)
        ax.plot(t, group["Gaussian HMM__pred"] * .98, color="#b76a16", alpha=.7, linestyle=":", label="HMM relevance decision")
        ax.set(ylim=(0, 1.05), xlabel="Time (s)", ylabel="Score",
               title=f"{example.selection}: IoU={example.temporal_iou:.2f} | {example.video_id} | {queries[int(example.query_id)]['query']}")
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    fig.suptitle("Post-hoc examples selected by held-out IoU; summary metrics include all 86 queries", y=1.01)
    save(fig, "08_example_timelines.png")

    example = chosen.iloc[1]
    group = predictions[(predictions.video_id == example.video_id) & (predictions.query_id == example.query_id)].sort_values("t")
    fold_detail = next(d for d in details if d["held_out"] == example.video_id)
    threshold = fold_detail["models"]["Gaussian HMM"]["threshold"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    t = group.t.to_numpy()
    axes[0].plot(t, group["Picture only"], label="Picture similarity")
    axes[0].plot(t, group["Speech only"], label="Speech similarity")
    axes[0].set(ylabel="Observed scores", ylim=(0, 1.05))
    axes[0].legend(loc="upper right")
    axes[1].plot(t, group["Gaussian HMM"], label="Posterior-derived relevance")
    axes[1].axhline(threshold, linestyle="--", color="#b76a16", label="Training-selected cutoff")
    axes[1].set(ylabel="Relevance score")
    axes[1].legend(loc="upper right")
    axes[2].step(t, group["Gaussian HMM__state"], where="post", color="#245d91", label="Viterbi path")
    axes[2].set(xlabel="Time (s)", ylabel="Hidden state ID")
    axes[2].set_yticks(range(fold_detail["models"]["Gaussian HMM"]["k"]))
    axes[2].legend(loc="upper right")
    for ax in axes:
        for start, end in _positive_intervals(t, group.rel_binary.to_numpy()):
            ax.axvspan(start, end, color="#cde5cd", alpha=.4)
    fig.suptitle(f"HMM observations, relevance and joint state path | {example.video_id}\nGreen bands: true answer; state IDs are arbitrary and differ from relevance labels", y=1.02)
    save(fig, "09_viterbi_decoding.png")

    _write_report(dataset, summary, details, cfg)


def _positive_intervals(t: np.ndarray, binary: np.ndarray) -> list[tuple[float, float]]:
    """Convert positive per-second rows into intervals for the benchmark plots."""
    padded = np.r_[0, binary.astype(int), 0]
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    return [(float(t[a]), float(t[b - 1] + step)) for a, b in zip(starts, ends)]


def _write_report(dataset: pd.DataFrame, summary: pd.DataFrame, details: list[dict[str, Any]], cfg: Any) -> None:
    output = cfg.review.output_dir
    columns = ["method", "precision", "recall", "f1", "pr_auc", "hit_at_1", "mrr", "temporal_iou"]
    table = summary[columns]
    markdown_table = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n"
    for row in table.itertuples(index=False, name=None):
        markdown_table += "| " + " | ".join([row[0], *[f"{v:.3f}" for v in row[1:]]]) + " |\n"
    by_method = summary.set_index("method")
    hmm, km = by_method.loc["Gaussian HMM"], by_method.loc["K-means"]
    convergence = sum(bool(d["models"]["Gaussian HMM"]["converged"]) for d in details)
    transitions = [np.trace(np.array(d["models"]["Gaussian HMM"]["transmat"])) / d["models"]["Gaussian HMM"]["k"] for d in details]
    dataset_notes = (Path(__file__).resolve().parents[1] / "DATASETS.md").read_text()
    report = f"""# Review 2: Clustering and Hidden Markov Models for Video Moment Retrieval

## 1. Problem identification and dataset selection (4 marks)

Needle locates the moments in a video that answer a natural-language query.
Pictures and speech give complementary evidence; individual similarity peaks can
be noisy. Can unsupervised grouping discover relevance patterns, and can a model
of temporal transitions produce useful, contiguous relevance decisions?

{dataset_notes}

The review uses eight existing picture, speech, keyword, distance-to-peak,
scene-change and speech-presence features (`config.yaml: review.features`).
The encoders remain frozen. StandardScaler is fitted separately on each training
fold. PCA is used only for an illustrative plot; it is fitted on that plot's
training videos and is not a predictive preprocessing step.

## 2. Implementation of clustering (4 marks)

K-means minimizes within-cluster squared Euclidean distance on standardized
features. Candidate cluster counts are {list(cfg.review.cluster_counts)}; the
highest training silhouette score selects k. Silhouette uses a fixed seeded
sample of at most {cfg.review.silhouette_sample:,} training rows. Gaussian mixture
models provide a second clustering method, using diagonal covariances and
training Bayesian Information Criterion (BIC; lower is better).

Neither clustering algorithm sees relevance labels during fitting or model
selection. For comparison with relevance predictors, each cluster is assigned its
relevant fraction using **training labels only**. K-means scores interpolate those
fractions using softmax of negative mean squared centroid distance; GMM scores
weight the fractions by component posterior probabilities. This calibration uses
labels: the complete retrieval systems are not purely unsupervised.

## 3. Implementation of a Hidden Markov Model (4 marks)

The Gaussian HMM is implemented in `analysis/hmm.py` with NumPy and SciPy-compatible
arrays, using a scikit-learn estimator interface. Observations are time-ordered
feature vectors; hidden states are learned patterns, not predefined relevance
labels. Parameters are the initial state probabilities, transition matrix, and
per-state Gaussian means/diagonal variances. K-means initializes the model;
Baum-Welch alternates scaled forward-backward expectation with parameter updates.
Variance floors and transition pseudocounts avoid degenerate distributions.

Candidate state counts {list(cfg.review.hmm_states)} are selected by training BIC.
Each (video, query) is a separate sequence; no transitions cross its boundary.
Training-label calibration maps state posteriors to relevance scores. Forward-
backward supplies smoothed scores; Viterbi supplies the most probable joint state
path, saved separately in the predictions. Relevance decisions threshold the
posterior-derived score; they are not the Viterbi state IDs.

The stopping criterion was met in {convergence}/{len(details)} selected fold models.
The mean diagonal transition probability is {np.mean(transitions):.3f}; inspect the
heatmap and state profiles before assigning semantic meaning to states. State IDs
are arbitrary and can permute between folds.

## 4. Comparative performance analysis (4 marks)

Evaluation holds out one entire video at a time. Every scaler, cluster/state count,
model, relevance mapping and decision threshold is fitted or selected without that
video's labels. Calibration and F1 threshold choice use the training rows in-sample,
so this is not an independently calibrated probability estimate. Earlier saved
supervised predictions used grouped inner cross-validation; their thresholds were
selected with that earlier protocol. These calibration differences must be kept in
mind when comparing methods. Baselines retain the configured fixed cutoff.

{markdown_table}

Numbers are **macro averages over held-out videos**, including per-video means
over queries for retrieval and IoU. They are not the pooled/query-averaged numbers
from the earlier report. Hit@k asks whether a relevant second is in the top k
seconds, not the top k distinct clips. MRR is the reciprocal rank of the first
relevant second. Temporal IoU is intersection/union of all predicted relevant
seconds and true relevant seconds for a query; it is not top-moment IoU. Ranking
scores are rounded to {cfg.review.ranking_decimals} decimal places for ranking and
threshold-free metrics, avoiding floating-point noise on posterior plateaus.
Ranking ties then use shared seeded random permutations. PR-AUC is average precision.

`comparison.csv` includes standard deviations and 95% percentile bootstrap
intervals, resampling whole videos. Only nine independent held-out videos are
available, so uncertainty is substantial. Saved supervised predictions are joined
by unique video/query/time keys and checked against current labels before reuse.

## 5. Result interpretation and presentation (4 marks)

The HMM achieves F1 {hmm.f1:.3f}, Hit@1 {hmm.hit_at_1:.3f}, and temporal IoU
{hmm.temporal_iou:.3f}. K-means achieves F1 {km.f1:.3f}, Hit@1 {km.hit_at_1:.3f},
and temporal IoU {km.temporal_iou:.3f}. The HMM-minus-K-means F1 difference is
{hmm.f1 - km.f1:+.3f}; its temporal IoU difference is {hmm.temporal_iou - km.temporal_iou:+.3f}.
Clustering optimizes feature geometry and the HMM optimizes sequence likelihood;
neither objective directly maximizes relevance F1 or retrieval rank. More persistent
states can make timelines smoother while also extending false positives. Improved
smoothness therefore does not automatically imply improved retrieval.

The timeline figure deliberately includes the lowest, middle and highest HMM IoU
queries, selected post-hoc to illustrate errors as well as successes. These examples
do not replace the full-dataset metrics. Study speech bias, correlated features
(a diagonal Gaussian assumption), small sample size, and local optima from one
seeded HMM initialization. Whole-video features and forward-backward use future
frames, so this is offline retrieval, not causal live streaming. Model selection on
correlated per-second rows makes BIC a practical heuristic, not proof of true states.

## Demo and reproduction

1. Open `notebooks/review_clustering_hmm.ipynb` and run all cells.
2. Explain the problem and class imbalance using `01_dataset.png`.
3. Show cluster selection, the PCA plot, and cluster profiles.
4. Explain transition/emission heatmaps, Baum-Welch convergence, and Viterbi.
5. Present `07_comparison.png`, then show a failure and success timeline.
6. Demonstrate the original app separately; the new models are analysis experiments
   and are not added to the app's live ranking controls.

Run `.venv/bin/python scripts/run_review.py` from the project root. Matching fold
artifacts are reused; `--force` reruns them. The manifest hashes the feature table,
settings, dependency versions and experiment/HMM source, preventing reuse after those change.

References: [NASA media guidance](https://www.nasa.gov/nasa-brand-center/images-and-media/),
[K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means),
[Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html),
[Rabiner's HMM tutorial](https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf).
"""
    (output / "REVIEW_REPORT.md").write_text(report)
    # A standalone browser presentation with tables and images; no frontend build needed.
    images = "".join(f'<figure><img src="figures/{p.name}" alt="{html.escape(p.stem)}"><figcaption>{html.escape(p.stem.replace("_", " "))}</figcaption></figure>' for p in sorted((output / "figures").glob("*.png")))
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Needle — Clustering &amp; HMM Review</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#182536;background:#f6f8fb}h1{line-height:1.2}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dbe2eb}.table{overflow:auto}figure{margin:32px 0;background:white;padding:16px;border-radius:12px}img{max-width:100%;height:auto}pre{white-space:pre-wrap;background:white;padding:24px}a{color:#245d91}</style>'
    page += '<h1>Needle: Clustering &amp; Hidden Markov Models</h1><p>Review 2 · 9 held-out videos · 86 queries · 28,900 feature rows</p><p>New algorithms are evaluated in the notebook; the original app continues to use fixed-weight fusion.</p><div class="table">'
    page += table.to_html(index=False, float_format=lambda value: f"{value:.3f}", border=0)
    page += '</div><p>Macro-video averages. Figures show video-bootstrap uncertainty. Read the methodology and limitations below.</p>' + images
    page += '<h2>Full review notes</h2><pre>' + html.escape(report) + '</pre></html>'
    (output / "review.html").write_text(page)
