"""Leakage-safe clustering/HMM experiments on the existing NASA benchmark."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from analysis.evaluate import best_f1_threshold, classification_metrics
from analysis.hmm import GaussianHMM

KEYS = ["video_id", "query_id", "t"]
NEW_METHODS = ("K-means", "Gaussian mixture", "Gaussian HMM")


def ordered_sequences(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Sort by video, query, time; reject duplicate keys and missing timestamps."""
    if frame.empty or frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Feature rows must have unique, nonempty video/query/time keys.")
    frame = frame.sort_values(KEYS, kind="stable").reset_index(drop=True)
    lengths = frame.groupby(["video_id", "query_id"], sort=False).size().tolist()
    return frame, lengths


def relevance_rates(membership: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Training-label calibration E[relevant | cluster/state]; no test labels."""
    weights = membership.sum(axis=0)
    return np.divide(
        membership.T @ labels, weights, out=np.full(len(weights), float(labels.mean())), where=weights > 0,
    )


def _kmeans_membership(model: KMeans, X: np.ndarray) -> np.ndarray:
    # Smooth interpolation between centroids avoids large score ties in retrieval.
    # Temperature is the number of standardized features, i.e. mean squared distance.
    return softmax(-(model.transform(X) ** 2) / X.shape[1], axis=1)


def fit_fold(dataset: pd.DataFrame, held_out: str, cfg: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit every transform/model on other videos; return held-out predictions."""
    options = cfg.review
    if options is None:
        raise ValueError("The config needs a 'review' section.")
    train, train_lengths = ordered_sequences(dataset[dataset.video_id != held_out])
    test, test_lengths = ordered_sequences(dataset[dataset.video_id == held_out])
    if len(train.video_id.unique()) < 2:
        raise ValueError("Review evaluation needs at least three videos.")
    features = list(options.features)
    scaler = StandardScaler().fit(train[features])
    X, Xt = scaler.transform(train[features]), scaler.transform(test[features])
    if not np.isfinite(X).all() or not np.isfinite(Xt).all():
        raise ValueError("Review features must contain only finite values.")
    y = train.rel_binary.to_numpy(dtype=int)
    result = test[KEYS + ["rel_binary"]].copy()
    details: dict[str, Any] = {
        "held_out": held_out, "training_videos": sorted(train.video_id.unique().tolist()),
        "features": features, "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "training_rows": len(train), "training_sequences": len(train_lengths),
        "test_rows": len(test), "test_sequences": len(test_lengths),
        "candidates": [], "models": {},
    }
    rng = np.random.default_rng(cfg.ml.seed)
    sample = rng.choice(len(X), min(len(X), options.silhouette_sample), replace=False)
    distances = pairwise_distances(X[sample])
    np.fill_diagonal(distances, 0)

    def silhouette(labels: np.ndarray) -> float:
        subset = labels[sample]
        if len(np.unique(subset)) < 2 or len(np.unique(subset)) == len(subset):
            return float("nan")
        return float(silhouette_score(distances, subset, metric="precomputed"))

    for method in NEW_METHODS:
        started = perf_counter()
        candidates = []
        counts = options.hmm_states if method == "Gaussian HMM" else options.cluster_counts
        for k in counts:
            common = {"n_init": options.n_init, "random_state": cfg.ml.seed}
            if method == "K-means":
                model = KMeans(n_clusters=k, **common).fit(X)
                diagnostic = {"silhouette": silhouette(model.labels_), "inertia": float(model.inertia_)}
                if not np.isfinite(diagnostic["silhouette"]):
                    raise ValueError("Silhouette sample must include at least two clusters.")
                criterion = -diagnostic["silhouette"]
            elif method == "Gaussian mixture":
                model = GaussianMixture(
                    n_components=k, covariance_type="diag", reg_covar=options.min_covar,
                    max_iter=options.max_iter, tol=options.tolerance, **common,
                ).fit(X)
                diagnostic = {"bic": float(model.bic(X)), "silhouette": silhouette(model.predict(X)),
                              "converged": bool(model.converged_)}
                criterion = diagnostic["bic"]
            else:
                model = GaussianHMM(
                    k, max_iter=options.max_iter, tol=options.tolerance,
                    min_covar=options.min_covar, pseudocount=options.pseudocount, **common,
                ).fit(X, train_lengths)
                diagnostic = {"bic": model.bic(X, train_lengths), "converged": bool(model.converged_),
                              "iterations": len(model.history_)}
                criterion = diagnostic["bic"]
            details["candidates"].append({"method": method, "k": int(k), **diagnostic})
            candidates.append((criterion, k, model, diagnostic))
        _, k, model, diagnostic = min(candidates, key=lambda entry: entry[0])
        if method == "K-means":
            train_membership = np.eye(k)[model.labels_]
            rates = relevance_rates(train_membership, y)
            train_scores = _kmeans_membership(model, X) @ rates
            scores = _kmeans_membership(model, Xt) @ rates
            state = model.predict(Xt)
            parameters = {"means": model.cluster_centers_.tolist()}
        elif method == "Gaussian mixture":
            rates = relevance_rates(model.predict_proba(X), y)
            train_scores = model.predict_proba(X) @ rates
            scores = model.predict_proba(Xt) @ rates
            state = model.predict(Xt)
            parameters = {"means": model.means_.tolist(), "covars": model.covariances_.tolist(),
                          "weights": model.weights_.tolist()}
        else:
            rates = relevance_rates(model.predict_proba(X, train_lengths), y)
            train_scores = model.predict_proba(X, train_lengths) @ rates
            scores = model.predict_proba(Xt, test_lengths) @ rates
            state = model.predict(Xt, test_lengths)
            parameters = {"means": model.means_.tolist(), "covars": model.covars_.tolist(),
                          "startprob": model.startprob_.tolist(), "transmat": model.transmat_.tolist(),
                          "log_likelihood_history": model.history_}
        threshold = best_f1_threshold(y, train_scores)
        elapsed = perf_counter() - started
        result[method] = scores
        result[f"{method}__pred"] = (scores >= threshold).astype(int)
        result[f"{method}__state"] = state
        details["models"][method] = {
            "k": int(k), "threshold": float(threshold), "state_relevance": rates.tolist(),
            "elapsed_s": elapsed, **diagnostic, **parameters,
        }
        print(f"  {method}: k={k}, {elapsed:.1f}s", flush=True)
    return result, details


def compare_methods(predictions: pd.DataFrame, dataset: pd.DataFrame, cfg: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool new held-out predictions with the saved, key-checked earlier results."""
    data = predictions.merge(
        dataset[KEYS + ["fused_baseline", "visual_norm", "speech_norm"]],
        on=KEYS, validate="one_to_one", how="left",
    )
    data["Fixed-weight fusion"] = data.fused_baseline
    data["Picture only"] = data.visual_norm
    data["Speech only"] = data.speech_norm
    methods = [*NEW_METHODS, "Fixed-weight fusion", "Picture only", "Speech only"]
    for method in methods[3:]:
        data[f"{method}__pred"] = (data[method] >= cfg.search.score_threshold).astype(int)
    prior_path = cfg.ml.tables_dir / "oof_predictions.csv.gz"
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        aligned = data[KEYS + ["rel_binary"]].merge(
            prior, on=KEYS, how="outer", validate="one_to_one", suffixes=("", "_old"), indicator=True,
        )
        if not aligned._merge.eq("both").all() or not aligned.rel_binary.eq(aligned.rel_binary_old).all():
            raise ValueError("Previous results do not match the current benchmark keys/labels.")
        # Merge, rather than positional assignment, to preserve key alignment.
        data = data.merge(prior[KEYS + [
            "cls_logistic", "cls_logistic_pred", "cls_knn", "cls_knn_pred",
            "cls_gradient_boosting", "cls_gradient_boosting_pred",
        ]], on=KEYS, validate="one_to_one")
        for key, name in [("logistic", "Logistic regression"), ("knn", "k-nearest neighbours"),
                          ("gradient_boosting", "Gradient boosting classifier")]:
            data[name] = data[f"cls_{key}"]
            data[f"{name}__pred"] = data[f"cls_{key}_pred"]
            methods.append(name)
    if data[methods].isna().any().any():
        raise ValueError("Comparison has missing predictions.")
    return data, evaluate_predictions(data, methods, cfg.ml.seed, ranking_decimals=cfg.review.ranking_decimals)


def evaluate_predictions(
    data: pd.DataFrame, methods: list[str], seed: int, *, ranking_decimals: int = 12,
) -> pd.DataFrame:
    """Compute common metrics per held-out video, including mean per-query IoU."""
    rows = []
    for video, frame in data.groupby("video_id"):
        for method in methods:
            rng = np.random.default_rng(seed)
            query_metrics = []
            for qid, query in frame.groupby("query_id", sort=True):
                truth = query.rel_binary.to_numpy(dtype=bool)
                predicted = query[f"{method}__pred"].to_numpy(dtype=bool)
                # A shared seeded permutation provides reproducible, unbiased tie-breaking.
                perm = rng.permutation(len(query))
                # Saturated posteriors differ by a few ulps after CSV round-trips.
                # Those differences carry no ranking evidence: apply fixed numerical
                # precision before shared random tie-breaking, for every method.
                scores = np.round(query[method].to_numpy(), decimals=ranking_decimals)
                order = perm[np.argsort(-scores[perm], kind="stable")]
                relevant = np.flatnonzero(truth[order])
                union = (truth | predicted).sum()
                query_metrics.append({
                    "hit_at_1": float(truth[order[:1]].any()),
                    "hit_at_5": float(truth[order[:5]].any()),
                    "mrr": float(1 / (relevant[0] + 1)) if len(relevant) else 0.0,
                    "temporal_iou": float((truth & predicted).sum() / union) if union else 1.0,
                })
            rows.append({
                "video_id": video, "method": method, "n_rows": len(frame),
                "n_queries": len(query_metrics),
                **classification_metrics(frame.rel_binary.to_numpy(),
                                         np.round(frame[method].to_numpy(), decimals=ranking_decimals),
                                         frame[f"{method}__pred"].to_numpy()),
                **pd.DataFrame(query_metrics).mean().to_dict(),
            })
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Macro-video means/std and video-bootstrap 95% intervals (only nine groups)."""
    metrics = [c for c in folds if c not in ["video_id", "method", "n_rows", "n_queries"]]
    rows = []
    for method, frame in folds.groupby("method", sort=False):
        values = frame[metrics].to_numpy()
        rng = np.random.default_rng(cfg.ml.seed)
        draws = rng.integers(len(frame), size=(cfg.review.bootstrap_samples, len(frame)))
        bootstrap = values[draws].mean(axis=1)
        row: dict[str, Any] = {"method": method, "n_videos": len(frame)}
        for i, metric in enumerate(metrics):
            row[metric] = float(values[:, i].mean())
            row[f"{metric}_std"] = float(values[:, i].std(ddof=1))
            row[f"{metric}_lo"], row[f"{metric}_hi"] = np.percentile(bootstrap[:, i], [2.5, 97.5]).tolist()
        rows.append(row)
    return pd.DataFrame(rows)


def run_review(cfg: Any, *, force: bool = False) -> None:
    """Run/reuse independently cached video folds; write tables, figures, report."""
    if cfg.review is None:
        raise ValueError("The config needs a 'review' section.")
    dataset_path = cfg.ml.tables_dir / "dataset.csv.gz"
    if not dataset_path.exists():
        raise FileNotFoundError("Build the existing feature table with scripts/run_analysis.py --step build first.")
    dataset, _ = ordered_sequences(pd.read_csv(dataset_path))
    output = cfg.review.output_dir
    folds_dir = output / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict

    settings = {
        "review": asdict(cfg.review), "seed": cfg.ml.seed,
        "libraries": {name: importlib.metadata.version(name) for name in
                      ["numpy", "pandas", "scipy", "scikit-learn"]},
    }
    digest = hashlib.sha256(json.dumps(settings, default=str, sort_keys=True).encode())
    for path in [dataset_path, Path(__file__), Path(__file__).with_name("hmm.py")]:
        digest.update(path.read_bytes())
    signature = digest.hexdigest()
    predictions, metadata = [], []
    with threadpool_limits(limits=1):
        for video in sorted(dataset.video_id.unique()):
            # Slugs are used only as filenames; reject unsafe identifiers from new datasets.
            if Path(video).name != video or video in (".", "..") or "\\" in video:
                raise ValueError("Unsafe video identifier in feature table.")
            table_path, meta_path = folds_dir / f"{video}.csv.gz", folds_dir / f"{video}.json"
            cached = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            if not force and table_path.exists() and cached.get("signature") == signature:
                print(f"Reuse {video}", flush=True)
                prediction, detail = pd.read_csv(table_path), cached["detail"]
            else:
                print(f"Hold out {video}", flush=True)
                prediction, detail = fit_fold(dataset, video, cfg)
                prediction.to_csv(table_path, index=False)
                meta_path.write_text(json.dumps({"signature": signature, "detail": detail}, indent=2))
            predictions.append(prediction)
            metadata.append(detail)
    data, folds = compare_methods(pd.concat(predictions, ignore_index=True), dataset, cfg)
    summary = summarize(folds, cfg)
    data.to_csv(output / "predictions.csv.gz", index=False)
    folds.to_csv(output / "per_video.csv", index=False)
    summary.to_csv(output / "comparison.csv", index=False)
    pd.DataFrame([
        {"held_out": d["held_out"], **candidate} for d in metadata for candidate in d["candidates"]
    ]).to_csv(output / "model_selection.csv", index=False)
    (output / "model_details.json").write_text(json.dumps(metadata, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "signature": signature, "settings": settings, "n_rows": len(dataset),
        "n_videos": int(dataset.video_id.nunique()), "n_queries": int(dataset.query_id.nunique()),
        "positive_rows": int(dataset.rel_binary.sum()),
    }, default=str, indent=2))
    from analysis.review_plots import make_review_artifacts

    make_review_artifacts(dataset, data, summary, metadata, cfg)
    print(summary[["method", "f1", "pr_auc", "hit_at_1", "mrr", "temporal_iou"]].to_string(index=False))
    print(f"Saved review artifacts to {output}", flush=True)
