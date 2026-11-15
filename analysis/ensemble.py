"""Review 3 (PBL-3): ensemble learning on the existing relevance feature table.

Every model predicts whether one second of a video answers a query (``rel_binary``)
from the same 31 engineered features used in Review 1. Evaluation holds out one whole
video at a time, exactly as before, so the results are directly comparable with the
Review 1 classifiers and the Review 2 clustering/HMM models.

Ensembles implemented:

* CART: a single decision tree, depth chosen by grouped inner cross-validation.
* Bagging: bootstrap-aggregated CART trees.
* Random forest and Extra trees: bagging plus random feature subsets / random splits.
* AdaBoost (SAMME) over depth-2 trees, and histogram gradient boosting.
* Soft voting: an unweighted average of heterogeneous models' probabilities.
* Stacking: a logistic meta-learner on out-of-fold base-model log-odds.
* Bayesian model averaging: pseudo-BMA+ weights from out-of-fold log predictive
  densities with a Bayesian bootstrap over queries (Yao et al., 2018).

No learner uses class re-weighting: every model estimates P(relevant) so that
log scores (BMA) and averaged probabilities (voting) are meaningful; the 8.3%
class imbalance is handled by choosing the F1 threshold instead. Smoothing
width and the F1 decision threshold for every method are chosen on
grouped inner out-of-fold predictions of the training videos only.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit, softmax
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from analysis.evaluate import best_f1_threshold, choose_smoothing_window, smooth_by_query
from analysis.features import FEATURES
from analysis.review import KEYS, evaluate_predictions, ordered_sequences

# Learners fitted per fold. Logistic regression and k-NN reuse the Review 1 settings and
# are inputs to the combiners only; their own results are already reported by Review 1.
LEARNERS: dict[str, str] = {
    "logistic": "Logistic regression (base)",
    "knn": "k-NN (base)",
    "cart": "CART tree",
    "bagging": "Bagging",
    "random_forest": "Random forest",
    "extra_trees": "Extra trees",
    "adaboost": "AdaBoost",
    "gradient_boosting": "Hist. gradient boosting",
}
COMBINERS: dict[str, str] = {
    "voting": "Soft voting",
    "stacking": "Stacking",
    "bma": "Bayesian model averaging",
}
REPORTED: dict[str, str] = {k: v for k, v in LEARNERS.items() if k not in ("logistic", "knn")} | COMBINERS
ENSEMBLE_METHODS: tuple[str, ...] = tuple(REPORTED.values())
PROB_CLIP = 1e-4


def make_learner(name: str, ens: Any, seed: int, *, depth: int | None = None) -> Any:
    """Build one unfitted base learner from the ``ensemble`` config section."""
    leaf = ens.min_samples_leaf
    if name == "logistic":
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(
            C=ens.logistic_C, max_iter=2000, random_state=seed))])
    if name == "knn":
        return Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(
            n_neighbors=ens.knn_neighbors, n_jobs=1))])
    if name == "cart":
        return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf * 4, random_state=seed)
    if name == "bagging":
        return BaggingClassifier(
            DecisionTreeClassifier(min_samples_leaf=leaf, random_state=seed),
            n_estimators=ens.n_estimators, bootstrap=True, n_jobs=1, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=ens.n_estimators, min_samples_leaf=leaf, max_features="sqrt",
                                      n_jobs=1, random_state=seed)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=ens.n_estimators, min_samples_leaf=leaf, max_features="sqrt",
                                    n_jobs=1, random_state=seed)
    if name == "adaboost":
        options = dict(ens.adaboost)
        return AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=options.pop("max_depth"), random_state=seed),
            algorithm="SAMME", random_state=seed, **options)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(early_stopping=False,
                                              random_state=seed, **ens.gradient_boosting)
    raise ValueError(f"Unknown ensemble learner: {name}")


def _out_of_fold(model: Any, X: np.ndarray, y: np.ndarray, splits: list) -> np.ndarray:
    """Grouped out-of-fold positive-class probabilities on the training videos."""
    oof = np.full(len(y), np.nan)
    for fit_idx, val_idx in splits:
        oof[val_idx] = clone(model).fit(X[fit_idx], y[fit_idx]).predict_proba(X[val_idx])[:, 1]
    if np.isnan(oof).any():
        raise ValueError("Inner folds did not cover every training row.")
    return oof


def _log_odds(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, PROB_CLIP, 1 - PROB_CLIP))


def pseudo_bma_weights(
    oof: np.ndarray, y: np.ndarray, groups: np.ndarray, draws: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-BMA+ model weights (Yao, Vehtari, Simpson and Gelman, 2018).

    ``oof`` is (rows, models) out-of-fold probabilities. The expected log predictive
    density of each model is summed per independent unit (one query), then the
    Bayesian bootstrap (Dirichlet(1) weights over units) regularizes the
    softmax-of-ELPD weights. Returns (weights, per-model ELPD).
    """
    p = np.clip(oof, PROB_CLIP, 1 - PROB_CLIP)
    pointwise = np.where(y[:, None] == 1, np.log(p), np.log1p(-p))
    unit_codes, units = pd.factorize(pd.Series(groups))
    elpd_units = np.zeros((len(units), oof.shape[1]))
    np.add.at(elpd_units, unit_codes, pointwise)
    rng = np.random.default_rng(seed)
    alpha = rng.dirichlet(np.ones(len(units)), size=draws)          # (draws, units)
    z = len(units) * alpha @ elpd_units                               # (draws, models)
    return softmax(z, axis=1).mean(axis=0), elpd_units.sum(axis=0)


def _staged_ap(name: str, model: Any, X: np.ndarray, y: np.ndarray, sizes: tuple[int, ...]) -> dict[int, float]:
    """Held-out PR-AUC after the first k members of a fitted ensemble."""
    if len(np.unique(y)) < 2:
        return {}
    out: dict[int, float] = {}
    if name in ("bagging", "random_forest", "extra_trees"):
        features = getattr(model, "estimators_features_", None)
        running = np.zeros(len(X))
        for i, tree in enumerate(model.estimators_, start=1):
            cols = features[i - 1] if features is not None else slice(None)
            running += tree.predict_proba(X[:, cols])[:, 1]
            if i in sizes:
                out[i] = float(average_precision_score(y, running / i))
    elif name in ("adaboost", "gradient_boosting"):
        for i, proba in enumerate(model.staged_predict_proba(X), start=1):
            if i in sizes:
                out[i] = float(average_precision_score(y, proba[:, 1]))
    return out


def fit_fold(dataset: pd.DataFrame, held_out: str, cfg: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit every learner and combiner on the other videos; score the held-out video."""
    ens, seed = cfg.ensemble, cfg.ml.seed
    if ens is None:
        raise ValueError("The config needs an 'ensemble' section.")
    train, _ = ordered_sequences(dataset[dataset.video_id != held_out])
    test, _ = ordered_sequences(dataset[dataset.video_id == held_out])
    videos = train.video_id.unique()
    if len(videos) < 2:
        raise ValueError("Ensemble evaluation needs at least three videos.")
    X, Xt = train[FEATURES].to_numpy(float), test[FEATURES].to_numpy(float)
    if not np.isfinite(X).all() or not np.isfinite(Xt).all():
        raise ValueError("Features must contain only finite values.")
    y, yt = train.rel_binary.to_numpy(int), test.rel_binary.to_numpy(int)
    groups = train.video_id.to_numpy()
    seq, seq_t = (train.video_id + "|" + train.query_id.astype(str)).to_numpy(), (test.video_id + "|" + test.query_id.astype(str)).to_numpy()
    splits = list(GroupKFold(n_splits=min(ens.inner_folds, len(videos))).split(X, y, groups))

    details: dict[str, Any] = {
        "held_out": held_out, "training_videos": sorted(videos.tolist()),
        "training_rows": len(train), "test_rows": len(test), "features": FEATURES,
        "learners": {}, "methods": {}, "size_curve": {}, "importances": {},
    }
    learners = list(dict.fromkeys([*ens.stacking_members, *ens.voting_members,
                                   *[k for k in LEARNERS if k in REPORTED]]))
    oof: dict[str, np.ndarray] = {}
    test_raw: dict[str, np.ndarray] = {}

    # CART depth: grouped inner-CV PR-AUC on training videos only.
    depth_scores = {int(d): float(average_precision_score(
        y, _out_of_fold(make_learner("cart", ens, seed, depth=d), X, y, splits))) for d in ens.cart_depths}
    cart_depth = max(depth_scores, key=depth_scores.get)
    details["cart_depth_scores"], details["cart_depth"] = depth_scores, cart_depth

    for name in learners:
        model = make_learner(name, ens, seed, depth=cart_depth if name == "cart" else None)
        oof[name] = _out_of_fold(model, X, y, splits)
        started = perf_counter()
        model.fit(X, y)
        fit_s = perf_counter() - started
        started = perf_counter()
        test_raw[name] = model.predict_proba(Xt)[:, 1]
        details["learners"][name] = {
            "fit_s": fit_s, "predict_s": perf_counter() - started,
            "inner_pr_auc": float(average_precision_score(y, oof[name])),
        }
        curve = _staged_ap(name, model, Xt, yt, ens.size_curve)
        if curve:
            details["size_curve"][name] = curve
        if name == "bagging":
            details["importances"][name] = np.mean([t.feature_importances_ for t in model.estimators_], axis=0).tolist()
        elif hasattr(model, "feature_importances_"):
            details["importances"][name] = model.feature_importances_.tolist()
        print(f"  [{held_out}] {name}: fit {fit_s:.1f}s", flush=True)

    # Soft voting: equal-weight probability average.
    vote = list(ens.voting_members)
    oof["voting"] = np.mean([oof[m] for m in vote], axis=0)
    test_raw["voting"] = np.mean([test_raw[m] for m in vote], axis=0)

    # Stacking: logistic meta-learner on base log-odds; its own inner OOF uses the same splits.
    stack = list(ens.stacking_members)
    Z, Zt = (np.column_stack([_log_odds(source[m]) for m in stack]) for source in (oof, test_raw))
    meta = Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=2000, random_state=seed))])
    oof["stacking"] = _out_of_fold(meta, Z, y, splits)
    meta.fit(Z, y)
    test_raw["stacking"] = meta.predict_proba(Zt)[:, 1]
    coef = meta.named_steps["model"].coef_[0] / meta.named_steps["scale"].scale_

    # Bayesian model averaging (pseudo-BMA+); units are training queries.
    weights, elpd = pseudo_bma_weights(np.column_stack([oof[m] for m in stack]), y,
                                       seq, ens.bma_bootstrap, seed)
    oof["bma"] = np.column_stack([oof[m] for m in stack]) @ weights
    test_raw["bma"] = np.column_stack([test_raw[m] for m in stack]) @ weights
    details["voting_members"] = vote
    details["stacking"] = {"members": stack, "coef_per_logodds": coef.tolist(),
                           "standardized_coef": meta.named_steps["model"].coef_[0].tolist(),
                           "intercept": float(meta.named_steps["model"].intercept_[0])}
    details["bma"] = {"members": stack, "weights": weights.tolist(), "elpd": elpd.tolist()}

    result = test[KEYS + ["rel_binary"]].copy()
    for key in ("logistic", "knn"):
        result[f"{LEARNERS[key]}__raw"] = test_raw[key]
    for key, label in REPORTED.items():
        window = choose_smoothing_window("classification", oof[key], y, seq, ens_windows(cfg))
        threshold = best_f1_threshold(y, smooth_by_query(oof[key], seq, window))
        score = smooth_by_query(test_raw[key], seq_t, window)
        result[label] = score
        result[f"{label}__pred"] = (score >= threshold).astype(int)
        result[f"{label}__raw"] = test_raw[key]
        details["methods"][label] = {
            "key": key, "smooth_window": int(window), "threshold": float(threshold),
            "inner_pr_auc": float(average_precision_score(y, oof[key])),
        }
    return result, details


def ens_windows(cfg: Any) -> tuple[int, ...]:
    """Candidate output-smoothing widths, shared with Review 1."""
    return tuple(cfg.ml.output_smooth_windows)


def _fold_job(dataset: pd.DataFrame, video: str, cfg: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    with threadpool_limits(limits=1):
        return fit_fold(dataset, video, cfg)


def paired_differences(folds: pd.DataFrame, reference: str, methods: list[str], metrics: list[str],
                       draws: int, seed: int) -> pd.DataFrame:
    """Per-video paired differences (method - reference) with video-bootstrap 95% CIs."""
    rows = []
    base = folds[folds.method == reference].set_index("video_id")
    for method in methods:
        other = folds[folds.method == method].set_index("video_id").loc[base.index]
        for metric in metrics:
            diff = (other[metric] - base[metric]).to_numpy()
            rng = np.random.default_rng(seed)
            boot = diff[rng.integers(len(diff), size=(draws, len(diff)))].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({"method": method, "reference": reference, "metric": metric,
                         "mean_diff": float(diff.mean()), "lo": float(lo), "hi": float(hi),
                         "videos_better": int((diff > 0).sum()), "n_videos": len(diff)})
    return pd.DataFrame(rows)


def compare_with_previous(predictions: pd.DataFrame, cfg: Any) -> tuple[pd.DataFrame, list[str]]:
    """Join with Review 1/2 held-out predictions on (video, query, second) keys."""
    data = predictions.copy()
    methods = list(ENSEMBLE_METHODS)
    previous = cfg.review.output_dir / "predictions.csv.gz" if cfg.review else None
    if previous is not None and previous.exists():
        earlier = pd.read_csv(previous)
        earlier_methods = [c[: -len("__pred")] for c in earlier if c.endswith("__pred")]
        keep = KEYS + ["rel_binary"] + earlier_methods + [f"{m}__pred" for m in earlier_methods]
        merged = data.merge(earlier[keep], on=KEYS, how="outer", validate="one_to_one",
                            suffixes=("", "_old"), indicator=True)
        if not merged._merge.eq("both").all() or not merged.rel_binary.eq(merged.rel_binary_old).all():
            raise ValueError("Earlier review predictions do not match the benchmark keys/labels.")
        data = merged.drop(columns=["_merge", "rel_binary_old"])
        methods += earlier_methods
    if data[methods].isna().any().any():
        raise ValueError("Comparison has missing predictions.")
    return data, methods


def summarize(folds: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    """Macro-video means, standard deviations and video-bootstrap 95% intervals."""
    metrics = [c for c in folds if c not in ["video_id", "method", "n_rows", "n_queries"]]
    rows = []
    for method, frame in folds.groupby("method", sort=False):
        values = frame[metrics].to_numpy()
        idx = np.random.default_rng(seed).integers(len(frame), size=(draws, len(frame)))
        boot = values[idx].mean(axis=1)
        row: dict[str, Any] = {"method": method, "n_videos": len(frame)}
        for i, metric in enumerate(metrics):
            row[metric] = float(values[:, i].mean())
            row[f"{metric}_std"] = float(values[:, i].std(ddof=1))
            row[f"{metric}_lo"], row[f"{metric}_hi"] = np.percentile(boot[:, i], [2.5, 97.5]).tolist()
        rows.append(row)
    return pd.DataFrame(rows)


def run_ensemble(cfg: Any, *, force: bool = False) -> pd.DataFrame:
    """Run (or reuse cached) leave-one-video-out folds; write tables, figures and report."""
    ens = cfg.ensemble
    if ens is None:
        raise ValueError("The config needs an 'ensemble' section.")
    dataset_path = cfg.ml.tables_dir / "dataset.csv.gz"
    if not dataset_path.exists():
        raise FileNotFoundError("Build the feature table first: scripts/run_analysis.py --step build.")
    dataset, _ = ordered_sequences(pd.read_csv(dataset_path))
    output = ens.output_dir
    folds_dir = output / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "ensemble": asdict(ens), "seed": cfg.ml.seed, "windows": ens_windows(cfg),
        "libraries": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn"]},
    }
    digest = hashlib.sha256(json.dumps(settings, default=str, sort_keys=True).encode())
    for path in [dataset_path, Path(__file__), Path(__file__).with_name("evaluate.py")]:
        digest.update(path.read_bytes())
    signature = digest.hexdigest()

    videos = sorted(dataset.video_id.unique())
    for video in videos:
        if Path(video).name != video or video in (".", "..") or "\\" in video:
            raise ValueError("Unsafe video identifier in feature table.")
    cached: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    todo = []
    for video in videos:
        table, meta = folds_dir / f"{video}.csv.gz", folds_dir / f"{video}.json"
        stored = json.loads(meta.read_text()) if meta.exists() else {}
        if not force and table.exists() and stored.get("signature") == signature:
            print(f"Reuse {video}", flush=True)
            cached[video] = (pd.read_csv(table), stored["detail"])
        else:
            todo.append(video)
    if todo:
        print(f"Fitting {len(todo)} held-out folds with {min(ens.n_jobs, len(todo))} workers", flush=True)
        results = Parallel(n_jobs=min(ens.n_jobs, len(todo)), verbose=0)(
            delayed(_fold_job)(dataset, video, cfg) for video in todo)
        for video, (prediction, detail) in zip(todo, results):
            prediction.to_csv(folds_dir / f"{video}.csv.gz", index=False)
            (folds_dir / f"{video}.json").write_text(json.dumps({"signature": signature, "detail": detail}, indent=2))
            cached[video] = (prediction, detail)
    predictions = pd.concat([cached[v][0] for v in videos], ignore_index=True)
    details = [cached[v][1] for v in videos]

    data, methods = compare_with_previous(predictions, cfg)
    folds = evaluate_predictions(data, methods, cfg.ml.seed,
                                 ranking_decimals=cfg.review.ranking_decimals if cfg.review else 12)
    summary = summarize(folds, ens.bma_bootstrap * 2, cfg.ml.seed)
    reference = "Logistic regression" if "Logistic regression" in methods else ENSEMBLE_METHODS[0]
    paired = paired_differences(folds, reference, list(ENSEMBLE_METHODS),
                                ["f1", "pr_auc", "hit_at_1", "temporal_iou"], ens.bma_bootstrap * 2, cfg.ml.seed)

    data.to_csv(output / "predictions.csv.gz", index=False)
    folds.to_csv(output / "per_video.csv", index=False)
    summary.to_csv(output / "comparison.csv", index=False)
    paired.to_csv(output / "paired_vs_logistic.csv", index=False)
    (output / "model_details.json").write_text(json.dumps(details, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "signature": signature, "settings": settings, "n_rows": len(dataset),
        "n_videos": len(videos), "n_queries": int(dataset.query_id.nunique()),
        "positive_rows": int(dataset.rel_binary.sum()), "n_features": len(FEATURES),
    }, default=str, indent=2))

    from analysis.ensemble_plots import make_ensemble_artifacts

    make_ensemble_artifacts(dataset, data, folds, summary, paired, details, cfg)
    print(summary[["method", "f1", "pr_auc", "hit_at_1", "mrr", "temporal_iou"]].round(3).to_string(index=False))
    print(f"Saved ensemble artifacts to {output}", flush=True)
    return summary
