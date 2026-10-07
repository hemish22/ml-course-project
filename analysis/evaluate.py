"""Leave-one-video-out evaluation of the regression and classification models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from joblib import Parallel, delayed

from analysis.features import FEATURES, KEYS
from analysis.models import CLASSIFIERS, REGRESSORS, make_model, param_grid


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R² and Spearman rank correlation."""
    rho = spearmanr(y_true, y_pred).statistic if np.ptp(y_pred) > 1e-9 else 0.0
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(0.0 if np.isnan(rho) else rho),
    }


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Threshold-dependent and threshold-free classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Probability threshold that maximizes F1 on the given (training) predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5


def _proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def smooth_by_query(values: np.ndarray, query_ids: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average of ``values`` within each query's timeline (rows must be time-ordered).

    Moments are runs of consecutive seconds, so averaging a model's per-second output over
    neighbouring seconds removes isolated spikes. ``window`` is in seconds (frames).
    """
    if window <= 1:
        return np.asarray(values, dtype=np.float64)
    series = pd.Series(np.asarray(values, dtype=np.float64))
    smoothed = series.groupby(np.asarray(query_ids)).transform(
        lambda x: x.rolling(window, center=True, min_periods=1).mean()
    )
    return smoothed.to_numpy()


def choose_smoothing_window(
    task: str, raw: np.ndarray, y: np.ndarray, query_ids: np.ndarray, windows: tuple[int, ...]
) -> int:
    """Pick the smoothing width that scores best on (out-of-fold) training predictions."""
    def score(window: int) -> float:
        smoothed = smooth_by_query(raw, query_ids, window)
        if task == "regression":
            return -float(np.sqrt(mean_squared_error(y, smoothed)))
        return float(average_precision_score(y, smoothed))

    return max(windows, key=score)


@dataclass
class ExperimentResult:
    """Everything produced by one leave-one-video-out run."""

    oof: pd.DataFrame
    best_params: pd.DataFrame
    regression_folds: pd.DataFrame
    classification_folds: pd.DataFrame
    importance: pd.DataFrame
    learning_curve: pd.DataFrame
    thresholds: pd.DataFrame


def _tune(name: str, task: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: object) -> Any:
    """Grid-search one model with grouped inner CV and return the refit best estimator."""
    folds = min(cfg.ml.inner_folds, len(np.unique(groups)))
    search = GridSearchCV(
        make_model(name, task, cfg.ml.seed),
        param_grid(name, cfg.ml.grids),
        scoring="neg_root_mean_squared_error" if task == "regression" else "average_precision",
        cv=GroupKFold(n_splits=folds),
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search


def _run_fold(
    held_out: str,
    fold_index: int,
    df: pd.DataFrame,
    cfg: object,
    features: list[str],
    windows: tuple[int, ...],
    learning_sizes: tuple[int, ...],
    light: bool,
) -> dict[str, Any]:
    """Train, tune and score every model with ``held_out`` as the unseen video."""
    rng = np.random.default_rng(cfg.ml.seed + fold_index)
    param_rows: list[dict] = []
    reg_rows: list[dict] = []
    cls_rows: list[dict] = []
    imp_rows: list[dict] = []
    curve_rows: list[dict] = []
    threshold_rows: list[dict] = []
    train = df[df["video_id"] != held_out]
    test = df[df["video_id"] == held_out]
    X_train, X_test = train[features], test[features]
    qid_train, qid_test = train["query_id"].to_numpy(), test["query_id"].to_numpy()
    groups = train["video_id"].to_numpy()
    part = test[KEYS + ["rel_graded", "rel_binary"]].copy()
    part["fold_video"] = held_out
    train_videos = sorted(train["video_id"].unique())

    for task, models, target in (
        ("regression", REGRESSORS, "rel_graded"),
        ("classification", CLASSIFIERS, "rel_binary"),
    ):
        y_train, y_test = train[target].to_numpy(), test[target].to_numpy()
        prefix = "reg" if task == "regression" else "cls"

        dummy = make_model("dummy", task, cfg.ml.seed).fit(X_train, y_train)
        if task == "regression":
            part["reg_dummy"] = dummy.predict(X_test)
            reg_rows.append({"fold_video": held_out, "model": "dummy", **regression_metrics(y_test, part["reg_dummy"])})
        else:
            part["cls_dummy"] = _proba(dummy, X_test)
            part["cls_dummy_pred"] = 0
            cls_rows.append({
                "fold_video": held_out, "model": "dummy", "threshold": 0.5,
                **classification_metrics(y_test, part["cls_dummy"], part["cls_dummy_pred"]),
            })

        for name in models:
            search = _tune(name, task, X_train, y_train, groups, cfg)
            best = search.best_estimator_
            param_rows.append({
                "fold_video": held_out, "task": task, "model": name,
                "params": str(search.best_params_), "inner_cv_score": float(search.best_score_),
            })
            inner_cv = GroupKFold(n_splits=min(cfg.ml.inner_folds, len(train_videos)))
            if task == "regression":
                inner = cross_val_predict(clone(best), X_train, y_train, groups=groups, cv=inner_cv)
                window = choose_smoothing_window(task, inner, y_train, qid_train, windows)
                raw_prediction = best.predict(X_test)
                prediction = smooth_by_query(raw_prediction, qid_test, window)
                part[f"reg_{name}_raw"] = raw_prediction
                part[f"reg_{name}"] = prediction
                reg_rows.append({"fold_video": held_out, "model": name, **regression_metrics(y_test, prediction)})
                threshold_rows.append({"fold_video": held_out, "model": name, "task": task, "smooth_window": window})
                scoring = "neg_root_mean_squared_error"
            else:
                inner = cross_val_predict(
                    clone(best), X_train, y_train, groups=groups, cv=inner_cv, method="predict_proba",
                )[:, 1]
                window = choose_smoothing_window(task, inner, y_train, qid_train, windows)
                threshold = best_f1_threshold(y_train, smooth_by_query(inner, qid_train, window))
                raw_proba = _proba(best, X_test)
                proba = smooth_by_query(raw_proba, qid_test, window)
                part[f"cls_{name}_raw"] = raw_proba
                part[f"cls_{name}"] = proba
                part[f"cls_{name}_pred"] = (proba >= threshold).astype(int)
                cls_rows.append({
                    "fold_video": held_out, "model": name, "threshold": threshold,
                    **classification_metrics(y_test, proba, part[f"cls_{name}_pred"]),
                })
                threshold_rows.append({"fold_video": held_out, "model": name, "task": task,
                                       "threshold": threshold, "smooth_window": window})
                scoring = "average_precision"
            if light:
                continue

            importance = permutation_importance(
                best, X_test, y_test, scoring=scoring,
                n_repeats=cfg.ml.permutation_repeats, random_state=cfg.ml.seed, n_jobs=1,
            )
            for feature, mean, std in zip(features, importance.importances_mean, importance.importances_std):
                imp_rows.append({"task": task, "model": name, "fold_video": held_out,
                                 "feature": feature, "importance": float(mean), "std": float(std)})

            for size in learning_sizes:
                if size > len(train_videos):
                    continue
                for _ in range(2):
                    subset = rng.choice(train_videos, size=size, replace=False)
                    sub = train[train["video_id"].isin(subset)]
                    model = clone(best).fit(sub[features], sub[target])
                    if task == "regression":
                        score = regression_metrics(y_test, model.predict(X_test))["rmse"]
                    else:
                        score = classification_metrics(
                            y_test, _proba(model, X_test), (_proba(model, X_test) >= 0.5).astype(int)
                        )["pr_auc"]
                    curve_rows.append({"task": task, "model": name, "n_train_videos": size, "score": score})

    return {
        "part": part, "params": param_rows, "reg": reg_rows, "cls": cls_rows,
        "importance": imp_rows, "curve": curve_rows, "thresholds": threshold_rows,
    }


def run_experiment(
    df: pd.DataFrame,
    cfg: object,
    learning_sizes: tuple[int, ...] = (1, 2, 4, 6),
    features: list[str] | None = None,
    smooth: bool = True,
    light: bool = False,
) -> ExperimentResult:
    """Leave-one-video-out regression and classification with tuning, thresholds and importances.

    Every prediction in ``oof`` comes from a model that never saw that video. Hyper-parameters,
    the output-smoothing width and the F1 threshold are chosen only on the training videos of each fold.

    Args:
        df: Feature table from ``build_dataset``.
        cfg: Application configuration.
        learning_sizes: Training-video counts for the learning curve.
        features: Feature columns to use (default: all).
        smooth: Smooth model outputs over time before scoring (width chosen on training videos).
        light: Skip permutation importance and learning curves (used for ablation variants).
    """
    features = features or FEATURES
    windows = tuple(cfg.ml.output_smooth_windows) if smooth else (1,)
    videos = sorted(df["video_id"].unique())
    results = Parallel(n_jobs=cfg.ml.n_jobs, verbose=5)(
        delayed(_run_fold)(video, index, df, cfg, features, windows, learning_sizes, light)
        for index, video in enumerate(videos)
    )
    flat = lambda key: [row for result in results for row in result[key]]  # noqa: E731

    return ExperimentResult(
        oof=pd.concat([result["part"] for result in results], ignore_index=True),
        best_params=pd.DataFrame(flat("params")),
        regression_folds=pd.DataFrame(flat("reg")),
        classification_folds=pd.DataFrame(flat("cls")),
        importance=pd.DataFrame(flat("importance")),
        learning_curve=pd.DataFrame(flat("curve")),
        thresholds=pd.DataFrame(flat("thresholds")),
    )


def summarise_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of every metric across held-out videos, per model."""
    metrics = [c for c in folds.columns if c not in ("fold_video", "model", "threshold")]
    grouped = folds.groupby("model")[metrics]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std"))
    return summary.reset_index()


def pooled_metrics(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics over all out-of-fold predictions pooled together (regression, classification)."""
    reg, cls = [], []
    for name in ["dummy", *REGRESSORS]:
        reg.append({"model": name, **regression_metrics(oof["rel_graded"].to_numpy(), oof[f"reg_{name}"].to_numpy())})
    for name in ["dummy", *CLASSIFIERS]:
        cls.append({"model": name, **classification_metrics(
            oof["rel_binary"].to_numpy(), oof[f"cls_{name}"].to_numpy(), oof[f"cls_{name}_pred"].to_numpy())})
    return pd.DataFrame(reg), pd.DataFrame(cls)


def retrieval_metrics(
    oof: pd.DataFrame, dataset: pd.DataFrame, ks: tuple[int, ...], seed: int
) -> pd.DataFrame:
    """Rank each query's seconds by a score and report hit@k and MRR of the first true second.

    Compared methods: fixed-weight fusion, picture only, speech only, random, and every model.
    """
    frame = oof.merge(dataset[KEYS + ["fused_baseline", "visual_raw", "speech_raw"]], on=KEYS)
    scores: dict[str, str] = {
        "Fixed-weight fusion": "fused_baseline",
        "Picture only": "visual_raw",
        "Speech only": "speech_raw",
    }
    scores.update({f"Regression: {label}": f"reg_{key}" for key, label in REGRESSORS.items()})
    scores.update({f"Classification: {label}": f"cls_{key}" for key, label in CLASSIFIERS.items()})
    rng = np.random.default_rng(seed)
    rows = []
    for label, column in scores.items():
        rows.append({"method": label, **_summarise_queries(_rank_stats(frame, column, ks, rng), rng)})
    random_runs = [
        _rank_stats(frame.assign(_random=rng.random(len(frame))), "_random", ks, rng).set_index("query_id")
        for _ in range(20)
    ]
    averaged = sum(run.drop(columns="video_id") for run in random_runs) / len(random_runs)
    averaged["video_id"] = random_runs[0]["video_id"]
    rows.append({"method": "Random", **_summarise_queries(averaged.reset_index(), rng)})
    return pd.DataFrame(rows)


def _summarise_queries(per_query: pd.DataFrame, rng: np.random.Generator, resamples: int = 2000) -> dict[str, float]:
    """Mean of each metric over queries with a bootstrap 95% interval (resampling queries)."""
    metrics = [c for c in per_query.columns if c not in ("query_id", "video_id")]
    values = per_query[metrics].to_numpy()
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    boot = values[draws].mean(axis=1)
    out: dict[str, float] = {"n_queries": float(len(values))}
    for i, metric in enumerate(metrics):
        out[metric] = float(values[:, i].mean())
        out[f"{metric}_lo"], out[f"{metric}_hi"] = (float(v) for v in np.percentile(boot[:, i], [2.5, 97.5]))
    return out


def _rank_stats(frame: pd.DataFrame, column: str, ks: tuple[int, ...], rng: np.random.Generator) -> pd.DataFrame:
    out = []
    for query_id, group in frame.groupby("query_id"):
        noise = rng.random(len(group)) * 1e-9  # random tie-breaking, e.g. zeros in silence
        order = np.argsort(-(group[column].to_numpy() + noise), kind="stable")
        labels = group["rel_binary"].to_numpy()[order]
        first = int(np.argmax(labels)) + 1 if labels.any() else len(labels) + 1
        out.append({
            "query_id": query_id, "video_id": group["video_id"].iat[0],
            **{f"hit_at_{k}": float(labels[:k].any()) for k in ks}, "mrr": 1.0 / first,
        })
    return pd.DataFrame(out)


def per_video_retrieval(
    oof: pd.DataFrame, dataset: pd.DataFrame, methods: dict[str, str], ks: tuple[int, ...], seed: int
) -> pd.DataFrame:
    """hit@k and MRR per video for the chosen ``{label: score column}`` methods."""
    frame = oof.merge(dataset[KEYS + ["fused_baseline", "visual_raw", "speech_raw"]], on=KEYS)
    rng = np.random.default_rng(seed)
    rows = []
    for label, column in methods.items():
        stats = _rank_stats(frame, column, ks, rng)
        grouped = stats.drop(columns="query_id").groupby("video_id").mean().reset_index()
        grouped.insert(0, "method", label)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def alpha_sweep(dataset: pd.DataFrame, alphas: tuple[float, ...], ks: tuple[int, ...], seed: int) -> pd.DataFrame:
    """Retrieval quality of the hand-set fusion ``alpha * picture + (1 - alpha) * speech``."""
    rng = np.random.default_rng(seed)
    rows = []
    for alpha in alphas:
        frame = dataset[KEYS + ["rel_binary"]].copy()
        frame["score"] = alpha * dataset["visual_norm"] + (1 - alpha) * dataset["speech_norm"]
        stats = _rank_stats(frame, "score", ks, rng).drop(columns=["query_id", "video_id"]).mean().to_dict()
        rows.append({"alpha": alpha, **stats})
    return pd.DataFrame(rows)


def summarise_variant(label: str, oof: pd.DataFrame, dataset: pd.DataFrame, cfg: object) -> pd.DataFrame:
    """One row per model: pooled held-out metrics plus search ranking (Hit@1, Hit@5, MRR)."""
    ks = (1, 5)
    ranking = retrieval_metrics(oof, dataset, ks, cfg.ml.seed).set_index("method")
    rows = []
    for key, name in REGRESSORS.items():
        rows.append({
            "variant": label, "task": "regression", "model": key,
            **regression_metrics(oof["rel_graded"].to_numpy(), oof[f"reg_{key}"].to_numpy()),
            **{m: float(ranking.loc[f"Regression: {name}", m]) for m in ("hit_at_1", "hit_at_5", "mrr")},
        })
    for key, name in CLASSIFIERS.items():
        rows.append({
            "variant": label, "task": "classification", "model": key,
            **classification_metrics(oof["rel_binary"].to_numpy(), oof[f"cls_{key}"].to_numpy(),
                                     oof[f"cls_{key}_pred"].to_numpy()),
            **{m: float(ranking.loc[f"Classification: {name}", m]) for m in ("hit_at_1", "hit_at_5", "mrr")},
        })
    return pd.DataFrame(rows)
