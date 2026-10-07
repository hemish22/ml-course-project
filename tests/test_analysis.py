import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from analysis.benchmark import BenchmarkQuery, load_benchmark, relevance_labels, validate_against_index  # noqa: E402
from analysis.evaluate import (  # noqa: E402
    alpha_sweep,
    best_f1_threshold,
    classification_metrics,
    regression_metrics,
    retrieval_metrics,
    run_experiment,
)
from analysis.features import FEATURES, features_for_query, frame_change  # noqa: E402
from cmvs.search import ScoreTimeline  # noqa: E402


def test_relevance_labels_are_one_inside_and_decay_outside() -> None:
    graded, binary = relevance_labels(np.arange(10, dtype=float), start=3.0, end=6.0, decay_s=2.0)

    assert binary.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert graded[3:6].tolist() == [1.0, 1.0, 1.0]
    assert graded[2] == pytest.approx(np.exp(-0.5))
    assert graded[6] == pytest.approx(1.0 * np.exp(0.0))  # distance from end is 0 at t == end
    assert graded[0] < graded[1] < graded[2]


def test_load_and_validate_benchmark(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"video": "v", "query": " a cat ", "start": 1, "end": 5}]))
    queries = load_benchmark(path)
    index = tmp_path / "index" / "v"
    index.mkdir(parents=True)
    (index / "meta.json").write_text(json.dumps({"duration": 10.0}))

    validate_against_index(queries, tmp_path / "index")

    assert queries[0].query == "a cat" and queries[0].query_id == 0
    late = [BenchmarkQuery(0, "v", "x", 5.0, 30.0)]
    with pytest.raises(ValueError, match="only"):
        validate_against_index(late, tmp_path / "index")
    path.write_text(json.dumps([{"video": "v", "query": "x", "start": 5, "end": 5}]))
    with pytest.raises(ValueError):
        load_benchmark(path)


def test_frame_change_is_zero_for_identical_and_positive_for_different_frames() -> None:
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    change = frame_change(embeddings)

    assert change.tolist() == pytest.approx([0.0, 0.0, 1.0])


def _timeline(n: int = 12) -> ScoreTimeline:
    t = np.arange(n, dtype=np.float32)
    visual = np.linspace(0, 1, n).astype(np.float32)
    speech = np.where(t < 6, 0.5, 0.0).astype(np.float32)
    meta_t = {"start": [0.0], "end": [6.0], "text": ["three words matter"], "seg_idx": [0]}
    return ScoreTimeline(t, visual, speech, 0.6 * visual + 0.4 * speech, {}, meta_t,
                         visual_raw=visual * 0.3, transcript_raw=speech)


def test_features_for_query_has_every_feature_and_both_labels() -> None:
    query = BenchmarkQuery(7, "vid", "three words here", 2.0, 5.0)
    embeddings = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (12, 1))

    ml = SimpleNamespace(baseline_alpha=0.6, smooth_window=2, relevance_decay_s=5.0, near_window=3,
                         wide_window=5, scene_cut_threshold=0.5)
    frame = features_for_query(_timeline(), embeddings, 12.0, query, ml)

    assert set(FEATURES) <= set(frame.columns)
    assert len(frame) == 12 and frame["query_id"].eq(7).all()
    assert frame["in_speech"].tolist() == [1] * 6 + [0] * 6
    assert frame["query_words"].eq(3).all()
    assert frame["rel_binary"].sum() == 3
    assert not frame[FEATURES].isna().any().any()
    assert frame["keyword_overlap"].tolist() == pytest.approx([2 / 3] * 6 + [0.0] * 6)
    assert frame["visual_peak_dist"].iloc[-1] == pytest.approx(0.0)  # visual score peaks at the last second
    assert frame["visual_vs_peak"].max() == pytest.approx(1.0)


def test_best_f1_threshold_separates_clean_scores() -> None:
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    threshold = best_f1_threshold(y, proba)

    assert 0.3 < threshold <= 0.8


def test_metrics_reward_perfect_predictions() -> None:
    y = np.array([0, 1, 0, 1])

    assert regression_metrics(np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0]))["rmse"] == 0.0
    scores = classification_metrics(y, y.astype(float), y)
    assert scores["f1"] == 1.0 and scores["roc_auc"] == 1.0 and scores["pr_auc"] == 1.0


def _synthetic_dataset(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for video in ("a", "b", "c"):
        for query in range(3):
            qid = len(rows) // 30
            t = np.arange(30)
            inside = (t >= 10) & (t < 15)
            data = {name: rng.normal(size=30) for name in FEATURES}
            data["visual_norm"] = inside * 0.8 + rng.random(30) * 0.3
            data["speech_norm"] = inside * 0.6 + rng.random(30) * 0.3
            frame = pd.DataFrame(data)
            frame["query_id"], frame["video_id"], frame["t"] = qid, video, t.astype(float)
            frame["rel_binary"] = inside.astype(int)
            frame["rel_graded"] = np.where(inside, 1.0, 0.1)
            frame["fused_baseline"] = 0.6 * frame["visual_norm"] + 0.4 * frame["speech_norm"]
            frame["visual_raw"], frame["speech_raw"] = frame["visual_norm"], frame["speech_norm"]
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _tiny_cfg() -> SimpleNamespace:
    grids = {
        "ridge": {"alpha": [1.0]}, "random_forest": {"n_estimators": [10], "max_depth": [3], "min_samples_leaf": [2]},
        "gradient_boosting": {"learning_rate": [0.1], "max_depth": [3], "max_iter": [20]},
        "logistic": {"C": [1.0]}, "knn": {"n_neighbors": [5]},
    }
    return SimpleNamespace(ml=SimpleNamespace(seed=0, n_jobs=1, inner_folds=2, permutation_repeats=1, grids=grids,
                                               output_smooth_windows=(1, 3)))


def test_experiment_predictions_come_from_held_out_videos_and_learn_the_signal() -> None:
    data = _synthetic_dataset()

    result = run_experiment(data, _tiny_cfg(), learning_sizes=(1, 2))

    assert len(result.oof) == len(data)
    assert set(result.oof["fold_video"]) == {"a", "b", "c"}
    assert (result.oof["fold_video"] == result.oof["video_id"]).all()  # leave-one-video-out
    summary = result.classification_folds.groupby("model")["roc_auc"].mean()
    assert summary["logistic"] > 0.9 and summary["dummy"] == 0.5
    assert set(result.best_params["model"]) >= {"ridge", "random_forest", "gradient_boosting", "logistic", "knn"}
    assert {"n_train_videos", "score"} <= set(result.learning_curve.columns)


def test_retrieval_metrics_rank_informative_scores_above_random() -> None:
    data = _synthetic_dataset()
    oof = data[["query_id", "video_id", "t", "rel_binary", "rel_graded"]].copy()
    for name in ("ridge", "random_forest", "gradient_boosting"):
        oof[f"reg_{name}"] = data["fused_baseline"]
    for name in ("logistic", "knn", "gradient_boosting"):
        oof[f"cls_{name}"] = data["fused_baseline"]

    table = retrieval_metrics(oof, data, (1, 5), seed=0).set_index("method")
    sweep = alpha_sweep(data, (0.0, 0.5, 1.0), (1,), seed=0)

    assert table.loc["Fixed-weight fusion", "hit_at_1"] > table.loc["Random", "hit_at_1"]
    assert table.loc["Fixed-weight fusion", "hit_at_1_lo"] <= table.loc["Fixed-weight fusion", "hit_at_1_hi"]
    assert "query_id" not in table.columns
    assert len(sweep) == 3


def test_smoothing_averages_within_each_query_only() -> None:
    from analysis.evaluate import choose_smoothing_window, smooth_by_query

    values = np.array([0.0, 3.0, 0.0, 9.0, 0.0, 9.0])
    queries = np.array([0, 0, 0, 1, 1, 1])

    smoothed = smooth_by_query(values, queries, 3)

    assert smoothed[:3].tolist() == pytest.approx([1.5, 1.0, 1.5])  # never mixes query 0 into query 1
    assert smoothed[3:].tolist() == pytest.approx([4.5, 6.0, 4.5])
    assert smooth_by_query(values, queries, 1).tolist() == values.tolist()
    noisy = np.tile([0.0, 1.0], 20)
    truth = np.full(40, 0.5)
    assert choose_smoothing_window("regression", noisy, truth, np.zeros(40), (1, 3, 5)) in (3, 5)
