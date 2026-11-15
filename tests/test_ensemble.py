"""PBL-3 ensemble learning: combiner maths, learner construction and held-out isolation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from analysis.ensemble import (
    ENSEMBLE_METHODS,
    LEARNERS,
    fit_fold,
    make_learner,
    paired_differences,
    pseudo_bma_weights,
)
from analysis.features import FEATURES


def _ensemble_options(**overrides) -> SimpleNamespace:
    options = dict(
        inner_folds=2, cart_depths=(2, 4), logistic_C=1.0, knn_neighbors=5, n_estimators=8,
        min_samples_leaf=2, adaboost={"n_estimators": 8, "learning_rate": .5, "max_depth": 2},
        gradient_boosting={"max_iter": 10, "learning_rate": .1, "max_leaf_nodes": 7, "l2_regularization": 1.0},
        voting_members=("logistic", "random_forest", "gradient_boosting"),
        stacking_members=tuple(LEARNERS), bma_bootstrap=50, size_curve=(1, 4, 8), n_jobs=1,
    )
    options.update(overrides)
    return SimpleNamespace(**options)


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(ensemble=_ensemble_options(),
                           ml=SimpleNamespace(seed=3, output_smooth_windows=(1, 3)))


def _dataset(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for video in ["a", "b", "c", "d"]:
        for query in range(2):
            t = np.arange(40)
            y = ((t >= 12) & (t < 20)).astype(int)
            signal = y + rng.normal(0, .4, (len(t), len(FEATURES))).T
            frame = pd.DataFrame(signal.T, columns=FEATURES)
            frame.insert(0, "t", t)
            frame.insert(0, "query_id", query + {"a": 0, "b": 10, "c": 20, "d": 30}[video])
            frame.insert(0, "video_id", video)
            frame["rel_binary"] = y
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_bma_weights_are_a_distribution_and_prefer_the_better_model() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = np.where(y == 1, .9, .1)
    poor = np.full(400, .5)
    groups = np.repeat(np.arange(40), 10)
    weights, elpd = pseudo_bma_weights(np.column_stack([good, poor]), y, groups, 200, 0)
    assert weights.sum() == pytest.approx(1)
    assert weights[0] > .99 and elpd[0] > elpd[1]
    equal, _ = pseudo_bma_weights(np.column_stack([good, good]), y, groups, 200, 0)
    np.testing.assert_allclose(equal, [.5, .5])


def test_bma_bootstrap_keeps_close_models_mixed() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    a = np.clip(np.where(y == 1, .7, .3) + rng.normal(0, .05, 200), .01, .99)
    b = np.clip(np.where(y == 1, .7, .3) + rng.normal(0, .05, 200), .01, .99)
    weights, _ = pseudo_bma_weights(np.column_stack([a, b]), y, np.repeat(np.arange(20), 10), 500, 1)
    assert .05 < weights[0] < .95


@pytest.mark.parametrize("name", list(LEARNERS))
def test_every_learner_fits_and_returns_probabilities(name: str) -> None:
    data = _dataset()
    X, y = data[FEATURES].to_numpy(), data.rel_binary.to_numpy()
    with threadpool_limits(limits=1):
        model = make_learner(name, _ensemble_options(), 0, depth=3).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and np.all((proba >= 0) & (proba <= 1))


def test_unknown_learner_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        make_learner("svm", _ensemble_options(), 0)


def test_fold_outputs_every_method_and_never_uses_held_out_labels() -> None:
    data = _dataset()
    flipped = data.copy()
    held = flipped.video_id == "d"
    flipped.loc[held, "rel_binary"] = 1 - flipped.loc[held, "rel_binary"]
    with threadpool_limits(limits=1):
        first, detail = fit_fold(data, "d", _cfg())
        second, changed = fit_fold(flipped, "d", _cfg())
    assert set(first.video_id) == {"d"} and detail["training_videos"] == ["a", "b", "c"]
    for method in ENSEMBLE_METHODS:
        assert set(first[f"{method}__pred"].unique()) <= {0, 1}
        assert first[method].between(0, 1).all()
    pd.testing.assert_frame_equal(first.drop(columns="rel_binary"), second.drop(columns="rel_binary"))
    assert detail["methods"] == changed["methods"]
    assert detail["bma"]["weights"] == changed["bma"]["weights"]
    assert sum(detail["bma"]["weights"]) == pytest.approx(1)


def test_ensembles_beat_chance_on_separable_synthetic_data() -> None:
    with threadpool_limits(limits=1):
        result, _ = fit_fold(_dataset(), "d", _cfg())
    for method in ("Random forest", "Stacking", "Soft voting"):
        pred = result[f"{method}__pred"].to_numpy()
        truth = result.rel_binary.to_numpy()
        assert (pred == truth).mean() > .85, method


def test_paired_differences_are_per_video() -> None:
    folds = pd.DataFrame({"video_id": ["a", "b", "a", "b"], "method": ["ref", "ref", "new", "new"],
                          "f1": [.5, .6, .7, .6]})
    row = paired_differences(folds, "ref", ["new"], ["f1"], 100, 0).iloc[0]
    assert row.mean_diff == pytest.approx(.1) and row.videos_better == 1 and row.lo <= row.mean_diff <= row.hi
