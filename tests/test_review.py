"""Check HMM inference against exact enumeration and audit held-out isolation."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from analysis.hmm import GaussianHMM, forward_backward, sequence_slices
from analysis.review import evaluate_predictions, fit_fold, ordered_sequences


def test_forward_backward_matches_all_hidden_paths() -> None:
    emission = np.array([[.9, .2], [.3, .8], [.7, .4]])
    start = np.array([.6, .4])
    transition = np.array([[.8, .2], [.3, .7]])
    paths = list(product(range(2), repeat=3))
    probabilities = []
    for states in paths:
        probability = start[states[0]] * emission[0, states[0]]
        for t in range(1, len(states)):
            probability *= transition[states[t - 1], states[t]] * emission[t, states[t]]
        probabilities.append(probability)
    normalizer = sum(probabilities)
    expected_posterior = np.array([
        [sum(p for states, p in zip(paths, probabilities) if states[t] == k) / normalizer for k in range(2)]
        for t in range(3)
    ])
    expected_transitions = np.array([
        [sum(p * sum(states[t] == i and states[t + 1] == j for t in range(2))
             for states, p in zip(paths, probabilities)) / normalizer for j in range(2)]
        for i in range(2)
    ])
    likelihood, posterior, transitions = forward_backward(np.log(emission), start, transition)
    assert np.exp(likelihood) == pytest.approx(normalizer)
    np.testing.assert_allclose(posterior, expected_posterior)
    np.testing.assert_allclose(transitions, expected_transitions)


@pytest.mark.parametrize("lengths", [[0, 3], [2, 2], [1.5, 1.5], [], [[3]]])
def test_invalid_sequence_boundaries_are_rejected(lengths: list) -> None:
    with pytest.raises(ValueError, match="length"):
        sequence_slices(lengths, 3)


def _fixed_hmm() -> GaussianHMM:
    model = GaussianHMM(2, max_iter=10, tol=1e-4, min_covar=.01, pseudocount=.01, n_init=2, random_state=7)
    model.startprob_ = np.array([.6, .4])
    model.transmat_ = np.array([[.8, .2], [.3, .7]])
    model.means_ = np.array([[-1.], [1.]])
    model.covars_ = np.array([[.8], [.6]])
    model.n_features_in_ = 1
    return model


def test_viterbi_matches_exhaustive_joint_path_and_sequences_reset() -> None:
    model = _fixed_hmm()
    X = np.array([[-.9], [.2], [1.]])
    paths = list(product(range(2), repeat=3))
    emission = model._log_emission(X)
    def joint(states: tuple[int, ...]) -> float:
        value = np.log(model.startprob_[states[0]]) + emission[0, states[0]]
        return value + sum(np.log(model.transmat_[states[t - 1], states[t]]) + emission[t, states[t]] for t in range(1, 3))
    assert model.predict(X, [3]).tolist() == list(max(paths, key=joint))
    np.testing.assert_allclose(model.predict_proba(X, [1, 2]), np.vstack([
        model.predict_proba(X[:1], [1]), model.predict_proba(X[1:], [2]),
    ]))
    assert model.score(X, [1, 2]) == pytest.approx(model.score(X[:1], [1]) + model.score(X[1:], [2]))


@pytest.mark.parametrize("method", ["score", "predict", "predict_proba"])
def test_hmm_rejects_wrong_feature_count(method: str) -> None:
    with pytest.raises(ValueError, match="feature count"):
        getattr(_fixed_hmm(), method)(np.ones((3, 2)), [3])


def test_hmm_fitting_and_long_sequence_probabilities_are_finite() -> None:
    rng = np.random.default_rng(2)
    X = np.concatenate([rng.normal(-2, .2, (80, 2)), rng.normal(2, .2, (80, 2))])
    with threadpool_limits(limits=1):
        model = GaussianHMM(2, max_iter=20, tol=1e-5, min_covar=.01, pseudocount=.01, n_init=2, random_state=2).fit(X, [160])
    np.testing.assert_allclose(model.transmat_.sum(axis=1), 1)
    assert model.startprob_.sum() == pytest.approx(1)
    assert np.min(model.covars_) >= .01
    assert model.history_[-1] >= model.history_[0]
    long_X = np.tile(X, (20, 1))
    posterior = model.predict_proba(long_X, [len(long_X)])
    assert np.isfinite(posterior).all() and np.isfinite(model.score(long_X, [len(long_X)]))
    np.testing.assert_allclose(posterior.sum(axis=1), 1)
    assert (np.argmax(posterior[:80], axis=1) == np.argmax(posterior[0])).mean() > .95
    assert np.argmax(posterior[0]) != np.argmax(posterior[100])


def test_held_out_labels_cannot_change_models_or_predictions() -> None:
    rng = np.random.default_rng(4)
    frames = []
    for video in ["a", "b", "c"]:
        for query in range(2):
            t = np.arange(30)
            y = ((t >= 10) & (t < 16)).astype(int)
            frames.append(pd.DataFrame({"video_id": video, "query_id": query, "t": t,
                                        "x": y * 3 + rng.normal(0, .2, len(t)),
                                        "z": y * 2 + rng.normal(0, .3, len(t)), "rel_binary": y}))
    dataset = pd.concat(frames, ignore_index=True)
    options = SimpleNamespace(features=("x", "z"), cluster_counts=(2,), hmm_states=(2,), n_init=2,
                              max_iter=10, tolerance=.001, min_covar=.05, pseudocount=.01, silhouette_sample=100)
    cfg = SimpleNamespace(review=options, ml=SimpleNamespace(seed=4))
    flipped = dataset.copy()
    flipped.loc[flipped.video_id == "c", "rel_binary"] = 1 - flipped.loc[flipped.video_id == "c", "rel_binary"]
    with threadpool_limits(limits=1):
        first, metadata = fit_fold(dataset, "c", cfg)
        second, changed_metadata = fit_fold(flipped, "c", cfg)
    pd.testing.assert_frame_equal(first.drop(columns="rel_binary"), second.drop(columns="rel_binary"))
    assert metadata["training_videos"] == ["a", "b"]
    assert metadata["scaler_mean"] == changed_metadata["scaler_mean"]
    for method in metadata["models"]:
        for key in ["means", "state_relevance", "threshold", "k"]:
            assert metadata["models"][method][key] == changed_metadata["models"][method][key]


def test_ordering_uses_video_and_query_and_rejects_duplicate_seconds() -> None:
    frame = pd.DataFrame({"video_id": ["b", "a", "a"], "query_id": [0, 0, 0], "t": [0, 1, 0]})
    ordered, lengths = ordered_sequences(frame)
    assert lengths == [2, 1] and ordered.video_id.tolist() == ["a", "a", "b"]
    with pytest.raises(ValueError, match="unique"):
        ordered_sequences(pd.concat([frame, frame.iloc[[0]]]))


def test_temporal_iou_and_retrieval_use_all_predicted_seconds() -> None:
    data = pd.DataFrame({"video_id": ["v"] * 4, "query_id": [0] * 4, "t": range(4),
                         "rel_binary": [0, 1, 1, 0], "model": [.1, .9, .8, .7], "model__pred": [0, 1, 1, 1]})
    row = evaluate_predictions(data, ["model"], 42).iloc[0]
    assert row.temporal_iou == pytest.approx(2 / 3)
    assert row.hit_at_1 == 1 and row.mrr == 1 and row.f1 == pytest.approx(.8)


def test_posterior_roundoff_cannot_change_retrieval_rankings() -> None:
    data = pd.DataFrame({"video_id": ["v"] * 8, "query_id": [0] * 8, "t": range(8),
                         "rel_binary": [0, 0, 1, 1, 0, 0, 0, 0], "model": [.4] * 8,
                         "model__pred": [1] * 8})
    perturbed = data.copy()
    perturbed["model"] += np.arange(8) * 1e-16
    original = evaluate_predictions(data, ["model"], 42)
    changed = evaluate_predictions(perturbed, ["model"], 42)
    pd.testing.assert_frame_equal(original, changed)
