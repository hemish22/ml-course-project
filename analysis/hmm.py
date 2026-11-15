"""Diagonal Gaussian HMM: Baum-Welch, scaled forward-backward, and Viterbi.

No encoder training or external HMM library is involved. Sequence lengths are
mandatory, so unrelated query/video timelines never create artificial transitions.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted


def sequence_slices(lengths: list[int] | np.ndarray, n_rows: int) -> list[slice]:
    """Validate positive integer lengths and return their disjoint row slices."""
    raw = np.asarray(lengths)
    if raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("Sequence lengths must be a nonempty array of integers.")
    if np.any(raw <= 0) or int(raw.sum()) != n_rows:
        raise ValueError("Positive sequence lengths must sum to the number of rows.")
    ends = np.cumsum(raw)
    return [slice(int(a), int(b)) for a, b in zip(np.r_[0, ends[:-1]], ends)]


def forward_backward(
    log_emission: np.ndarray, start: np.ndarray, transition: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return sequence log likelihood, state posteriors, and expected transitions.

    Emissions are shifted per timestamp and forward probabilities are scaled each
    step. The backward recursion uses the same scales to avoid underflow.
    """
    shifts = log_emission.max(axis=1)
    emission = np.exp(log_emission - shifts[:, None])
    n, k = emission.shape
    alpha, beta = np.empty((n, k)), np.ones((n, k))
    scales = np.empty(n)
    alpha[0] = start * emission[0]
    scales[0] = alpha[0].sum()
    alpha[0] /= scales[0]
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ transition) * emission[t]
        scales[t] = alpha[t].sum()
        alpha[t] /= scales[t]
    for t in range(n - 2, -1, -1):
        beta[t] = transition @ (emission[t + 1] * beta[t + 1]) / scales[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = (
        alpha[:-1, :, None] * transition[None, :, :]
        * (emission[1:] * beta[1:])[:, None, :] / scales[1:, None, None]
    ).sum(axis=0)
    return float(np.log(scales).sum() + shifts.sum()), gamma, xi


class GaussianHMM(BaseEstimator):
    """Unsupervised diagonal Gaussian HMM with explicit sequence boundaries.

    Fit accepts no labels. K-means initializes Gaussian means and transition
    counts; Baum-Welch estimates all parameters by maximum likelihood with small
    transition pseudocounts and a variance floor. A single deterministic seeded
    initialization is used; local optima remain a limitation.
    """

    def __init__(
        self, n_components: int, *, max_iter: int, tol: float, min_covar: float,
        pseudocount: float, n_init: int, random_state: int,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.min_covar = min_covar
        self.pseudocount = pseudocount
        self.n_init = n_init
        self.random_state = random_state

    def _validate(self, X: np.ndarray, lengths: list[int] | np.ndarray) -> tuple[np.ndarray, list[slice]]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.shape[1] or not np.isfinite(X).all():
            raise ValueError("HMM input must be a finite two-dimensional feature matrix.")
        return X, sequence_slices(lengths, len(X))

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError("HMM feature count differs from the fitted data.")
        difference = X[:, None, :] - self.means_[None, :, :]
        return -0.5 * (
            np.log(2 * np.pi * self.covars_).sum(axis=1)[None, :]
            + (difference ** 2 / self.covars_[None, :, :]).sum(axis=2)
        )

    def fit(self, X: np.ndarray, lengths: list[int] | np.ndarray) -> GaussianHMM:
        """Estimate parameters from unlabeled, time-ordered training sequences."""
        X, slices = self._validate(X, lengths)
        if (self.n_components < 2 or self.n_components > len(X) or self.max_iter < 1
                or self.min_covar <= 0 or self.pseudocount <= 0 or self.tol < 0):
            raise ValueError("Invalid HMM state count, iterations, tolerance, or regularization.")
        km = KMeans(n_clusters=self.n_components, n_init=self.n_init, random_state=self.random_state).fit(X)
        labels = km.labels_
        self.means_ = km.cluster_centers_.copy()
        self.covars_ = np.array([
            np.maximum(X[labels == s].var(axis=0), self.min_covar)
            if np.any(labels == s) else np.maximum(X.var(axis=0), self.min_covar)
            for s in range(self.n_components)
        ])
        starts = np.full(self.n_components, self.pseudocount)
        counts = np.full((self.n_components, self.n_components), self.pseudocount)
        for sl in slices:
            states = labels[sl]
            starts[states[0]] += 1
            np.add.at(counts, (states[:-1], states[1:]), 1)
        self.startprob_ = starts / starts.sum()
        self.transmat_ = counts / counts.sum(axis=1, keepdims=True)
        self.history_: list[float] = []
        self.converged_ = False
        for _ in range(self.max_iter):
            log_emission = self._log_emission(X)
            gamma = np.empty((len(X), self.n_components))
            start_counts = np.full(self.n_components, self.pseudocount)
            trans_counts = np.full_like(counts, self.pseudocount)
            likelihood = 0.0
            for sl in slices:
                score, posterior, xi = forward_backward(log_emission[sl], self.startprob_, self.transmat_)
                likelihood += score
                gamma[sl] = posterior
                start_counts += posterior[0]
                trans_counts += xi
            self.history_.append(likelihood)
            # The current parameters produced this likelihood. Stop before another update.
            if len(self.history_) > 1 and abs(likelihood - self.history_[-2]) / len(X) < self.tol:
                self.converged_ = True
                break
            self.startprob_ = start_counts / start_counts.sum()
            self.transmat_ = trans_counts / trans_counts.sum(axis=1, keepdims=True)
            weights = np.maximum(gamma.sum(axis=0), np.finfo(float).eps)
            self.means_ = gamma.T @ X / weights[:, None]
            self.covars_ = np.maximum(gamma.T @ (X * X) / weights[:, None] - self.means_ ** 2, self.min_covar)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X: np.ndarray, lengths: list[int] | np.ndarray) -> np.ndarray:
        """Smooth state probabilities within each complete offline video timeline."""
        check_is_fitted(self, "n_features_in_")
        X, slices = self._validate(X, lengths)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("HMM feature count differs from the fitted data.")
        log_emission = self._log_emission(X)
        return np.concatenate([
            forward_backward(log_emission[sl], self.startprob_, self.transmat_)[1] for sl in slices
        ])

    def score(self, X: np.ndarray, lengths: list[int] | np.ndarray) -> float:
        """Sum independent sequence log likelihoods under the fitted model."""
        check_is_fitted(self, "n_features_in_")
        X, slices = self._validate(X, lengths)
        log_emission = self._log_emission(X)
        return sum(forward_backward(log_emission[sl], self.startprob_, self.transmat_)[0] for sl in slices)

    def bic(self, X: np.ndarray, lengths: list[int] | np.ndarray) -> float:
        """Training BIC: -2 log L + free-parameter count * log(observations)."""
        k, d = self.n_components, np.shape(X)[1]
        parameters = (k - 1) + k * (k - 1) + 2 * k * d
        return float(-2 * self.score(X, lengths) + parameters * np.log(len(X)))

    def predict(self, X: np.ndarray, lengths: list[int] | np.ndarray) -> np.ndarray:
        """Decode the most probable joint hidden-state path with Viterbi."""
        check_is_fitted(self, "n_features_in_")
        X, slices = self._validate(X, lengths)
        emission = self._log_emission(X)
        paths = []
        for sl in slices:
            obs = emission[sl]
            delta = np.log(self.startprob_) + obs[0]
            back = np.zeros((len(obs), self.n_components), dtype=int)
            for t in range(1, len(obs)):
                candidates = delta[:, None] + np.log(self.transmat_)
                back[t] = candidates.argmax(axis=0)
                delta = candidates.max(axis=0) + obs[t]
            path = np.empty(len(obs), dtype=int)
            path[-1] = delta.argmax()
            for t in range(len(obs) - 2, -1, -1):
                path[t] = back[t + 1, path[t + 1]]
            paths.append(path)
        return np.concatenate(paths)
