"""The three regression and three classification models, each an sklearn Pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REGRESSORS: dict[str, str] = {
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}
CLASSIFIERS: dict[str, str] = {
    "logistic": "Logistic regression",
    "knn": "k-nearest neighbours",
    "gradient_boosting": "Gradient boosting",
}


def make_model(name: str, task: str, seed: int, params: dict[str, Any] | None = None) -> Pipeline:
    """Build one model as ``Pipeline([scale, model])`` so grid keys are always ``model__<param>``.

    Args:
        name: A key of ``REGRESSORS`` / ``CLASSIFIERS``, or ``"dummy"`` for the mean/prior baseline.
        task: ``"regression"`` or ``"classification"``.
        seed: Random seed for stochastic models.
        params: Optional ``model__<param>`` overrides (e.g. a grid search winner).
    """
    scaled = Pipeline  # alias for readability below
    if task == "regression":
        table = {
            "dummy": ("passthrough", DummyRegressor(strategy="mean")),
            "ridge": (StandardScaler(), Ridge()),
            "random_forest": ("passthrough", RandomForestRegressor(n_jobs=1, random_state=seed)),
            "gradient_boosting": ("passthrough", HistGradientBoostingRegressor(random_state=seed)),
        }
    elif task == "classification":
        table = {
            "dummy": ("passthrough", DummyClassifier(strategy="prior")),
            "logistic": (
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
            ),
            "knn": (StandardScaler(), KNeighborsClassifier(n_jobs=1)),
            "gradient_boosting": (
                "passthrough",
                HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
            ),
        }
    else:
        raise ValueError(f"Unknown task: {task}")
    if name not in table:
        raise ValueError(f"Unknown {task} model: {name}")
    scaler, estimator = table[name]
    pipeline = scaled([("scale", scaler), ("model", estimator)])
    if params:
        pipeline.set_params(**params)
    return pipeline


def param_grid(name: str, grids: dict[str, dict[str, list]]) -> dict[str, list]:
    """Prefix a configured grid with ``model__`` for the pipeline."""
    return {f"model__{key}": list(values) for key, values in grids[name].items()}
