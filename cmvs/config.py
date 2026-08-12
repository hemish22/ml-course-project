"""Configuration loading for Cross-Modal Video Search."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by the application."""

    raw: Path
    processed: Path
    index: Path
    results: Path


@dataclass(frozen=True)
class ExtractConfig:
    """Video and audio extraction settings."""

    fps: float
    frame_width: int
    jpeg_quality: int
    use_scene_detect: bool
    audio_sample_rate: int


@dataclass(frozen=True)
class ModelsConfig:
    """Frozen model identifiers and inference settings."""

    clip_name: str
    clip_pretrained: str
    whisper_size: str
    whisper_compute_type: str
    text_model: str


@dataclass(frozen=True)
class EncodeConfig:
    """Embedding generation settings."""

    batch_size: int


@dataclass(frozen=True)
class SearchConfig:
    """Retrieval, fusion, and moment grouping settings."""

    alpha: float
    top_k_frames: int
    score_threshold: float
    merge_gap_s: float
    min_moment_s: float
    max_results: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation and ablation settings."""

    alpha_values: tuple[float, ...]
    pooling_methods: tuple[str, ...]
    clip_models: tuple[str, ...]
    moment_iou_thresholds: tuple[float, ...]
    plot_dpi: int


@dataclass(frozen=True)
class Config:
    """The complete immutable application configuration."""

    paths: PathsConfig
    extract: ExtractConfig
    models: ModelsConfig
    encode: EncodeConfig
    search: SearchConfig
    evaluation: EvaluationConfig


def _require_mapping(value: object, section: str) -> dict[str, Any]:
    """Return a YAML section as a mapping or raise a clear configuration error."""
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{section}' must be a mapping.")
    return value


def _build_config(data: dict[str, Any]) -> Config:
    """Convert parsed YAML configuration into immutable typed settings."""
    try:
        paths = _require_mapping(data["paths"], "paths")
        extract = _require_mapping(data["extract"], "extract")
        models = _require_mapping(data["models"], "models")
        encode = _require_mapping(data["encode"], "encode")
        search = _require_mapping(data["search"], "search")
        evaluation = _require_mapping(data["evaluation"], "evaluation")
        return Config(
            paths=PathsConfig(
                raw=Path(paths["raw"]),
                processed=Path(paths["processed"]),
                index=Path(paths["index"]),
                results=Path(paths["results"]),
            ),
            extract=ExtractConfig(**extract),
            models=ModelsConfig(**models),
            encode=EncodeConfig(**encode),
            search=SearchConfig(**search),
            evaluation=EvaluationConfig(
                alpha_values=tuple(evaluation["alpha_values"]),
                pooling_methods=tuple(evaluation["pooling_methods"]),
                clip_models=tuple(evaluation["clip_models"]),
                moment_iou_thresholds=tuple(evaluation["moment_iou_thresholds"]),
                plot_dpi=evaluation["plot_dpi"],
            ),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc


@lru_cache(maxsize=None)
def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and cache a validated immutable configuration from a YAML file.

    Args:
        path: Path to the project's configuration file.

    Returns:
        Parsed project configuration.

    Raises:
        FileNotFoundError: If the requested configuration file is absent.
        ValueError: If the YAML root or required sections are invalid.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping.")
    return _build_config(data)
