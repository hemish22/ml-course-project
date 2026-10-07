"""Hand-curated (query, video, true interval) benchmark and its per-second labels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BenchmarkQuery:
    """One search query with the interval of its true answer."""

    query_id: int
    video_id: str
    query: str
    start: float
    end: float


def load_benchmark(path: Path) -> list[BenchmarkQuery]:
    """Read ``queries.json`` (a list of video/query/start/end objects) and number the queries.

    Args:
        path: Benchmark JSON path from ``config.yaml``.

    Returns:
        Queries in file order with consecutive ``query_id`` values.
    """
    with Path(path).open(encoding="utf-8") as benchmark_file:
        rows = json.load(benchmark_file)
    queries = [
        BenchmarkQuery(index, row["video"], row["query"].strip(), float(row["start"]), float(row["end"]))
        for index, row in enumerate(rows)
    ]
    for query in queries:
        if not query.query or query.end <= query.start or query.start < 0:
            raise ValueError(f"Invalid benchmark row {query.query_id}: {query}")
    return queries


def validate_against_index(queries: list[BenchmarkQuery], index_dir: Path) -> None:
    """Check every query refers to an indexed video and lies inside its duration."""
    durations: dict[str, float] = {}
    for query in queries:
        if query.video_id not in durations:
            meta_path = Path(index_dir) / query.video_id / "meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Benchmark video '{query.video_id}' is not indexed.")
            with meta_path.open(encoding="utf-8") as meta_file:
                durations[query.video_id] = float(json.load(meta_file)["duration"])
        if query.end > durations[query.video_id] + 1.0:
            raise ValueError(
                f"Query {query.query_id} ends at {query.end}s but '{query.video_id}' "
                f"is only {durations[query.video_id]:.0f}s long."
            )


def relevance_labels(
    timestamps: np.ndarray, start: float, end: float, decay_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-second ground truth for one query.

    Args:
        timestamps: Frame timestamps in seconds.
        start: True interval start.
        end: True interval end.
        decay_s: Decay length; relevance outside the interval is ``exp(-distance / decay_s)``.

    Returns:
        ``(graded, binary)``: graded relevance in ``[0, 1]`` (regression target) and
        a 0/1 inside-the-interval flag (classification target).
    """
    t = np.asarray(timestamps, dtype=np.float64)
    inside = (t >= start) & (t < end)
    distance = np.where(t < start, start - t, np.where(t >= end, t - end, 0.0))
    graded = np.where(inside, 1.0, np.exp(-distance / decay_s))
    return graded.astype(np.float32), inside.astype(np.int8)
