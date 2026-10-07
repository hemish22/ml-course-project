"""Per-(query, second) feature table built from the frozen search pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from analysis.benchmark import BenchmarkQuery, relevance_labels, validate_against_index
from cmvs.fuse import align_transcript_to_frames
from cmvs.index import load_index
from cmvs.search import ScoreTimeline, score_video

FEATURES: list[str] = [
    "visual_raw",
    "speech_raw",
    "visual_norm",
    "speech_norm",
    "fused_baseline",
    "visual_z",
    "speech_z",
    "visual_rank",
    "speech_rank",
    "visual_smooth",
    "speech_smooth",
    "frame_change",
    "in_speech",
    "rel_pos",
    "query_words",
]
KEYS = ["query_id", "video_id", "t"]
TARGETS = ["rel_graded", "rel_binary"]


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Rank of each value as a fraction in ``[0, 1]`` (1 = highest)."""
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float32)
    order = values.argsort(kind="stable").argsort(kind="stable")
    return (order / (values.size - 1)).astype(np.float32)


def _zscore(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score using only ``mask`` rows for the statistics; masked-out rows become 0."""
    keep = np.ones(values.shape, dtype=bool) if mask is None else mask.astype(bool)
    if keep.sum() < 2:
        return np.zeros_like(values, dtype=np.float32)
    mean, std = values[keep].mean(), values[keep].std()
    out = np.zeros_like(values, dtype=np.float32)
    out[keep] = (values[keep] - mean) / max(float(std), 1e-6)
    return out


def frame_change(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance between each frame embedding and the previous one (0 for the first)."""
    change = np.zeros(len(embeddings), dtype=np.float32)
    if len(embeddings) > 1:
        change[1:] = 1.0 - np.einsum("ij,ij->i", embeddings[1:], embeddings[:-1])
    return np.clip(change, 0.0, 2.0)


def features_for_query(
    timeline: ScoreTimeline,
    embeddings: np.ndarray,
    duration: float,
    query: BenchmarkQuery,
    baseline_alpha: float,
    smooth_window: int,
    decay_s: float,
) -> pd.DataFrame:
    """Turn one query's score timeline into labeled per-second feature rows."""
    t = timeline.timestamps.astype(np.float64)
    segments = timeline.transcript_meta
    in_speech = align_transcript_to_frames(
        timeline.timestamps,
        np.asarray(segments["start"], dtype=np.float32),
        np.asarray(segments["end"], dtype=np.float32),
        np.ones(len(segments["start"]), dtype=np.float32),
    ) > 0
    width = 2 * smooth_window + 1
    graded, binary = relevance_labels(t, query.start, query.end, decay_s)
    visual_raw = np.asarray(timeline.visual_raw, dtype=np.float32)
    speech_raw = np.asarray(timeline.transcript_raw, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "query_id": query.query_id,
            "video_id": query.video_id,
            "t": t,
            "visual_raw": visual_raw,
            "speech_raw": speech_raw,
            "visual_norm": timeline.visual,
            "speech_norm": timeline.transcript,
            "fused_baseline": baseline_alpha * timeline.visual + (1 - baseline_alpha) * timeline.transcript,
            "visual_z": _zscore(visual_raw),
            "speech_z": _zscore(speech_raw, in_speech),
            "visual_rank": _percentile_rank(visual_raw),
            "speech_rank": _percentile_rank(speech_raw),
            "visual_smooth": uniform_filter1d(timeline.visual, width, mode="nearest"),
            "speech_smooth": uniform_filter1d(timeline.transcript, width, mode="nearest"),
            "frame_change": frame_change(embeddings),
            "in_speech": in_speech.astype(np.int8),
            "rel_pos": t / max(duration, 1e-6),
            "query_words": len(query.query.split()),
            "rel_graded": graded,
            "rel_binary": binary,
        }
    )
    return frame


def build_dataset(cfg: object, queries: list[BenchmarkQuery]) -> pd.DataFrame:
    """Score every benchmark query against its video and assemble the feature table.

    All frames and segments are scored (not just the usual top-k) so raw similarities
    exist for every second.
    """
    validate_against_index(queries, Path(cfg.paths.index))
    exhaustive = replace(cfg, search=replace(cfg.search, top_k_frames=10**6))
    tables: list[pd.DataFrame] = []
    embedding_cache: dict[str, np.ndarray] = {}
    durations: dict[str, float] = {}
    for query in queries:
        index_dir = Path(cfg.paths.index) / query.video_id
        if query.video_id not in embedding_cache:
            index, meta = load_index(index_dir, "visual")
            embedding_cache[query.video_id] = index.reconstruct_n(0, index.ntotal)
            durations[query.video_id] = float(len(meta["timestamp"])) / cfg.extract.fps
        timeline = score_video(query.query, query.video_id, exhaustive, cfg.ml.baseline_alpha)
        tables.append(
            features_for_query(
                timeline,
                embedding_cache[query.video_id],
                durations[query.video_id],
                query,
                cfg.ml.baseline_alpha,
                cfg.ml.smooth_window,
                cfg.ml.relevance_decay_s,
            )
        )
    return pd.concat(tables, ignore_index=True)
