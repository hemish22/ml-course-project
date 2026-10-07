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

FEATURES_BASE: list[str] = [
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
FEATURES_CONTEXT: list[str] = [
    "visual_near",
    "speech_near",
    "visual_wide",
    "speech_wide",
    "fused_smooth",
    "fused_near",
    "visual_peak_dist",
    "speech_peak_dist",
    "fused_peak_dist",
    "visual_vs_peak",
    "speech_vs_peak",
    "fused_vs_peak",
    "local_rank_fused",
    "scene_cuts_near",
    "keyword_overlap",
    "keyword_overlap_near",
]
FEATURES: list[str] = FEATURES_BASE + FEATURES_CONTEXT
KEYS = ["query_id", "video_id", "t"]
_STOPWORDS = frozenset(
    "the a an and or of to in on at for with from by is are was were be as it its this that these those "
    "into over about after before up down out off than then so but not no".split()
)
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


def _content_words(text: str) -> set[str]:
    """Lower-cased words longer than 2 letters that are not stop words."""
    words = ("".join(c if c.isalnum() else " " for c in text.lower())).split()
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def keyword_overlap(timeline: ScoreTimeline, query_text: str) -> np.ndarray:
    """Share of the query's content words found in the transcript segment covering each second."""
    words = _content_words(query_text)
    segments = timeline.transcript_meta
    overlap = np.zeros(len(timeline.timestamps), dtype=np.float32)
    if not words or not segments["start"]:
        return overlap
    shares = np.array([len(words & _content_words(text)) / len(words) for text in segments["text"]], dtype=np.float32)
    return align_transcript_to_frames(
        timeline.timestamps,
        np.asarray(segments["start"], dtype=np.float32),
        np.asarray(segments["end"], dtype=np.float32),
        shares,
    )


def _peak_distance(smooth: np.ndarray, t: np.ndarray, empty_value: float) -> np.ndarray:
    """Seconds from each frame to the highest-scoring frame (``empty_value`` if everything is 0)."""
    if smooth.max() <= 0:
        return np.full(len(t), empty_value, dtype=np.float32)
    return np.abs(t - t[int(np.argmax(smooth))]).astype(np.float32)


def _vs_peak(smooth: np.ndarray) -> np.ndarray:
    top = float(smooth.max())
    return (smooth / top if top > 0 else np.zeros_like(smooth)).astype(np.float32)


def _local_rank(values: np.ndarray, radius: int) -> np.ndarray:
    """Fraction of frames within +/-radius seconds that score below each frame."""
    out = np.zeros(len(values), dtype=np.float32)
    for i in range(len(values)):
        window = values[max(0, i - radius) : i + radius + 1]
        out[i] = float((window < values[i]).sum()) / max(len(window) - 1, 1)
    return out


def features_for_query(
    timeline: ScoreTimeline,
    embeddings: np.ndarray,
    duration: float,
    query: BenchmarkQuery,
    ml: object,
) -> pd.DataFrame:
    """Turn one query's score timeline into labeled per-second feature rows.

    Args:
        timeline: Calibrated per-frame scores for the query.
        embeddings: Frame embeddings of the video, for the frame-change feature.
        duration: Video length in seconds.
        query: The benchmark query with its true interval.
        ml: ``config.ml`` settings (alpha, windows, decay, scene-cut threshold).
    """
    baseline_alpha, smooth_window, decay_s = ml.baseline_alpha, ml.smooth_window, ml.relevance_decay_s
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
    fused = (baseline_alpha * timeline.visual + (1 - baseline_alpha) * timeline.transcript).astype(np.float32)
    change = frame_change(embeddings)
    near, wide = 2 * ml.near_window + 1, 2 * ml.wide_window + 1
    visual_smooth = uniform_filter1d(timeline.visual, width, mode="nearest")
    speech_smooth = uniform_filter1d(timeline.transcript, width, mode="nearest")
    fused_smooth = uniform_filter1d(fused, width, mode="nearest")
    overlap = keyword_overlap(timeline, query.query)
    frame = pd.DataFrame(
        {
            "query_id": query.query_id,
            "video_id": query.video_id,
            "t": t,
            "visual_raw": visual_raw,
            "speech_raw": speech_raw,
            "visual_norm": timeline.visual,
            "speech_norm": timeline.transcript,
            "fused_baseline": fused,
            "visual_z": _zscore(visual_raw),
            "speech_z": _zscore(speech_raw, in_speech),
            "visual_rank": _percentile_rank(visual_raw),
            "speech_rank": _percentile_rank(speech_raw),
            "visual_smooth": visual_smooth,
            "speech_smooth": speech_smooth,
            "frame_change": change,
            "in_speech": in_speech.astype(np.int8),
            "rel_pos": t / max(duration, 1e-6),
            "query_words": len(query.query.split()),
            "visual_near": uniform_filter1d(timeline.visual, near, mode="nearest"),
            "speech_near": uniform_filter1d(timeline.transcript, near, mode="nearest"),
            "visual_wide": uniform_filter1d(timeline.visual, wide, mode="nearest"),
            "speech_wide": uniform_filter1d(timeline.transcript, wide, mode="nearest"),
            "fused_smooth": fused_smooth,
            "fused_near": uniform_filter1d(fused, near, mode="nearest"),
            "visual_peak_dist": _peak_distance(visual_smooth, t, duration),
            "speech_peak_dist": _peak_distance(speech_smooth, t, duration),
            "fused_peak_dist": _peak_distance(fused_smooth, t, duration),
            "visual_vs_peak": _vs_peak(visual_smooth),
            "speech_vs_peak": _vs_peak(speech_smooth),
            "fused_vs_peak": _vs_peak(fused_smooth),
            "local_rank_fused": _local_rank(fused_smooth, ml.wide_window),
            "scene_cuts_near": pd.Series((change > ml.scene_cut_threshold).astype(np.float32)).rolling(
                near, center=True, min_periods=1).sum().to_numpy(),
            "keyword_overlap": overlap,
            "keyword_overlap_near": uniform_filter1d(overlap, near, mode="nearest"),
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
                cfg.ml,
            )
        )
    return pd.concat(tables, ignore_index=True)
