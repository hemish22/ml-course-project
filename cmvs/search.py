"""Cross-modal query APIs returning timestamped video moments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cmvs.encode_text import encode_query_text
from cmvs.encode_visual import encode_query_visual
from cmvs.fuse import align_transcript_to_frames, fuse_scores, minmax_normalize
from cmvs.group import group_frames_into_moments
from cmvs.index import load_index, search as search_index


@dataclass(frozen=True)
class Moment:
    """A ranked, timestamped multimodal search result."""

    video_id: str
    start: float
    end: float
    score: float
    visual_score: float
    transcript_score: float
    thumbnail_path: str
    transcript_text: str


def _scatter_scores(size: int, scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Scatter sparse FAISS scores back to their dense metadata-row timeline."""
    dense = np.zeros(size, dtype=np.float32)
    for score, row_id in zip(scores.ravel(), ids.ravel()):
        if row_id >= 0:
            dense[int(row_id)] = max(dense[int(row_id)], float(score))
    return dense


def _overlapping_text(meta: dict[str, Any], start: float, end: float) -> str:
    """Join transcript segments that overlap a result moment."""
    return " ".join(
        text
        for segment_start, segment_end, text in zip(
            meta["start"], meta["end"], meta["text"]
        )
        if segment_start < end and segment_end > start
    )


@dataclass(frozen=True)
class ScoreTimeline:
    """Calibrated per-frame scores for one query over one video."""

    timestamps: np.ndarray
    visual: np.ndarray
    transcript: np.ndarray
    fused: np.ndarray
    visual_meta: dict[str, Any]
    transcript_meta: dict[str, Any]


def score_video(
    query: str, video_id: str, cfg: object, alpha: float | None = None
) -> ScoreTimeline:
    """Score every frame of one indexed video against a query.

    Args:
        query: Natural-language search query.
        video_id: Indexed video identifier.
        cfg: Application configuration.
        alpha: Optional override for the configured visual fusion weight.

    Returns:
        Min-max-normalized visual and transcript scores plus their fusion.
    """
    index_dir = Path(cfg.paths.index) / video_id
    visual_index, visual_meta = load_index(index_dir, "visual")
    transcript_index, transcript_meta = load_index(index_dir, "transcript")
    frame_timestamps = np.asarray(visual_meta["timestamp"], dtype=np.float32)
    if frame_timestamps.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return ScoreTimeline(frame_timestamps, empty, empty, empty, visual_meta, transcript_meta)

    visual_scores, visual_ids = search_index(
        visual_index,
        encode_query_visual(query, cfg),
        cfg.search.top_k_frames,
    )
    dense_visual = _scatter_scores(len(frame_timestamps), visual_scores, visual_ids)

    if transcript_index.ntotal:
        transcript_scores, transcript_ids = search_index(
            transcript_index,
            encode_query_text(query, cfg),
            cfg.search.top_k_frames,
        )
        dense_segments = _scatter_scores(len(transcript_meta["seg_idx"]), transcript_scores, transcript_ids)
    else:
        dense_segments = np.empty(0, dtype=np.float32)

    aligned_transcript = align_transcript_to_frames(
        frame_timestamps,
        np.asarray(transcript_meta["start"], dtype=np.float32),
        np.asarray(transcript_meta["end"], dtype=np.float32),
        dense_segments,
    )
    normalized_visual = minmax_normalize(dense_visual)
    normalized_transcript = minmax_normalize(aligned_transcript)
    selected_alpha = cfg.search.alpha if alpha is None else alpha
    return ScoreTimeline(
        frame_timestamps,
        normalized_visual,
        normalized_transcript,
        fuse_scores(normalized_visual, normalized_transcript, selected_alpha),
        visual_meta,
        transcript_meta,
    )


def moments_from_timeline(timeline: ScoreTimeline, video_id: str, cfg: object) -> list[Moment]:
    """Group a score timeline into ranked, timestamped moments."""
    if timeline.timestamps.size == 0:
        return []
    grouped = group_frames_into_moments(
        timeline.timestamps,
        timeline.fused,
        timeline.visual,
        timeline.transcript,
        cfg,
    )
    return [
        Moment(
            video_id=video_id,
            start=moment["start"],
            end=moment["end"],
            score=moment["score"],
            visual_score=moment["visual_score"],
            transcript_score=moment["transcript_score"],
            thumbnail_path=timeline.visual_meta["path"][moment["peak_idx"]],
            transcript_text=_overlapping_text(timeline.transcript_meta, moment["start"], moment["end"]),
        )
        for moment in grouped
    ]


def search_video(
    query: str, video_id: str, cfg: object, alpha: float | None = None
) -> list[Moment]:
    """Search one indexed video and return its ranked fused moments.

    Args:
        query: Natural-language search query.
        video_id: Indexed video identifier.
        cfg: Application configuration.
        alpha: Optional override for the configured visual fusion weight.

    Returns:
        Ranked timestamped search moments.
    """
    return moments_from_timeline(score_video(query, video_id, cfg, alpha), video_id, cfg)


def search_library(query: str, cfg: object, alpha: float | None = None) -> list[Moment]:
    """Search every completely indexed video and return globally ranked moments.

    Args:
        query: Natural-language search query.
        cfg: Application configuration.
        alpha: Optional override for the configured visual fusion weight.

    Returns:
        Globally ranked moments, capped by ``cfg.search.max_results``.
    """
    index_root = Path(cfg.paths.index)
    if not index_root.exists():
        return []
    moments = [
        moment
        for child in sorted(index_root.iterdir())
        if child.is_dir() and (child / "meta.json").exists()
        for moment in search_video(query, child.name, cfg, alpha)
    ]
    moments.sort(key=lambda moment: moment.score, reverse=True)
    return moments[: cfg.search.max_results]
