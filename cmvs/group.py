"""Convert high-scoring timeline frames into non-duplicated moments."""

from __future__ import annotations

from typing import Any

import numpy as np


def _make_moment(
    indices: np.ndarray,
    timestamps: np.ndarray,
    fused: np.ndarray,
    visual: np.ndarray,
    transcript: np.ndarray,
    min_moment_s: float,
) -> dict[str, Any]:
    """Summarize one contiguous run of kept frames around its peak."""
    peak_idx = int(indices[np.argmax(fused[indices])])
    start = float(timestamps[indices].min())
    end = float(timestamps[indices].max())
    if end - start < min_moment_s:
        peak_time = float(timestamps[peak_idx])
        start = max(0.0, peak_time - min_moment_s / 2.0)
        end = start + min_moment_s
    return {
        "start": start,
        "end": end,
        "score": float(fused[peak_idx]),
        "visual_score": float(visual[peak_idx]),
        "transcript_score": float(transcript[peak_idx]),
        "peak_idx": peak_idx,
    }


def group_frames_into_moments(
    timestamps: np.ndarray,
    fused: np.ndarray,
    visual: np.ndarray,
    transcript: np.ndarray,
    cfg: object,
) -> list[dict]:
    """Group qualifying frames into ranked, temporally distinct moments.

    Args:
        timestamps: Frame timestamps in seconds.
        fused: Fused score per frame.
        visual: Normalized visual score per frame.
        transcript: Normalized aligned transcript score per frame.
        cfg: Application configuration containing grouping thresholds.

    Returns:
        Moment dictionaries sorted by their peak fused score.
    """
    timeline = np.asarray(timestamps, dtype=np.float32)
    fused_scores = np.asarray(fused, dtype=np.float32)
    visual_scores = np.asarray(visual, dtype=np.float32)
    transcript_scores = np.asarray(transcript, dtype=np.float32)
    if not (
        timeline.shape == fused_scores.shape == visual_scores.shape == transcript_scores.shape
    ) or timeline.ndim != 1:
        raise ValueError("All frame timeline and score arrays must be matching one-dimensional arrays.")

    kept = np.flatnonzero(fused_scores >= cfg.search.score_threshold)
    if kept.size == 0:
        return []
    kept = kept[np.argsort(timeline[kept], kind="stable")]
    split_after = np.flatnonzero(np.diff(timeline[kept]) > cfg.search.merge_gap_s) + 1
    groups = np.split(kept, split_after)
    moments = [
        _make_moment(
            indices,
            timeline,
            fused_scores,
            visual_scores,
            transcript_scores,
            cfg.search.min_moment_s,
        )
        for indices in groups
    ]
    moments.sort(key=lambda moment: moment["score"], reverse=True)
    return moments[: cfg.search.max_results]
