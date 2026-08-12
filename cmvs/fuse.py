"""Timeline alignment and calibrated visual/transcript score fusion."""

from __future__ import annotations

import numpy as np


def align_transcript_to_frames(
    frame_timestamps: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    seg_scores: np.ndarray,
) -> np.ndarray:
    """Assign each frame the score of its covering transcript segment.

    Transcript VAD segments may have silent gaps. A frame receives a score only
    when it is in a segment's half-open ``[start, end)`` interval.

    Args:
        frame_timestamps: Frame timestamps in seconds.
        seg_starts: Sorted transcript segment start times.
        seg_ends: Transcript segment end times.
        seg_scores: Query similarity scores for transcript segments.

    Returns:
        One transcript score per frame, with zero for silence or no transcript.
    """
    frames = np.asarray(frame_timestamps, dtype=np.float32)
    starts = np.asarray(seg_starts, dtype=np.float32)
    ends = np.asarray(seg_ends, dtype=np.float32)
    scores = np.asarray(seg_scores, dtype=np.float32)
    if starts.shape != ends.shape or starts.shape != scores.shape:
        raise ValueError("Transcript starts, ends, and scores must have matching shapes.")
    if starts.ndim != 1 or frames.ndim != 1:
        raise ValueError("Frame and transcript timestamps must be one-dimensional.")

    aligned = np.zeros(frames.shape, dtype=np.float32)
    if starts.size == 0:
        return aligned
    candidate_indices = np.searchsorted(starts, frames, side="right") - 1
    valid = candidate_indices >= 0
    valid_indices = candidate_indices[valid]
    covered = frames[valid] < ends[valid_indices]
    aligned_positions = np.flatnonzero(valid)[covered]
    aligned[aligned_positions] = scores[candidate_indices[aligned_positions]]
    return aligned


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize a score array independently to ``[0, 1]``.

    Empty and constant score arrays become zeros so fusion remains stable rather
    than producing division-by-zero values.
    """
    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low = float(values.min())
    high = float(values.max())
    if np.isclose(low, high):
        return np.zeros_like(values, dtype=np.float32)
    return (values - low) / (high - low)


def fuse_scores(visual: np.ndarray, transcript: np.ndarray, alpha: float) -> np.ndarray:
    """Weight already normalized visual and transcript scores on one timeline.

    Args:
        visual: Min-max-normalized visual scores.
        transcript: Min-max-normalized aligned transcript scores.
        alpha: Visual-modality weight in the inclusive interval ``[0, 1]``.

    Returns:
        Fused score per frame.
    """
    visual_scores = np.asarray(visual, dtype=np.float32)
    transcript_scores = np.asarray(transcript, dtype=np.float32)
    if visual_scores.shape != transcript_scores.shape:
        raise ValueError("Visual and transcript score arrays must have matching shapes.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    return alpha * visual_scores + (1.0 - alpha) * transcript_scores
