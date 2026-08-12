"""MSR-VTT clip-level retrieval evaluation over pre-indexed frame embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cmvs.encode_visual import encode_query_visual
from cmvs.index import load_index
from cmvs.utils import l2_normalize


def pool_frame_embeddings(frame_embeddings: np.ndarray, method: str) -> np.ndarray:
    """Pool one clip's normalized frame vectors into a normalized clip vector.

    Args:
        frame_embeddings: Frame-vector matrix for one clip.
        method: Configured ``mean`` or ``max`` aggregation method.

    Returns:
        One shape ``(D,)`` unit clip vector.
    """
    vectors = np.asarray(frame_embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Each evaluated clip must contain at least one frame embedding.")
    if method == "mean":
        pooled = vectors.mean(axis=0)
    elif method == "max":
        pooled = vectors.max(axis=0)
    else:
        raise ValueError(f"Unknown pooling method: {method}")
    return l2_normalize(pooled)[0]


def load_indexed_frame_embeddings(video_ids: Sequence[str], cfg: object) -> list[np.ndarray]:
    """Read complete visual vector matrices from exact FAISS indices.

    Args:
        video_ids: Indexed video IDs in dataset ground-truth order.
        cfg: Application configuration.

    Returns:
        One frame-vector matrix for each requested video.
    """
    vectors: list[np.ndarray] = []
    for video_id in video_ids:
        index, _ = load_index(Path(cfg.paths.index) / video_id, "visual")
        vectors.append(index.reconstruct_n(0, index.ntotal))
    return vectors


def _retrieval_metrics(similarities: np.ndarray) -> dict[str, float]:
    """Compute standard paired query-to-clip retrieval metrics."""
    if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1]:
        raise ValueError("Similarity matrix must be square with paired query/clip order.")
    ranks = []
    for query_idx, scores in enumerate(similarities):
        ranked_ids = np.argsort(-scores, kind="stable")
        ranks.append(int(np.flatnonzero(ranked_ids == query_idx)[0]) + 1)
    rank_array = np.asarray(ranks)
    return {
        "recall_at_1": float(np.mean(rank_array <= 1)),
        "recall_at_5": float(np.mean(rank_array <= 5)),
        "recall_at_10": float(np.mean(rank_array <= 10)),
        "median_rank": float(np.median(rank_array)),
    }


def evaluate_msrvtt(
    frame_embeddings: Sequence[np.ndarray],
    captions: Sequence[str],
    cfg: object,
    pooling: str,
    output_path: Path | None = None,
    write_result: bool = True,
) -> dict[str, float | str | int]:
    """Evaluate paired MSR-VTT caption-to-clip retrieval and write its metrics.

    The caller supplies the official 1K-A ordered test split (or a smaller
    development subset). Dataset downloading is deliberately outside this API.

    Args:
        frame_embeddings: Per-clip frame matrices in ground-truth order.
        captions: First caption per matching clip in the same order.
        cfg: Application configuration.
        pooling: One configured frame-pooling method.
        output_path: Optional result JSON destination.

    Returns:
        Retrieval metric mapping.
    """
    if len(frame_embeddings) != len(captions) or not captions:
        raise ValueError("Frame embeddings and captions must be non-empty and equally sized.")
    clip_vectors = np.stack([pool_frame_embeddings(vectors, pooling) for vectors in frame_embeddings])
    query_vectors = np.concatenate([encode_query_visual(caption, cfg) for caption in captions])
    metrics: dict[str, float | str | int] = {
        "n_clips": len(captions),
        "pooling": pooling,
        **_retrieval_metrics(query_vectors @ clip_vectors.T),
    }
    if write_result:
        destination = output_path or (Path(cfg.paths.results) / "msrvtt.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as results_file:
            json.dump(metrics, results_file, indent=2)
    return metrics
