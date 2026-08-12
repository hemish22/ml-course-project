"""Exact FAISS inner-product indices and their metadata sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _faiss() -> Any:
    """Import FAISS only when an index operation is requested."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss-cpu is required for vector indexing. "
            "Install the project's requirements before building an index."
        ) from exc
    return faiss


def build_index(vectors: np.ndarray) -> Any:
    """Build an exact inner-product index from normalized embedding rows.

    Args:
        vectors: Two-dimensional float32 unit embedding matrix.

    Returns:
        A FAISS ``IndexFlatIP`` containing rows in their original order.
    """
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Expected a non-empty-dimensional two-dimensional vector matrix.")
    if not np.isfinite(values).all():
        raise ValueError("Index vectors must be finite.")
    if values.shape[0] and not np.allclose(
        np.linalg.norm(values, axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("Index vectors must be L2-normalized before indexing.")

    index = _faiss().IndexFlatIP(values.shape[1])
    index.add(np.ascontiguousarray(values))
    return index


def _assert_metadata_order(index: Any, meta: dict[str, Any]) -> None:
    """Assert each metadata array maps one-to-one to FAISS row order."""
    array_lengths = [len(value) for value in meta.values() if isinstance(value, list)]
    if not array_lengths:
        raise ValueError("Index metadata must contain at least one ordered array.")
    if len(set(array_lengths)) != 1:
        raise AssertionError("Metadata arrays must all have the same row count.")
    if index.ntotal != array_lengths[0]:
        raise AssertionError(
            f"FAISS has {index.ntotal} rows but metadata has {array_lengths[0]} rows."
        )


def save_index(index: Any, meta: dict[str, Any], dir: Path, kind: str) -> None:
    """Save a FAISS index and its same-order JSON metadata sidecar.

    Args:
        index: FAISS index to serialize.
        meta: Parallel metadata arrays in index row order.
        dir: Destination per-video index directory.
        kind: Modality name, such as ``visual`` or ``transcript``.
    """
    _assert_metadata_order(index, meta)
    dir.mkdir(parents=True, exist_ok=True)
    _faiss().write_index(index, str(dir / f"{kind}.faiss"))
    with (dir / f"{kind}_meta.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(meta, metadata_file, ensure_ascii=False)


def load_index(dir: Path, kind: str) -> tuple[Any, dict[str, Any]]:
    """Load an exact index and assert its metadata still maps every row.

    Args:
        dir: Per-video index directory.
        kind: Modality name, such as ``visual`` or ``transcript``.

    Returns:
        The FAISS index and its ordered metadata mapping.
    """
    index = _faiss().read_index(str(dir / f"{kind}.faiss"))
    with (dir / f"{kind}_meta.json").open(encoding="utf-8") as metadata_file:
        meta = json.load(metadata_file)
    if not isinstance(meta, dict):
        raise ValueError(f"{kind}_meta.json must contain a JSON object.")
    _assert_metadata_order(index, meta)
    return index, meta


def search(index: Any, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Search an exact index while safely clamping the requested result count.

    Args:
        index: A FAISS index built from normalized vectors.
        query_vec: One or more normalized query rows.
        k: Requested nearest-neighbor count.

    Returns:
        ``(scores, ids)`` arrays without FAISS overflow IDs.
    """
    query = np.asarray(query_vec, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2:
        raise ValueError("Query vectors must be one- or two-dimensional.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    safe_k = min(k, index.ntotal)
    if safe_k == 0:
        return (
            np.empty((query.shape[0], 0), dtype=np.float32),
            np.empty((query.shape[0], 0), dtype=np.int64),
        )
    scores, ids = index.search(np.ascontiguousarray(query), safe_k)
    return scores, ids
