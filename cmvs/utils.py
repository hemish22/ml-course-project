"""Small dependency-light utilities shared by pipeline modules."""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import re
import time
from typing import Iterable, Iterator

import numpy as np

LOGGER = logging.getLogger(__name__)


def get_device() -> str:
    """Return the preferred PyTorch inference device without requiring it at import time."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def slugify(name: str) -> str:
    """Convert a source filename or label into a stable, filesystem-safe video ID."""
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or "video"


def read_jsonl(path: Path) -> list[dict]:
    """Read newline-delimited JSON objects, ignoring blank lines."""
    with path.open(encoding="utf-8") as jsonl_file:
        return [json.loads(line) for line in jsonl_file if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write JSON-serializable mappings as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row, ensure_ascii=False))
            jsonl_file.write("\n")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 vectors with zero-row protection."""
    values = np.asarray(x, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("Expected a one- or two-dimensional embedding array.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Log elapsed wall-clock seconds for the enclosed operation."""
    started_at = time.perf_counter()
    try:
        yield
    finally:
        LOGGER.info("%s completed in %.2fs", label, time.perf_counter() - started_at)
