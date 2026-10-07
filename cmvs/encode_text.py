"""Sentence-transformer embeddings for transcript segments and text queries."""

from __future__ import annotations

from typing import Any

import numpy as np

from cmvs.utils import l2_normalize

_TEXT_MODEL: Any | None = None


def _load_text_model(cfg: object) -> Any:
    """Load the configured sentence-transformer model once for the process."""
    global _TEXT_MODEL
    if _TEXT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for transcript encoding. "
                "Install the project's requirements before encoding text."
            ) from exc
        _TEXT_MODEL = SentenceTransformer(cfg.models.text_model)
    return _TEXT_MODEL


def _embedding_dim(model: Any) -> int:
    """Return embedding width across sentence-transformers API renames."""
    getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return getter()


def _encode_texts(texts: list[str], model: Any) -> np.ndarray:
    """Embed text with a sentence-transformer and enforce vector normalization."""
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    return l2_normalize(np.asarray(embeddings, dtype=np.float32))


def encode_segments(texts: list[str], cfg: object) -> np.ndarray:
    """Encode transcript segment texts as normalized float32 embeddings.

    Args:
        texts: Segment text in transcript-row order.
        cfg: Application configuration containing the text model identifier.

    Returns:
        One unit-length embedding row per input segment.
    """
    model = _load_text_model(cfg)
    if not texts:
        return np.empty((0, int(_embedding_dim(model))), dtype=np.float32)
    return _encode_texts(texts, model)


def encode_query_text(text: str, cfg: object) -> np.ndarray:
    """Encode a natural-language transcript query as a normalized row vector.

    Args:
        text: Natural-language search query.
        cfg: Application configuration containing the text model identifier.

    Returns:
        A shape ``(1, D)`` float32 unit vector.
    """
    return _encode_texts([text], _load_text_model(cfg))
