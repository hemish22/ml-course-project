from types import SimpleNamespace

import numpy as np

import cmvs.encode_text as encode_text


class FakeSentenceTransformer:
    """Small model replacement that avoids downloading a text encoder."""

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return np.asarray([[3.0, 4.0, float(index + 1)] for index, _ in enumerate(texts)])

    def get_sentence_embedding_dimension(self) -> int:
        return 3


def test_encode_segments_returns_normalized_float32_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(encode_text, "_load_text_model", lambda _: FakeSentenceTransformer())

    vectors = encode_text.encode_segments(["one", "two"], SimpleNamespace())

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_encode_text_handles_empty_transcript_and_query(monkeypatch) -> None:
    monkeypatch.setattr(encode_text, "_load_text_model", lambda _: FakeSentenceTransformer())

    empty = encode_text.encode_segments([], SimpleNamespace())
    query = encode_text.encode_query_text("some words", SimpleNamespace())

    assert empty.shape == (0, 3)
    assert query.shape == (1, 3)
    assert np.allclose(np.linalg.norm(query, axis=1), 1.0, atol=1e-5)
