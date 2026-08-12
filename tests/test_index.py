from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from cmvs.index import build_index, load_index, save_index, search


class FakeIndexFlatIP:
    """Tiny NumPy replacement for the FAISS flat-IP API used by these tests."""

    def __init__(self, dimension: int) -> None:
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return self.vectors.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.copy()

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        similarities = query @ self.vectors.T
        ids = np.argsort(-similarities, axis=1)[:, :k]
        return np.take_along_axis(similarities, ids, axis=1), ids.astype(np.int64)


@pytest.fixture
def fake_faiss(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("faiss")
    saved: dict[str, FakeIndexFlatIP] = {}
    module.IndexFlatIP = FakeIndexFlatIP
    module.write_index = lambda index, path: saved.__setitem__(path, index)
    module.read_index = lambda path: saved[path]
    monkeypatch.setitem(sys.modules, "faiss", module)
    return module


def test_index_round_trip_and_self_retrieval(fake_faiss: ModuleType, tmp_path: Path) -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = build_index(vectors)
    metadata = {"frame_idx": [0, 1], "timestamp": [0.0, 1.0], "path": ["a.jpg", "b.jpg"]}

    save_index(index, metadata, tmp_path, "visual")
    loaded_index, loaded_meta = load_index(tmp_path, "visual")
    scores, ids = search(loaded_index, vectors[1], k=99)

    assert loaded_meta == metadata
    assert ids[0, 0] == 1
    assert scores[0, 0] == pytest.approx(1.0)
    assert ids.shape == (1, 2)


def test_save_index_rejects_misaligned_metadata(fake_faiss: ModuleType, tmp_path: Path) -> None:
    index = build_index(np.array([[1.0, 0.0]], dtype=np.float32))

    with pytest.raises(AssertionError, match="metadata"):
        save_index(index, {"frame_idx": [0, 1]}, tmp_path, "visual")


def test_build_index_requires_normalized_vectors(fake_faiss: ModuleType) -> None:
    with pytest.raises(ValueError, match="L2-normalized"):
        build_index(np.array([[2.0, 0.0]], dtype=np.float32))
