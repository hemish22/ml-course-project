from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

import cmvs.encode_visual as encode_visual


class FakeVisualModel:
    """Small deterministic CLIP stand-in used without loading a real model."""

    class visual:
        output_dim = 3

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return torch.stack((images[:, 0], images[:, 1], images[:, 2]), dim=1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.to(dtype=torch.float32)


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(encode=SimpleNamespace(batch_size=2))


def _preprocess(image: Image.Image) -> torch.Tensor:
    return torch.tensor(np.asarray(image, dtype=np.float32)[0, 0, :])


def test_encode_images_batches_and_normalizes(
    monkeypatch, tmp_path: Path
) -> None:
    paths: list[str] = []
    for index, colour in enumerate(((3, 4, 0), (0, 5, 12), (8, 0, 6))):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (1, 1), colour).save(path)
        paths.append(str(path))
    monkeypatch.setattr(
        encode_visual,
        "load_clip",
        lambda _: (FakeVisualModel(), _preprocess, None),
    )
    monkeypatch.setattr(encode_visual, "get_device", lambda: "cpu")

    vectors = encode_visual.encode_images(paths, _cfg())

    assert vectors.shape == (3, 3)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_encode_query_visual_is_a_normalized_row_vector(monkeypatch) -> None:
    monkeypatch.setattr(
        encode_visual,
        "load_clip",
        lambda _: (FakeVisualModel(), None, lambda _: torch.tensor([[3, 4, 0]])),
    )
    monkeypatch.setattr(encode_visual, "get_device", lambda: "cpu")

    vector = encode_visual.encode_query_visual("a board", _cfg())

    assert vector.shape == (1, 3)
    assert vector.dtype == np.float32
    assert np.allclose(np.linalg.norm(vector, axis=1), 1.0, atol=1e-5)
