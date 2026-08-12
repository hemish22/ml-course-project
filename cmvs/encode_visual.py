"""CLIP image and query embedding functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from cmvs.utils import get_device, l2_normalize

if TYPE_CHECKING:
    from torch import nn


_CLIP_CACHE: tuple[Any, Callable[[Any], Any], Callable[[list[str]], Any], str] | None = None


def load_clip(cfg: object) -> tuple["nn.Module", Callable[[Any], Any], Any]:
    """Load and cache the configured CLIP model, image preprocessor, and tokenizer.

    Args:
        cfg: Application configuration containing CLIP model settings.

    Returns:
        The model in evaluation mode, its image preprocessor, and text tokenizer.
    """
    global _CLIP_CACHE
    if _CLIP_CACHE is None:
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "open_clip_torch is required for visual encoding. "
                "Install the project's requirements before encoding."
            ) from exc

        device = get_device()
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.models.clip_name,
            pretrained=cfg.models.clip_pretrained,
            device=device,
        )
        model.eval()
        _CLIP_CACHE = (model, preprocess, open_clip.get_tokenizer(cfg.models.clip_name), device)
    model, preprocess, tokenizer, _ = _CLIP_CACHE
    return model, preprocess, tokenizer


def _as_normalized_numpy(embeddings: Any) -> np.ndarray:
    """Move a Torch embedding tensor to a normalized float32 NumPy array."""
    return l2_normalize(embeddings.detach().cpu().numpy().astype(np.float32, copy=False))


def encode_images(paths: list[str], cfg: object) -> np.ndarray:
    """Encode image paths into normalized CLIP embeddings.

    Args:
        paths: Frame JPEG paths in the order that should be indexed.
        cfg: Application configuration containing encoding settings.

    Returns:
        A float32 matrix of unit-length image embeddings, one row per path.
    """
    import torch
    from PIL import Image
    from tqdm.auto import tqdm

    model, preprocess, _ = load_clip(cfg)
    device = get_device()
    encoded_batches: list[np.ndarray] = []
    for start in tqdm(range(0, len(paths), cfg.encode.batch_size), desc="Encoding frames"):
        batch_paths = paths[start : start + cfg.encode.batch_size]
        images = []
        for path in batch_paths:
            with Image.open(Path(path)) as image:
                images.append(preprocess(image.convert("RGB")))
        image_batch = torch.stack(images).to(device)
        with torch.no_grad():
            encoded_batches.append(_as_normalized_numpy(model.encode_image(image_batch)))

    if encoded_batches:
        return np.concatenate(encoded_batches, axis=0)
    output_dim = int(model.visual.output_dim)
    return np.empty((0, output_dim), dtype=np.float32)


def encode_query_visual(text: str, cfg: object) -> np.ndarray:
    """Encode a text query into a normalized CLIP text embedding.

    Args:
        text: Natural-language visual query.
        cfg: Application configuration containing CLIP model settings.

    Returns:
        A shape ``(1, D)`` float32 unit vector.
    """
    import torch

    model, _, tokenizer = load_clip(cfg)
    tokens = tokenizer([text]).to(get_device())
    with torch.no_grad():
        return _as_normalized_numpy(model.encode_text(tokens))
