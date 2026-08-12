from pathlib import Path
from types import SimpleNamespace

import numpy as np

import cmvs.search as search_module


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(index=tmp_path / "index"),
        search=SimpleNamespace(alpha=0.6, top_k_frames=10, score_threshold=0.3, merge_gap_s=2.0, min_moment_s=1.0, max_results=10),
    )


def test_search_video_fuses_scores_and_returns_metadata(monkeypatch, tmp_path: Path) -> None:
    visual_meta = {"timestamp": [0.0, 1.0, 2.0], "path": ["0.jpg", "1.jpg", "2.jpg"]}
    transcript_meta = {"seg_idx": [0], "start": [0.0], "end": [2.0], "text": ["hello world"]}
    visual_index = SimpleNamespace(ntotal=3)
    transcript_index = SimpleNamespace(ntotal=1)
    monkeypatch.setattr(
        search_module,
        "load_index",
        lambda _, kind: (visual_index, visual_meta) if kind == "visual" else (transcript_index, transcript_meta),
    )
    monkeypatch.setattr(search_module, "encode_query_visual", lambda *_: np.array([[1.0]], dtype=np.float32))
    monkeypatch.setattr(search_module, "encode_query_text", lambda *_: np.array([[1.0]], dtype=np.float32))
    monkeypatch.setattr(
        search_module,
        "search_index",
        lambda index, *_: (np.array([[0.9, 0.6]], dtype=np.float32), np.array([[1, 2]]))
        if index is visual_index
        else (np.array([[0.8]], dtype=np.float32), np.array([[0]])),
    )

    results = search_module.search_video("hello", "video", _cfg(tmp_path))

    assert len(results) == 1
    assert results[0].video_id == "video"
    assert results[0].thumbnail_path == "1.jpg"
    assert results[0].transcript_text == "hello world"
