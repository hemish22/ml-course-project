import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import cmvs.pipeline as pipeline


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(processed=tmp_path / "processed", index=tmp_path / "index"),
        extract=SimpleNamespace(fps=1.0),
        models=SimpleNamespace(clip_name="clip", whisper_size="whisper", text_model="text"),
    )


def test_ingest_is_resumable_per_completed_video(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    calls: list[str] = []
    frames = [{"frame_idx": 0, "timestamp": 0.0, "path": "frame.jpg"}]
    segments = [{"seg_idx": 0, "start": 0.0, "end": 1.0, "text": "hello"}]
    def fake_extract_frames(_: Path, directory: Path, __: object) -> list[dict]:
        calls.append("frames")
        pipeline.write_jsonl(directory / "frames.jsonl", frames)
        return frames

    monkeypatch.setattr(pipeline, "extract_frames", fake_extract_frames)
    monkeypatch.setattr(pipeline, "extract_audio", lambda *_: calls.append("audio") or tmp_path / "audio.wav")
    monkeypatch.setattr(pipeline, "transcribe", lambda *_: calls.append("transcribe") or segments)
    monkeypatch.setattr(pipeline, "encode_images", lambda *_: calls.append("visual") or np.array([[1.0, 0.0]], dtype=np.float32))
    monkeypatch.setattr(pipeline, "encode_segments", lambda *_: calls.append("text") or np.array([[1.0, 0.0]], dtype=np.float32))
    monkeypatch.setattr(pipeline, "build_index", lambda vectors: SimpleNamespace(ntotal=len(vectors)))
    monkeypatch.setattr(pipeline, "probe_duration", lambda _: 1.0)

    def fake_save(_: object, meta: dict, directory: Path, kind: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{kind}.faiss").touch()
        (directory / f"{kind}_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr(pipeline, "save_index", fake_save)

    assert pipeline.ingest(tmp_path / "A Video.mp4", cfg) == "a-video"
    assert pipeline.ingest(tmp_path / "A Video.mp4", cfg) == "a-video"
    assert calls == ["frames", "audio", "transcribe", "visual", "text"]
