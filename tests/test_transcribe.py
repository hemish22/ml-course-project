from pathlib import Path
from types import SimpleNamespace

import cmvs.transcribe as transcribe_module


class Segment:
    """Small stand-in for a faster-whisper segment."""

    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class FakeWhisper:
    """Deterministic test-only transcription model."""

    def transcribe(self, _: str, vad_filter: bool) -> tuple[list[Segment], object]:
        assert vad_filter is True
        return [Segment(0.0, 1.5, " hello "), Segment(2.0, 4.0, "world")], object()


def test_transcribe_returns_ordered_segment_rows(monkeypatch, tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    wav_path.touch()
    monkeypatch.setattr(transcribe_module, "_load_whisper", lambda _: FakeWhisper())

    rows = transcribe_module.transcribe(wav_path, SimpleNamespace())

    assert rows == [
        {"seg_idx": 0, "start": 0.0, "end": 1.5, "text": "hello"},
        {"seg_idx": 1, "start": 2.0, "end": 4.0, "text": "world"},
    ]


def test_transcribe_missing_audio_is_valid_empty_transcript(tmp_path: Path) -> None:
    assert transcribe_module.transcribe(tmp_path / "absent.wav", SimpleNamespace()) == []
