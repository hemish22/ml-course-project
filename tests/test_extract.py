from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from cmvs.extract import extract_audio, extract_frames, probe_duration
from cmvs.utils import read_jsonl


@pytest.fixture
def cfg() -> SimpleNamespace:
    return SimpleNamespace(
        extract=SimpleNamespace(fps=1.0, frame_width=336, jpeg_quality=88, audio_sample_rate=16000)
    )


def test_extract_frames_writes_ordered_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: SimpleNamespace
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        frames_dir = tmp_path / "processed" / "frames"
        (frames_dir / "000002.jpg").touch()
        (frames_dir / "000001.jpg").touch()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("cmvs.extract.subprocess.run", fake_run)

    rows = extract_frames(tmp_path / "clip.mp4", tmp_path / "processed", cfg)

    assert [row["timestamp"] for row in rows] == [0.0, 1.0]
    assert read_jsonl(tmp_path / "processed" / "frames.jsonl") == rows
    assert commands[0][0] == "ffmpeg"
    assert "fps=1.0,scale='min(336,iw)':-2" in commands[0]


def test_extract_audio_uses_mono_configured_sample_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: SimpleNamespace
) -> None:
    captured: list[str] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured.extend(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("cmvs.extract.subprocess.run", fake_run)

    result = extract_audio(tmp_path / "clip.mp4", tmp_path / "audio.wav", cfg)

    assert result == tmp_path / "audio.wav"
    assert captured[captured.index("-ac") + 1] == "1"
    assert captured[captured.index("-ar") + 1] == "16000"
    assert "-vn" in captured


def test_probe_duration_parses_ffprobe_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "12.5\n", "")

    monkeypatch.setattr("cmvs.extract.subprocess.run", fake_run)

    assert probe_duration(tmp_path / "clip.mp4") == 12.5


def test_missing_ffmpeg_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: SimpleNamespace
) -> None:
    def missing_binary(*_: object, **__: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr("cmvs.extract.subprocess.run", missing_binary)

    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        extract_audio(tmp_path / "clip.mp4", tmp_path / "audio.wav", cfg)
