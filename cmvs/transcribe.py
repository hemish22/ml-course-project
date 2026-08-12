"""Whisper transcription into timestamped segment rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cmvs.utils import get_device

_WHISPER_MODEL: Any | None = None


def _load_whisper(cfg: object) -> Any:
    """Load the configured Whisper model once for the process."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is required for transcription. "
                "Install the project's requirements before transcribing."
            ) from exc
        _WHISPER_MODEL = WhisperModel(
            cfg.models.whisper_size,
            device=get_device(),
            compute_type=cfg.models.whisper_compute_type,
        )
    return _WHISPER_MODEL


def transcribe(wav_path: Path, cfg: object) -> list[dict]:
    """Transcribe a WAV file into ordered Whisper segment rows.

    A missing audio file is treated as a valid video with no audio track. Silent
    audio produces no Whisper segments and likewise returns an empty list.

    Args:
        wav_path: Extracted mono WAV audio path.
        cfg: Application configuration containing Whisper settings.

    Returns:
        Rows containing ``seg_idx``, ``start``, ``end``, and ``text``.
    """
    if not wav_path.exists():
        return []

    segments, _ = _load_whisper(cfg).transcribe(str(wav_path), vad_filter=True)
    rows: list[dict] = []
    for segment in segments:
        start = float(segment.start)
        end = float(segment.end)
        if end <= start:
            continue
        rows.append(
            {
                "seg_idx": len(rows),
                "start": start,
                "end": end,
                "text": str(segment.text).strip(),
            }
        )
    return rows
