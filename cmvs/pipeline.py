"""Resumable orchestration for indexing one source video."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

from cmvs.encode_text import encode_segments
from cmvs.encode_visual import encode_images
from cmvs.extract import extract_audio, extract_frames, probe_duration
from cmvs.index import build_index, save_index
from cmvs.transcribe import transcribe
from cmvs.utils import read_jsonl, slugify, write_jsonl


def _index_exists(index_dir: Path, kind: str) -> bool:
    """Return whether both persistent files for one modality are present."""
    return (index_dir / f"{kind}.faiss").exists() and (index_dir / f"{kind}_meta.json").exists()


def _extract_optional_audio(video_path: Path, wav_path: Path, cfg: object) -> Path | None:
    """Extract audio, allowing the specific ffmpeg no-audio-stream outcome."""
    try:
        return extract_audio(video_path, wav_path, cfg)
    except RuntimeError as exc:
        message = str(exc).lower()
        no_audio_markers = ("matches no streams", "does not contain any stream")
        if any(marker in message for marker in no_audio_markers):
            return None
        raise


def _write_video_meta(
    destination: Path,
    video_id: str,
    video_path: Path,
    duration: float,
    frames: list[dict],
    segments: list[dict],
    cfg: object,
) -> None:
    """Write the per-video metadata contract after both indices are ready."""
    meta = {
        "video_id": video_id,
        "source_path": str(video_path),
        "duration": duration,
        "fps_sampled": cfg.extract.fps,
        "n_frames": len(frames),
        "n_segments": len(segments),
        "clip_model": cfg.models.clip_name,
        "whisper_model": cfg.models.whisper_size,
        "text_model": cfg.models.text_model,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with destination.open("w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, ensure_ascii=False)


def ingest(
    video_path: Path,
    cfg: object,
    force: bool = False,
    on_stage: Callable[[str], None] | None = None,
) -> str:
    """Extract, embed, and index a video, skipping completed stages by default.

    Args:
        video_path: Source video path.
        cfg: Application configuration.
        force: Rebuild every artifact even if its output already exists.
        on_stage: Optional callback told which stage is starting: ``frames``,
            ``speech``, ``picture``, ``text`` or ``finish``.

    Returns:
        Stable video identifier derived from the source filename.
    """
    def announce(stage: str) -> None:
        if on_stage is not None:
            on_stage(stage)

    video_id = slugify(video_path.name)
    processed_dir = cfg.paths.processed / video_id
    index_dir = cfg.paths.index / video_id
    frame_manifest = processed_dir / "frames.jsonl"
    transcript_manifest = processed_dir / "transcript.jsonl"
    audio_path = processed_dir / "audio.wav"
    meta_path = index_dir / "meta.json"

    is_complete = (
        frame_manifest.exists()
        and transcript_manifest.exists()
        and _index_exists(index_dir, "visual")
        and _index_exists(index_dir, "transcript")
        and meta_path.exists()
    )
    if is_complete and not force:
        return video_id

    announce("frames")
    if frame_manifest.exists() and not force:
        frames = read_jsonl(frame_manifest)
    else:
        frames = extract_frames(video_path, processed_dir, cfg)

    announce("speech")
    if transcript_manifest.exists() and not force:
        segments = read_jsonl(transcript_manifest)
    else:
        audio_for_transcription: Path | None
        if audio_path.exists() and not force:
            audio_for_transcription = audio_path
        else:
            audio_for_transcription = _extract_optional_audio(video_path, audio_path, cfg)
        segments = transcribe(audio_for_transcription, cfg) if audio_for_transcription else []
        write_jsonl(transcript_manifest, segments)

    announce("picture")
    if force or not _index_exists(index_dir, "visual"):
        visual_vectors = encode_images([row["path"] for row in frames], cfg)
        save_index(
            build_index(visual_vectors),
            {
                "frame_idx": [row["frame_idx"] for row in frames],
                "timestamp": [row["timestamp"] for row in frames],
                "path": [row["path"] for row in frames],
            },
            index_dir,
            "visual",
        )

    announce("text")
    if force or not _index_exists(index_dir, "transcript"):
        transcript_vectors = encode_segments([row["text"] for row in segments], cfg)
        save_index(
            build_index(transcript_vectors),
            {
                "seg_idx": [row["seg_idx"] for row in segments],
                "start": [row["start"] for row in segments],
                "end": [row["end"] for row in segments],
                "text": [row["text"] for row in segments],
            },
            index_dir,
            "transcript",
        )

    announce("finish")
    if force or not meta_path.exists():
        try:
            duration = probe_duration(video_path)
        except RuntimeError:
            duration = len(frames) / cfg.extract.fps
        index_dir.mkdir(parents=True, exist_ok=True)
        _write_video_meta(meta_path, video_id, video_path, duration, frames, segments, cfg)
    return video_id
