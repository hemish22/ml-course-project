"""Frame, audio, and duration extraction through ffmpeg command-line tools."""

from __future__ import annotations

from pathlib import Path
import subprocess

from cmvs.utils import write_jsonl


def _run_command(command: list[str], binary_name: str) -> subprocess.CompletedProcess[str]:
    """Run a multimedia command and surface actionable errors."""
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{binary_name} is required but was not found on PATH. "
            f"Install {binary_name} and try again."
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or "no error output"
        raise RuntimeError(f"{binary_name} failed: {details}") from exc


def extract_frames(video_path: Path, out_dir: Path, cfg: object) -> list[dict]:
    """Extract sampled JPEG frames and write their ordered timestamp manifest.

    Args:
        video_path: Source video to sample.
        out_dir: Per-video processed-data directory.
        cfg: Application configuration with extraction settings.

    Returns:
        Frame rows in ``frames.jsonl`` order.
    """
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = frames_dir / "%06d.jpg"
    video_filter = (
        f"fps={cfg.extract.fps},"
        f"scale='min({cfg.extract.frame_width},iw)':-2"
    )
    _run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            video_filter,
            "-q:v",
            str(cfg.extract.jpeg_quality),
            str(frame_pattern),
        ],
        "ffmpeg",
    )

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"ffmpeg produced no JPEG frames for {video_path}.")
    rows = [
        {
            "frame_idx": frame_idx,
            "timestamp": frame_idx / cfg.extract.fps,
            "path": str(frame_path),
        }
        for frame_idx, frame_path in enumerate(frame_paths)
    ]
    write_jsonl(out_dir / "frames.jsonl", rows)
    return rows


def extract_audio(video_path: Path, out_wav: Path, cfg: object) -> Path:
    """Extract a mono WAV audio track at the configured sample rate.

    Args:
        video_path: Source video containing an optional audio stream.
        out_wav: Destination WAV path.
        cfg: Application configuration with audio extraction settings.

    Returns:
        The destination WAV path.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            str(cfg.extract.audio_sample_rate),
            "-vn",
            str(out_wav),
        ],
        "ffmpeg",
    )
    return out_wav


def probe_duration(video_path: Path) -> float:
    """Return video duration in seconds using ffprobe.

    Args:
        video_path: Source video to inspect.

    Returns:
        Duration in seconds.

    Raises:
        RuntimeError: If ffprobe is unavailable or cannot determine a duration.
    """
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        "ffprobe",
    )
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe did not return a duration for {video_path}.") from exc
