"""HTTP API and media server for the cross-modal video search frontend."""

from __future__ import annotations

from contextlib import asynccontextmanager
import shutil
import subprocess
import uuid
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, AsyncIterator, Iterator

if sys.platform == "darwin":  # faiss + torch each bundle libomp and segfault together
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from cmvs.config import Config, load_config
from cmvs.encode_text import encode_query_text
from cmvs.encode_visual import encode_query_visual
from cmvs.pipeline import ingest
from cmvs.search import Moment, ScoreTimeline, moments_from_timeline, score_video
from cmvs.utils import read_jsonl, slugify

_RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK_BYTES = 1024 * 512
_MAX_UPLOAD_BYTES = 2 * 1024**3
_UPLOAD_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_INGEST_LOCK = threading.Lock()  # one video is indexed at a time
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_FRAME_PATTERN = re.compile(r"^\d+\.jpg$")
_SEARCH_LOCK = threading.Lock()  # torch encoders are shared module-level singletons
_CFG: Config = load_config()


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load both query encoders once so the first search is not slow."""
    encode_query_visual("warm up", _CFG)
    encode_query_text("warm up", _CFG)
    yield


app = FastAPI(title="Cross-Modal Video Search", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)


def _check_id(video_id: str) -> str:
    """Reject identifiers that could escape the data directories."""
    if not _ID_PATTERN.match(video_id) or video_id.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid video id.")
    return video_id


def _read_meta(video_id: str) -> dict[str, Any]:
    """Read one completed video's metadata or raise 404."""
    meta_path = Path(_CFG.paths.index) / _check_id(video_id) / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' is not indexed.")
    with meta_path.open(encoding="utf-8") as meta_file:
        return json.load(meta_file)


def _video_summary(meta: dict[str, Any]) -> dict[str, Any]:
    """Shape the metadata contract into what the frontend needs."""
    video_id = meta["video_id"]
    poster = max(1, round(meta["n_frames"] * 0.04) + 1)  # ffmpeg numbers frames from 1
    return {
        "id": video_id,
        "title": video_id.replace("-", " ").replace("_", " ").title(),
        "duration": meta["duration"],
        "fps": meta["fps_sampled"],
        "n_frames": meta["n_frames"],
        "n_segments": meta["n_segments"],
        "has_speech": meta["n_segments"] > 0,
        "video_url": f"/media/video/{video_id}",
        "poster_url": f"/media/frame/{video_id}/{poster:06d}.jpg",
    }


def _moment_payload(moment: Moment, duration: float, frame_step: float) -> dict[str, Any]:
    """Serialize a moment, adding the playable clip range.

    Frames are samples taken every ``frame_step`` seconds, so the clip extends
    one step past the last matching frame to cover what that frame represents.
    """
    return {
        "start": moment.start,
        "end": moment.end,
        "clip_start": max(0.0, moment.start),
        "clip_end": min(duration, moment.end + frame_step),
        "score": moment.score,
        "visual": moment.visual_score,
        "transcript": moment.transcript_score,
        "thumbnail": f"/media/frame/{moment.video_id}/{Path(moment.thumbnail_path).name}",
        "text": moment.transcript_text,
    }


def _trace_payload(timeline: ScoreTimeline, alpha: float) -> dict[str, list[float]]:
    """Per-frame scores for the relevance trace, weighted by the chosen alpha."""
    return {
        "t": [round(float(v), 2) for v in timeline.timestamps],
        "visual": [round(float(alpha * v), 3) for v in timeline.visual],
        "transcript": [round(float((1.0 - alpha) * v), 3) for v in timeline.transcript],
    }


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """Defaults and active model names for the controls."""
    return {
        "alpha": _CFG.search.alpha,
        "score_threshold": _CFG.search.score_threshold,
        "max_results": _CFG.search.max_results,
        "models": {
            "clip": _CFG.models.clip_name,
            "whisper": _CFG.models.whisper_size,
            "text": _CFG.models.text_model,
        },
    }


@app.get("/api/videos")
def list_videos() -> list[dict[str, Any]]:
    """Every completely indexed video."""
    index_root = Path(_CFG.paths.index)
    if not index_root.exists():
        return []
    return [
        _video_summary(_read_meta(child.name))
        for child in sorted(index_root.iterdir())
        if child.is_dir() and (child / "meta.json").exists()
    ]


@app.get("/api/videos/{video_id}/transcript")
def get_transcript(video_id: str) -> list[dict[str, Any]]:
    """Transcript segments for click-to-seek (empty for videos without speech)."""
    _read_meta(video_id)
    manifest = Path(_CFG.paths.processed) / video_id / "transcript.jsonl"
    return read_jsonl(manifest) if manifest.exists() else []


@app.get("/api/search")
def search(
    video: str,
    q: str = Query(min_length=1, max_length=300),
    alpha: float | None = Query(default=None, ge=0.0, le=1.0),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    limit: int | None = Query(default=None, ge=1, le=50),
) -> dict[str, Any]:
    """Fuse picture and speech scores for a query and return ranked moments."""
    meta = _read_meta(video)
    selected_alpha = _CFG.search.alpha if alpha is None else alpha
    cfg = replace(
        _CFG,
        search=replace(
            _CFG.search,
            score_threshold=_CFG.search.score_threshold if threshold is None else threshold,
            max_results=_CFG.search.max_results if limit is None else limit,
        ),
    )
    with _SEARCH_LOCK:
        timeline = score_video(q.strip(), video, cfg, selected_alpha)
        moments = moments_from_timeline(timeline, video, cfg)
    frame_step = 1.0 / meta["fps_sampled"]
    return {
        "query": q.strip(),
        "alpha": selected_alpha,
        "moments": [_moment_payload(m, meta["duration"], frame_step) for m in moments],
        "trace": _trace_payload(timeline, selected_alpha),
    }


def _stream_file(path: Path, start: int, end: int) -> Iterator[bytes]:
    """Yield bytes ``start..end`` (inclusive) of a file in chunks."""
    with path.open("rb") as source:
        source.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = source.read(min(_CHUNK_BYTES, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@app.get("/media/video/{video_id}")
def get_video(video_id: str, request: Request) -> Response:
    """Serve the source video with HTTP range support.

    Browsers seek by requesting byte ranges; a server that ignores them makes every
    seek restart playback at 0:00. Ranges are handled here rather than by
    ``FileResponse`` because only recent Starlette versions implement them.
    """
    source = Path(_read_meta(video_id)["source_path"])
    if not source.exists():
        raise HTTPException(status_code=404, detail="Source video file is missing.")
    size = source.stat().st_size
    base_headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}

    requested = request.headers.get("range")
    if requested is None:
        return StreamingResponse(
            _stream_file(source, 0, size - 1),
            media_type="video/mp4",
            headers={**base_headers, "Content-Length": str(size)},
        )
    match = _RANGE_PATTERN.match(requested.strip())
    if not match or (not match.group(1) and not match.group(2)):
        raise HTTPException(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    if match.group(1):
        first = int(match.group(1))
        last = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
    else:  # suffix range: the final N bytes
        first = max(0, size - int(match.group(2)))
        last = size - 1
    if first > last or first >= size:
        raise HTTPException(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    return StreamingResponse(
        _stream_file(source, first, last),
        status_code=206,
        media_type="video/mp4",
        headers={
            **base_headers,
            "Content-Range": f"bytes {first}-{last}/{size}",
            "Content-Length": str(last - first + 1),
        },
    )


@app.get("/media/frame/{video_id}/{name}")
def get_frame(video_id: str, name: str) -> FileResponse:
    """Serve one extracted frame thumbnail."""
    _read_meta(video_id)
    if not _FRAME_PATTERN.match(name):
        raise HTTPException(status_code=400, detail="Invalid frame name.")
    frame = Path(_CFG.paths.processed) / video_id / "frames" / name
    if not frame.exists():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(frame, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


def _probe_codec(path: Path, stream: str) -> str | None:
    """Return the codec name of the first video (``v:0``) or audio (``a:0``) stream."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
         "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else None


def _make_playable(source: Path, destination: Path) -> None:
    """Write a browser-playable, seekable MP4: remux if already H.264/AAC, else re-encode."""
    if _probe_codec(source, "v:0") is None:
        raise RuntimeError("The file has no video stream.")
    already_compatible = _probe_codec(source, "v:0") == "h264" and _probe_codec(source, "a:0") in (None, "aac")
    codec_args = ["-c", "copy"] if already_compatible else [
        "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-crf", "24", "-preset", "fast",
        "-c:a", "aac", "-b:a", "96k",
    ]
    try:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), *codec_args,
             "-movflags", "+faststart", str(destination)],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required but was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg could not read this file: {exc.stderr.strip()[-300:]}") from exc


def _update_job(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id].update(fields)


def _run_ingest_job(job_id: str, uploaded: Path, video_id: str) -> None:
    """Prepare and index one uploaded video, reporting stages to the job record."""
    final = Path(_CFG.paths.raw) / f"{video_id}.mp4"
    _update_job(job_id, state="queued", stage="queued")
    try:
        with _INGEST_LOCK:
            _update_job(job_id, state="running", stage="prepare")
            staged = final.with_suffix(".part.mp4")
            _make_playable(uploaded, staged)
            staged.replace(final)
            if uploaded != final:
                uploaded.unlink(missing_ok=True)
            ingest(final, _CFG, on_stage=lambda stage: _update_job(job_id, stage=stage))
        _update_job(job_id, state="done", stage="done")
    except Exception as exc:  # surfaced to the person uploading, never swallowed
        uploaded.unlink(missing_ok=True)
        _update_job(job_id, state="error", error=str(exc) or exc.__class__.__name__)


@app.post("/api/upload")
async def upload_video(request: Request, filename: str = Query(min_length=1, max_length=200)) -> dict[str, str]:
    """Receive a video as the raw request body, then index it in the background."""
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in _UPLOAD_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{suffix or name}'. Use MP4, MOV, MKV, WEBM or AVI.")
    video_id = slugify(name)
    if (Path(_CFG.paths.index) / video_id / "meta.json").exists():
        raise HTTPException(status_code=409, detail=f"'{video_id}' is already in the library. Remove it first to upload it again.")
    with _JOBS_LOCK:
        if any(j["video_id"] == video_id and j["state"] in ("queued", "running") for j in _JOBS.values()):
            raise HTTPException(status_code=409, detail=f"'{video_id}' is already being processed.")

    raw_dir = Path(_CFG.paths.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)
    uploaded = raw_dir / f"{video_id}.upload{suffix}"
    received = 0
    try:
        with uploaded.open("wb") as target:
            async for chunk in request.stream():
                received += len(chunk)
                if received > _MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File is larger than the 2 GB upload limit.")
                target.write(chunk)
    except BaseException:
        uploaded.unlink(missing_ok=True)
        raise
    if received == 0:
        uploaded.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {"id": job_id, "video_id": video_id, "state": "queued", "stage": "queued", "error": None}
    threading.Thread(target=_run_ingest_job, args=(job_id, uploaded, video_id), daemon=True).start()
    return {"job_id": job_id, "video_id": video_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Progress of one upload-and-index job."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job.")
        return dict(job)


@app.delete("/api/videos/{video_id}")
def remove_video(video_id: str) -> dict[str, str]:
    """Remove a video's index and extracted frames from the library (the source file stays)."""
    _read_meta(video_id)
    with _JOBS_LOCK:
        if any(j["video_id"] == video_id and j["state"] in ("queued", "running") for j in _JOBS.values()):
            raise HTTPException(status_code=409, detail="This video is still being processed.")
    for root in (Path(_CFG.paths.index), Path(_CFG.paths.processed)):
        shutil.rmtree(root / video_id, ignore_errors=True)
    return {"removed": video_id}
