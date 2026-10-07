from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import server.main as server  # noqa: E402
from cmvs.search import Moment, ScoreTimeline  # noqa: E402


def test_moment_payload_pads_clip_by_one_frame_and_clamps_to_duration() -> None:
    moment = Moment("vid", 10.0, 14.0, 0.9, 0.8, 0.5, "data/processed/vid/frames/000011.jpg", "hi")

    payload = server._moment_payload(moment, duration=20.0, frame_step=1.0)
    clamped = server._moment_payload(moment, duration=14.5, frame_step=1.0)

    assert payload["clip_start"] == 10.0
    assert payload["clip_end"] == 15.0
    assert clamped["clip_end"] == 14.5
    assert payload["thumbnail"] == "/media/frame/vid/000011.jpg"


def test_trace_payload_weights_modalities_by_alpha() -> None:
    timeline = ScoreTimeline(
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        {},
        {},
    )

    trace = server._trace_payload(timeline, alpha=0.75)

    assert trace["visual"] == [0.75, 0.0]
    assert trace["transcript"] == [0.0, 0.25]


def test_unindexed_video_and_path_tricks_are_rejected(monkeypatch, tmp_path) -> None:
    cfg = SimpleNamespace(paths=SimpleNamespace(index=tmp_path, processed=tmp_path))
    monkeypatch.setattr(server, "_CFG", cfg)
    client = TestClient(server.app)  # no context manager: skips model warm-up

    assert client.get("/api/search", params={"video": "missing", "q": "x"}).status_code == 404
    assert client.get("/media/frame/missing/000001.jpg").status_code == 404
    assert client.get("/api/videos").json() == []
    assert server._check_id("ok-id_1") == "ok-id_1"
    with pytest.raises(server.HTTPException):
        server._check_id("../etc")


def test_video_endpoint_honours_byte_ranges(monkeypatch, tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(bytes(range(100)))
    index = tmp_path / "vid"
    index.mkdir()
    (index / "meta.json").write_text(
        '{"video_id": "vid", "source_path": "%s", "duration": 1, "fps_sampled": 1, '
        '"n_frames": 1, "n_segments": 0}' % video
    )
    monkeypatch.setattr(server, "_CFG", SimpleNamespace(paths=SimpleNamespace(index=tmp_path, processed=tmp_path)))
    client = TestClient(server.app)

    whole = client.get("/media/video/vid")
    middle = client.get("/media/video/vid", headers={"Range": "bytes=10-19"})
    open_ended = client.get("/media/video/vid", headers={"Range": "bytes=90-"})
    suffix = client.get("/media/video/vid", headers={"Range": "bytes=-5"})
    beyond = client.get("/media/video/vid", headers={"Range": "bytes=500-"})

    assert whole.status_code == 200 and len(whole.content) == 100
    assert middle.status_code == 206 and middle.content == bytes(range(10, 20))
    assert middle.headers["content-range"] == "bytes 10-19/100"
    assert open_ended.status_code == 206 and open_ended.content == bytes(range(90, 100))
    assert suffix.content == bytes(range(95, 100))
    assert beyond.status_code == 416


def _server_cfg(tmp_path):
    return SimpleNamespace(
        paths=SimpleNamespace(
            raw=tmp_path / "raw", processed=tmp_path / "processed", index=tmp_path / "index"
        )
    )


def test_upload_rejects_unsupported_and_empty_files(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(server, "_CFG", _server_cfg(tmp_path))
    client = TestClient(server.app)

    bad_type = client.post("/api/upload", params={"filename": "notes.txt"}, content=b"hello")
    empty = client.post("/api/upload", params={"filename": "clip.mp4"}, content=b"")

    assert bad_type.status_code == 415
    assert empty.status_code == 400
    assert not list((tmp_path / "raw").glob("*"))


def test_upload_runs_job_and_reports_stages(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(server, "_CFG", _server_cfg(tmp_path))
    seen: list[str] = []

    def fake_playable(source, destination) -> None:
        destination.write_bytes(source.read_bytes())

    def fake_ingest(video_path, cfg, on_stage=None):
        for stage in ("frames", "speech"):
            on_stage(stage)
            seen.append(stage)
        return video_path.stem

    monkeypatch.setattr(server, "_make_playable", fake_playable)
    monkeypatch.setattr(server, "ingest", fake_ingest)
    monkeypatch.setattr(server, "threading", SimpleNamespace(Thread=_InlineThread))
    client = TestClient(server.app)

    response = client.post("/api/upload", params={"filename": "My Clip.MP4"}, content=b"video-bytes")
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()

    assert response.json()["video_id"] == "my-clip"
    assert job["state"] == "done" and seen == ["frames", "speech"]
    assert (tmp_path / "raw" / "my-clip.mp4").read_bytes() == b"video-bytes"
    assert not list((tmp_path / "raw").glob("*.upload*"))


def test_failed_job_reports_error_and_removes_upload(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(server, "_CFG", _server_cfg(tmp_path))

    def broken(source, destination) -> None:
        raise RuntimeError("The file has no video stream.")

    monkeypatch.setattr(server, "_make_playable", broken)
    monkeypatch.setattr(server, "threading", SimpleNamespace(Thread=_InlineThread))
    client = TestClient(server.app)

    job_id = client.post("/api/upload", params={"filename": "junk.mp4"}, content=b"x").json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()

    assert job["state"] == "error" and "no video stream" in job["error"]
    assert not list((tmp_path / "raw").glob("*"))


def test_remove_video_deletes_index_and_frames_but_keeps_source(monkeypatch, tmp_path) -> None:
    cfg = _server_cfg(tmp_path)
    monkeypatch.setattr(server, "_CFG", cfg)
    for root in (cfg.paths.index, cfg.paths.processed):
        (root / "vid").mkdir(parents=True)
    (cfg.paths.index / "vid" / "meta.json").write_text(
        '{"video_id": "vid", "source_path": "x", "duration": 1, "fps_sampled": 1, "n_frames": 1, "n_segments": 0}'
    )
    cfg.paths.raw.mkdir()
    (cfg.paths.raw / "vid.mp4").write_bytes(b"keep")
    client = TestClient(server.app)

    assert client.delete("/api/videos/vid").status_code == 200
    assert not (cfg.paths.index / "vid").exists() and not (cfg.paths.processed / "vid").exists()
    assert (cfg.paths.raw / "vid.mp4").exists()
    assert client.delete("/api/videos/vid").status_code == 404


class _InlineThread:
    """Run the thread target immediately so tests can assert the finished job."""

    def __init__(self, target, args=(), daemon=None) -> None:
        self._target, self._args = target, args

    def start(self) -> None:
        self._target(*self._args)
