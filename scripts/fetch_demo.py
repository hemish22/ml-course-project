"""Download the demo video (a public-domain NASA news episode) and ingest it."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

if sys.platform == "darwin":  # faiss + torch each bundle libomp and segfault together
    os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmvs.config import load_config
from cmvs.pipeline import ingest

ITEM = "space-weather-forecasts-in-stereo-on-this-week-nasa-KpKwKLP2Ans"
SOURCE_URL = f"https://archive.org/download/{ITEM}/{ITEM}.ogv"
VIDEO_NAME = "this-week-at-nasa.mp4"


def main() -> None:
    """Fetch, shrink to a web-friendly MP4, and index the demo video."""
    cfg = load_config()
    target = Path(cfg.paths.raw) / VIDEO_NAME
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as workdir:
            download = Path(workdir) / "episode.ogv"
            print(f"Downloading {SOURCE_URL}")
            urllib.request.urlretrieve(SOURCE_URL, download)
            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-y", "-i", str(download),
                    "-vf", "scale=854:-2", "-c:v", "libx264", "-crf", "25", "-preset", "fast",
                    "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(target),
                ],
                check=True,
            )
    print(ingest(target, cfg))


if __name__ == "__main__":
    main()
