"""Command-line entry point for resumably indexing videos."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

if sys.platform == "darwin":  # faiss + torch each bundle libomp and segfault together
    os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmvs.config import load_config
from cmvs.pipeline import ingest


def main() -> None:
    """Parse CLI arguments and ingest one video or every file in a directory."""
    parser = argparse.ArgumentParser(description="Index videos for cross-modal search.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--video", type=Path, help="One video file to index.")
    target.add_argument("--dir", type=Path, help="Directory of video files to index.")
    parser.add_argument("--force", action="store_true", help="Rebuild existing artifacts.")
    args = parser.parse_args()
    cfg = load_config()
    videos = [args.video] if args.video else sorted(path for path in args.dir.iterdir() if path.is_file())
    for video_path in videos:
        print(ingest(video_path, cfg, force=args.force))


if __name__ == "__main__":
    main()
