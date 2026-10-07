"""Command-line entry point for video-moment search."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

if sys.platform == "darwin":  # faiss + torch each bundle libomp and segfault together
    os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmvs.config import load_config
from cmvs.search import Moment, search_library, search_video


def _format_timestamp(seconds: float) -> str:
    """Format seconds as a compact ``mm:ss`` timestamp."""
    minutes, remaining_seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def _print_results(results: list[Moment]) -> None:
    """Print a stable plain-text table for search results."""
    print("rank  time         fused  visual  transcript  text")
    for rank, moment in enumerate(results, start=1):
        snippet = moment.transcript_text[:60]
        print(
            f"{rank:>4}  {_format_timestamp(moment.start)}–{_format_timestamp(moment.end)}  "
            f"{moment.score:>5.3f}  {moment.visual_score:>6.3f}  "
            f"{moment.transcript_score:>10.3f}  {snippet}"
        )


def main() -> None:
    """Parse CLI arguments, execute a search, and print its result table."""
    parser = argparse.ArgumentParser(description="Search indexed video moments.")
    parser.add_argument("--q", required=True, help="Natural-language query.")
    parser.add_argument("--video", help="Optional video ID; omit to search the library.")
    parser.add_argument("--alpha", type=float, help="Optional visual fusion weight override.")
    args = parser.parse_args()
    cfg = load_config()
    results = (
        search_video(args.q, args.video, cfg, args.alpha)
        if args.video
        else search_library(args.q, cfg, args.alpha)
    )
    _print_results(results)


if __name__ == "__main__":
    main()
