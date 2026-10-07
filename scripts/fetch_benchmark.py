"""Download the public-domain NASA episodes used by the regression/classification benchmark.

Requires the ``yt-dlp`` command (not a project dependency; only needed once to fetch data).
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmvs.config import load_config

# NASA "This Week @NASA" news episodes: (short file name, YouTube id).
EPISODES: list[tuple[str, str]] = [
    ("mars-mission-briefed", "8Hgw084eUHc"),
    ("a-remote-possibility", "eC9gfgB6Ft4"),
    ("new-explorers-new-roadmap", "0bjmVGiPlls"),
    ("smoke-and-fire", "TJ5e9O6yKlU"),
    ("hubble-exoplanet", "1bJDPqd8V88"),
    ("cosmic-phenomenon", "0M9Sxm7peC8"),
    ("administrator-chats-astronauts", "zvfNUGkSQ_k"),
    ("commercial-crew-milestone", "emQej1NKTtU"),
    ("artemis-moonwalk-practice", "qb2ONVgj-3Q"),
]

FORMAT = "bv*[height<=480][vcodec^=avc1]+ba[ext=m4a]/b[height<=480][ext=mp4]/b[height<=480]"


def main() -> None:
    """Fetch every episode that is not already in the raw video directory."""
    raw_dir = Path(load_config().paths.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, video_id in EPISODES:
        target = raw_dir / f"{name}.mp4"
        if target.exists():
            continue
        print(f"Downloading {name}")
        try:
            subprocess.run(
                [
                    "yt-dlp", "--no-update", "-q", "--no-warnings", "-f", FORMAT,
                    "--merge-output-format", "mp4", "-o", str(target),
                    f"https://www.youtube.com/watch?v={video_id}",
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            print(f"  skipped {name}: download failed (video may be unavailable)")


if __name__ == "__main__":
    main()
