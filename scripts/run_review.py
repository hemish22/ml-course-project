"""Run the clustering and Gaussian HMM review without re-encoding the videos."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.review import run_review  # noqa: E402
from cmvs.config import load_config  # noqa: E402


def main() -> None:
    """Execute cached folds by default or explicitly recompute with --force."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_review(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
