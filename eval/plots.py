"""Publication-ready visualizations for evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def plot_alpha_sweep(
    rows: Sequence[dict[str, object]], output_path: Path, dpi: int
) -> None:
    """Save a labelled line chart of fused mean IoU against alpha.

    Args:
        rows: Ablation rows containing ``configuration``, ``alpha``, and ``mean_iou``.
        output_path: PNG destination.
        dpi: Configured raster resolution.
    """
    import matplotlib.pyplot as plt

    points = sorted(
        (
            float(row["alpha"]),
            float(row["mean_iou"]),
        )
        for row in rows
        if row.get("configuration") == "fused" and row.get("alpha") is not None
    )
    if not points:
        raise ValueError("No fused alpha-sweep rows were supplied for plotting.")
    alphas, values = zip(*points)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(alphas, values, marker="o", label="Fused retrieval")
    axis.set_xlabel("Visual weight (alpha)")
    axis.set_ylabel("Mean temporal IoU")
    axis.set_title("Fusion-weight ablation")
    axis.grid(alpha=0.3)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
