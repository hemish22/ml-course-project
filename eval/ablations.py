"""Configured retrieval ablations and CSV/plot reporting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from eval.charades import evaluate_charades
from eval.msrvtt import evaluate_msrvtt
from eval.plots import plot_alpha_sweep


def _moment_row(configuration: str, alpha: float, report: dict[str, dict[str, float | int]]) -> dict[str, object]:
    """Flatten overall Charades metrics into one ablation-table row."""
    return {"configuration": configuration, "alpha": alpha, **report["overall"]}


def _write_rows(rows: Sequence[dict[str, object]], destination: Path) -> None:
    """Write heterogeneous ablation rows as a stable CSV table."""
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_ablations(
    charades_examples: Sequence[dict[str, Any]],
    cfg: object,
    msrvtt_frame_embeddings: Sequence[np.ndarray] | None = None,
    msrvtt_captions: Sequence[str] | None = None,
    encoder_evaluator: Callable[[str], dict[str, float | int]] | None = None,
) -> list[dict[str, object]]:
    """Run configured modality, alpha, pooling, and encoder ablations.

    ``encoder_evaluator`` must reindex the supplied corpus for each configured
    CLIP model before returning metrics. It is a callback because changing an
    encoder invalidates existing visual indices.

    Args:
        charades_examples: Moment-retrieval examples.
        cfg: Application configuration.
        msrvtt_frame_embeddings: Optional paired clip frame embeddings.
        msrvtt_captions: Optional paired MSR-VTT captions.
        encoder_evaluator: Optional model-name-to-metrics reindexing callback.

    Returns:
        Complete ablation rows, also persisted as CSV and an alpha plot.
    """
    rows: list[dict[str, object]] = []
    rows.append(_moment_row("visual_only", 1.0, evaluate_charades(charades_examples, cfg, alpha=1.0, write_result=False)))
    rows.append(_moment_row("transcript_only", 0.0, evaluate_charades(charades_examples, cfg, alpha=0.0, write_result=False)))
    fused_rows: list[dict[str, object]] = []
    for alpha in cfg.evaluation.alpha_values:
        fused_rows.append(
            _moment_row("fused", alpha, evaluate_charades(charades_examples, cfg, alpha=alpha, write_result=False))
        )
    rows.extend(fused_rows)
    rows.append(
        {
            **max(fused_rows, key=lambda row: float(row["mean_iou"])),
            "configuration": "fused_best",
        }
    )

    if msrvtt_frame_embeddings is not None and msrvtt_captions is not None:
        for pooling in cfg.evaluation.pooling_methods:
            metrics = evaluate_msrvtt(
                msrvtt_frame_embeddings, msrvtt_captions, cfg, pooling, write_result=False
            )
            rows.append({"configuration": "pooling", **metrics})

    if encoder_evaluator is not None:
        for model_name in cfg.evaluation.clip_models:
            rows.append({"configuration": "encoder", "clip_model": model_name, **encoder_evaluator(model_name)})

    csv_path = Path(cfg.paths.results) / "ablations.csv"
    _write_rows(rows, csv_path)
    plot_alpha_sweep(rows, Path(cfg.paths.results) / "alpha_sweep.png", cfg.evaluation.plot_dpi)
    return rows
