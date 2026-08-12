"""Charades-style timestamped moment-retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from cmvs.index import load_index
from cmvs.search import search_video


def temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Calculate temporal intersection-over-union for two intervals."""
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return intersection / union if union > 0.0 else 0.0


def _contains_speech(example: dict[str, Any], cfg: object) -> bool:
    """Use an explicit label or infer speech from transcript overlap."""
    if "contains_speech" in example:
        return bool(example["contains_speech"])
    _, transcript_meta = load_index(Path(cfg.paths.index) / example["video_id"], "transcript")
    return any(
        segment_start < example["end"] and segment_end > example["start"]
        for segment_start, segment_end in zip(transcript_meta["start"], transcript_meta["end"])
    )


def _summarize_ious(ious: list[float], thresholds: Sequence[float]) -> dict[str, float | int]:
    """Compute R@1 IoU metrics for one query bucket."""
    if not ious:
        return {"n_queries": 0, "mean_iou": 0.0, **{f"r1_iou_at_{t:g}": 0.0 for t in thresholds}}
    values = list(ious)
    return {
        "n_queries": len(values),
        "mean_iou": sum(values) / len(values),
        **{f"r1_iou_at_{threshold:g}": sum(iou >= threshold for iou in values) / len(values) for threshold in thresholds},
    }


def evaluate_charades(
    examples: Sequence[dict[str, Any]], cfg: object, alpha: float | None = None,
    output_path: Path | None = None, write_result: bool = True,
) -> dict[str, dict[str, float | int]]:
    """Evaluate top-1 moment retrieval and speech/no-speech query buckets.

    Each example must include ``query``, ``video_id``, ``start``, and ``end``.

    Args:
        examples: Ground-truth moment examples.
        cfg: Application configuration.
        alpha: Optional visual fusion-weight override.
        output_path: Optional JSON result destination.

    Returns:
        Overall, speech, and no-speech retrieval metric mappings.
    """
    all_ious: list[float] = []
    speech_ious: list[float] = []
    no_speech_ious: list[float] = []
    for example in examples:
        results = search_video(example["query"], example["video_id"], cfg, alpha)
        iou = (
            temporal_iou(results[0].start, results[0].end, float(example["start"]), float(example["end"]))
            if results
            else 0.0
        )
        all_ious.append(iou)
        (speech_ious if _contains_speech(example, cfg) else no_speech_ious).append(iou)
    thresholds = cfg.evaluation.moment_iou_thresholds
    report = {
        "overall": _summarize_ious(all_ious, thresholds),
        "speech": _summarize_ious(speech_ious, thresholds),
        "no_speech": _summarize_ious(no_speech_ious, thresholds),
    }
    if write_result:
        destination = output_path or (Path(cfg.paths.results) / "charades.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as results_file:
            json.dump(report, results_file, indent=2)
    return report
