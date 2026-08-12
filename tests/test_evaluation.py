from pathlib import Path
from types import SimpleNamespace

import numpy as np

import eval.ablations as ablations
import eval.charades as charades
import eval.msrvtt as msrvtt
from cmvs.search import Moment


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(results=tmp_path / "results", index=tmp_path / "index"),
        evaluation=SimpleNamespace(
            alpha_values=(0.0, 1.0),
            pooling_methods=("mean", "max"),
            clip_models=("model-a", "model-b"),
            moment_iou_thresholds=(0.5, 0.7),
            plot_dpi=200,
        ),
    )


def test_msrvtt_metrics_write_json_without_dataset_download(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(
        msrvtt,
        "encode_query_visual",
        lambda caption, _: np.array([[1.0, 0.0]], dtype=np.float32)
        if caption == "first"
        else np.array([[0.0, 1.0]], dtype=np.float32),
    )

    metrics = msrvtt.evaluate_msrvtt(
        [np.array([[1.0, 0.0]], dtype=np.float32), np.array([[0.0, 1.0]], dtype=np.float32)],
        ["first", "second"],
        cfg,
        "mean",
    )

    assert metrics["recall_at_1"] == 1.0
    assert (cfg.paths.results / "msrvtt.json").exists()


def test_charades_reports_speech_buckets(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    moment = Moment("video", 1.0, 3.0, 1.0, 1.0, 0.0, "frame.jpg", "speech")
    monkeypatch.setattr(charades, "search_video", lambda *_args, **_kwargs: [moment])

    report = charades.evaluate_charades(
        [
            {"query": "a", "video_id": "video", "start": 1.0, "end": 3.0, "contains_speech": True},
            {"query": "b", "video_id": "video", "start": 0.0, "end": 1.0, "contains_speech": False},
        ],
        cfg,
    )

    assert report["speech"]["r1_iou_at_0.7"] == 1.0
    assert report["no_speech"]["mean_iou"] == 0.0
    assert (cfg.paths.results / "charades.json").exists()


def test_ablations_write_csv_and_request_encoder_reindex(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(
        ablations,
        "evaluate_charades",
        lambda _examples, _cfg, alpha, write_result: {"overall": {"mean_iou": alpha, "n_queries": 1}},
    )
    plotted: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        ablations,
        "plot_alpha_sweep",
        lambda _rows, path, dpi: plotted.append((path, dpi)),
    )

    rows = ablations.run_ablations([], cfg, encoder_evaluator=lambda model: {"mean_iou": len(model)})

    assert (cfg.paths.results / "ablations.csv").exists()
    assert len([row for row in rows if row["configuration"] == "encoder"]) == 2
    assert plotted == [(cfg.paths.results / "alpha_sweep.png", 200)]
