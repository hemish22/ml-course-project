"""Build the benchmark feature table, train/evaluate the six models, and draw the charts.

Resumable: each step is skipped when its output exists, unless ``--force`` is given.

    python scripts/run_analysis.py            # everything
    python scripts/run_analysis.py --step build
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

if sys.platform == "darwin":  # faiss + torch each bundle libomp and segfault together
    os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from analysis.benchmark import load_benchmark
from analysis.evaluate import (
    alpha_sweep,
    per_video_retrieval,
    pooled_metrics,
    retrieval_metrics,
    run_experiment,
    summarise_folds,
)
from analysis.features import build_dataset
from cmvs.config import load_config


def write_retrieval_tables(cfg: object, tables: Path) -> None:
    """Recompute retrieval comparisons from saved out-of-fold predictions (fast, no training)."""
    dataset = pd.read_csv(tables / "dataset.csv.gz")
    oof = pd.read_csv(tables / "oof_predictions.csv.gz")
    ks = cfg.ml.retrieval_ks
    retrieval_metrics(oof, dataset, ks, cfg.ml.seed).to_csv(tables / "retrieval.csv", index=False)
    methods = {
        "Fixed-weight fusion": "fused_baseline",
        "Picture only": "visual_raw",
        "Speech only": "speech_raw",
        "Regression: Gradient boosting": "reg_gradient_boosting",
        "Classification: Gradient boosting": "cls_gradient_boosting",
    }
    per_video_retrieval(oof, dataset, methods, ks, cfg.ml.seed).to_csv(tables / "retrieval_per_video.csv", index=False)
    alpha_sweep(dataset, cfg.evaluation.alpha_values, ks, cfg.ml.seed).to_csv(tables / "alpha_sweep.csv", index=False)
    print("Saved retrieval tables")


def main() -> None:
    """Run the requested analysis steps."""
    parser = argparse.ArgumentParser(description="Regression and classification analysis.")
    parser.add_argument("--step", choices=["build", "train", "retrieval", "plots", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Recompute outputs that already exist.")
    args = parser.parse_args()
    cfg = load_config()
    tables = Path(cfg.ml.tables_dir)
    tables.mkdir(parents=True, exist_ok=True)
    dataset_path = tables / "dataset.csv.gz"
    oof_path = tables / "oof_predictions.csv.gz"

    if args.step in ("build", "all") and (args.force or not dataset_path.exists()):
        queries = load_benchmark(cfg.ml.benchmark)
        dataset = build_dataset(cfg, queries)
        dataset.to_csv(dataset_path, index=False)
        print(f"Built {len(dataset):,} rows from {len(queries)} queries -> {dataset_path}")

    if args.step in ("train", "all") and (args.force or not oof_path.exists()):
        dataset = pd.read_csv(dataset_path)
        result = run_experiment(dataset, cfg)
        result.oof.to_csv(oof_path, index=False)
        result.best_params.to_csv(tables / "best_params.csv", index=False)
        result.importance.to_csv(tables / "feature_importance_folds.csv", index=False)
        result.learning_curve.to_csv(tables / "learning_curve.csv", index=False)
        result.thresholds.to_csv(tables / "thresholds.csv", index=False)
        result.regression_folds.to_csv(tables / "regression_folds.csv", index=False)
        result.classification_folds.to_csv(tables / "classification_folds.csv", index=False)
        summarise_folds(result.regression_folds).to_csv(tables / "regression_summary.csv", index=False)
        summarise_folds(result.classification_folds).to_csv(tables / "classification_summary.csv", index=False)
        pooled_reg, pooled_cls = pooled_metrics(result.oof)
        pooled_reg.to_csv(tables / "regression_pooled.csv", index=False)
        pooled_cls.to_csv(tables / "classification_pooled.csv", index=False)
        print(f"Saved tables to {tables}")

    if args.step in ("train", "retrieval", "all") and oof_path.exists():
        write_retrieval_tables(cfg, tables)

    if args.step in ("plots", "all"):
        from analysis.plots import make_all_plots

        make_all_plots(cfg)


if __name__ == "__main__":
    main()
