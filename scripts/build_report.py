"""Build the integrated PBL-1/2/3 project report (Word) from the template and saved results.

Usage:
    python scripts/build_report.py            # cover taken from report/cover.docx

Every number in the report is read from the saved result tables, so rerunning the
experiments and then this script keeps the report consistent with the code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from docx_kit import Report  # noqa: E402
from report_content import write_report  # noqa: E402


def load_results() -> dict:
    """Collect every table the report quotes."""
    ml, rv, en = ROOT / "results" / "ml", ROOT / "results" / "review", ROOT / "results" / "ensemble"
    data = pd.read_csv(ml / "dataset.csv.gz")
    return {
        "dataset": data,
        "queries": json.loads((ROOT / "benchmark" / "queries.json").read_text()),
        "reg": pd.read_csv(ml / "regression_summary.csv"),
        "cls": pd.read_csv(ml / "classification_summary.csv"),
        "cls_pooled": pd.read_csv(ml / "classification_pooled.csv"),
        "retrieval": pd.read_csv(ml / "retrieval.csv"),
        "ablation": pd.read_csv(ml / "ablation.csv"),
        "importance": pd.read_csv(ml / "feature_importance_folds.csv"),
        "learning": pd.read_csv(ml / "learning_curve.csv"),
        "review": pd.read_csv(rv / "comparison.csv"),
        "review_details": json.loads((rv / "model_details.json").read_text()),
        "review_selection": pd.read_csv(rv / "model_selection.csv"),
        "ens": pd.read_csv(en / "comparison.csv"),
        "ens_folds": pd.read_csv(en / "per_video.csv"),
        "ens_paired": pd.read_csv(en / "paired_vs_logistic.csv"),
        "ens_details": json.loads((en / "model_details.json").read_text()),
        "ens_cost": pd.read_csv(en / "training_cost.csv"),
        "ens_importance": pd.read_csv(en / "feature_importance.csv", index_col=0),
        "ens_manifest": json.loads((en / "manifest.json").read_text()),
        "paths": {"ml": ml / "figures", "rv": rv / "figures", "en": en / "figures",
                  "rp": ROOT / "results" / "report" / "figures"},
        "root": ROOT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT / "report" / "cover.docx",
                        help="Document whose cover page (everything before its first section break) is kept.")
    parser.add_argument("--out", type=Path, default=ROOT / "report" / "ML_Project_Report.docx")
    parser.add_argument("--cover-elements", type=int, default=None,
                        help="Body elements kept as the cover (default: up to the first section break).")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = Report(args.template, args.cover_elements)
    write_report(report, load_results())
    report.finish(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
