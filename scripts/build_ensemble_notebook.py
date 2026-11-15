"""Create and execute notebooks/pbl3_ensemble.ipynb (PBL-3 ensemble learning walkthrough)."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "notebooks" / "pbl3_ensemble.ipynb"

CELLS = [
    new_markdown_cell(
        "# PBL-3: Ensemble learning for video moment retrieval\n\n"
        "Task: decide whether each second of a video answers a query (`rel_binary`) from 31 engineered features "
        "(Review 1 feature table). Every number below is a **held-out** result: leave-one-video-out over 9 NASA "
        "videos, with grouped inner cross-validation for CART depth, stacking, BMA weights, output smoothing and "
        "thresholds.\n\nRubric: problem & dataset (4) · ensemble implementation (8) · comparative analysis (4) · "
        "interpretation & presentation (4)."),
    new_code_cell(
        "import os, sys, json\nfrom pathlib import Path\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "os.chdir(ROOT); sys.path.insert(0, str(ROOT))\n"
        "import pandas as pd\nfrom IPython.display import Image, display, Markdown\n"
        "pd.set_option('display.precision', 3)\n"
        "RERUN = False  # True: refit all folds (about 10 minutes with 9 cores)\n"
        "if RERUN:\n    from cmvs.config import load_config\n    from analysis.ensemble import run_ensemble\n"
        "    run_ensemble(load_config('config.yaml'))\n"
        "OUT = ROOT / 'results' / 'ensemble'\nFIG = OUT / 'figures'"),
    new_markdown_cell("## 1. Problem and dataset"),
    new_code_cell(
        "data = pd.read_csv('results/ml/dataset.csv.gz')\n"
        "print(f'{len(data):,} rows | {data.video_id.nunique()} videos | {data.query_id.nunique()} queries | '\n"
        "      f'{data.rel_binary.mean():.1%} relevant')\n"
        "from analysis.features import FEATURES\nprint(len(FEATURES), 'features:', ', '.join(FEATURES))"),
    new_markdown_cell(
        "## 2. Ensemble implementations (`analysis/ensemble.py`)\n\n"
        "| Method | Idea |\n|---|---|\n| CART tree | single Gini tree, depth by inner CV |\n"
        "| Bagging | 300 bootstrap trees averaged (variance reduction) |\n"
        "| Random forest / Extra trees | bagging + random features / random thresholds |\n"
        "| AdaBoost (SAMME) | 300 re-weighted depth-2 trees (bias reduction) |\n"
        "| Hist. gradient boosting | 300 trees fitted to log-loss gradients |\n"
        "| Soft voting | mean probability of logistic, RF, HGB |\n"
        "| Stacking | logistic meta-learner on out-of-fold log-odds of 8 learners |\n"
        "| Bayesian model averaging | pseudo-BMA+ weights from out-of-fold log score, Bayesian bootstrap |"),
    new_code_cell(
        "from cmvs.config import load_config\nfrom analysis.ensemble import make_learner, LEARNERS\n"
        "cfg = load_config('config.yaml')\nfor key in LEARNERS:\n    print(f'{key:18s}', make_learner(key, cfg.ensemble, cfg.ml.seed, depth=4))"),
    new_code_cell("display(Image(str(ROOT / 'results/report/figures/workflow_pbl3.png'), width=900))"),
    new_code_cell("display(Image(str(FIG / '03_cart_depth.png'), width=800))\ndisplay(Image(str(FIG / '04_cart_tree.png'), width=1000))"),
    new_code_cell("display(Image(str(FIG / '02_ensemble_size.png'), width=750))"),
    new_code_cell("display(Image(str(FIG / '06_combiner_weights.png'), width=1000))"),
    new_markdown_cell("## 3. Comparative performance analysis (same folds and metrics as Reviews 1 and 2)"),
    new_code_cell(
        "summary = pd.read_csv(OUT / 'comparison.csv')\n"
        "cols = ['method','precision','recall','f1','f1_lo','f1_hi','pr_auc','roc_auc','hit_at_1','mrr','temporal_iou']\n"
        "summary[cols].sort_values('f1', ascending=False).reset_index(drop=True)"),
    new_code_cell("display(Image(str(FIG / '01_comparison.png'), width=1000))"),
    new_code_cell(
        "paired = pd.read_csv(OUT / 'paired_vs_logistic.csv')\n"
        "paired.pivot(index='method', columns='metric', values='mean_diff').round(3)"),
    new_code_cell("display(Image(str(FIG / '09_paired_differences.png'), width=1000))\ndisplay(Image(str(FIG / '08_pr_curves.png'), width=600))"),
    new_code_cell("display(pd.read_csv(OUT / 'training_cost.csv'))\ndisplay(Image(str(FIG / '11_cost_vs_quality.png'), width=650))"),
    new_markdown_cell("## 4. Result interpretation"),
    new_code_cell("for name in ['07_feature_importance.png', '05_diversity.png', '10_per_video_f1.png', '12_confusion.png', '13_example_timelines.png']:\n    display(Image(str(FIG / name), width=900))"),
    new_code_cell(
        "s = summary.set_index('method'); best = s.loc[s.index.isin(LEARNERS.values()) | s.index.isin(['Soft voting','Stacking','Bayesian model averaging'])].f1.idxmax()\n"
        "p = paired[(paired.method == best) & (paired.metric == 'f1')].iloc[0]\n"
        "display(Markdown(f\"**Best ensemble: {best}** — F1 {s.loc[best,'f1']:.3f}, PR-AUC {s.loc[best,'pr_auc']:.3f} \"\n"
        "  f\"(logistic regression: {s.loc['Logistic regression','f1']:.3f} / {s.loc['Logistic regression','pr_auc']:.3f}). \"\n"
        "  f\"Paired ΔF1 {p.mean_diff:+.3f} [{p.lo:+.3f}, {p.hi:+.3f}], better on {p.videos_better}/9 videos: a small, marginally \"\n"
        "  \"significant gain. Homogeneous tree ensembles fix single-tree variance (CART PR-AUC → forests) but do not beat \"\n"
        "  \"logistic regression alone; combining a linear model with randomized trees (BMA/stacking/voting) does best.\"))"),
]


def main() -> None:
    PATH.write_text(nbformat.writes(new_notebook(cells=CELLS, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})))
    from nbclient import NotebookClient

    os.chdir(ROOT / "notebooks")
    nb = nbformat.read(PATH, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    nbformat.write(nb, PATH)
    print(f"Executed {PATH}")


if __name__ == "__main__":
    main()
