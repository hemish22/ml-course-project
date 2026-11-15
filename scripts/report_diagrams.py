"""Draw the workflow diagrams used in the integrated project report (results/report/figures)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "report" / "figures"
PALETTE = {"data": "#dbe8f5", "prep": "#e7f1e2", "model": "#fdebd3", "eval": "#efe3f4", "out": "#f3f3f3"}


def _box(ax, x, y, w, h, text, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=PALETTE[kind], ec="#333333", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)
    return (x, y, w, h)


def _arrow(ax, a, b, side="h"):
    ax_, ay, aw, ah = a
    bx, by, bw, bh = b
    if side == "h":
        start, end = (ax_ + aw, ay + ah / 2), (bx, by + bh / 2)
    else:
        start, end = (ax_ + aw / 2, ay), (bx + bw / 2, by + bh)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, lw=1.2, color="#333333"))


def _canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def _legend(ax, x, y):
    for i, (kind, label) in enumerate([("data", "Data"), ("prep", "Preprocessing / features"),
                                       ("model", "Models"), ("eval", "Evaluation"), ("out", "Outputs")]):
        ax.add_patch(FancyBboxPatch((x + i * 2.4, y), .35, .25, boxstyle="round,pad=0.01", fc=PALETTE[kind], ec="#333"))
        ax.text(x + i * 2.4 + .45, y + .12, label, va="center", fontsize=8)


def pbl1() -> None:
    fig, ax = _canvas(16, 6.4)
    v = _box(ax, .2, 4.3, 2.4, 1.2, "9 NASA videos\n(public domain)\n+ 86 labelled queries", "data")
    f = _box(ax, 3.2, 4.3, 2.4, 1.2, "ffmpeg: 1 frame/s\n+ 16 kHz audio", "prep")
    e = _box(ax, 6.2, 4.3, 3.0, 1.2, "Frozen encoders\nCLIP ViT-B/32 (frames)\nWhisper → MiniLM (speech)", "model")
    s = _box(ax, 9.8, 4.3, 2.8, 1.2, "Per-query similarity\ntimelines (picture,\nspeech) per second", "prep")
    ft = _box(ax, 13.2, 4.3, 2.6, 1.2, "31 engineered features\n+ targets rel_graded,\nrel_binary (28,900 rows)", "prep")
    for a, b in [(v, f), (f, e), (e, s), (s, ft)]:
        _arrow(ax, a, b)
    cv = _box(ax, 12.6, 2.3, 3.2, 1.3, "Leave-one-video-out (9 folds)\ninner GroupKFold (3) grid\nsearch, smoothing, threshold", "eval")
    _arrow(ax, ft, cv, "v")
    reg = _box(ax, 8.3, 2.6, 3.6, 1.0, "Regression: Ridge, Random\nforest, Gradient boosting", "model")
    cls = _box(ax, 8.3, 1.3, 3.6, 1.0, "Classification: Logistic,\nk-NN, Gradient boosting", "model")
    ax.add_patch(FancyArrowPatch((12.6, 2.95), (11.9, 3.1), arrowstyle="-|>", mutation_scale=13, color="#333"))
    ax.add_patch(FancyArrowPatch((12.6, 2.95), (11.9, 1.8), arrowstyle="-|>", mutation_scale=13, color="#333"))
    m = _box(ax, 4.3, 1.9, 3.3, 1.3, "Metrics: RMSE, MAE, R²;\nPrecision, Recall, F1,\nROC-AUC, PR-AUC; Hit@k, MRR", "eval")
    ax.add_patch(FancyArrowPatch((8.3, 3.1), (7.6, 2.7), arrowstyle="-|>", mutation_scale=13, color="#333"))
    ax.add_patch(FancyArrowPatch((8.3, 1.8), (7.6, 2.4), arrowstyle="-|>", mutation_scale=13, color="#333"))
    o = _box(ax, .2, 1.9, 3.4, 1.3, "Outputs: 19 figures, tables,\nlearned ranker vs hand-set\nfusion (α = 0.6)", "out")
    _arrow(ax, m, o) if False else ax.add_patch(FancyArrowPatch((4.3, 2.55), (3.6, 2.55), arrowstyle="-|>", mutation_scale=13, color="#333"))
    _legend(ax, .3, .3)
    ax.set_title("PBL-1 workflow: feature construction, supervised regression and classification", fontsize=12)
    fig.savefig(OUT / "workflow_pbl1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def pbl2(kind: str) -> None:
    fig, ax = _canvas(16, 4.6)
    d = _box(ax, .2, 2.4, 2.6, 1.3, "Feature table\n(28,900 rows, 9 videos)", "data")
    if kind == "unit3":
        p = _box(ax, 3.3, 2.4, 2.8, 1.3, "8 selected features\nStandardScaler fitted\non training fold only", "prep")
        m1 = _box(ax, 6.7, 3.0, 3.0, 1.1, "K-means (k ∈ 2..5)\nselect k by silhouette", "model")
        m2 = _box(ax, 6.7, 1.6, 3.0, 1.1, "Gaussian mixture (diag)\nselect k by BIC", "model")
        c = _box(ax, 10.3, 2.4, 2.8, 1.3, "Label-free fit, then\ncluster relevance rate\nfrom training labels", "prep")
        title = "PBL-2 Unit 3 workflow: clustering with training-only calibration"
    else:
        p = _box(ax, 3.3, 2.4, 2.8, 1.3, "Order by (video, query, t)\n86 sequences; no\ncross-sequence transitions", "prep")
        m1 = _box(ax, 6.7, 3.0, 3.0, 1.1, "Gaussian HMM, K-means init\nBaum-Welch (EM), states 2..3", "model")
        m2 = _box(ax, 6.7, 1.6, 3.0, 1.1, "Select state count by BIC;\nforward-backward + Viterbi", "model")
        c = _box(ax, 10.3, 2.4, 2.8, 1.3, "State posteriors ×\ntraining state relevance\n→ score, F1 threshold", "prep")
        title = "PBL-2 Unit 4 workflow: Gaussian Hidden Markov Model on per-query sequences"
    ev = _box(ax, 13.6, 2.4, 2.3, 1.3, "Held-out video:\nF1, PR-AUC, Hit@1,\nMRR, temporal IoU", "eval")
    _arrow(ax, d, p)
    for mm in (m1, m2):
        ax.add_patch(FancyArrowPatch((6.1, 3.05), (6.7, mm[1] + .55), arrowstyle="-|>", mutation_scale=13, color="#333"))
        ax.add_patch(FancyArrowPatch((9.7, mm[1] + .55), (10.3, 3.05), arrowstyle="-|>", mutation_scale=13, color="#333"))
    _arrow(ax, c, ev)
    ax.text(8.2, .9, "Repeated for each of the 9 leave-one-video-out folds", ha="center", fontsize=9, style="italic")
    _legend(ax, .3, .1)
    ax.set_title(title, fontsize=12)
    fig.savefig(OUT / f"workflow_pbl2_{kind}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def pbl3() -> None:
    fig, ax = _canvas(16, 8.2)
    d = _box(ax, .2, 6.2, 2.6, 1.3, "Feature table: 31 features,\nrel_binary, 9 videos", "data")
    s = _box(ax, 3.4, 6.2, 3.0, 1.3, "Outer split: hold out 1 video\n(8 train / 1 test), ×9", "prep")
    i = _box(ax, 7.0, 6.2, 3.4, 1.3, "Inner GroupKFold (3) over\ntraining videos → out-of-fold\nprobabilities for each learner", "prep")
    _arrow(ax, d, s)
    _arrow(ax, s, i)
    learners = ["CART\n(depth by CV)", "Bagging\n300 trees", "Random forest\n300 trees", "Extra trees\n300 trees",
                "AdaBoost\n300 stumps-2", "Hist. gradient\nboosting 300", "Logistic\n(base)", "k-NN\n(base)"]
    boxes = []
    for k, label in enumerate(learners):
        boxes.append(_box(ax, .2 + k * 1.98, 3.9, 1.8, 1.2, label, "model"))
    for b in boxes:
        ax.add_patch(FancyArrowPatch((8.7, 6.2), (b[0] + .9, 5.1), arrowstyle="-|>", mutation_scale=10, lw=.8, color="#555"))
    v = _box(ax, 1.2, 1.7, 3.6, 1.3, "Soft voting\nmean P of Logistic, RF, HGB", "model")
    st = _box(ax, 6.0, 1.7, 3.8, 1.3, "Stacking\nlogistic meta-learner on\nout-of-fold log-odds (8 inputs)", "model")
    bm = _box(ax, 11.0, 1.7, 3.8, 1.3, "Bayesian model averaging\npseudo-BMA+ weights from\nout-of-fold log score", "model")
    for target in (v, st, bm):
        ax.add_patch(FancyArrowPatch((8.0, 3.9), (target[0] + target[2] / 2, 3.0), arrowstyle="-|>", mutation_scale=12, color="#333"))
    sel = _box(ax, 11.6, 6.2, 4.2, 1.3, "Per method, training only:\nsmoothing width (1–15 s) and\nF1 threshold on inner OOF", "eval")
    _arrow(ax, i, sel)
    ev = _box(ax, 1.2, .1, 13.6, 1.0, "Held-out video: Precision, Recall, F1, ROC-AUC, PR-AUC, Hit@1, MRR, temporal IoU → "
                                    "compare with PBL-1 / PBL-2 and paired video bootstrap", "eval")
    for src in (v, st, bm):
        ax.add_patch(FancyArrowPatch((src[0] + src[2] / 2, 1.7), (8.1, 1.1), arrowstyle="-|>", mutation_scale=12, color="#333"))
    ax.set_title("PBL-3 workflow: ensemble learning with leakage-safe nested validation", fontsize=12)
    fig.savefig(OUT / "workflow_pbl3.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pbl1()
    pbl2("unit3")
    pbl2("unit4")
    pbl3()
    print(f"Saved diagrams to {OUT}")


if __name__ == "__main__":
    main()
