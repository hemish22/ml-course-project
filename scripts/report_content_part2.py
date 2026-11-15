"""Chapters 4-6, references and appendix of the integrated report (compact)."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING

from docx_kit import Report

ENSEMBLES = ["CART tree", "Bagging", "Random forest", "Extra trees", "AdaBoost", "Hist. gradient boosting",
             "Soft voting", "Stacking", "Bayesian model averaging"]
GITHUB = "https://github.com/hemish22/ml-course-project"
COLAB = "https://colab.research.google.com/github/hemish22/ml-course-project/blob/main/notebooks/"


def counts(c: dict, symbol: str) -> str:
    return ", ".join(f"{symbol} = {k} in {v} fold{'s' if v != 1 else ''}" for k, v in sorted(c.items()))


def f3(x: float) -> str:
    return f"{x:.3f}"


def _paired(R: dict, method: str, metric: str) -> pd.Series:
    p = R["ens_paired"]
    return p[(p.method == method) & (p.metric == metric)].iloc[0]


def abstract_conclusion(R: dict) -> str:
    d = _paired(R, R["best"], "f1")
    return (
        f"The gain is small (+{d.mean_diff:.3f} F1, better on {int(d.videos_better)} of 9 videos) and only "
        "marginally significant: engineered context features, more than the choice of learner, drive performance. "
        "The project delivers a reproducible, leakage-safe comparison of all five units on one real retrieval task "
        "and a working search application."
    )


# --------------------------------------------------------------------------- chapter 4
def chapter4(r: Report, R: dict) -> None:
    P, e, det = R["paths"], R["ens"].set_index("method"), R["ens_details"]
    best = R["best"]
    depths = pd.Series([x["cart_depth"] for x in det]).value_counts().to_dict()
    members = det[0]["stacking"]["members"]
    bma = np.mean([x["bma"]["weights"] for x in det], axis=0)
    cost = R["ens_cost"].set_index("learner")
    imp = R["ens_importance"].mean(axis=1).sort_values(ascending=False)
    lr = e.loc["Logistic regression"]
    dbf, dbp = _paired(R, best, "f1"), _paired(R, best, "pr_auc")

    r.chapter_heading("Advanced Predictive Modelling and Ensemble Learning (Unit 5)")
    r.h2("4.1 Overview of Allotted Unit 5 Topics")
    r.para(
        "Ensembles combine base models so that the combination generalizes better than any member. **Bagging, "
        "random forests and extra trees** reduce variance by averaging many unstable trees; **boosting** (AdaBoost, "
        "gradient boosting) reduces bias by adding weak learners that correct the current ensemble, an adaptive "
        "basis-function model whose basis functions are trees; **soft voting, stacking and Bayesian model averaging "
        "(BMA)** combine heterogeneous models. The single CART tree is the reference point. The task is the PBL-1 "
        "classification problem, so results compare directly with Chapters 2 and 3."
    )
    r.h2("4.2 Dataset Description and Preprocessing")
    r.bullets([
        f"Same dataset ({len(R['dataset']):,} rows, 31 features, target `rel_binary`) and the same nine leave-one-video-out folds.",
        "Trees use raw features (invariant to scaling); no learner uses class re-weighting, so outputs are probabilities "
        "suitable for averaging and BMA; imbalance is handled by the F1 threshold.",
        "**Nested validation:** grouped 3-fold out-of-fold (OOF) predictions on the eight training videos select CART "
        "depth, train the stacking meta-learner, compute BMA weights and choose each method's smoothing and threshold.",
    ])
    r.h2("4.3 EDA and Feature Analysis")
    r.para(
        f"Dataset-level EDA is in Section 2.4. Impurity importance averaged over the tree ensembles ranks "
        f"{imp.index[0]} first ({imp.iloc[0]:.2f}), then {imp.index[1]} and {imp.index[2]}: how far a second is from "
        "the strongest speech evidence matters more than raw similarity, as in PBL 1. Held-out probabilities of "
        "logistic regression and the tree ensembles correlate only moderately (Spearman ≈ 0.7), so their errors "
        "are partly independent, the precondition for a combination to beat its best member."
    )
    r.h2("4.4 Proposed Workflow / Architecture")
    r.figure_block(P["rp"] / "workflow_pbl3.png", "Unit 5 ensemble-learning workflow with nested validation", 4.6,
                   "Inner OOF predictions train the combiners and tune decisions, and the held-out video is used only "
                   "for evaluation (`analysis/ensemble.py`, settings in `config.yaml`).")
    r.h2("4.5 Algorithm Theory and Mathematical Formulation")
    r.equation("G(S) = 1 − Σ_c p_c², Δ = G(S) − (|S_L|/|S|)G(S_L) − (|S_R|/|S|)G(S_R)",
               f"CART [7] chooses the split maximizing the Gini decrease Δ; depth from {{4, 6, 8, 12}} by inner CV "
               f"({counts(depths, 'depth')}).")
    r.equation("p̂(x) = (1/B) Σ_b T_b(x),      Var[p̂] = ρσ² + (1 − ρ)σ²/B",
               "Bagging [8] averages B = 300 bootstrap trees; random forests [9] lower the tree correlation ρ by "
               "sampling √p features per split, and extra trees [10] also randomize thresholds.")
    r.equation("α_m = η ln((1 − ε_m)/ε_m),      F_m(x) = F_{m−1}(x) + ν h_m(x)",
               "AdaBoost (SAMME) [11] re-weights misclassified rows and votes with weights α_m (300 depth-2 trees, "
               "η = 0.5); gradient boosting [12] fits each tree to the log-loss gradient (300 iterations, ν = 0.05).")
    r.equation("p_vote = (1/M) Σ_m p_m,      p_stack = σ(γ₀ + Σ_m γ_m logit p_m)",
               "Soft voting averages logistic regression, random forest and gradient boosting; stacking [13] learns "
               "γ with a logistic meta-learner on the OOF log-odds of all eight base learners.")
    r.equation("w_m = E_{α∼Dir(1)}[softmax_m(G Σ_g α_g ℓ_{mg})]",
               "BMA predicts p(c | x, D) = Σ_m w_m p(c | x, M_m). Pseudo-BMA+ [14] weights models by their OOF log "
               "predictive density ℓ per training query g, regularized by 1,000 Bayesian-bootstrap draws.")
    r.h2("4.6 Results and Performance Evaluation")
    cols = ["precision", "recall", "f1", "pr_auc", "hit_at_1", "temporal_iou"]
    order = sorted(ENSEMBLES, key=lambda m: -e.loc[m, "f1"])
    r.table_block("Ensemble results on held-out videos (mean over 9 videos; sorted by F1)",
                  ["Method", "Precision", "Recall", "F1", "PR-AUC", "Hit@1", "IoU"],
                  [[m] + [f3(e.loc[m, c]) for c in cols] for m in order],
                  widths=[2.0, 0.8, 0.7, 0.6, 0.75, 0.65, 0.6], bold_rows=[0],
                  interpretation=f"{best} is best on F1 ({e.loc[best, 'f1']:.3f}), PR-AUC ({e.loc[best, 'pr_auc']:.3f}) "
                                 f"and IoU. The depth-4 CART tree matches the forests on F1 ({e.loc['CART tree', 'f1']:.3f}) "
                                 f"but ranks coarsely (PR-AUC {e.loc['CART tree', 'pr_auc']:.3f}); averaging trees mainly "
                                 "repairs this ranking weakness, and PR-AUC flattens beyond 50–100 trees.")
    r.figure_block(P["en"] / "06_combiner_weights.png", "Stacking meta-learner coefficients and BMA weights across folds", 5.75,
                   f"BMA gives logistic regression {bma[members.index('logistic')]:.2f} of the weight, random forest "
                   f"{bma[members.index('random_forest')]:.2f} and extra trees {bma[members.index('extra_trees')]:.2f}; "
                   "stacking makes the same choice of a linear model plus randomized trees.")
    r.h2("4.7 Comparison with Previous Models")
    rows = []
    for m in ENSEMBLES:
        a, b = _paired(R, m, "f1"), _paired(R, m, "pr_auc")
        rows.append([m, f"{a.mean_diff:+.3f}", f"[{a.lo:+.3f}, {a.hi:+.3f}]", f"{int(a.videos_better)}/9",
                     f"{b.mean_diff:+.3f}", f"[{b.lo:+.3f}, {b.hi:+.3f}]"])
    r.table_block("Paired per-video differences from PBL-1 logistic regression (95% video-bootstrap CI)",
                  ["Ensemble", "ΔF1", "95% CI", "Better", "ΔPR-AUC", "95% CI"], rows,
                  widths=[1.75, 0.6, 1.1, 0.55, 0.7, 1.1],
                  interpretation=f"Only {best} has intervals excluding zero for F1 ([{dbf.lo:+.3f}, {dbf.hi:+.3f}], "
                                 f"better on {int(dbf.videos_better)}/9 videos) and PR-AUC ([{dbp.lo:+.4f}, {dbp.hi:+.3f}]); "
                                 "the lower bounds are almost zero, so the gain is small and marginally significant. "
                                 "Homogeneous tree ensembles are indistinguishable from logistic regression.")
    r.figure_block(P["en"] / "01_comparison.png", "Held-out comparison of all models from PBL 1, PBL 2 and PBL 3", 5.75,
                   f"Ensembles and PBL-1 classifiers form one group (F1 ≈ 0.74–0.77) well above the unsupervised PBL-2 "
                   f"models and the baselines; {best} improves on logistic regression by "
                   f"+{e.loc[best, 'f1'] - lr.f1:.3f} F1 and on K-means by +{e.loc[best, 'f1'] - e.loc['K-means', 'f1']:.3f}.")
    r.h2("4.8 Discussion, Findings and Limitations")
    r.bullets([
        "Tree ensembles fix a single tree's high variance but alone do not beat a regularized logistic regression; "
        "combining a linear model with randomized trees gives the best results.",
        f"Cost does not buy quality: histogram boosting fits in {cost.loc['Hist. gradient boosting', 'fit_s']:.1f} s per "
        f"fold and extra trees in {cost.loc['Extra trees', 'fit_s']:.1f} s, while bagging needs "
        f"{cost.loc['Bagging', 'fit_s']:.0f} s for lower PR-AUC; stacking and BMA also need the inner CV.",
        "Calibration matters for BMA: with class-balanced weights, k-NN took most of the weight only because it was "
        "the one calibrated model, so re-weighting was removed from all learners.",
        "Limitations: nine videos give little statistical power, pseudo-BMA+ approximates the model posterior, and "
        "ensemble sizes were fixed rather than tuned.",
    ])
    r.h2("4.9 Chapter Conclusion")
    r.para(
        f"Nine ensemble methods were evaluated under the same leakage-safe protocol. {best} reached the project's "
        f"best F1 ({e.loc[best, 'f1']:.3f}), PR-AUC ({e.loc[best, 'pr_auc']:.3f}) and IoU "
        f"({e.loc[best, 'temporal_iou']:.3f}), a small but consistent gain that comes from model diversity."
    )


# --------------------------------------------------------------------------- chapter 5
def chapter5(r: Report, R: dict) -> None:
    e = R["ens"].set_index("method")
    rt = R["retrieval"].set_index("method")
    best = R["best"]
    r.chapter_heading("Integrated Discussion and Findings")
    rows = [
        ["Baseline", "Fixed-weight fusion", "None", f3(e.loc["Fixed-weight fusion", "f1"]),
         f3(e.loc["Fixed-weight fusion", "pr_auc"]), f3(e.loc["Fixed-weight fusion", "hit_at_1"])],
        ["PBL 1 / Unit 1", "Random forest regression", "Graded", "—", "—",
         f3(rt.loc["Regression: Random forest", "hit_at_1"]) + "*"],
        ["PBL 1 / Unit 2", "Logistic regression", "Binary", f3(e.loc["Logistic regression", "f1"]),
         f3(e.loc["Logistic regression", "pr_auc"]), f3(e.loc["Logistic regression", "hit_at_1"])],
        ["PBL 2 / Unit 3", "K-means", "None", f3(e.loc["K-means", "f1"]), f3(e.loc["K-means", "pr_auc"]),
         f3(e.loc["K-means", "hit_at_1"])],
        ["PBL 2 / Unit 4", "Gaussian HMM", "None", f3(e.loc["Gaussian HMM", "f1"]), f3(e.loc["Gaussian HMM", "pr_auc"]),
         f3(e.loc["Gaussian HMM", "hit_at_1"])],
        ["PBL 3 / Unit 5", best, "Binary", f3(e.loc[best, "f1"]), f3(e.loc[best, "pr_auc"]), f3(e.loc[best, "hit_at_1"])],
    ]
    r.table_block("Best model of each unit on the shared benchmark (mean over 9 held-out videos)",
                  ["Unit", "Best model", "Labels", "F1", "PR-AUC", "Hit@1"], rows,
                  widths=[1.2, 2.1, 0.75, 0.55, 0.7, 0.65],
                  interpretation="*Regression Hit@1 is pooled over 86 queries. Supervision dominates: supervised models "
                                 "are about 0.25 F1 above the fixed rule and 0.13–0.22 above the unsupervised ones, while "
                                 "supervised models differ among themselves by only 0.01–0.03.")
    r.h2("5.1 Critical Findings")
    r.bullets([
        "**Features matter more than models:** context features raised logistic F1 by about 0.1; switching among the best models changes it by at most 0.02.",
        "**Learning beats the hand-set rule** on F1, PR-AUC and IoU; the Hit@1 gain is consistent but within the uncertainty of 86 queries.",
        "**Unsupervised structure is real but incomplete:** K-means and the HMM find relevant-rich clusters/states without labels, but do not optimize relevance.",
        f"**Ensembles add a small, marginally significant gain** through diversity ({best}); every model family identifies speech strength near its peak as the key evidence.",
    ])
    r.h2("5.2 Study Limitations")
    r.para(
        "Nine videos from one domain and 86 queries give low statistical power; labels were written from transcripts "
        "(speech bias) with one interval per query; frozen encoders propagate transcription errors; and features use "
        "whole-video context, so the models suit offline retrieval only."
    )


# --------------------------------------------------------------------------- chapter 6
def chapter6(r: Report, R: dict) -> None:
    e = R["ens"].set_index("method")
    reg = R["reg"].set_index("model")
    best = R["best"]
    r.chapter_heading("Overall Conclusion and Future Scope")
    r.para(
        "Needle, a working cross-modal video search application, was used to study all five units on one real "
        f"problem: regression of graded relevance (R² {reg.loc['random_forest', 'r2_mean']:.2f}), classification "
        f"(logistic F1 {e.loc['Logistic regression', 'f1']:.3f}), label-free clustering (K-means F1 "
        f"{e.loc['K-means', 'f1']:.3f}), a from-scratch Gaussian HMM (F1 {e.loc['Gaussian HMM', 'f1']:.3f}) and nine "
        f"ensembles, the best of which ({best}) reached F1 {e.loc[best, 'f1']:.3f} and PR-AUC "
        f"{e.loc[best, 'pr_auc']:.3f}. Every supervised model clearly beats the hand-designed fusion rule (F1 "
        f"{e.loc['Fixed-weight fusion', 'f1']:.3f}). All results are leave-one-video-out estimates with nested "
        "selection, reproducible from the saved code, results and automated tests."
    )
    r.h2("6.1 Future Scope")
    r.bullets([
        "Label more videos from other domains, annotated by watching the picture, to reduce speech bias and raise statistical power.",
        "Tune ensemble sizes and boosting rates within the nested protocol and calibrate probabilities before averaging.",
        "Deploy the best learned ranker in the web application, and try supervised sequence models (e.g. CRFs) that join HMM-style persistence with supervised accuracy.",
    ])


# --------------------------------------------------------------------------- references
REFERENCES = [
    "A. Radford et al., \"Learning transferable visual models from natural language supervision,\" in Proc. ICML, PMLR vol. 139, pp. 8748–8763, 2021.",
    "A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, \"Robust speech recognition via large-scale weak supervision,\" in Proc. ICML, PMLR vol. 202, pp. 28492–28518, 2023.",
    "N. Reimers and I. Gurevych, \"Sentence-BERT: Sentence embeddings using Siamese BERT-networks,\" in Proc. EMNLP-IJCNLP, pp. 3982–3992, 2019.",
    "J. Johnson, M. Douze, and H. Jégou, \"Billion-scale similarity search with GPUs,\" IEEE Trans. Big Data, vol. 7, no. 3, pp. 535–547, 2021.",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" J. Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
    "T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, 2nd ed. New York, NY, USA: Springer, 2009.",
    "L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, Classification and Regression Trees. Belmont, CA, USA: Wadsworth, 1984.",
    "L. Breiman, \"Bagging predictors,\" Machine Learning, vol. 24, no. 2, pp. 123–140, 1996.",
    "L. Breiman, \"Random forests,\" Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
    "P. Geurts, D. Ernst, and L. Wehenkel, \"Extremely randomized trees,\" Machine Learning, vol. 63, no. 1, pp. 3–42, 2006.",
    "Y. Freund and R. E. Schapire, \"A decision-theoretic generalization of on-line learning and an application to boosting,\" J. Computer and System Sciences, vol. 55, no. 1, pp. 119–139, 1997.",
    "J. H. Friedman, \"Greedy function approximation: A gradient boosting machine,\" Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001.",
    "D. H. Wolpert, \"Stacked generalization,\" Neural Networks, vol. 5, no. 2, pp. 241–259, 1992.",
    "Y. Yao, A. Vehtari, D. Simpson, and A. Gelman, \"Using stacking to average Bayesian predictive distributions,\" Bayesian Analysis, vol. 13, no. 3, pp. 917–1007, 2018.",
    "L. R. Rabiner, \"A tutorial on hidden Markov models and selected applications in speech recognition,\" Proc. IEEE, vol. 77, no. 2, pp. 257–286, 1989.",
    "A. P. Dempster, N. M. Laird, and D. B. Rubin, \"Maximum likelihood from incomplete data via the EM algorithm,\" J. Royal Statistical Society B, vol. 39, no. 1, pp. 1–38, 1977.",
    "P. J. Rousseeuw, \"Silhouettes: A graphical aid to the interpretation and validation of cluster analysis,\" J. Computational and Applied Mathematics, vol. 20, pp. 53–65, 1987.",
    "G. Schwarz, \"Estimating the dimension of a model,\" Annals of Statistics, vol. 6, no. 2, pp. 461–464, 1978.",
]


def references(r: Report, R: dict) -> None:
    r.page_break()
    r.title("References", toc=True)
    for i, ref in enumerate(REFERENCES, start=1):
        p = r.doc.add_paragraph(style="Report Body")
        pf = p.paragraph_format
        pf.left_indent, pf.first_line_indent = Inches(0.45), Inches(-0.45)
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
        pf.space_after = Pt(3)
        p.add_run(f"[{i}]\t{ref}")


# --------------------------------------------------------------------------- appendix
def excerpt(path: Path, names: list[str]) -> str:
    """Source of the named functions / methods of ``path``, without docstrings (stays in sync with the code)."""
    source = path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            skip = set()
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                skip = set(range(body[0].lineno, body[0].end_lineno + 1))
            text = "\n".join(lines[i - 1] for i in range(start, node.end_lineno + 1) if i not in skip)
            found[node.name] = textwrap.dedent(text)
    missing = [n for n in names if n not in found]
    if missing:
        raise KeyError(f"{path}: functions not found: {missing}")
    return "\n\n".join(found[n] for n in names)


def appendix(r: Report, R: dict) -> None:
    root = R["root"]
    r.page_break()
    r.title("Appendix A", toc=True, space_after=2)
    r.title("Coding", size=14)
    r.para(align=WD_ALIGN_PARAGRAPH.LEFT, text=
        f"Only the core PBL code is listed; the complete project is at {GITHUB}. Executed notebooks open in Google "
        f"Colab: Units 1–2 {COLAB}analysis.ipynb, Units 3–4 {COLAB}review_clustering_hmm.ipynb, Unit 5 "
        f"{COLAB}pbl3_ensemble.ipynb. Each reads the saved feature table, so no video download is needed."
    )
    sections = [
        ("Chapter 2", "Implementation Methodology and Algorithm Configuration", [
            ("analysis/models.py", ["make_model"]),
            ("analysis/evaluate.py", ["best_f1_threshold"]),
        ]),
        ("Chapter 3", "Unit 3 – Proposed Workflow and Implementation Methodology", [
            ("analysis/review.py", ["relevance_rates", "_kmeans_membership"]),
        ]),
        ("Chapter 3", "Unit 4 – Proposed Workflow and Implementation Methodology", [
            ("analysis/hmm.py", ["forward_backward", "predict"]),
        ]),
        ("Chapter 4", "Implementation Methodology and Hyperparameter Configuration", [
            ("analysis/ensemble.py", ["make_learner", "pseudo_bma_weights"]),
        ]),
    ]
    for label, title, files in sections:
        r.h3(f"{label}: {title}")
        for name, functions in files:
            r.para(f"`{name}`: {', '.join(functions)}")
            r.code(excerpt(root / name, functions))
