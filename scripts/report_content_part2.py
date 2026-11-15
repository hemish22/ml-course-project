"""Chapters 4-6, references and appendix of the integrated report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from docx_kit import Report

ENSEMBLES = ["CART tree", "Bagging", "Random forest", "Extra trees", "AdaBoost", "Hist. gradient boosting",
             "Soft voting", "Stacking", "Bayesian model averaging"]
GITHUB = "https://github.com/hemish22/ml-course-project"



def counts(c: dict, symbol: str) -> str:
    """{2: 7, 3: 2} -> 'k = 2 in 7 folds, k = 3 in 2 folds'."""
    return ", ".join(f"{symbol} = {k} in {v} fold{'s' if v != 1 else ''}" for k, v in sorted(c.items()))

def f3(x: float) -> str:
    return f"{x:.3f}"


def _paired(R: dict, method: str, metric: str) -> pd.Series:
    p = R["ens_paired"]
    return p[(p.method == method) & (p.metric == metric)].iloc[0]


def abstract_conclusion(R: dict) -> str:
    best = R["best"]
    d = _paired(R, best, "f1")
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
    names = {"logistic": "Logistic regression (base)", "knn": "k-NN (base)", "cart": "CART tree", "bagging": "Bagging",
             "random_forest": "Random forest", "extra_trees": "Extra trees", "adaboost": "AdaBoost",
             "gradient_boosting": "Hist. gradient boosting"}
    bma = np.mean([x["bma"]["weights"] for x in det], axis=0)
    windows = pd.DataFrame([{"m": m, **v} for x in det for m, v in x["methods"].items()])
    win = windows.groupby("m").smooth_window.agg(lambda s: int(s.mode().iloc[0]))
    thr = windows.groupby("m").threshold.mean()
    inner = windows.groupby("m").inner_pr_auc.mean()
    cost = R["ens_cost"].set_index("learner")
    imp = R["ens_importance"].mean(axis=1).sort_values(ascending=False)
    lr = e.loc["Logistic regression"]
    dbf, dbp = _paired(R, best, "f1"), _paired(R, best, "pr_auc")

    r.chapter_heading("Advanced Predictive Modelling and Ensemble Learning (Unit 5)")
    r.h2("4.1 Overview of Allotted Unit 5 Topics")
    r.para(
        "Unit 5 studies ensemble learning: combining many base models so that the combination generalizes better "
        "than any member. Ensembles reduce error through two mechanisms. **Variance reduction** (bagging, random "
        "forests, extra trees) averages many unstable, low-bias trees trained on perturbed data. **Bias reduction** "
        "(boosting) adds weak learners sequentially, each correcting the errors of the current ensemble; boosting "
        "is an adaptive basis-function model whose basis functions (trees) are learned from data. A third family "
        "combines **heterogeneous** models: soft voting averages their probabilities, stacking learns the "
        "combination with a meta-model, and Bayesian model averaging (BMA) weights models by their predictive "
        "evidence. The single CART decision tree, the building block of all tree ensembles, is included as the "
        "reference point."
    )
    r.para(
        "The ensemble task is the PBL-1 classification problem: predict whether a second belongs to the answer "
        "interval. This makes the results directly comparable with the logistic, k-NN and gradient-boosting "
        "classifiers of Chapter 2 and the clustering and HMM models of Chapter 3, using the same folds and metric code."
    )
    r.h2("4.2 Dataset Description and Preprocessing")
    r.para(
        f"The existing dataset (Chapter 2) is used: {len(R['dataset']):,} rows, all 31 features, target "
        "`rel_binary`, leave-one-video-out outer folds. Preprocessing specific to Unit 5:"
    )
    r.bullets([
        "**No scaling for trees:** CART-based learners are invariant to monotone feature transformations, so raw "
        "features are used; the logistic and k-NN base learners keep their standardization pipeline.",
        "**No class re-weighting:** unlike PBL 1, no learner uses balanced class weights, so every model outputs a "
        "probability estimate of relevance. This is required for meaningful probability averaging (voting) and "
        "log-score model weights (BMA). The 8.3% imbalance is handled by choosing the F1-maximizing threshold.",
        "**Nested validation:** within each outer fold, a 3-fold GroupKFold over the eight training videos produces "
        "out-of-fold (OOF) probabilities for every base learner. These OOF predictions select the CART depth, train "
        "the stacking meta-learner, compute BMA weights, and choose each method's output smoothing width (1–15 s) and "
        "decision threshold. The held-out video is used only for the final evaluation.",
        "**Meta-features for stacking:** base probabilities are clipped to [10⁻⁴, 1 − 10⁻⁴] and converted to log-odds, "
        "then standardized before the logistic meta-learner.",
    ])
    r.h2("4.3 EDA and Feature Analysis")
    r.para(
        "Dataset-level EDA was presented in Section 2.4 and is not repeated. Two analyses specific to ensembles are "
        "added: the importance that tree ensembles assign to features, and the diversity among base learners, which "
        "determines how much a combination can gain."
    )
    r.figure_block(P["en"] / "07_feature_importance.png", "Mean impurity-based feature importance of the tree ensembles", 5.8,
                   f"All tree models rely most on {imp.index[0]} (mean share {imp.iloc[0]:.2f}), followed by "
                   f"{imp.index[1]} and {imp.index[2]}: how far a second is from the strongest speech evidence matters more "
                   "than the raw similarity value. This agrees with the PBL-1 permutation importance (Figure 2.9) and "
                   "confirms that the engineered temporal-context features carry the signal.")
    r.figure_block(P["en"] / "05_diversity.png", "Diversity of base learners: Spearman correlation of held-out probabilities", 5.2,
                   "Correlations range from about 0.43 to 0.85. The single CART tree is the most different from all "
                   "others (its piecewise-constant output ranks seconds coarsely); random forest and extra trees are the "
                   "most similar pair. Moderate correlation between logistic regression and the tree ensembles "
                   "(≈ 0.68–0.72) means their errors are partly independent, which is the precondition for a "
                   "heterogeneous combination to beat its best member.")
    r.h2("4.4 Proposed Workflow / Architecture")
    r.figure_block(P["rp"] / "workflow_pbl3.png", "Unit 5 ensemble-learning workflow with nested validation", 6.3)
    r.para(
        "**Stage 1 – outer split:** one of nine videos is held out. **Stage 2 – inner OOF predictions:** for each of "
        "eight base learners, three models are fitted on two thirds of the training videos and predict the remaining "
        "third, giving honest probabilities for every training row; the base learner is then refitted on all eight "
        "training videos. **Stage 3 – combiners:** soft voting averages three members; stacking fits a logistic "
        "meta-learner on the OOF log-odds; BMA computes posterior-style weights from the OOF log predictive density. "
        "**Stage 4 – calibration of decisions:** for each of the nine reported methods, the smoothing width and F1 "
        "threshold are chosen on the OOF predictions. **Stage 5 – evaluation:** the held-out video is scored and the "
        "results of all folds are compared with PBL-1 and PBL-2 models using paired, video-level bootstrap intervals. "
        "The implementation is `analysis/ensemble.py`, run by `scripts/run_ensemble.py`; settings are in the "
        "`ensemble` section of `config.yaml`."
    )
    r.h2("4.5 Algorithm Theory and Mathematical Formulation")
    r.h3("4.5.1 CART decision tree")
    r.equation("G(S) = 1 − Σ_c p_c(S)²,      Δ = G(S) − (|S_L|/|S|)·G(S_L) − (|S_R|/|S|)·G(S_R)",
               "CART grows a binary tree greedily, choosing at each node the feature and threshold that maximize the "
               "decrease Δ in Gini impurity G. A leaf predicts the fraction of relevant training rows it contains. "
               f"Depth was selected from {{4, 6, 8, 12}} by inner-CV PR-AUC (chosen: {counts(depths, 'depth')}); leaves hold at least 20 rows.")
    r.h3("4.5.2 Bagging, random forest and extra trees")
    r.equation("p̂(x) = (1/B) Σ_{b=1}^{B} T_b(x),      Var[p̂] = ρσ² + ((1 − ρ)/B)σ²",
               "Each tree T_b is trained on a bootstrap sample. Averaging B trees with pairwise correlation ρ and "
               "variance σ² removes the second variance term as B grows; random forests lower ρ further by drawing "
               "√p = 5 candidate features at each split, and extra trees also draw split thresholds at random. "
               "B = 300 trees with minimum leaf size 5.")
    r.h3("4.5.3 AdaBoost (SAMME)")
    r.equation("ε_m = Σᵢ wᵢ 1[h_m(xᵢ) ≠ cᵢ] / Σᵢ wᵢ,    α_m = η · ln((1 − ε_m)/ε_m),    wᵢ ← wᵢ · exp(α_m 1[h_m(xᵢ) ≠ cᵢ])",
               "Each round fits a depth-2 tree h_m to the re-weighted data, so later trees focus on misclassified "
               "seconds; the ensemble votes with weights α_m. 300 rounds, learning rate η = 0.5 (for two classes "
               "SAMME reduces to discrete AdaBoost).")
    r.h3("4.5.4 Gradient boosting")
    r.equation("F_m(x) = F_{m−1}(x) + ν · h_m(x),    h_m ≈ argmin_h Σᵢ (gᵢ − h(xᵢ))²,   gᵢ = cᵢ − σ(F_{m−1}(xᵢ))",
               "Gradient boosting performs functional gradient descent on the log-loss: every new tree fits the "
               "negative gradient (residual) of the current model. The histogram variant bins features into at most "
               "255 levels for speed. 300 iterations, ν = 0.05, at most 31 leaves, L2 leaf regularization 1.0.")
    r.h3("4.5.5 Soft voting and stacking")
    r.equation("p_vote(x) = (1/M) Σ_m p_m(x),      p_stack(x) = σ(γ₀ + Σ_m γ_m · logit p_m(x))",
               "Soft voting gives equal weight to logistic regression, random forest and gradient boosting. Stacking "
               "learns the weights γ_m with a logistic meta-learner trained on out-of-fold predictions of all eight "
               "base learners, so it can down-weight redundant or poorly calibrated members.")
    r.h3("4.5.6 Bayesian model averaging (pseudo-BMA+)")
    r.equation("p(c | x, D) = Σ_m p(c | x, M_m) · p(M_m | D)",
               "BMA averages model predictions weighted by posterior model probabilities. The exact marginal "
               "likelihood is intractable for trees, so the pseudo-BMA+ approximation of Yao et al. [19] is used:")
    r.equation("w_m = E_{α∼Dir(1)} [ softmax_m( G · Σ_g α_g ℓ_{mg} ) ]",
               "ℓ_{mg} is the out-of-fold log predictive density of model m summed over the rows of training query g "
               f"(G ≈ {len(R['queries']) * 8 // 9} queries per fold), and the Bayesian bootstrap (Dirichlet weights α, "
               "1,000 draws) accounts for uncertainty in those scores so that near-equal models share weight.")
    r.h3("4.5.7 Model selection and settings")
    rows = [
        ["CART tree", "max_depth ∈ {4, 6, 8, 12} (inner CV); min leaf 20", "Grouped inner CV (PR-AUC)"],
        ["Bagging", "300 trees, bootstrap, unpruned, min leaf 5", "Fixed"],
        ["Random forest", "300 trees, max_features = √p, min leaf 5", "Fixed"],
        ["Extra trees", "300 trees, random thresholds, max_features = √p, min leaf 5", "Fixed"],
        ["AdaBoost", "300 rounds, depth-2 trees, η = 0.5, SAMME", "Fixed"],
        ["Hist. gradient boosting", "300 iterations, ν = 0.05, 31 leaves, λ = 1.0", "Fixed"],
        ["Soft voting", "Logistic + random forest + gradient boosting", "Fixed"],
        ["Stacking", "Logistic meta-learner on 8 OOF log-odds", "Inner OOF"],
        ["Bayesian model averaging", "Pseudo-BMA+ over 8 members, 1,000 bootstrap draws", "Inner OOF"],
        ["All methods", "Output smoothing 1–15 s and F1 threshold", "Inner OOF"],
    ]
    r.table_block("Ensemble configurations and how their settings were selected", ["Method", "Configuration", "Selection"],
                  rows, widths=[1.7, 3.0, 1.4],
                  interpretation="Ensemble sizes were fixed at 300 because the size curves (Figure 4.4) are flat beyond "
                                 "about 50–100 members, so tuning them would add cost without benefit. The logistic and "
                                 "k-NN base learners use C = 0.1 and k = 151, the values chosen in every PBL-1 fold.")
    r.h2("4.6 Results and Performance Evaluation")
    cols = ["precision", "recall", "f1", "pr_auc", "roc_auc", "hit_at_1", "mrr", "temporal_iou"]
    rows = [[m] + [f3(e.loc[m, c]) for c in cols] for m in ENSEMBLES]
    order = np.argsort([-e.loc[m, "f1"] for m in ENSEMBLES])
    rows = [rows[i] for i in order]
    r.table_block("Ensemble results on held-out videos (mean over 9 videos; sorted by F1)",
                  ["Method", "Prec.", "Recall", "F1", "PR-AUC", "ROC-AUC", "Hit@1", "MRR", "IoU"], rows,
                  widths=[1.47, 0.53, 0.55, 0.53, 0.53, 0.55, 0.53, 0.53, 0.53], bold_rows=[0],
                  interpretation=f"{best} is the best ensemble on F1 ({e.loc[best, 'f1']:.3f}), PR-AUC "
                                 f"({e.loc[best, 'pr_auc']:.3f}) and temporal IoU ({e.loc[best, 'temporal_iou']:.3f}); soft "
                                 f"voting is a close second on PR-AUC ({e.loc['Soft voting', 'pr_auc']:.3f}). Among single "
                                 "ensemble types, random forest and extra trees rank best (PR-AUC ≈ 0.84), and "
                                 f"histogram gradient boosting has the highest Hit@1 ({e.loc['Hist. gradient boosting', 'hit_at_1']:.3f}). "
                                 f"Notably the shallow CART tree matches the F1 of the forests ({e.loc['CART tree', 'f1']:.3f}) "
                                 f"after temporal smoothing, but its PR-AUC ({e.loc['CART tree', 'pr_auc']:.3f}) is clearly "
                                 "lower: a single tree makes good yes/no decisions but ranks seconds coarsely.")
    r.figure_block(P["en"] / "02_ensemble_size.png", "Held-out PR-AUC as a function of the number of ensemble members", 5.8,
                   "With one member, bagging, random forest and AdaBoost score PR-AUC ≈ 0.57–0.60 (a single unpruned "
                   "tree overfits). Averaging 10 trees adds about 0.2, and the curves flatten after 50–100 members "
                   "(random forest 0.84 at 300). Gradient boosting starts higher (0.77 after one round, because its "
                   "first tree already uses the log-odds prior) and saturates sooner. The wide bands show that "
                   "differences between videos are much larger than differences between ensemble sizes.")
    r.figure_block(P["en"] / "03_cart_depth.png", "CART depth selection: inner-CV PR-AUC per fold and selected depth", 5.8,
                   f"Inner-CV PR-AUC peaks at shallow depth and decreases for deeper trees, and depth 4 was chosen in "
                   f"most folds ({counts(depths, 'depth')}). Deep single trees overfit the training videos; this is the variance "
                   "problem that bagging and forests solve by averaging.")
    r.figure_block(P["en"] / "04_cart_tree.png", "Top three levels of a CART tree trained on eight videos", 6.3,
                   "The root split asks whether the second is within 20.5 s of the speech peak (12.5% of rows, 55% of "
                   "them relevant, versus 2% on the other side). The next splits use the ±5 s fused score (fused_near), "
                   "the local speech window and the speech z-score. The tree reads directly as rules, e.g. \"within 20 s "
                   "of the speech peak and fused_near > 0.61 ⇒ 89% relevant\", the interpretability advantage a single "
                   "tree has over the ensembles.")
    r.figure_block(P["en"] / "06_combiner_weights.png", "Stacking meta-learner coefficients and BMA weights across folds", 6.3,
                   f"BMA assigns most weight to logistic regression ({bma[members.index('logistic')]:.2f}), with "
                   f"random forest ({bma[members.index('random_forest')]:.2f}) and extra trees "
                   f"({bma[members.index('extra_trees')]:.2f}) taking the rest; the other members receive ≈ 0 because their "
                   "out-of-fold probabilities are less well calibrated (lower log score). Stacking spreads weight more widely: "
                   "logistic regression, extra trees and random forest dominate, while gradient boosting receives a small "
                   "negative weight because it is largely redundant given the others. Both combiners therefore choose "
                   "the same complementary pair: a linear model plus a randomized tree ensemble.")
    r.figure_block(P["en"] / "08_pr_curves.png", "Pooled precision–recall curves of the ensembles, logistic regression and fixed fusion", 5.6,
                   "All ensembles dominate the fixed fusion rule over the entire recall range. Combiners (voting, "
                   "stacking, BMA) and the forests overlap closely with logistic regression; the single CART tree "
                   "has a step-shaped curve and the lowest precision at high recall.")
    r.figure_block(P["en"] / "12_confusion.png", f"Pooled confusion matrices of {best} and logistic regression", 5.8,
                   "Both models recover about 80% of relevant seconds (1,904 versus 1,899 of 2,385); the ensemble "
                   "produces about 10% fewer false positives (690 versus 764), which explains its small precision and "
                   "F1 advantage.")
    r.figure_block(P["en"] / "13_example_timelines.png", f"Held-out timelines of {best} for the lowest-, median- and highest-IoU queries", 6.3,
                   "In the best case the decision covers the true interval almost exactly (IoU 0.97). In the median case "
                   "it finds the right segment but stops early. In the worst case the model confidently selects a "
                   "different segment in which an astronaut also talks about astronaut qualities, a plausible answer "
                   "that the single annotated interval does not credit, illustrating label ambiguity rather than a "
                   "random error.")
    r.h2("4.7 Comparison with Previous Models")
    r.para(
        "All models below were evaluated on exactly the same nine held-out videos, the same 28,900 rows and the same "
        "metric implementation, so the comparison is direct. Regression models (Unit 1) predict a different target "
        "and are therefore compared through ranking metrics in Chapter 5 only."
    )
    comp = [("PBL 3", m) for m in [best, "Soft voting", "Stacking", "Random forest", "Hist. gradient boosting", "CART tree"]]
    comp = list(dict.fromkeys(comp))
    comp += [("PBL 1", m) for m in ["Logistic regression", "Gradient boosting classifier", "k-nearest neighbours"]]
    comp += [("PBL 2", m) for m in ["K-means", "Gaussian HMM"]] + [("Baseline", "Fixed-weight fusion")]
    rows = [[u, m, f3(e.loc[m, "f1"]), f"{e.loc[m, 'f1_lo']:.3f}–{e.loc[m, 'f1_hi']:.3f}", f3(e.loc[m, "pr_auc"]),
             f3(e.loc[m, "hit_at_1"]), f3(e.loc[m, "temporal_iou"])] for u, m in comp]
    r.table_block("Comparison of Unit 5 ensembles with PBL-1, PBL-2 and baseline models (same folds and metrics)",
                  ["Unit", "Method", "F1", "F1 95% CI", "PR-AUC", "Hit@1", "IoU"], rows,
                  widths=[0.6, 2.0, 0.55, 1.0, 0.7, 0.6, 0.55], bold_rows=[0],
                  interpretation=f"The best ensemble improves on every earlier model: +{e.loc[best, 'f1'] - lr.f1:.3f} F1 and "
                                 f"+{e.loc[best, 'pr_auc'] - lr.pr_auc:.3f} PR-AUC over logistic regression, "
                                 f"+{e.loc[best, 'f1'] - e.loc['Gradient boosting classifier', 'f1']:.3f} F1 over the PBL-1 "
                                 f"gradient-boosting classifier, +{e.loc[best, 'f1'] - e.loc['K-means', 'f1']:.3f} over K-means and "
                                 f"+{e.loc[best, 'f1'] - e.loc['Fixed-weight fusion', 'f1']:.3f} over fixed fusion. The "
                                 "confidence intervals of the supervised models overlap heavily, so the unpaired intervals "
                                 "alone cannot separate them.")
    rows = []
    for m in ENSEMBLES:
        a, b = _paired(R, m, "f1"), _paired(R, m, "pr_auc")
        rows.append([m, f"{a.mean_diff:+.3f}", f"[{a.lo:+.3f}, {a.hi:+.3f}]", f"{int(a.videos_better)}/9",
                     f"{b.mean_diff:+.3f}", f"[{b.lo:+.3f}, {b.hi:+.3f}]"])
    r.table_block("Paired per-video differences from PBL-1 logistic regression (95% video-bootstrap CI)",
                  ["Ensemble", "ΔF1", "95% CI", "Videos better", "ΔPR-AUC", "95% CI"], rows,
                  widths=[1.5, 0.62, 1.12, 0.66, 0.73, 1.12],
                  interpretation=f"Pairing removes the large between-video variation. Only {best} has a 95% interval that "
                                 f"excludes zero for both F1 ([{dbf.lo:+.3f}, {dbf.hi:+.3f}], better on "
                                 f"{int(dbf.videos_better)} of 9 videos) and PR-AUC ([{dbp.lo:+.4f}, {dbp.hi:+.3f}]); the "
                                 "lower bounds are almost exactly zero and a percentile bootstrap over only nine videos is "
                                 "optimistic, so the improvement should be described as small and marginally significant. "
                                 "Every homogeneous tree ensemble is statistically indistinguishable from logistic regression.")
    r.figure_block(P["en"] / "09_paired_differences.png", "Paired differences of every ensemble from logistic regression", 6.3,
                   "Homogeneous ensembles (top) scatter around zero or below for PR-AUC and Hit@1; heterogeneous "
                   "combiners (bottom) shift to the right, and only BMA clears zero on F1 and PR-AUC. No ensemble "
                   "improves Hit@1, which is already near its ceiling (0.94).")
    r.figure_block(P["en"] / "01_comparison.png", "Held-out comparison of all models from PBL 1, PBL 2 and PBL 3", 6.3,
                   "Ensembles (green) and PBL-1 supervised models (blue) form one cluster at F1 ≈ 0.74–0.77, well above "
                   "the unsupervised PBL-2 models (orange, 0.55–0.63) and the hand-set baselines (grey, 0.33–0.53).")
    r.figure_block(P["en"] / "10_per_video_f1.png", "Per-video F1 of the best ensemble and earlier models", 5.8,
                   "Per-video F1 ranges from about 0.6 to 0.9 depending on the video, a much larger spread than the "
                   "differences between supervised models; all supervised models rise and fall together, and the "
                   "ensemble's advantage is a small, mostly consistent shift.")
    r.h2("4.8 Discussion, Findings and Limitations")
    rows = [[n, f"{cost.loc[n, 'fit_s']:.2f}", f"{cost.loc[n, 'predict_s'] * 1000:.1f}",
             f3(e.loc[n, "pr_auc"]) if n in e.index else f3(lr.pr_auc)]
            for n in ["CART tree", "Logistic regression (base)", "Hist. gradient boosting", "Extra trees", "Random forest",
                      "AdaBoost", "Bagging"]]
    r.table_block("Training and prediction cost per fold (one CPU core, ≈ 25,000 training rows)",
                  ["Learner", "Fit time (s)", "Predict time (ms)", "Held-out PR-AUC"], rows, widths=[2.2, 1.1, 1.3, 1.3],
                  interpretation="Histogram gradient boosting and extra trees reach forest-level quality in 1–3 s, while "
                                 "bagging of unpruned trees takes about 90 s for lower PR-AUC. Stacking and BMA also need "
                                 "the inner cross-validation (three extra fits of every member), so they cost roughly four "
                                 "times the sum of their members — about 10 minutes per fold for all eight learners.")
    r.figure_block(P["en"] / "11_cost_vs_quality.png", "Training cost versus held-out PR-AUC of the tree ensembles", 5.4,
                   "Quality does not increase with cost: the cheapest strong learners (gradient boosting, extra trees) "
                   "are as good as the most expensive (bagging, AdaBoost).")
    r.bullets([
        "**Performance:** ensembles of trees repair the weaknesses of a single tree (PR-AUC 0.80 → 0.84) but, on their "
        "own, do not beat a well-regularized logistic regression on the engineered features. The best results come "
        "from combining heterogeneous models, because a linear model and randomized trees make partly independent errors.",
        "**Generalization:** every number is a leave-one-video-out estimate with nested selection, so no tuning used the "
        "evaluated video. Variation between videos (F1 ≈ 0.6–0.9) dominates variation between models.",
        "**Interpretability:** a depth-4 CART tree gives readable rules at the same F1, while forests, boosting and "
        "combiners are opaque; impurity importance and combiner weights partially explain them.",
        "**Calibration matters for BMA:** an earlier run with class-balanced weights let the only unweighted learner "
        "(k-NN) dominate the BMA weights simply because its probabilities were calibrated; removing re-weighting from "
        "all learners fixed this, which is why the final design handles imbalance through thresholds.",
        "**Limitations:** nine videos give little statistical power; labels favour speech; pseudo-BMA+ approximates "
        "the true model posterior; ensemble sizes and boosting rates were fixed rather than tuned; and ensembles cost "
        "10–1000 times more compute than logistic regression for a gain of about one F1 point.",
    ])
    r.h2("4.9 Chapter Conclusion")
    r.para(
        f"Unit 5 contributed nine ensemble methods implemented under the same leakage-safe protocol as the rest of the "
        f"project. The best method, {best}, reached F1 {e.loc[best, 'f1']:.3f}, PR-AUC {e.loc[best, 'pr_auc']:.3f} and "
        f"temporal IoU {e.loc[best, 'temporal_iou']:.3f}, the highest values of any model in the project, with a small "
        "but consistent paired improvement over logistic regression. The study also shows why: diversity between a "
        "linear model and tree ensembles is what a combination can exploit, while homogeneous ensembles mainly fix the "
        "high variance of individual trees."
    )


# --------------------------------------------------------------------------- chapter 5
def chapter5(r: Report, R: dict) -> None:
    e = R["ens"].set_index("method")
    rt = R["retrieval"].set_index("method")
    reg = R["reg"].set_index("model")
    best = R["best"]
    r.chapter_heading("Integrated Discussion and Findings")
    r.para(
        "The five units attacked the same question — which seconds of a video answer a query — with progressively "
        "different learning paradigms. Because they share one dataset, one leave-one-video-out protocol and one "
        "metric implementation (for Units 2–5), their results can be compared directly. Unit 1 regression is "
        "compared through query ranking only, since it predicts a graded target."
    )
    rows = [
        ["Baseline", "Fixed-weight fusion (α = 0.6)", "None", f3(e.loc["Fixed-weight fusion", "f1"]),
         f3(e.loc["Fixed-weight fusion", "pr_auc"]), f3(e.loc["Fixed-weight fusion", "hit_at_1"])],
        ["PBL 1 / Unit 1", "Random forest regression", "Graded relevance", "—", "—",
         f3(rt.loc["Regression: Random forest", "hit_at_1"]) + "*"],
        ["PBL 1 / Unit 2", "Logistic regression", "Binary labels", f3(e.loc["Logistic regression", "f1"]),
         f3(e.loc["Logistic regression", "pr_auc"]), f3(e.loc["Logistic regression", "hit_at_1"])],
        ["PBL 2 / Unit 3", "K-means (k by silhouette)", "None (labels only to name clusters)", f3(e.loc["K-means", "f1"]),
         f3(e.loc["K-means", "pr_auc"]), f3(e.loc["K-means", "hit_at_1"])],
        ["PBL 2 / Unit 4", "Gaussian HMM", "None (labels only to name states)", f3(e.loc["Gaussian HMM", "f1"]),
         f3(e.loc["Gaussian HMM", "pr_auc"]), f3(e.loc["Gaussian HMM", "hit_at_1"])],
        ["PBL 3 / Unit 5", best, "Binary labels", f3(e.loc[best, "f1"]), f3(e.loc[best, "pr_auc"]), f3(e.loc[best, "hit_at_1"])],
    ]
    r.table_block("Best model of each unit on the shared benchmark (mean over 9 held-out videos)",
                  ["Unit", "Best model", "Supervision", "F1", "PR-AUC", "Hit@1"], rows,
                  widths=[1.1, 1.8, 1.6, 0.55, 0.65, 0.6],
                  interpretation="*Hit@1 for regression is pooled over 86 queries (Table 2.4) rather than averaged per video. "
                                 "Supervision is the dominant factor: every supervised model is about 0.25 F1 above the "
                                 "fixed rule and 0.13–0.22 above the unsupervised models, while the differences among "
                                 "supervised models, ensembles included, are about 0.01–0.03.")
    r.h2("5.1 Critical Findings")
    r.bullets([
        "**Features matter more than models.** Adding temporal context features raised logistic F1 from 0.64 to 0.75 "
        "(Table 2.5); switching among the best supervised models or ensembles changes F1 by at most about 0.02.",
        "**Learning beats the hand-set rule.** All learned supervised models clearly surpass fixed fusion on F1, PR-AUC "
        "and temporal IoU; for top-1 ranking the gain (≈ 0.06) is consistent but within the uncertainty of 86 queries.",
        "**Unsupervised structure is real but incomplete.** K-means and the HMM discover a relevant-rich cluster/state "
        "without labels and beat fixed fusion on F1, yet trail supervised models because their objectives "
        "(compactness, likelihood) are not relevance.",
        "**Temporal modelling helps decisions more than ranking.** Smoothing outputs and the HMM's persistent states "
        "give contiguous predictions, but saturated posteriors weaken ranking (HMM Hit@1 0.57).",
        "**Ensembles add a small, marginally significant gain.** Combining a linear model with randomized trees via "
        f"{best} gives the best scores of the project; homogeneous tree ensembles alone do not beat logistic regression.",
        f"**Explainability converges.** Every family — logistic coefficients, tree importances, cluster centres, HMM "
        "emission means — identifies speech strength relative to its peak and the distance to that peak as the key evidence.",
    ])
    r.h2("5.2 Study Limitations")
    r.bullets([
        "Only nine videos from one domain (NASA news) and 86 queries; between-video variation is large and statistical power is low.",
        "Answer intervals were written from transcripts, biasing labels toward speech and possibly crediting a single interval where several are valid (Figure 4.10).",
        "Pretrained encoders were kept frozen by design; errors of Whisper on names and jargon propagate to every model.",
        "Features use whole-video context, so all models are suitable for offline retrieval, not live streams.",
        "The MSR-VTT and Charades evaluation harness is implemented but was not run, so no external benchmark results are claimed.",
    ])


# --------------------------------------------------------------------------- chapter 6
def chapter6(r: Report, R: dict) -> None:
    e = R["ens"].set_index("method")
    reg = R["reg"].set_index("model")
    best = R["best"]
    r.chapter_heading("Overall Conclusion and Future Scope")
    r.h2("6.1 Conclusion")
    r.para(
        "This project built Needle, a working cross-modal video search application, and used its relevance signals to "
        "study all five units of the course on one real problem. With frozen CLIP, Whisper and MiniLM encoders and 31 "
        "engineered features, the project achieved: regression of graded relevance with R² "
        f"{reg.loc['random_forest', 'r2_mean']:.2f} (Unit 1); classification of relevant seconds with F1 "
        f"{e.loc['Logistic regression', 'f1']:.3f} and PR-AUC {e.loc['Logistic regression', 'pr_auc']:.3f} (Unit 2); "
        f"label-free clustering reaching F1 {e.loc['K-means', 'f1']:.3f} (Unit 3); a from-scratch Gaussian HMM with "
        f"Baum–Welch and Viterbi decoding reaching F1 {e.loc['Gaussian HMM', 'f1']:.3f} (Unit 4); and nine ensemble "
        f"methods, the best of which ({best}) reached F1 {e.loc[best, 'f1']:.3f} and PR-AUC {e.loc[best, 'pr_auc']:.3f} "
        "(Unit 5). Every learned supervised model improved substantially on the hand-designed fusion rule "
        f"(F1 {e.loc['Fixed-weight fusion', 'f1']:.3f}). All results are leave-one-video-out estimates with nested model "
        "selection, and the code, saved results, figures and 74 automated tests make them reproducible."
    )
    r.h2("6.2 Future Scope")
    r.bullets([
        "**Better data:** label more videos from other domains (lectures, sports, silent footage) and annotate answers by watching the picture, to reduce speech bias and increase statistical power.",
        "**Tuning:** tune ensemble sizes, boosting learning rates and stacking regularization within the nested protocol, and add calibrated probabilities (isotonic or Platt) before averaging.",
        "**Deployment:** integrate the best learned ranker into the web application as an optional scoring mode, with absolute relevance thresholds so that queries with no true answer return nothing.",
        "**Sequential models:** use supervised sequence models (e.g. conditional random fields or HMMs with relevance-aware states) that combine the HMM's temporal persistence with supervised accuracy.",
        "**Interpretability:** report SHAP explanations per returned clip so that users see whether picture or speech evidence produced each match.",
        "**External benchmarks:** run the implemented MSR-VTT and Charades evaluations, and test larger encoders (ViT-L/14, larger Whisper models).",
    ])


# --------------------------------------------------------------------------- references
def references(r: Report, R: dict) -> None:
    r.page_break()
    r.title("References", toc=True)
    refs = [
        "A. Radford et al., \"Learning transferable visual models from natural language supervision,\" in Proc. 38th Int. Conf. Machine Learning (ICML), PMLR vol. 139, pp. 8748–8763, 2021.",
        "C. Schuhmann et al., \"LAION-5B: An open large-scale dataset for training next generation image-text models,\" in Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks, vol. 35, 2022.",
        "A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, \"Robust speech recognition via large-scale weak supervision,\" in Proc. 40th Int. Conf. Machine Learning (ICML), PMLR vol. 202, pp. 28492–28518, 2023.",
        "N. Reimers and I. Gurevych, \"Sentence-BERT: Sentence embeddings using Siamese BERT-networks,\" in Proc. EMNLP-IJCNLP, pp. 3982–3992, 2019.",
        "W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, \"MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,\" in Advances in Neural Information Processing Systems, vol. 33, pp. 5776–5788, 2020.",
        "J. Johnson, M. Douze, and H. Jégou, \"Billion-scale similarity search with GPUs,\" IEEE Trans. Big Data, vol. 7, no. 3, pp. 535–547, 2021.",
        "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" J. Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
        "T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, 2nd ed. New York, NY, USA: Springer, 2009.",
        "K. P. Murphy, Machine Learning: A Probabilistic Perspective. Cambridge, MA, USA: MIT Press, 2012.",
        "L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, Classification and Regression Trees. Belmont, CA, USA: Wadsworth, 1984.",
        "L. Breiman, \"Bagging predictors,\" Machine Learning, vol. 24, no. 2, pp. 123–140, 1996.",
        "L. Breiman, \"Random forests,\" Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
        "P. Geurts, D. Ernst, and L. Wehenkel, \"Extremely randomized trees,\" Machine Learning, vol. 63, no. 1, pp. 3–42, 2006.",
        "Y. Freund and R. E. Schapire, \"A decision-theoretic generalization of on-line learning and an application to boosting,\" J. Computer and System Sciences, vol. 55, no. 1, pp. 119–139, 1997.",
        "J. Zhu, H. Zou, S. Rosset, and T. Hastie, \"Multi-class AdaBoost,\" Statistics and Its Interface, vol. 2, no. 3, pp. 349–360, 2009.",
        "J. H. Friedman, \"Greedy function approximation: A gradient boosting machine,\" Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001.",
        "G. Ke et al., \"LightGBM: A highly efficient gradient boosting decision tree,\" in Advances in Neural Information Processing Systems, vol. 30, 2017.",
        "D. H. Wolpert, \"Stacked generalization,\" Neural Networks, vol. 5, no. 2, pp. 241–259, 1992.",
        "Y. Yao, A. Vehtari, D. Simpson, and A. Gelman, \"Using stacking to average Bayesian predictive distributions,\" Bayesian Analysis, vol. 13, no. 3, pp. 917–1007, 2018.",
        "J. A. Hoeting, D. Madigan, A. E. Raftery, and C. T. Volinsky, \"Bayesian model averaging: A tutorial,\" Statistical Science, vol. 14, no. 4, pp. 382–417, 1999.",
        "S. Lloyd, \"Least squares quantization in PCM,\" IEEE Trans. Information Theory, vol. 28, no. 2, pp. 129–137, 1982.",
        "P. J. Rousseeuw, \"Silhouettes: A graphical aid to the interpretation and validation of cluster analysis,\" J. Computational and Applied Mathematics, vol. 20, pp. 53–65, 1987.",
        "G. Schwarz, \"Estimating the dimension of a model,\" Annals of Statistics, vol. 6, no. 2, pp. 461–464, 1978.",
        "A. P. Dempster, N. M. Laird, and D. B. Rubin, \"Maximum likelihood from incomplete data via the EM algorithm,\" J. Royal Statistical Society: Series B, vol. 39, no. 1, pp. 1–38, 1977.",
        "L. R. Rabiner, \"A tutorial on hidden Markov models and selected applications in speech recognition,\" Proc. IEEE, vol. 77, no. 2, pp. 257–286, 1989.",
        "A. J. Viterbi, \"Error bounds for convolutional codes and an asymptotically optimum decoding algorithm,\" IEEE Trans. Information Theory, vol. 13, no. 2, pp. 260–269, 1967.",
        "T. Saito and M. Rehmsmeier, \"The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,\" PLoS ONE, vol. 10, no. 3, e0118432, 2015.",
    ]
    for i, ref in enumerate(refs, start=1):
        p = r.doc.add_paragraph(style="Report Body")
        p.paragraph_format.left_indent = __import__("docx").shared.Inches(0.45)
        p.paragraph_format.first_line_indent = __import__("docx").shared.Inches(-0.45)
        p.add_run(f"[{i}]\t{ref}")


# --------------------------------------------------------------------------- appendix
def appendix(r: Report, R: dict) -> None:
    root = R["root"]
    r.page_break()
    r.title("Appendix A", toc=True, space_after=2)
    r.title("Coding", size=14)
    r.para(
        f"Complete source code, saved results and executed notebooks: {GITHUB}. "
        "Google Colab: open the notebooks `notebooks/analysis.ipynb` (Units 1–2), `notebooks/review_clustering_hmm.ipynb` "
        "(Units 3–4) and `notebooks/pbl3_ensemble.ipynb` (Unit 5) from the repository through Colab's \"GitHub\" tab "
        "(https://colab.research.google.com/github/hemish22/ml-course-project/blob/main/notebooks/pbl3_ensemble.ipynb, "
        "and likewise for the other notebooks). Each notebook reads the saved feature table, so no video download is needed."
    )
    r.para("Reproduction commands (from the project root):")
    r.code("pip install -r requirements-review.txt\n"
           "python scripts/run_analysis.py      # Units 1-2: regression and classification\n"
           "python scripts/run_review.py        # Units 3-4: clustering and Gaussian HMM\n"
           "python scripts/run_ensemble.py      # Unit 5: ensemble learning\n"
           "pytest                              # automated tests")
    sections = [
        ("Chapter 2", "Implementation Methodology and Algorithm Configuration",
         ["analysis/features.py", "analysis/models.py", "analysis/evaluate.py"]),
        ("Chapter 3", "Unit 3 – Proposed Workflow and Implementation Methodology", ["analysis/review.py"]),
        ("Chapter 3", "Unit 4 – Proposed Workflow and Implementation Methodology", ["analysis/hmm.py"]),
        ("Chapter 4", "Implementation Methodology and Hyperparameter Configuration",
         ["config.yaml", "analysis/ensemble.py"]),
    ]
    for label, title, files in sections:
        r.h2(f"{label}: {title}")
        for name in files:
            r.h3(f"File: {name}")
            text = (root / name).read_text()
            if name == "config.yaml":
                text = text[text.index("ml:"):]
            r.code(text)
