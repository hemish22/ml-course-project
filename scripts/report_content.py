"""Text, tables and figures of the integrated report. Numbers come from saved results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH

from docx_kit import Report

ENSEMBLES = ["CART tree", "Bagging", "Random forest", "Extra trees", "AdaBoost", "Hist. gradient boosting",
             "Soft voting", "Stacking", "Bayesian model averaging"]
PBL1_CLS = ["Logistic regression", "k-nearest neighbours", "Gradient boosting classifier"]
PBL2 = ["K-means", "Gaussian mixture", "Gaussian HMM"]
BASELINES = ["Fixed-weight fusion", "Speech only", "Picture only"]
GITHUB = "https://github.com/hemish22/ml-course-project"



def counts(c: dict, symbol: str) -> str:
    """{2: 7, 3: 2} -> 'k = 2 in 7 folds, k = 3 in 2 folds'."""
    return ", ".join(f"{symbol} = {k} in {v} fold{'s' if v != 1 else ''}" for k, v in sorted(c.items()))

def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# --------------------------------------------------------------------------- front matter
def front_matter(r: Report, R: dict) -> None:
    r.new_section("lowerRoman", start=5)
    r.title("Abstract", toc=True)
    r.para(abstract_text(R))
    r.page_break()
    r.title("Table of Contents")
    r.toc('TOC \\h \\z \\t "Chapter Heading,1,Front Heading,1,Section Heading,2"', "Right-click and choose Update Field to build the table of contents.")
    r.page_break()
    r.title("List of Figures", toc=True)
    r.toc('TOC \\h \\z \\t "Figure Caption,1"', "Right-click and choose Update Field to build the list of figures.")
    r.page_break()
    r.title("List of Tables", toc=True)
    r.toc('TOC \\h \\z \\t "Table Caption,1"', "Right-click and choose Update Field to build the list of tables.")
    r.page_break()
    r.title("Abbreviations", toc=True)
    abbreviations = [
        ("AdaBoost", "Adaptive Boosting"), ("AP", "Average Precision (area under the precision–recall curve)"),
        ("AUC", "Area Under the Curve"), ("BIC", "Bayesian Information Criterion"),
        ("BMA", "Bayesian Model Averaging"), ("CART", "Classification and Regression Trees"),
        ("CLIP", "Contrastive Language–Image Pre-training"), ("CV", "Cross-Validation"),
        ("ELPD", "Expected Log Predictive Density"), ("EM", "Expectation–Maximization"),
        ("EDA", "Exploratory Data Analysis"), ("FAISS", "Facebook AI Similarity Search"),
        ("GMM", "Gaussian Mixture Model"), ("HGB", "Histogram-based Gradient Boosting"),
        ("HMM", "Hidden Markov Model"), ("IoU", "Intersection over Union"),
        ("k-NN", "k-Nearest Neighbours"), ("LOVO", "Leave-One-Video-Out"),
        ("MAE", "Mean Absolute Error"), ("MRR", "Mean Reciprocal Rank"),
        ("OOF", "Out-of-Fold (predictions)"), ("PCA", "Principal Component Analysis"),
        ("PR", "Precision–Recall"), ("R²", "Coefficient of Determination"),
        ("RF", "Random Forest"), ("RMSE", "Root Mean Squared Error"),
        ("ROC", "Receiver Operating Characteristic"),
        ("SAMME", "Stagewise Additive Modelling using a Multi-class Exponential loss"),
        ("ViT", "Vision Transformer"),
    ]
    r.table_block("Abbreviations used in this report", ["Abbreviation", "Expansion"],
                  abbreviations, widths=[1.4, 4.6])
    r.table -= 1  # the abbreviation list is not a numbered results table
    _drop_last_caption_numbering(r)


def _drop_last_caption_numbering(r: Report) -> None:
    """Turn the most recent 'Table x.y:' caption into a plain, unlisted heading line."""
    body = r.doc.element.body
    for p in reversed(r.doc.paragraphs):
        if p.style.name == "Table Caption":
            p.style = r.doc.styles["Report Body"]
            for run in p.runs:
                run.text = ""
            p.runs[0].text = ""
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            break
    del body


def abstract_text(R: dict) -> str:
    e = R["ens"].set_index("method")
    best = R["best"]
    rev = R["review"].set_index("method")
    reg = R["reg"].set_index("model")
    cls = R["cls"].set_index("model")
    return (
        "Finding the moment in a long video that answers a natural-language query is slow when done manually. "
        "This project, Needle, builds a cross-modal video search system and applies five families of "
        "machine-learning techniques to one problem: deciding, for every second of a video, whether it answers a "
        "query. Frozen encoders (CLIP for frames, Whisper and MiniLM for speech) yield signals from which 31 "
        "features were engineered for a labelled benchmark "
        f"of {len(R['dataset']):,} query–second observations covering 86 queries over nine public-domain NASA "
        f"videos, of which {pct(R['dataset'].rel_binary.mean())} are relevant. All models were evaluated by "
        "leave-one-video-out validation. In PBL 1, regression models (ridge, random forest, gradient boosting) "
        f"predicted graded relevance with R² up to {reg.loc['random_forest', 'r2_mean']:.2f}, and classifiers "
        f"(logistic regression, k-nearest neighbours, gradient boosting) reached F1 up to {cls.loc['logistic', 'f1_mean']:.2f}. "
        "In PBL 2, K-means, Gaussian mixture clustering and a from-scratch Gaussian hidden Markov model "
        "(Baum–Welch, Viterbi) on per-query sequences found relevance structure without labels "
        f"(F1 {rev.loc['K-means', 'f1']:.2f}, {rev.loc['Gaussian mixture', 'f1']:.2f} and "
        f"{rev.loc['Gaussian HMM', 'f1']:.2f}). In PBL 3, CART, bagging, random forests, extra trees, AdaBoost, "
        "gradient boosting, soft voting, stacking and Bayesian model averaging were compared under nested "
        f"validation; the best ensemble, {best}, achieved F1 {e.loc[best, 'f1']:.3f} and PR-AUC "
        f"{e.loc[best, 'pr_auc']:.3f}, against {rev.loc['Logistic regression', 'f1']:.3f} and "
        f"{rev.loc['Logistic regression', 'pr_auc']:.3f} for logistic regression. " + R["abstract_conclusion"]
    )


# --------------------------------------------------------------------------- chapter 1
def chapter1(r: Report, R: dict) -> None:
    r.new_section("decimal", start=1)
    r.chapter_heading("Introduction", first=True)
    r.h2("1.1 Background, Motivation and Scope")
    r.para(
        "Video has become a primary medium for lectures, news, meetings and scientific outreach, yet finding a "
        "particular moment inside a long recording still requires scrubbing through the timeline. Text search "
        "over titles or descriptions cannot answer questions such as \"where does the rover arm appear?\" or "
        "\"when do they explain how to become an astronaut?\". Answering them requires matching the query against "
        "what is shown (the frames) and what is said (the speech) at every point in time."
    )
    r.para(
        "Needle is a cross-modal video moment retrieval system built for this purpose. A video is sampled at one "
        "frame per second and embedded with the OpenCLIP ViT-B/32 image encoder; its audio is transcribed by "
        "Whisper and each transcript segment is embedded with the all-MiniLM-L6-v2 sentence encoder. A query is "
        "embedded by the matching text encoders, compared with both indexes using exact FAISS inner-product search, "
        "and the two similarity timelines are calibrated, fused and grouped into timestamped clips. A FastAPI "
        "backend and a Next.js web interface let a user upload a video, search it and play the returned moments."
    )
    r.para(
        "The stakeholders are students and educators searching lecture recordings, journalists and archivists "
        "searching news footage, and any user of long-form video. The scope of the machine-learning work is "
        "deliberately bounded: the pretrained encoders are never fine-tuned. All learning in this project happens "
        "on top of their outputs, using features engineered from the per-second similarity signals. This keeps "
        "the experiments affordable on a laptop CPU and makes every unit's models directly comparable on one task."
    )
    r.h2("1.2 Integrated Problem Statement")
    r.para(
        "**Problem.** Given a video and a natural-language query, decide for every second of the video how relevant "
        "it is to the query, so that the relevant interval can be located and ranked first. The hand-designed "
        "baseline fuses calibrated picture and speech similarity with a fixed weight (α = 0.6). The project asks "
        "whether learned models can replace this fixed rule, and which learning paradigm does so best."
    )
    r.para(
        "All five units use the same original dataset, the labelled query–second relevance table described in "
        "Chapter 2, so their results are directly comparable. Unit 1 frames the task as regression of a graded "
        "relevance score; Unit 2 as binary classification of relevant seconds; Unit 3 asks whether unsupervised "
        "clustering of the feature space recovers relevant seconds; Unit 4 treats each (video, query) pair as a time "
        "series and models it with a hidden Markov model, so no separate sequential dataset was required; Unit 5 "
        "combines many learners into ensembles to improve on the single models."
    )
    r.h2("1.3 Project Objectives")
    r.bullets([
        "**Objective 1 (Unit 1 – regression):** engineer relevance features from the frozen encoders and fit ridge, "
        "random-forest and gradient-boosting regressors to predict graded relevance; target R² well above the mean "
        "baseline on unseen videos.",
        "**Objective 2 (Unit 2 – classification):** classify each second as inside or outside the answer interval "
        "with logistic regression, k-nearest neighbours and gradient boosting; report precision, recall, F1, ROC-AUC "
        "and PR-AUC and test whether learned rankers beat hand-set fusion.",
        "**Objective 3 (Unit 3 – clustering):** apply K-means and Gaussian mixture models without labels, select the "
        "number of clusters by silhouette and BIC, and measure how well clusters align with relevance.",
        "**Objective 4 (Unit 4 – sequential modelling):** implement a Gaussian hidden Markov model from first "
        "principles (Baum–Welch, forward–backward, Viterbi) on per-query time sequences and assess temporal "
        "relevance decoding.",
        "**Objective 5 (Unit 5 – ensemble learning):** implement CART, bagging, random forest, extra trees, AdaBoost, "
        "gradient boosting, soft voting, stacking and Bayesian model averaging, compare them with all earlier models "
        "under identical folds and metrics, and quantify whether ensembling gives a statistically supported gain.",
    ])
    r.h2("1.4 Unit-wise Topic and Dataset Allocation")
    r.table_block("Unit-wise topic and dataset allocation", ["PBL", "Unit", "Allotted topic / algorithm", "Dataset"], [
        ["PBL 1", "1", "Regression: ridge, random forest, gradient boosting", "Original (query–second relevance table)"],
        ["PBL 1", "2", "Classification: logistic regression, k-NN, gradient boosting", "Original"],
        ["PBL 2", "3", "Clustering: K-means, Gaussian mixture model", "Original"],
        ["PBL 2", "4", "Sequential modelling: Gaussian hidden Markov model", "Original, ordered as 86 per-query time sequences"],
        ["PBL 3", "5", "Ensemble learning: CART, bagging, random forest, extra trees, AdaBoost, gradient boosting, voting, stacking, BMA",
         "Original"],
    ], widths=[0.7, 0.5, 3.0, 1.8],
        interpretation="One dataset serves every unit. The Unit 4 problem uses the same rows, ordered in time "
                       "within each (video, query) pair, which is a natural sequence because relevance persists "
                       "over consecutive seconds.")


# --------------------------------------------------------------------------- chapter 2
def chapter2(r: Report, R: dict) -> None:
    d, P = R["dataset"], R["paths"]
    reg, cls = R["reg"].set_index("model"), R["cls"].set_index("model")
    r.chapter_heading("Data Analysis and Fundamental Machine Learning Techniques – (Units 1 and 2)")
    r.h2("2.1 Overview of Allotted Topics")
    r.para(
        "Unit 1 covers supervised regression and Unit 2 supervised classification. Both are natural fits: the "
        "benchmark gives, for every second, a graded relevance target (for regression) and a binary in/out label "
        "(for classification). Three models of increasing flexibility were used for each task. Ridge and logistic "
        "regression are linear models with L2 regularization and serve as interpretable references. k-nearest "
        "neighbours is a non-parametric, instance-based classifier. Random forests and gradient boosting capture "
        "non-linear interactions, for example that a high speech similarity matters more when the second is close "
        "to the speech peak. Each model was compared with a trivial baseline (mean or class prior) and with the "
        "hand-set fusion rule used by the application."
    )
    r.h2("2.2 Dataset Description")
    lengths = d.groupby(["video_id", "query_id"]).size()
    r.para(
        "**Name and source.** Needle relevance benchmark, built by this project from nine public-domain NASA "
        "video episodes (NASA media is not copyrighted in the United States) downloaded with "
        "`scripts/fetch_benchmark.py`. The annotations are in `benchmark/queries.json` and the feature table in "
        f"`results/ml/dataset.csv.gz`. Code and data: {GITHUB}."
    )
    r.para(
        f"**Size.** {len(d):,} rows, one per (query, second); {d.video_id.nunique()} videos; {d.query_id.nunique()} "
        f"hand-written queries, each with the time interval of its true answer; sequences of {lengths.min()}–"
        f"{lengths.max()} seconds (median {int(lengths.median())}). **Targets:** `rel_binary` (1 inside the "
        "answer interval) and `rel_graded` (1 inside, decaying as exp(−distance/5 s) outside). "
        f"{int(d.rel_binary.sum()):,} rows ({pct(d.rel_binary.mean())}) are relevant. **Data types:** all 31 features are "
        "numeric (continuous similarities, ranks, distances, counts and one binary speech indicator). **Missing "
        f"values:** none ({int(d.isna().sum().sum())} missing cells); seconds in silence receive a speech score of 0 by design."
    )
    r.para(
        "**Limitations.** Answer intervals were written from transcripts, so labels favour speech evidence; all "
        "videos come from one domain (NASA news); and 86 queries over nine videos give only nine independent "
        "evaluation groups, so confidence intervals are wide."
    )
    r.table_block("Engineered feature groups (31 features)", ["Group", "Features", "Meaning"], [
        ["Raw and calibrated similarity", "visual_raw, speech_raw, visual_norm, speech_norm, fused_baseline",
         "Cosine similarity of the query with the frame / transcript; per-query min–max calibration; hand-set fusion (α = 0.6)"],
        ["Within-video standardization", "visual_z, speech_z, visual_rank, speech_rank",
         "z-score and percentile rank of the similarity within the video"],
        ["Temporal context", "visual_smooth, speech_smooth, *_near (±5 s), *_wide (±10 s), fused_smooth, fused_near",
         "Moving averages around the second"],
        ["Peak structure", "*_peak_dist, *_vs_peak, local_rank_fused",
         "Distance (s) to the strongest peak and score relative to it; local rank"],
        ["Content and structure", "frame_change, scene_cuts_near, in_speech, rel_pos, query_words",
         "Visual change, nearby scene cuts, speech presence, position in video, query length"],
        ["Lexical overlap", "keyword_overlap, keyword_overlap_near", "Shared content words between query and spoken text"],
    ], widths=[1.5, 2.3, 2.2])
    r.h2("2.3 Data Preprocessing")
    r.para(
        "Preprocessing converts raw videos into the feature table: (1) ffmpeg extracts one frame per second "
        "(336 px wide) and a 16 kHz mono audio track; (2) frames are encoded by CLIP ViT-B/32 and transcript "
        "segments by MiniLM, and every embedding is L2-normalized so that inner product equals cosine similarity; "
        "(3) for each query, picture and speech similarities are spread onto a per-second timeline, speech scores "
        "being assigned to the seconds their segment covers; (4) the 31 features are computed per query; "
        "(5) targets are attached from the annotated intervals."
    )
    r.para(
        "No rows required cleaning or imputation. Scaling is part of each model pipeline (StandardScaler for ridge, "
        "logistic regression and k-NN; none for tree models) and is fitted on training videos only. The split is "
        "leave-one-video-out: each of the nine folds trains on eight videos and tests on the unseen ninth, so no "
        "second of a test video, and no query, is ever seen during training. Hyper-parameters were tuned by "
        "3-fold GroupKFold grid search inside the training videos, using PR-AUC (classification) or RMSE "
        "(regression). Two further choices were also made on training videos only: the width of a moving-average "
        "smoothing applied to model outputs (1–15 s) and the classification threshold maximizing F1."
    )
    r.h2("2.4 Exploratory Data Analysis (EDA)")
    r.figure_block(P["ml"] / "01_class_balance.png", "Class balance of the binary relevance label", 5.6,
                   f"Only {pct(d.rel_binary.mean())} of seconds are relevant. A classifier that always answers "
                   f"\"not relevant\" is {pct(1 - d.rel_binary.mean())} accurate, so accuracy is not used as the main "
                   "metric; F1, PR-AUC and balanced accuracy are reported instead and thresholds are tuned for F1.")
    r.figure_block(P["ml"] / "02_feature_distributions.png", "Distributions of key features for relevant and background seconds", 6.0,
                   "Relevant seconds have clearly higher calibrated speech similarity and smaller distance to the "
                   "speech peak, while picture similarity overlaps more between the two classes. This anticipates "
                   "the finding that speech-derived and context features carry most of the signal.")
    corr = d.select_dtypes("number").drop(columns=["query_id", "t"]).corr()["rel_binary"].drop(["rel_binary", "rel_graded"])
    top = corr.abs().sort_values(ascending=False).index[:3]
    r.figure_block(P["ml"] / "03_correlation_heatmap.png", "Correlation heatmap of features and targets", 6.0,
                   "Many features are strongly correlated with each other (raw, calibrated and smoothed versions of "
                   "the same signal), which motivates regularized linear models and tree ensembles. The features most "
                   f"correlated with the label are {', '.join(top)} (|r| up to {corr.abs().max():.2f}).")
    r.figure_block(P["ml"] / "04_example_traces.png", "Picture and speech similarity traces for example queries", 6.0,
                   "Relevant intervals (shaded) usually coincide with a broad rise in speech similarity, whereas "
                   "picture similarity is noisier and peaks at isolated frames. Temporal context features "
                   "(windows, distance to peak) were introduced to capture this shape.")
    r.h2("2.5 Proposed System Architecture / Workflow")
    r.figure_block(P["rp"] / "workflow_pbl1.png", "PBL-1 workflow from raw video to evaluated regression and classification models", 6.3)
    r.para(
        "The workflow has five stages. **Input:** nine videos and 86 labelled queries. **Representation:** frames "
        "and audio are extracted and encoded by frozen CLIP, Whisper and MiniLM models. **Feature engineering:** "
        "per-second similarity timelines are converted into 31 features and two targets. **Learning:** within each "
        "leave-one-video-out fold, three regressors and three classifiers are tuned by grouped inner cross-validation, "
        "and output smoothing and thresholds are selected. **Evaluation and outputs:** predictions on held-out "
        "videos are scored with regression, classification and retrieval metrics and compared with the baselines."
    )
    r.h2("2.6 Algorithm Theory and Mathematical Formulation")
    r.para("Let xᵢ ∈ ℝ³¹ be the feature vector of row i, yᵢ ∈ [0, 1] its graded target and cᵢ ∈ {0, 1} its label.")
    r.h3("2.6.1 Ridge regression")
    r.equation("β̂ = argmin_β Σᵢ (yᵢ − β₀ − xᵢᵀβ)² + λ‖β‖²₂",
               "The L2 penalty λ (grid 0.01–100) shrinks coefficients of the many correlated features.")
    r.h3("2.6.2 Logistic regression")
    r.equation("P(cᵢ = 1 | xᵢ) = σ(β₀ + xᵢᵀβ) = 1 / (1 + e^−(β₀ + xᵢᵀβ))",
               "Parameters minimize the class-balanced log-loss plus (1/C)‖β‖²; C was tuned over {0.1, 1, 10}.")
    r.h3("2.6.3 k-nearest neighbours")
    r.equation("P(c = 1 | x) = (1/k) Σ_{j ∈ N_k(x)} c_j",
               "N_k(x) are the k closest standardized training rows (k ∈ {15, 51, 151}).")
    r.h3("2.6.4 Random forest and gradient boosting")
    r.equation("f_RF(x) = (1/B) Σ_{b=1}^{B} T_b(x),        F_m(x) = F_{m−1}(x) + ν·h_m(x)",
               "A random forest averages B decorrelated regression trees; gradient boosting adds trees h_m fitted "
               "to the negative gradient of the loss with learning rate ν. Both are revisited in Chapter 4.")
    r.h3("2.6.5 Evaluation metrics")
    r.equation("RMSE = √((1/n) Σ (yᵢ − ŷᵢ)²),    R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²")
    r.equation("Precision = TP/(TP+FP),  Recall = TP/(TP+FN),  F1 = 2·P·R/(P+R)")
    r.equation("Hit@k = 1[∃ relevant second in top k],    MRR = mean(1 / rank of first relevant second)",
               "PR-AUC is the average precision over the ranked seconds; Hit@k and MRR are computed per query.")
    r.h2("2.7 Results and Performance Evaluation")
    names = {"dummy": "Mean baseline", "ridge": "Ridge regression", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting"}
    r.table_block("Regression results (mean ± std over 9 held-out videos)", ["Model", "RMSE", "MAE", "R²", "Spearman ρ"], [
        [names[m], f"{reg.loc[m, 'rmse_mean']:.3f} ± {reg.loc[m, 'rmse_std']:.3f}", f3(reg.loc[m, "mae_mean"]),
         f"{reg.loc[m, 'r2_mean']:.3f} ± {reg.loc[m, 'r2_std']:.3f}", f3(reg.loc[m, "spearman_mean"])]
        for m in ["dummy", "ridge", "random_forest", "gradient_boosting"]
    ], widths=[1.7, 1.3, 0.9, 1.3, 1.0],
        interpretation=f"Both tree ensembles reach R² ≈ {reg.loc['random_forest', 'r2_mean']:.2f}, explaining about "
                       "three quarters of the variance in graded relevance on unseen videos, against −0.03 for the "
                       "mean baseline. Ridge is clearly weaker (R² "
                       f"{reg.loc['ridge', 'r2_mean']:.2f}), showing that the relation is non-linear.")
    r.figure_block(P["ml"] / "06_reg_pred_vs_actual.png", "Predicted versus actual graded relevance (held-out videos)", 6.0,
                   "Points for the tree models concentrate along the diagonal at both ends of the scale; ridge "
                   "compresses predictions toward the middle, the typical bias of a linear fit to a bounded, "
                   "non-linear target.")
    cnames = {"dummy": "Prior baseline", "logistic": "Logistic regression", "knn": "k-nearest neighbours", "gradient_boosting": "Gradient boosting"}
    r.table_block("Classification results (mean over 9 held-out videos)",
                  ["Model", "Precision", "Recall", "F1", "Bal. acc.", "ROC-AUC", "PR-AUC"], [
        [cnames[m], f3(cls.loc[m, "precision_mean"]), f3(cls.loc[m, "recall_mean"]), f3(cls.loc[m, "f1_mean"]),
         f3(cls.loc[m, "balanced_accuracy_mean"]), f3(cls.loc[m, "roc_auc_mean"]), f3(cls.loc[m, "pr_auc_mean"])]
        for m in ["dummy", "logistic", "knn", "gradient_boosting"]
    ], widths=[1.7, 0.8, 0.7, 0.7, 0.8, 0.8, 0.8],
        interpretation=f"Logistic regression is the best classifier (F1 {cls.loc['logistic', 'f1_mean']:.3f}, PR-AUC "
                       f"{cls.loc['logistic', 'pr_auc_mean']:.3f}), narrowly ahead of gradient boosting and k-NN. "
                       "With rich context features, a linear decision boundary in log-odds is already adequate for "
                       "the binary task. All models greatly exceed the prior baseline (PR-AUC ≈ 0.10).")
    r.figure_block(P["ml"] / "10_cls_confusion.png", "Pooled confusion matrices of the three classifiers", 6.0,
                   "About four in five relevant seconds are recovered by every model, at the cost of some false "
                   "positives adjacent to true intervals; most errors are boundary errors of a few seconds rather "
                   "than detections in the wrong part of the video.")
    r.figure_block(P["ml"] / "12_cls_pr.png", "Precision–recall curves of the classifiers", 5.6,
                   "The curves lie far above the prevalence line; precision stays above 0.8 up to roughly 0.7 recall "
                   "for logistic regression, which is the operating region relevant to returning a short list of clips.")
    rt = R["retrieval"].set_index("method")
    rows = []
    for label, key in [("Random", "Random"), ("Picture only", "Picture only"), ("Speech only", "Speech only"),
                       ("Hand-set fusion (α = 0.6)", "Fixed-weight fusion"), ("Ridge regression", "Regression: Ridge regression"),
                       ("Random forest regression", "Regression: Random forest"),
                       ("Logistic regression", "Classification: Logistic regression"),
                       ("Gradient boosting classifier", "Classification: Gradient boosting")]:
        row = rt.loc[key]
        rows.append([label, f"{row.hit_at_1:.3f} ({row.hit_at_1_lo:.2f}–{row.hit_at_1_hi:.2f})", f3(row.hit_at_5), f3(row.mrr)])
    r.table_block("Ranking quality over 86 queries (95% bootstrap interval for Hit@1)", ["Method", "Hit@1", "Hit@5", "MRR"],
                  rows, widths=[2.4, 1.8, 0.9, 0.9],
                  interpretation="Learned rankers put a true second first for 91–93% of queries versus 87% for hand-set "
                                 "fusion and 60% for picture-only search. The direction is consistent, but the "
                                 "intervals overlap, so the improvement in Hit@1 is not statistically established on 86 queries.")
    ab = R["ablation"]
    lg = ab[(ab.model == "logistic")].set_index("variant")
    rf = ab[(ab.model == "random_forest")].set_index("variant")
    r.table_block("Ablation of feature engineering and output smoothing (pooled, held-out)",
                  ["Variant", "RF regression R²", "Logistic F1", "Logistic PR-AUC", "Logistic Hit@1"], [
        [v, f3(rf.loc[v, "r2"]), f3(lg.loc[v, "f1"]), f3(lg.loc[v, "pr_auc"]), f3(lg.loc[v, "hit_at_1"])]
        for v in ["Original 15 features", "+ context features", "+ output smoothing"]
    ], widths=[2.0, 1.1, 1.0, 1.1, 1.0],
        interpretation="Adding the 16 context features nearly doubles R² (0.39 → 0.70) and raises F1 from 0.64 to "
                       "0.75; smoothing the outputs adds a small further gain. Feature engineering mattered more than the choice of model.")
    r.figure_block(P["ml"] / "14_cls_importance.png", "Permutation importance of features for the classifiers", 6.0,
                   "Speech-derived context dominates: the within-video speech z-score and wide speech window for "
                   "logistic regression, and distance to the speech peak for gradient boosting. Raw similarities "
                   "matter little once their context is known.")
    r.figure_block(P["ml"] / "09_learning_curves.png", "Learning curves: held-out quality versus number of training videos", 6.0,
                   "Classification PR-AUC rises from 0.79 with one training video to 0.85 with six for logistic "
                   "regression, with diminishing returns, suggesting that more labelled videos would still help.")
    r.h2("2.8 Discussion and Findings")
    r.bullets([
        "All six models clearly beat their baselines; context features (windows, distance to peak, z-scores) explain most of the gain.",
        "For regression the non-linear tree ensembles are best (R² ≈ 0.74); for classification the linear logistic "
        "model is as good as the non-linear ones, so the binary boundary is close to linear in the engineered features.",
        "Learned rankers improve Hit@1 over hand-set fusion by 4–6 points, but not significantly with 86 queries.",
        "Speech-only search almost matches fusion partly because labels were derived from transcripts (label bias).",
        "Challenges were class imbalance (handled by F1-tuned thresholds and PR-AUC), strongly correlated features, "
        "and the risk of leakage between seconds of one video, which leave-one-video-out validation prevents.",
    ])
    r.h2("2.9 Chapter Conclusion")
    r.para(
        f"PBL 1 established a reliable baseline for the rest of the project: a shared 31-feature dataset, a "
        f"leakage-safe leave-one-video-out protocol, and reference scores of R² {reg.loc['random_forest', 'r2_mean']:.2f} "
        f"(regression) and F1 {cls.loc['logistic', 'f1_mean']:.3f} / PR-AUC {cls.loc['logistic', 'pr_auc_mean']:.3f} "
        "(logistic classification). Later chapters reuse exactly these folds and metrics, so every new model can be "
        "compared against logistic regression on equal terms."
    )


# --------------------------------------------------------------------------- chapter 3
def chapter3(r: Report, R: dict) -> None:
    d, P = R["dataset"], R["paths"]
    rv = R["review"].set_index("method")
    det = R["review_details"]
    km_k = pd.Series([x["models"]["K-means"]["k"] for x in det]).value_counts().to_dict()
    gm_k = pd.Series([x["models"]["Gaussian mixture"]["k"] for x in det]).value_counts().to_dict()
    hm_k = pd.Series([x["models"]["Gaussian HMM"]["k"] for x in det]).value_counts().to_dict()
    conv = sum(bool(x["models"]["Gaussian HMM"]["converged"]) for x in det)
    diag = np.mean([np.trace(np.array(x["models"]["Gaussian HMM"]["transmat"])) / x["models"]["Gaussian HMM"]["k"] for x in det])
    r.chapter_heading("Unsupervised Learning and Sequential Data Modelling – (Units 3 and 4)")
    r.h2("3.1 Overview of Allotted Unit 3 and Unit 4 Topics")
    r.para(
        "**Unit 3 (clustering):** K-means and Gaussian mixture models (GMM) group seconds by feature similarity "
        "without using labels. The question is whether natural groups in the feature space coincide with relevant "
        "moments. **Unit 4 (sequential modelling):** a Gaussian hidden Markov model (HMM) treats each (video, query) "
        "pair as a time series of feature vectors emitted by a small number of hidden states. Unlike clustering, "
        "the HMM models persistence: a relevant moment lasts several seconds, so neighbouring seconds should share "
        "a state. The HMM was implemented from first principles in `analysis/hmm.py` (scaled forward–backward, "
        "Baum–Welch, Viterbi) behind a scikit-learn style interface."
    )
    r.h2("3.2 Unit 3 – Dataset Description and Preprocessing")
    feats = ["visual_norm", "speech_norm", "visual_vs_peak", "speech_vs_peak", "keyword_overlap", "fused_peak_dist",
             "frame_change", "in_speech"]
    r.para(
        "The existing dataset of Chapter 2 is used. Additional preprocessing specific to clustering: (1) a "
        f"compact subset of eight interpretable features was selected ({', '.join(feats)}) because distance-based "
        "clustering degrades with many redundant, correlated dimensions; (2) the features were standardized with a "
        "StandardScaler fitted on the training videos of each fold only; (3) principal component analysis was "
        "fitted on training videos solely to visualize clusters in two dimensions and is not part of the models."
    )
    r.h2("3.3 Unit 3 – Theory, EDA, Implementation and Results")
    r.para(
        "Implementation (`analysis/review.py`): within each leave-one-video-out fold, K-means is fitted for "
        "k ∈ {2, 3, 4, 5} with 5 restarts and k is chosen by the highest silhouette score on a fixed sample of "
        "2,000 training rows; the GMM uses diagonal covariances with the same candidates and is selected by lowest "
        "BIC. Labels are never used for fitting or model selection. To compare with supervised models, each cluster "
        "is then assigned its relevant fraction among training rows, and a held-out second's score is the "
        "membership-weighted average of those fractions (softmax of negative squared centroid distance for K-means, "
        "posterior probabilities for the GMM). The F1 threshold is chosen on training rows."
    )
    r.figure_block(P["rv"] / "03_cluster_projection.png", "PCA projection of a held-out video coloured by K-means cluster and by ground truth", 6.2,
                   "The relevant seconds (right) occupy one edge of the projected cloud, and one K-means cluster (left) "
                   "covers the same region. Clustering therefore recovers part of the relevance structure without "
                   "labels, although the cluster also contains many background seconds.")
    r.figure_block(P["rv"] / "04_cluster_interpretation.png", "K-means cluster centres and training-label relevance of each cluster", 6.2,
                   "The relevant-rich cluster has high calibrated speech similarity, high score-relative-to-peak and "
                   "small distance to the fused peak; the other cluster is background. The relevance fraction of the "
                   "best cluster is well above the 8.3% base rate, which is what makes the cluster score informative.")
    r.h2("3.4 Unit 3 – Algorithm Theory and Mathematical Formulation")
    r.equation("J = Σ_{k=1}^{K} Σ_{i ∈ C_k} ‖xᵢ − μ_k‖²",
               "K-means (Lloyd's algorithm) alternates assigning each xᵢ to the nearest centre μ_k and recomputing centres.")
    r.equation("p(x) = Σ_{k=1}^{K} π_k 𝒩(x | μ_k, diag(σ²_k))",
               "The GMM is fitted by EM: the E-step computes responsibilities γ_ik ∝ π_k 𝒩(xᵢ | μ_k, σ²_k); the M-step "
               "re-estimates π_k, μ_k and σ²_k from them, with a variance floor of 0.05.")
    r.equation("s(i) = (b(i) − a(i)) / max(a(i), b(i))",
               "Silhouette: a(i) is the mean distance to the own cluster and b(i) to the nearest other cluster.")
    r.equation("BIC = −2 ln L̂ + p ln n",
               "Lower BIC balances fit (log-likelihood L̂) against the number of free parameters p.")
    r.equation("score(x) = Σ_k w_k(x) · ρ_k,      ρ_k = Σᵢ 1[i ∈ C_k] cᵢ / |C_k|",
               "Training-label calibration maps cluster memberships w_k(x) to a relevance score.")
    r.h2("3.5 Unit 3 Proposed System Architecture / Workflow")
    r.figure_block(P["rp"] / "workflow_pbl2_unit3.png", "Unit 3 clustering workflow", 6.3,
                   "Every transformation and selection step lies inside the fold, so the held-out video influences "
                   "neither the scaler, the number of clusters, the cluster relevance rates, nor the threshold.")
    r.h2("3.6 Unit 3 – Results and Performance Evaluation")
    r.figure_block(P["rv"] / "02_model_selection.png", "Training-only model selection: silhouette for K-means and BIC for GMM and HMM", 6.3,
                   f"K-means selected k = 2 in most folds ({counts(km_k, 'k')}); the GMM always preferred the largest "
                   f"candidate ({counts(gm_k, 'k')}), because BIC on 25,000 correlated rows keeps rewarding extra components.")
    rows = [[m, f3(rv.loc[m, "precision"]), f3(rv.loc[m, "recall"]), f3(rv.loc[m, "f1"]), f3(rv.loc[m, "pr_auc"]),
             f3(rv.loc[m, "hit_at_1"]), f3(rv.loc[m, "temporal_iou"])] for m in ["K-means", "Gaussian mixture", "Fixed-weight fusion"]]
    r.table_block("Unit 3 clustering results (mean over 9 held-out videos)",
                  ["Method", "Precision", "Recall", "F1", "PR-AUC", "Hit@1", "Temporal IoU"], rows,
                  widths=[1.7, 0.8, 0.7, 0.6, 0.7, 0.7, 0.9],
                  interpretation=f"K-means reaches F1 {rv.loc['K-means', 'f1']:.3f} and Hit@1 {rv.loc['K-means', 'hit_at_1']:.3f}, "
                                 "better than the fixed fusion rule even though its clusters were found without labels. "
                                 "The GMM is weaker; its five diagonal components fragment the relevant region.")
    r.h2("3.7 Unit 4 – Sequential Dataset Description")
    lengths = d.groupby(["video_id", "query_id"]).size()
    gaps = d.sort_values(["video_id", "query_id", "t"]).groupby(["video_id", "query_id"]).t.diff().dropna()
    r.para(
        "The sequential view of the original dataset is used. **Sequence identifier:** (video_id, query_id); "
        f"**number of sequences:** {len(lengths)}; **timestamp:** t in seconds; **sampling rate:** 1 Hz (one frame "
        f"per second); **sequence length:** {lengths.min()}–{lengths.max()} steps (median {int(lengths.median())}); "
        f"**missing timestamps:** none (every step is exactly {gaps.max():.0f} s after the previous). "
        "**Observation variables:** the same eight standardized features as Unit 3. **Hidden states:** not given; "
        "they are learned. **Limitations:** sequences of one query are short relative to the number of parameters, "
        "relevance occupies 8.3% of steps, and features use a whole-video context, so the model is suitable for "
        "offline retrieval rather than causal streaming."
    )
    r.h2("3.8 Unit 4 – Sequential Preprocessing and EDA")
    r.para(
        "Rows are sorted by (video, query, t) and duplicate or missing keys are rejected. Each (video, query) pair "
        "is an independent sequence: the forward–backward recursions restart at each boundary, so no transition "
        "is learned across queries or videos. Scaling uses training-fold statistics only. The example timelines "
        "(Figure 3.9) show that relevance appears as contiguous blocks of 5–30 seconds, which justifies a model with "
        "persistent states."
    )
    r.h2("3.9 Unit 4 – Mathematical Formulation")
    r.para("Let z_t ∈ {1..K} be the hidden state and x_t the observation at second t of a sequence of length T.")
    r.equation("p(x_{1:T}, z_{1:T}) = π_{z₁} ∏_{t=2}^{T} A_{z_{t−1} z_t} ∏_{t=1}^{T} 𝒩(x_t | μ_{z_t}, diag(σ²_{z_t}))",
               "π is the initial distribution, A the transition matrix and (μ_k, σ²_k) the Gaussian emission of state k.")
    r.equation("α_t(j) = [Σᵢ α_{t−1}(i) A_{ij}] b_j(x_t),     β_t(i) = Σ_j A_{ij} b_j(x_{t+1}) β_{t+1}(j)",
               "Scaled forward–backward recursions; γ_t(k) = α_t(k)β_t(k)/p(x_{1:T}) is the posterior of state k.")
    r.equation("ξ_t(i, j) ∝ α_t(i) A_{ij} b_j(x_{t+1}) β_{t+1}(j),    Â_{ij} = Σ_t ξ_t(i, j) / Σ_t γ_t(i)",
               "Baum–Welch (EM) re-estimates transitions, initial probabilities and emission means/variances from the "
               "expected counts, with a variance floor 0.05 and transition pseudocount 0.01.")
    r.equation("δ_t(j) = max_i δ_{t−1}(i) A_{ij} b_j(x_t),    ẑ_{1:T} = argmax p(z_{1:T} | x_{1:T})",
               "Viterbi decoding returns the most probable joint state path by dynamic programming.")
    r.para(
        "Inference goal: the relevance score of second t is Σ_k γ_t(k)ρ_k, where ρ_k is the training relevant "
        "fraction of state k (same calibration as Unit 3). Assumptions: first-order Markov dynamics, conditionally "
        "independent diagonal-Gaussian emissions given the state, and a stationary transition matrix shared by all "
        "queries. The number of states K ∈ {2, 3} is chosen by training BIC; initialization uses K-means."
    )
    r.h2("3.10 Unit 4 Proposed System Architecture / Workflow")
    r.figure_block(P["rp"] / "workflow_pbl2_unit4.png", "Unit 4 hidden Markov model workflow", 6.3,
                   "The HMM is fitted without labels on the training sequences; labels are used afterwards only to "
                   "interpret states, and the held-out video is decoded with the frozen model.")
    r.h2("3.11 Results and Performance Evaluation")
    r.figure_block(P["rv"] / "05_hmm_parameters.png", "Learned HMM transition matrix, emission means and state relevance (one fold)", 6.3,
                   f"Diagonal transitions dominate (mean self-transition {diag:.3f} across folds), so states persist "
                   "for many seconds. One state combines high speech similarity with short distance to the peak and "
                   "has the highest training relevance, mirroring the relevant K-means cluster.")
    r.figure_block(P["rv"] / "06_hmm_convergence.png", "Baum–Welch log-likelihood per training row by iteration", 5.8,
                   f"The log-likelihood increases monotonically, as EM guarantees, and the stopping criterion was met "
                   f"in {conv}/9 folds. The selected number of states was {counts(hm_k, 'K')}.")
    r.figure_block(P["rv"] / "09_viterbi_decoding.png", "HMM observations, posterior relevance and Viterbi state path for one query", 6.2,
                   "The Viterbi path switches to the relevant state around the true interval (green) and stays there, "
                   "producing contiguous decisions; it also stays in that state for some neighbouring background "
                   "seconds, which lowers precision.")
    rows = [[m, f3(rv.loc[m, "precision"]), f3(rv.loc[m, "recall"]), f3(rv.loc[m, "f1"]), f3(rv.loc[m, "pr_auc"]),
             f3(rv.loc[m, "hit_at_1"]), f3(rv.loc[m, "mrr"]), f3(rv.loc[m, "temporal_iou"])]
            for m in ["Gaussian HMM", "Fixed-weight fusion", "Speech only"]]
    r.table_block("Unit 4 hidden Markov model results (mean over 9 held-out videos)",
                  ["Method", "Precision", "Recall", "F1", "PR-AUC", "Hit@1", "MRR", "Temporal IoU"], rows,
                  widths=[1.5, 0.75, 0.65, 0.55, 0.65, 0.6, 0.6, 0.8],
                  interpretation=f"The HMM reaches F1 {rv.loc['Gaussian HMM', 'f1']:.3f} and temporal IoU "
                                 f"{rv.loc['Gaussian HMM', 'temporal_iou']:.3f}, above fixed fusion on both, but its "
                                 f"ranking metrics (Hit@1 {rv.loc['Gaussian HMM', 'hit_at_1']:.3f}) are lower because "
                                 "state posteriors saturate to plateaus that do not order seconds within a state.")
    r.figure_block(P["rv"] / "08_example_timelines.png", "Held-out timelines for the lowest-, middle- and highest-IoU queries", 6.3,
                   "The best case matches the interval closely; the worst case is a query answered only in speech "
                   "that the eight unsupervised features do not separate from nearby narration.")
    r.figure_block(P["rv"] / "07_comparison.png", "Comparison of Unit 3 and Unit 4 models with baselines and PBL-1 classifiers", 6.3,
                   "Unsupervised models sit between the hand-set baselines and the supervised classifiers. Error "
                   "bars (video bootstrap) are wide because only nine independent videos are available.")
    r.h2("3.12 Discussion, Findings and Limitations")
    r.bullets([
        "Clustering recovers relevance without labels: K-means beats fixed fusion on F1 and Hit@1, showing that relevant seconds form a distinct region of the engineered feature space.",
        "The HMM's states are persistent and interpretable, but its objective is sequence likelihood rather than relevance, so smoother timelines do not automatically give better retrieval; its posteriors saturate, which hurts ranking.",
        "State definitions are learned, not predefined: state IDs are arbitrary and can permute between folds, so states are only interpreted through training-label relevance rates.",
        "Sequential validation holds out whole videos, and sequences never cross query or video boundaries, so the reported numbers are honest generalization estimates; with nine videos, uncertainty remains large.",
        "Limitations: diagonal covariances ignore correlated features, BIC on correlated per-second rows is only a heuristic, and features use future context (offline only).",
    ])
    r.h2("3.13 Chapter Conclusion")
    r.para(
        f"Unit 3 showed that unsupervised clustering captures much of the relevance signal (K-means F1 "
        f"{rv.loc['K-means', 'f1']:.3f}), and Unit 4 delivered a from-scratch Gaussian HMM whose persistent states "
        f"decode contiguous moments (F1 {rv.loc['Gaussian HMM', 'f1']:.3f}). Both improve on the fixed fusion rule "
        "on F1 but remain below the supervised classifiers of Chapter 2, which motivates combining strong "
        "supervised learners through ensembles in Chapter 4."
    )


from report_content_part2 import chapter4, chapter5, chapter6, references, appendix  # noqa: E402


def write_report(r: Report, R: dict) -> None:
    e = R["ens"]
    ens = e[e.method.isin(ENSEMBLES)].sort_values(["f1", "pr_auc"], ascending=False)
    R["best"] = ens.method.iloc[0]
    from report_content_part2 import abstract_conclusion
    R["abstract_conclusion"] = abstract_conclusion(R)
    front_matter(r, R)
    chapter1(r, R)
    chapter2(r, R)
    chapter3(r, R)
    chapter4(r, R)
    chapter5(r, R)
    chapter6(r, R)
    references(r, R)
    appendix(r, R)
