"""Text, tables and figures of the integrated report (compact, < 30 pages). Numbers come from saved results."""

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


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# --------------------------------------------------------------------------- front matter
def front_matter(r: Report, R: dict) -> None:
    r.new_section("lowerRoman", start=5)
    r.title("Abstract", toc=True)
    r.para(abstract_text(R))
    r.page_break()
    r.title("Table of Contents")
    r.toc('TOC \\h \\z \\t "Chapter Heading,1,Front Heading,1,Section Heading,2"', "Table of contents")
    r.page_break()
    r.title("List of Figures", toc=True)
    r.toc('TOC \\h \\z \\t "Figure Caption,1"', "List of figures")
    r.title("List of Tables", toc=True)
    r.toc('TOC \\h \\z \\t "Table Caption,1"', "List of tables")
    r.title("Abbreviations", toc=True)
    pairs = [
        ("AP", "Average Precision"), ("BIC", "Bayesian Information Criterion"),
        ("BMA", "Bayesian Model Averaging"), ("CART", "Classification and Regression Trees"),
        ("CLIP", "Contrastive Language–Image Pre-training"), ("CV", "Cross-Validation"),
        ("EM", "Expectation–Maximization"), ("EDA", "Exploratory Data Analysis"),
        ("GMM", "Gaussian Mixture Model"), ("HMM", "Hidden Markov Model"),
        ("IoU", "Intersection over Union"), ("k-NN", "k-Nearest Neighbours"),
        ("MRR", "Mean Reciprocal Rank"), ("OOF", "Out-of-Fold"),
        ("PR-AUC", "Area under the PR curve"), ("RMSE", "Root Mean Squared Error"),
    ]
    half = len(pairs) // 2
    rows = [[a, b, c, d] for (a, b), (c, d) in zip(pairs[:half], pairs[half:])]
    r.plain_table(["Abbreviation", "Expansion", "Abbreviation", "Expansion"], rows, widths=[1.15, 1.75, 1.15, 1.7])


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
        "Finding a particular moment inside a long lecture, news or outreach video still means scrubbing through "
        "the timeline, and title search cannot answer questions such as \"where does the rover arm appear?\". "
        "Needle is a cross-modal video moment retrieval system: frames are sampled at 1 frame/s and embedded with "
        "OpenCLIP ViT-B/32 [1], speech is transcribed by Whisper [2] and embedded with MiniLM [3], and a query is "
        "matched against both indexes with exact FAISS search [4]. The resulting picture and speech similarity "
        "timelines are fused into timestamped clips served by a FastAPI and Next.js application. Stakeholders are "
        "students, educators, journalists and archivists. The encoders are never fine-tuned; all learning in this "
        "project happens on features engineered from their outputs, which keeps experiments affordable on a laptop."
    )
    r.h2("1.2 Integrated Problem Statement")
    r.para(
        "Given a video and a natural-language query, decide for every second how relevant it is, so that the "
        "answer interval can be located and ranked first. The hand-designed baseline fuses calibrated picture and "
        "speech similarity with a fixed weight (α = 0.6); the project asks whether learned models can replace it. "
        "All five units use one original dataset, the labelled query–second table of Chapter 2: Unit 1 regresses "
        "graded relevance, Unit 2 classifies relevant seconds, Unit 3 clusters the feature space, Unit 4 models "
        "each (video, query) pair as a time series with a hidden Markov model (so no separate sequential dataset "
        "was needed), and Unit 5 combines learners into ensembles."
    )
    r.h2("1.3 Project Objectives")
    r.bullets([
        "**Objective 1 (Unit 1):** predict graded relevance with ridge, random-forest and gradient-boosting regressors, well above the mean baseline on unseen videos.",
        "**Objective 2 (Unit 2):** classify relevant seconds with logistic regression, k-NN and gradient boosting, and beat hand-set fusion on F1 and PR-AUC.",
        "**Objective 3 (Unit 3):** find relevance structure without labels using K-means and Gaussian mixtures.",
        "**Objective 4 (Unit 4):** implement a Gaussian HMM from first principles (Baum–Welch, Viterbi) for temporal relevance decoding.",
        "**Objective 5 (Unit 5):** compare nine ensemble methods with all earlier models and test whether ensembling gives a statistically supported gain.",
    ])
    r.h2("1.4 Unit-wise Topic and Dataset Allocation")
    r.table_block("Unit-wise topic and dataset allocation", ["PBL", "Unit", "Allotted topic / algorithm", "Dataset"], [
        ["PBL 1", "1", "Regression: ridge, random forest, gradient boosting", "Original query–second table"],
        ["PBL 1", "2", "Classification: logistic regression, k-NN, gradient boosting", "Original"],
        ["PBL 2", "3", "Clustering: K-means, Gaussian mixture model", "Original"],
        ["PBL 2", "4", "Sequential modelling: Gaussian HMM", "Original, as 86 per-query sequences"],
        ["PBL 3", "5", "Ensembles: CART, bagging, RF, extra trees, AdaBoost, boosting, voting, stacking, BMA", "Original"],
    ], widths=[0.7, 0.5, 3.0, 1.8])


# --------------------------------------------------------------------------- chapter 2
def chapter2(r: Report, R: dict) -> None:
    d, P = R["dataset"], R["paths"]
    reg, cls = R["reg"].set_index("model"), R["cls"].set_index("model")
    rt = R["retrieval"].set_index("method")
    lengths = d.groupby(["video_id", "query_id"]).size()
    r.chapter_heading("Data Analysis and Fundamental Machine Learning Techniques – (Units 1 and 2)")
    r.h2("2.1 Overview of Allotted Topics")
    r.para(
        "Unit 1 (regression) and Unit 2 (classification) fit the benchmark naturally: every second has a graded "
        "relevance target and a binary in/out label. For each task three models of increasing flexibility were "
        "used, all implemented with scikit-learn [5]: L2-regularized linear models (ridge, logistic regression), "
        "k-nearest neighbours, and tree ensembles (random forest, gradient boosting) [6] that capture non-linear interactions. Each is compared with a trivial "
        "baseline and with the application's hand-set fusion rule."
    )
    r.h2("2.2 Dataset Description")
    r.para(
        f"**Needle relevance benchmark** (built by this project; {GITHUB}): nine public-domain NASA videos, "
        f"{d.query_id.nunique()} hand-written queries with annotated answer intervals, and {len(d):,} rows, one per "
        f"(query, second); sequences of {lengths.min()}–{lengths.max()} s. Targets: `rel_binary` (inside the answer) "
        f"and `rel_graded` (1 inside, decaying as exp(−distance/5 s) outside); {pct(d.rel_binary.mean())} of rows are "
        f"relevant. All 31 features are numeric and there are {int(d.isna().sum().sum())} missing values. "
        "Limitations: labels were written from transcripts (speech bias), one domain, and only nine independent videos."
    )
    r.table_block("Engineered feature groups (31 features)", ["Group", "Examples", "Meaning"], [
        ["Similarity", "visual_raw, speech_norm, fused_baseline", "Query–frame / transcript cosine; calibrated; fused"],
        ["Standardized", "visual_z, speech_rank", "z-score and percentile within the video"],
        ["Temporal context", "speech_near (±5 s), visual_wide (±10 s)", "Moving averages around the second"],
        ["Peak structure", "speech_peak_dist, fused_vs_peak", "Distance to and ratio with the strongest peak"],
        ["Content", "frame_change, in_speech, keyword_overlap", "Scene change, speech presence, shared words"],
    ], widths=[1.3, 2.3, 2.4])
    r.h2("2.3 Data Preprocessing")
    r.para(
        "ffmpeg extracts one frame per second and 16 kHz audio; frames and transcript segments are encoded and "
        "L2-normalized so that inner product equals cosine similarity; similarities are spread onto a per-second "
        "timeline and the 31 features and two targets are computed per query. No cleaning or imputation was "
        "needed. Scaling (StandardScaler for linear models and k-NN) is fitted on training videos only. Evaluation "
        "is **leave-one-video-out**: nine folds, each testing on an unseen video, with hyper-parameters, an output "
        "smoothing window (1–15 s) and the F1 threshold chosen by grouped 3-fold CV inside the training videos."
    )
    r.h2("2.4 Exploratory Data Analysis (EDA)")
    corr = d.select_dtypes("number").drop(columns=["query_id", "t"]).corr()["rel_binary"].drop(["rel_binary", "rel_graded"])
    r.figure_block(P["ml"] / "02_feature_distributions.png", "Distributions of key features for relevant and background seconds", 4.6,
                   f"Only {pct(d.rel_binary.mean())} of seconds are relevant, so F1 and PR-AUC replace accuracy. "
                   "Relevant seconds have higher speech similarity and lie close to the speech peak, while picture "
                   f"similarity overlaps more; the most label-correlated feature is {corr.abs().idxmax()} "
                   f"(|r| = {corr.abs().max():.2f}).")
    r.h2("2.5 Proposed System Architecture / Workflow")
    r.figure_block(P["rp"] / "workflow_pbl1.png", "PBL-1 workflow from raw video to evaluated models", 5.2,
                   "Videos and labelled queries are encoded by frozen models, turned into 31 features, and used to "
                   "tune three regressors and three classifiers inside each leave-one-video-out fold; held-out "
                   "predictions are scored and compared with the baselines.")
    r.h2("2.6 Algorithm Theory and Mathematical Formulation")
    r.equation("β̂ = argmin_β Σᵢ (yᵢ − xᵢᵀβ)² + λ‖β‖²",
               "Ridge regression; λ shrinks the many correlated coefficients. Logistic regression models "
               "P(c = 1 | x) = σ(β₀ + xᵀβ) with the same L2 penalty.")
    r.equation("f_RF(x) = (1/B) Σ_b T_b(x),      F_m(x) = F_{m−1}(x) + ν·h_m(x)",
               "A random forest averages B decorrelated trees; gradient boosting adds trees fitted to the "
               "negative gradient of the loss. k-NN predicts the relevant fraction among the k nearest rows.")
    r.equation("F1 = 2PR / (P + R),      R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²",
               "PR-AUC averages precision over the ranking; Hit@1 is the share of queries whose top second is relevant.")
    r.h2("2.7 Results and Performance Evaluation")
    names = {"dummy": "Mean baseline", "ridge": "Ridge", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting"}
    r.table_block("Regression results (mean ± std over 9 held-out videos)", ["Model", "RMSE", "MAE", "R²", "Spearman ρ"], [
        [names[m], f"{reg.loc[m, 'rmse_mean']:.3f} ± {reg.loc[m, 'rmse_std']:.3f}", f3(reg.loc[m, "mae_mean"]),
         f"{reg.loc[m, 'r2_mean']:.3f} ± {reg.loc[m, 'r2_std']:.3f}", f3(reg.loc[m, "spearman_mean"])]
        for m in ["dummy", "ridge", "random_forest", "gradient_boosting"]
    ], widths=[1.7, 1.3, 0.9, 1.3, 1.0],
        interpretation=f"Tree ensembles explain about three quarters of the variance (R² ≈ {reg.loc['random_forest', 'r2_mean']:.2f}) "
                       f"on unseen videos; ridge (R² {reg.loc['ridge', 'r2_mean']:.2f}) shows the relation is non-linear.")
    cnames = {"dummy": "Prior baseline", "logistic": "Logistic regression", "knn": "k-NN", "gradient_boosting": "Gradient boosting"}
    r.table_block("Classification results (mean over 9 held-out videos)",
                  ["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"], [
        [cnames[m], f3(cls.loc[m, "precision_mean"]), f3(cls.loc[m, "recall_mean"]), f3(cls.loc[m, "f1_mean"]),
         f3(cls.loc[m, "roc_auc_mean"]), f3(cls.loc[m, "pr_auc_mean"])]
        for m in ["dummy", "logistic", "knn", "gradient_boosting"]
    ], widths=[1.8, 0.85, 0.8, 0.75, 0.85, 0.85],
        interpretation=f"Logistic regression is best (F1 {cls.loc['logistic', 'f1_mean']:.3f}, PR-AUC "
                       f"{cls.loc['logistic', 'pr_auc_mean']:.3f}): with context features a linear boundary suffices. "
                       f"For ranking, it puts a relevant second first for {pct(rt.loc['Classification: Logistic regression', 'hit_at_1'])} "
                       f"of queries versus {pct(rt.loc['Fixed-weight fusion', 'hit_at_1'])} for hand-set fusion.")
    ab = R["ablation"]
    lg = ab[(ab.model == "logistic")].set_index("variant")
    rf = ab[(ab.model == "random_forest")].set_index("variant")
    r.table_block("Ablation of feature engineering and output smoothing (pooled, held-out)",
                  ["Variant", "RF regression R²", "Logistic F1", "Logistic PR-AUC"], [
        [v, f3(rf.loc[v, "r2"]), f3(lg.loc[v, "f1"]), f3(lg.loc[v, "pr_auc"])]
        for v in ["Original 15 features", "+ context features", "+ output smoothing"]
    ], widths=[2.3, 1.2, 1.1, 1.2],
        interpretation="Context features nearly double R² and raise F1 by about 0.1: feature engineering mattered "
                       "more than the choice of model.")
    r.h2("2.8 Discussion and Findings")
    r.bullets([
        "All six models clearly beat their baselines; speech-derived context features (windows, distance to peak, z-scores) carry most of the signal.",
        "Non-linear trees win for regression, while logistic regression matches them for classification. Learned rankers beat hand-set fusion on Hit@1, but not significantly with 86 queries.",
        "Challenges were class imbalance (F1-tuned thresholds, PR-AUC), correlated features and leakage between seconds of one video, prevented by leave-one-video-out validation.",
    ])
    r.h2("2.9 Chapter Conclusion")
    r.para(
        f"PBL 1 fixed the shared dataset, the leakage-safe protocol and the reference scores (R² "
        f"{reg.loc['random_forest', 'r2_mean']:.2f}; logistic F1 {cls.loc['logistic', 'f1_mean']:.3f}) against which "
        "every later model is compared on identical folds and metrics."
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
    lengths = d.groupby(["video_id", "query_id"]).size()
    cols = ["precision", "recall", "f1", "pr_auc", "hit_at_1", "temporal_iou"]
    head = ["Method", "Precision", "Recall", "F1", "PR-AUC", "Hit@1", "IoU"]
    widths = [1.9, 0.8, 0.7, 0.6, 0.75, 0.65, 0.6]

    r.chapter_heading("Unsupervised Learning and Sequential Data Modelling – (Units 3 and 4)")
    r.h2("3.1 Overview of Allotted Unit 3 and Unit 4 Topics")
    r.para(
        "Unit 3 asks whether K-means and Gaussian mixture models (GMM), fitted without labels, find a region of the "
        "feature space that coincides with relevant seconds. Unit 4 treats each (video, query) pair as a time series "
        "and fits a Gaussian hidden Markov model whose persistent hidden states can capture that a relevant moment "
        "lasts several seconds. The HMM was implemented from first principles in `analysis/hmm.py` (scaled "
        "forward–backward, Baum–Welch, Viterbi)."
    )
    r.h2("3.2 Unit 3 – Dataset Description and Preprocessing")
    r.para(
        "The Chapter 2 dataset is used with eight interpretable features (calibrated similarities, scores relative "
        "to the peak, keyword overlap, distance to the fused peak, frame change and speech presence), because "
        "distance-based clustering degrades with many redundant dimensions. Features are standardized on the "
        "training videos of each fold only; PCA is used for visualization alone."
    )
    r.h2("3.3 Unit 3 – Theory, EDA, Implementation and Results")
    r.para(
        "In each leave-one-video-out fold, K-means (k ∈ {2..5}) is selected by silhouette and the diagonal GMM by "
        "BIC, without labels. Each cluster is then given its relevant fraction among training rows, and a held-out "
        "second scores the membership-weighted average of those fractions."
    )
    r.figure_block(P["rv"] / "03_cluster_projection.png", "PCA projection of a held-out video by K-means cluster and by ground truth", 5.4,
                   "One cluster covers the region where relevant seconds lie, so clustering recovers part of the "
                   "relevance structure without labels, though that cluster also holds background seconds.")
    r.h2("3.4 Unit 3 – Algorithm Theory and Mathematical Formulation")
    r.equation("J = Σ_k Σ_{i ∈ C_k} ‖xᵢ − μ_k‖²,      p(x) = Σ_k π_k 𝒩(x | μ_k, diag(σ²_k))",
               "K-means minimizes within-cluster distance J (Lloyd's algorithm); the GMM is fitted by EM [16].")
    r.equation("s(i) = (b(i) − a(i)) / max(a(i), b(i)),      BIC = −2 ln L̂ + p ln n",
               "Silhouette [17] and BIC [18] select the number of clusters; the relevance score is "
               "Σ_k w_k(x) ρ_k with ρ_k the training relevant fraction of cluster k.")
    r.h2("3.5 Unit 3 Proposed System Architecture / Workflow")
    r.figure_block(P["rp"] / "workflow_pbl2_unit3.png", "Unit 3 clustering workflow", 5.0,
                   "Scaling, model selection, cluster relevance rates and the threshold are all fitted inside the fold.")
    r.h2("3.6 Unit 3 – Results and Performance Evaluation")
    r.table_block("Unit 3 clustering results (mean over 9 held-out videos)", head,
                  [[m] + [f3(rv.loc[m, c]) for c in cols] for m in ["K-means", "Gaussian mixture", "Fixed-weight fusion"]],
                  widths=widths,
                  interpretation=f"K-means ({counts(km_k, 'k')}) beats the fixed fusion rule on F1 and Hit@1 without "
                                 f"labels. The GMM ({counts(gm_k, 'k')}) is weaker: BIC on correlated rows keeps adding "
                                 "components that fragment the relevant region.")
    r.h2("3.7 Unit 4 – Sequential Dataset Description")
    r.para(
        f"Sequences are the (video_id, query_id) pairs of the original dataset: {len(lengths)} sequences of "
        f"{lengths.min()}–{lengths.max()} steps (median {int(lengths.median())}), sampled at 1 Hz with no missing "
        "timestamps. Observations are the eight standardized Unit 3 features; hidden states are learned, not given."
    )
    r.h2("3.8 Unit 4 – Sequential Preprocessing and EDA")
    r.para(
        "Rows are sorted by (video, query, t); recursions restart at every sequence boundary, so no transition "
        "crosses queries or videos. Relevance appears as contiguous blocks of 5–30 s, which motivates persistent states."
    )
    r.h2("3.9 Unit 4 – Mathematical Formulation")
    r.equation("p(x_{1:T}, z_{1:T}) = π_{z₁} ∏_t A_{z_{t−1} z_t} ∏_t 𝒩(x_t | μ_{z_t}, diag(σ²_{z_t}))",
               "π is the initial distribution, A the transition matrix and (μ_k, σ²_k) the emission of state k [15].")
    r.equation("α_t(j) = [Σᵢ α_{t−1}(i) A_{ij}] b_j(x_t),      Â_{ij} = Σ_t ξ_t(i, j) / Σ_t γ_t(i)",
               "Scaled forward–backward gives state posteriors γ and transition expectations ξ, which Baum–Welch "
               "uses to re-estimate the parameters.")
    r.equation("δ_t(j) = max_i δ_{t−1}(i) A_{ij} b_j(x_t)",
               "Viterbi returns the most probable state path. The score of second t is Σ_k γ_t(k) ρ_k; K ∈ {2, 3} is "
               "chosen by training BIC.")
    r.h2("3.10 Unit 4 Proposed System Architecture / Workflow")
    r.figure_block(P["rp"] / "workflow_pbl2_unit4.png", "Unit 4 hidden Markov model workflow", 5.0,
                   "The HMM is fitted without labels; labels only name the states, and the held-out video is decoded "
                   "with the frozen model.")
    r.h2("3.11 Results and Performance Evaluation")
    r.figure_block(P["rv"] / "05_hmm_parameters.png", "Learned HMM transition matrix, emission means and state relevance (one fold)", 5.75,
                   f"Self-transitions dominate (mean {diag:.3f}), so states persist for many seconds; one state combines "
                   f"high speech similarity with a short distance to the peak. Baum–Welch converged in {conv}/9 folds "
                   f"({counts(hm_k, 'K')}). Viterbi decoding then gives contiguous decisions around the true interval.")
    r.table_block("Unit 4 hidden Markov model results (mean over 9 held-out videos)", head,
                  [[m] + [f3(rv.loc[m, c]) for c in cols] for m in ["Gaussian HMM", "Fixed-weight fusion", "Speech only"]],
                  widths=widths,
                  interpretation=f"The HMM beats fixed fusion on F1 ({rv.loc['Gaussian HMM', 'f1']:.3f}) and IoU, but "
                                 "its ranking is weaker because state posteriors saturate into plateaus.")
    r.h2("3.12 Discussion, Findings and Limitations")
    r.bullets([
        "Relevant seconds form a distinct region of the feature space: unsupervised K-means beats the hand-set rule.",
        "The HMM's states are persistent and interpretable, but its objective is likelihood, not relevance; state IDs are arbitrary, so states are interpreted only through training relevance rates.",
        "Limitations: diagonal covariances, BIC on correlated rows is only a heuristic, and features use future context (offline use only).",
    ])
    r.h2("3.13 Chapter Conclusion")
    r.para(
        f"Clustering (K-means F1 {rv.loc['K-means', 'f1']:.3f}) and the from-scratch HMM (F1 "
        f"{rv.loc['Gaussian HMM', 'f1']:.3f}) both improve on fixed fusion without labels but remain below the "
        "supervised classifiers, which motivates combining supervised learners in Chapter 4."
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
