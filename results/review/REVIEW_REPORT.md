# Review 2: Clustering and Hidden Markov Models for Video Moment Retrieval

## 1. Problem identification and dataset selection (4 marks)

Needle locates the moments in a video that answer a natural-language query.
Pictures and speech give complementary evidence; individual similarity peaks can
be noisy. Can unsupervised grouping discover relevance patterns, and can a model
of temporal transitions produce useful, contiguous relevance decisions?

### Training datasets and model provenance

The model-training datasets are documented below. They have different roles from
the videos a user uploads to the web application for searching.

#### Datasets used to train the pretrained encoders

These encoders were trained by their original authors. This project loads their
released weights and keeps them frozen; we did not retrain them on these corpora.

| Model used in this project | Original training dataset / corpus | Purpose in our pipeline |
|---|---|---|
| OpenCLIP ViT-B/32, `laion2b_s34b_b79k` | **LAION-2B English**, the approximately two-billion image–text-pair English subset of LAION-5B. | Provides aligned visual and text embeddings for matching queries to frames. |
| Whisper `base`, through faster-whisper | **OpenAI's 680,000-hour multilingual audio–transcript corpus**, collected from the internet. The faster-whisper checkpoint is a CTranslate2 conversion of the released Whisper base model. | Converts speech into timestamped text. |
| Sentence Transformer `all-MiniLM-L6-v2` | **A mixed sentence-pair training collection with over one billion pairs/tuples**. Sources include Reddit comments, S2ORC, WikiAnswers, PAQ, Stack Exchange, MS MARCO, and other datasets listed in the model card. This sentence-embedding training builds on a pretrained MiniLM backbone. | Encodes queries and transcript segments for semantic similarity. |

Sources: [OpenCLIP model card](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K),
[Whisper training-data description](https://github.com/openai/whisper/blob/main/model-card.md#training-data),
[faster-whisper base conversion](https://huggingface.co/Systran/faster-whisper-base),
[MiniLM sentence-embedding training data](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#training-data).

#### Dataset used to train or fit models in this project

Our own fitted models use the **labeled query–second relevance feature dataset**
stored in `results/ml/dataset.csv.gz`. Its annotations are in
`benchmark/queries.json`; `analysis/features.py` constructs the feature table.

| Models fitted by us | Training input | Training target / objective |
|---|---|---|
| Ridge, Random Forest, Gradient Boosting regressors | 31 engineered relevance features per query–second row. | Predict graded relevance (`rel_graded`). |
| Logistic Regression, k-nearest neighbours, Gradient Boosting classifiers | The same 31-feature dataset. | Predict whether the second belongs to the answer interval (`rel_binary`). |
| K-means and Gaussian mixture model | Eight selected features from the same dataset, standardized on the training fold. | Discover clusters without labels. Training labels subsequently calibrate cluster relevance. |
| Gaussian Hidden Markov Model | The same eight-feature subset, ordered into separate query–video sequences. | Learn Gaussian emissions and state transitions without labels. Training labels subsequently calibrate state relevance. |

**Dataset size and origin:** 28,900 labeled query–second rows, 86 annotated queries,
and nine source recordings from NASA news. There are 2,385 relevant rows (8.3%).
The source recordings are the material from which this training benchmark was
constructed. Their role here is established by the saved feature table and
annotations, rather than by which videos happen to appear in the web app.

**Training/evaluation split:** leave one entire source video out per fold. Each
model is fitted on rows from eight source videos and evaluated on the remaining
video, for nine folds. Preprocessing, model selection and relevance calibration
use the training fold only. The 28,900 rows describe the complete benchmark;
they are not all used for training in any one fold.

**Why this dataset:** the rows directly represent the relevance prediction task,
and their temporal order supports the HMM. The benchmark provides both graded
and binary labels, allowing the existing and new models to be compared. Labels
were derived from transcripts, so the benchmark favors speech and has limited
domain diversity.

**Web-app uploads:** ingestion extracts frames, transcripts and embeddings and
builds search indexes. Uploading a video does not fit these analysis models or
update the pretrained encoder weights. A source recording may also be available
as a demo, but that does not define the training dataset.

MSR-VTT and Charades appear in evaluation-harness code, but the saved experiments
have not been trained or evaluated on those datasets.


The review uses eight existing picture, speech, keyword, distance-to-peak,
scene-change and speech-presence features (`config.yaml: review.features`).
The encoders remain frozen. StandardScaler is fitted separately on each training
fold. PCA is used only for an illustrative plot; it is fitted on that plot's
training videos and is not a predictive preprocessing step.

## 2. Implementation of clustering (4 marks)

K-means minimizes within-cluster squared Euclidean distance on standardized
features. Candidate cluster counts are [2, 3, 4, 5]; the
highest training silhouette score selects k. Silhouette uses a fixed seeded
sample of at most 2,000 training rows. Gaussian mixture
models provide a second clustering method, using diagonal covariances and
training Bayesian Information Criterion (BIC; lower is better).

Neither clustering algorithm sees relevance labels during fitting or model
selection. For comparison with relevance predictors, each cluster is assigned its
relevant fraction using **training labels only**. K-means scores interpolate those
fractions using softmax of negative mean squared centroid distance; GMM scores
weight the fractions by component posterior probabilities. This calibration uses
labels: the complete retrieval systems are not purely unsupervised.

## 3. Implementation of a Hidden Markov Model (4 marks)

The Gaussian HMM is implemented in `analysis/hmm.py` with NumPy and SciPy-compatible
arrays, using a scikit-learn estimator interface. Observations are time-ordered
feature vectors; hidden states are learned patterns, not predefined relevance
labels. Parameters are the initial state probabilities, transition matrix, and
per-state Gaussian means/diagonal variances. K-means initializes the model;
Baum-Welch alternates scaled forward-backward expectation with parameter updates.
Variance floors and transition pseudocounts avoid degenerate distributions.

Candidate state counts [2, 3] are selected by training BIC.
Each (video, query) is a separate sequence; no transitions cross its boundary.
Training-label calibration maps state posteriors to relevance scores. Forward-
backward supplies smoothed scores; Viterbi supplies the most probable joint state
path, saved separately in the predictions. Relevance decisions threshold the
posterior-derived score; they are not the Viterbi state IDs.

The stopping criterion was met in 9/9 selected fold models.
The mean diagonal transition probability is 0.905; inspect the
heatmap and state profiles before assigning semantic meaning to states. State IDs
are arbitrary and can permute between folds.

## 4. Comparative performance analysis (4 marks)

Evaluation holds out one entire video at a time. Every scaler, cluster/state count,
model, relevance mapping and decision threshold is fitted or selected without that
video's labels. Calibration and F1 threshold choice use the training rows in-sample,
so this is not an independently calibrated probability estimate. Earlier saved
supervised predictions used grouped inner cross-validation; their thresholds were
selected with that earlier protocol. These calibration differences must be kept in
mind when comparing methods. Baselines retain the configured fixed cutoff.

| method | precision | recall | f1 | pr_auc | hit_at_1 | mrr | temporal_iou |
|---|---|---|---|---|---|---|---|
| K-means | 0.646 | 0.633 | 0.633 | 0.696 | 0.930 | 0.942 | 0.491 |
| Gaussian mixture | 0.630 | 0.521 | 0.557 | 0.556 | 0.769 | 0.853 | 0.419 |
| Gaussian HMM | 0.575 | 0.557 | 0.554 | 0.473 | 0.571 | 0.733 | 0.415 |
| Fixed-weight fusion | 0.381 | 0.763 | 0.500 | 0.604 | 0.881 | 0.920 | 0.365 |
| Picture only | 0.216 | 0.778 | 0.332 | 0.368 | 0.613 | 0.721 | 0.218 |
| Speech only | 0.469 | 0.638 | 0.529 | 0.599 | 0.902 | 0.919 | 0.408 |
| Logistic regression | 0.728 | 0.811 | 0.761 | 0.845 | 0.944 | 0.948 | 0.626 |
| k-nearest neighbours | 0.688 | 0.820 | 0.743 | 0.812 | 0.918 | 0.938 | 0.612 |
| Gradient boosting classifier | 0.709 | 0.805 | 0.747 | 0.832 | 0.930 | 0.937 | 0.609 |


Numbers are **macro averages over held-out videos**, including per-video means
over queries for retrieval and IoU. They are not the pooled/query-averaged numbers
from the earlier report. Hit@k asks whether a relevant second is in the top k
seconds, not the top k distinct clips. MRR is the reciprocal rank of the first
relevant second. Temporal IoU is intersection/union of all predicted relevant
seconds and true relevant seconds for a query; it is not top-moment IoU. Ranking
scores are rounded to 12 decimal places for ranking and
threshold-free metrics, avoiding floating-point noise on posterior plateaus.
Ranking ties then use shared seeded random permutations. PR-AUC is average precision.

`comparison.csv` includes standard deviations and 95% percentile bootstrap
intervals, resampling whole videos. Only nine independent held-out videos are
available, so uncertainty is substantial. Saved supervised predictions are joined
by unique video/query/time keys and checked against current labels before reuse.

## 5. Result interpretation and presentation (4 marks)

The HMM achieves F1 0.554, Hit@1 0.571, and temporal IoU
0.415. K-means achieves F1 0.633, Hit@1 0.930,
and temporal IoU 0.491. The HMM-minus-K-means F1 difference is
-0.079; its temporal IoU difference is -0.076.
Clustering optimizes feature geometry and the HMM optimizes sequence likelihood;
neither objective directly maximizes relevance F1 or retrieval rank. More persistent
states can make timelines smoother while also extending false positives. Improved
smoothness therefore does not automatically imply improved retrieval.

The timeline figure deliberately includes the lowest, middle and highest HMM IoU
queries, selected post-hoc to illustrate errors as well as successes. These examples
do not replace the full-dataset metrics. Study speech bias, correlated features
(a diagonal Gaussian assumption), small sample size, and local optima from one
seeded HMM initialization. Whole-video features and forward-backward use future
frames, so this is offline retrieval, not causal live streaming. Model selection on
correlated per-second rows makes BIC a practical heuristic, not proof of true states.

## Demo and reproduction

1. Open `notebooks/review_clustering_hmm.ipynb` and run all cells.
2. Explain the problem and class imbalance using `01_dataset.png`.
3. Show cluster selection, the PCA plot, and cluster profiles.
4. Explain transition/emission heatmaps, Baum-Welch convergence, and Viterbi.
5. Present `07_comparison.png`, then show a failure and success timeline.
6. Demonstrate the original app separately; the new models are analysis experiments
   and are not added to the app's live ranking controls.

Run `.venv/bin/python scripts/run_review.py` from the project root. Matching fold
artifacts are reused; `--force` reruns them. The manifest hashes the feature table,
settings, dependency versions and experiment/HMM source, preventing reuse after those change.

References: [NASA media guidance](https://www.nasa.gov/nasa-brand-center/images-and-media/),
[K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means),
[Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html),
[Rabiner's HMM tutorial](https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf).
