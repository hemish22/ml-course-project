# Needle: Professor Review Presentation Script

**Duration:** approximately 5–7 minutes.

Use `notebooks/review_clustering_hmm.ipynb` as the main presentation. Follow the
professor's five criteria, then demonstrate the web application.

## Opening — 20 seconds

> Our project is Needle, a video-search system. A user enters a natural-language
> query, and the system finds relevant moments inside a video. It combines visual
> information with spoken content.
>
> For this review, we investigate whether clustering and temporal modeling can
> identify relevant video segments.

## 1. Problem Identification and Dataset Selection — 4 marks

**Time:** approximately 1 minute.

**Show:** the notebook's dataset section.

> The problem is that searching a long video manually takes time. Visual search
> can identify what appears on screen, while speech search can identify concepts
> mentioned in the audio.
>
> We distinguish between the pretrained models' training datasets and the dataset
> used for our own experiments.
>
> OpenCLIP was pretrained on LAION-2B English image–text pairs. Whisper was trained
> on a large multilingual audio–transcript corpus. MiniLM's sentence-embedding
> training used a mixture of over one billion sentence pairs. We reuse these
> models with their weights frozen.
>
> Our own models are fitted on a labeled relevance dataset containing 28,900
> query–second observations and 86 annotated queries, constructed from nine source
> recordings.
>
> Each row describes how relevant one second is to a particular query. Videos
> uploaded to the web app are indexed for searching; uploading does not train the
> models.

**Training-data sources:**

- [OpenCLIP model card](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)
- [Whisper training data](https://github.com/openai/whisper/blob/main/model-card.md#training-data)
- [MiniLM sentence-embedding training data](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#training-data)
- [Local dataset and model provenance](DATASETS.md)

## 2. Implementation of Clustering Algorithm — 4 marks

**Time:** approximately 1 minute.

**Show:** model selection, the cluster projection, and cluster profiles.

> We implemented K-means and a Gaussian mixture model.
>
> K-means groups observations with similar features by assigning them to nearby
> cluster centers. Gaussian mixtures provide soft membership probabilities.
>
> We standardize eight selected features using training data. We compare different
> cluster counts, choosing K-means clusters using silhouette scores and Gaussian
> mixture components using BIC.
>
> Cluster numbers do not automatically mean relevant or irrelevant. We use
> training labels afterward to estimate how strongly each cluster is associated
> with relevance.

**Point at the cluster-profile figure:**

> This figure helps explain what distinguishes the clusters—for example, their
> visual similarity, speech similarity, or keyword overlap.

## 3. Implementation of Hidden Markov Model — 4 marks

**Time:** approximately 1 minute.

**Show:** the transition heatmap and Viterbi figure.

> Clustering treats observations independently. A video also has temporal
> structure: neighboring seconds often belong to the same scene or topic.
>
> We implemented a Gaussian Hidden Markov Model. The observed values are our
> per-second features, and the hidden states represent learned patterns.
>
> The model learns initial-state probabilities, transition probabilities between
> states, and Gaussian feature distributions for each state.
>
> We implemented Baum–Welch for fitting, forward–backward for state probabilities,
> and Viterbi for the most probable state sequence.
>
> Each query–video timeline is a separate sequence, so transitions never cross
> between unrelated queries or videos.

**Point to a diagonal heatmap entry:**

> This represents the probability of staying in the same state at the next second.

## 4. Comparative Performance Analysis — 4 marks

**Time:** approximately 1 minute.

**Show:** the comparison chart.

> We evaluate by holding out an entire video at a time. Models train on eight
> source videos and are evaluated on the remaining one, repeated across all nine
> videos.
>
> This prevents seconds from the same video appearing in both training and
> testing.
>
> We measure precision, recall, F1, average precision, retrieval ranking, and
> temporal overlap. Accuracy alone would be misleading because only about 8.3%
> of observations are relevant.

**Measured mean held-out-video F1:**

| Method | F1 |
|---|---:|
| Fixed-weight fusion | 0.500 |
| K-means | 0.633 |
| Gaussian mixture | 0.557 |
| Gaussian HMM | 0.554 |
| Existing logistic regression | 0.761 |

> The earlier supervised classifier remains the strongest of these methods on
> F1. The report also documents differences in how the methods' decision thresholds
> were selected.

## 5. Result Interpretation and Presentation — 4 marks

**Time:** approximately 1 minute.

**Show:** one successful timeline and one failure.

> The results show that adding a temporal model does not automatically improve
> relevance prediction.
>
> The HMM learns feature patterns and transitions, while our evaluation measures
> relevance to a query. These objectives may not align perfectly.
>
> Possible limitations include correlated features, the Gaussian assumptions,
> and the small dataset. Our annotations were derived from transcripts, which
> also favors speech-based methods.
>
> The useful outcome is an implemented and evaluated comparison: we can explain
> where each method succeeds and where it fails.

## Web-App Demo — 30 seconds

> The application demonstrates the original retrieval pipeline. I'll select a
> video, enter a query, and show the returned timestamps and clips. The new
> clustering and HMM experiments are demonstrated in the notebook.

**Demo sequence:**

1. Select an indexed video.
2. Enter a query about something visible or spoken in that video.
3. Show a returned timestamp and play the matching clip.
4. Point out the separate visual and speech scores.

## Likely Viva Questions

### Did you train CLIP, Whisper, and MiniLM?

> We use their pretrained weights. We fit the smaller comparison, clustering,
> and HMM models ourselves.

### Why use an HMM?

> To model dependencies between neighboring seconds.

### Why is the HMM worse?

> Its learned states optimize sequence likelihood, which may not align with query
> relevance. The dataset size and modeling assumptions may also contribute.

### Is clustering completely unsupervised?

> Cluster fitting and cluster-count selection are unsupervised. Mapping clusters
> to relevance uses training labels.

### What would you improve next?

> Add more diverse recordings and independently annotated answer intervals, then
> evaluate on a separate benchmark.

## Presentation Reminders

- Describe the implementation, comparison, and measured results clearly.
- Distinguish external pretraining, models fitted in this project, and web-app
  inputs.
- Explain that the HMM performed worse on this benchmark; do not claim it improved
  relevance performance.
- Describe the encoders as pretrained and frozen; do not claim we trained them.
- Explain that the new clustering/HMM models are notebook experiments; the live
  app uses the existing fusion pipeline.

## Files to Keep Open

- [Review notebook](notebooks/review_clustering_hmm.ipynb)
- [Browser presentation](results/review/review.html)
- [Detailed review report](results/review/REVIEW_REPORT.md)
- [Training dataset explanation](DATASETS.md)

**Reference for the table:** [saved comparison results](results/review/comparison.csv).
If experiments are rerun with different settings or data, refresh the numbers in
this script before presenting.
