# Cross-Modal Video Search

Searches inside videos using frozen CLIP frame embeddings and Whisper transcript
segments. Visual and transcript scores are independently calibrated, fused on a
frame timeline, and grouped into timestamped moments. The pretrained encoders
are never fine-tuned.

The CLIP, Whisper, and MiniLM encoders remain frozen. The `analysis/` package
fits small regression, classification, clustering, and HMM models on their
existing features.

## Review 3 (PBL-3): ensemble learning

[The executed notebook](notebooks/pbl3_ensemble.ipynb) and
[the generated report](results/ensemble/ENSEMBLE_REPORT.md) cover the four PBL-3 criteria.
`analysis/ensemble.py` fits CART, bagging, random forest, extra trees, AdaBoost,
histogram gradient boosting, soft voting, stacking and Bayesian model averaging
(pseudo-BMA+) on the same 31 features and leave-one-video-out folds as Reviews 1–2,
with grouped inner cross-validation for CART depth, stacking, BMA weights, smoothing
and thresholds.

```bash
python scripts/run_ensemble.py               # ~10 min on 9 cores; cached folds are reused
python scripts/build_ensemble_notebook.py    # rebuilds and executes the notebook
scripts/render_report_pdf.sh "<template.docx>"  # integrated Word/PDF report in report/
```

Mean held-out-video F1: Bayesian model averaging **0.771** (PR-AUC 0.853), stacking
0.765, soft voting 0.763, CART 0.761, random forest 0.755, versus logistic regression
0.761 (0.845). The paired gain over logistic regression is small (+0.010 F1, 7/9
videos) and only marginally significant.

## Review 2: clustering and Hidden Markov Models

Start with **[the executed review notebook](notebooks/review_clustering_hmm.ipynb)**
or open **[the browser presentation](results/review/review.html)** directly.
[The review report](results/review/REVIEW_REPORT.md) follows the five 4-mark criteria.
The earlier notebook and results remain available separately.

The new experiments compare K-means, Gaussian mixtures, and a Gaussian HMM with
the original baselines and classifiers, holding out entire videos. The HMM is
implemented directly in `analysis/hmm.py` (Baum-Welch, forward-backward, Viterbi).
The models use training labels only to interpret clusters/states as relevance;
their fitting and cluster/state-count selection are unsupervised.

```bash
# With the existing feature table: no video download or encoder inference needed
python -m pip install -r requirements-review.txt
python scripts/run_review.py              # reuses matching cached folds
python scripts/execute_review_notebook.py # executes and saves rich notebook outputs
```

Mean held-out-video F1: K-means **0.633**, Gaussian mixture **0.557**, Gaussian HMM
**0.554**, fixed fusion **0.500**, and earlier logistic regression **0.761**.
The HMM does not outperform K-means or the earlier classifier on this benchmark.
See the report for calibration differences, uncertainty, metrics, and failure cases.
These models are analysis experiments; the app continues to use its existing fusion.

## Setup

Use Python 3.10+ (the local review was validated with 3.11), ensure `ffmpeg` and `ffprobe` are on
`PATH`, then install the project requirements:

```bash
pip install -r requirements.txt
```

Set all paths, models, inference settings, and retrieval thresholds in
`config.yaml`. The first actual inference downloads the configured model weights.

## Run

On macOS, `./run-local.command` installs missing local dependencies, starts both
servers, waits for startup, and opens the browser. It uses `.venv/` and stores
logs under `.runtime/`. The first setup needs internet access. Run it from your
own Terminal; restricted agent sessions may prohibit localhost sockets.

To start the components manually:

Index a video, then start the API and the web app (two terminals):

```bash
python scripts/fetch_demo.py                 # optional: public-domain NASA episode (about 11 min)
python scripts/ingest.py --video data/raw/lecture.mp4
uvicorn server.main:app --port 8000          # API and video/thumbnail server

cd web && npm install && npm run dev         # http://localhost:3000
```

Search from the terminal instead:

```bash
python scripts/query.py --q "person writing on the whiteboard" --video lecture --alpha 0.7
```

Ingestion is resumable: rerun the first command to reuse complete artifacts, or
add `--force` to rebuild them.

### Web app

The full video is shown until you search. A search swaps the player for the
best-matching clip only (bounded timeline, auto-play, stops at the end); pick
other results from the list or the timeline, or choose *Show full video* to go
back. The strip under the player shows every second of the video scored by
picture (blue) and speech (amber); the slider re-weights the two live.

Set `NEXT_PUBLIC_API_URL` in `web/.env.local` if the API is not on
`http://localhost:8000`. On macOS the entry points set `OMP_NUM_THREADS=1`
because FAISS and PyTorch otherwise crash on duplicate OpenMP runtimes.

## Regression and classification analysis

Three regression and three classification models are trained on features from the
frozen pipeline to predict, for every second of a video, how relevant it is to a query
(`analysis/`, results in `results/ml/`, walkthrough in `notebooks/analysis.ipynb`).

```bash
python scripts/fetch_benchmark.py            # needs yt-dlp; downloads 7 public-domain episodes
python scripts/ingest.py --dir data/raw      # index them (about 5 minutes)
python scripts/run_analysis.py               # features -> 6 models -> tables -> 19 charts
```

Labels are in `benchmark/queries.json`. The run takes roughly 10 to 20 minutes
(leave-one-video-out with hyper-parameter search; folds run in parallel, see `ml.n_jobs`); `--step plots` redraws charts only.

## Evaluation

The evaluation code intentionally does **not** download MSR-VTT or Charades.
Obtain the datasets under their respective licenses, ingest their videos, and
pass the official ordered split records to `eval.msrvtt.evaluate_msrvtt` or
`eval.charades.evaluate_charades`. Results are written under the configured
`paths.results` directory. `eval.ablations.run_ablations` creates
`ablations.csv` and `alpha_sweep.png`; encoder comparisons require a callback
that reindexes the corpus per configured CLIP model.

## Validation

```bash
pytest
```
