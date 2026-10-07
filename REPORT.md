# Needle: Cross-Modal Video Search

## 1. What it is

Needle finds moments inside a video from a plain-language query. You type
"baseball first pitch" or "astronauts with Elmo", and it returns ranked, timestamped
clips. It matches the query against both **what is shown** (the pictures) and
**what is said** (the speech), then combines the two.

The three encoders (CLIP, Whisper, MiniLM) are pretrained and never trained or
fine-tuned. The work is in the pipeline, the way the two signals are aligned in time
and fused, and the app built on top. Section 5 adds six small supervised models
(three regression, three classification) trained on features extracted from the
frozen encoders, to test whether a learned score can replace the hand-set fusion.

**Why two signals?** Pictures alone miss things that are only spoken ("advice for
becoming an astronaut"). Speech alone misses things that are only seen (a rover
arm, a baseball field). Each covers the other's blind spots.

## 2. How it works

```
 video ──► ffmpeg ──► 1 frame/second ──► CLIP image encoder ──┐
   │                                                           ├─► FAISS indexes
   └────► audio ──► Whisper (speech-to-text) ──► MiniLM text ──┘   (one per modality)

 query ──► CLIP text encoder ──► search picture index  ─┐
   └─────► MiniLM encoder   ──► search speech index    ─┴─► fuse ─► group ─► clips
```

| Stage | Model / tool | Output |
|---|---|---|
| Frames | ffmpeg, 1 per second, 336 px wide | one image per second |
| Picture embedding | OpenCLIP ViT-B/32 (laion2b) | 512-d unit vector per frame |
| Speech to text | faster-whisper `base`, voice-activity filter | timestamped segments |
| Speech embedding | all-MiniLM-L6-v2 | 384-d unit vector per segment |
| Index | FAISS `IndexFlatIP` (exact search) | cosine similarity |

All vectors are L2-normalized, so inner product equals cosine similarity.
FAISS row order is asserted to match the metadata arrays. Every stage is
resumable: re-running ingest on a finished video takes under a second.

### Query time: fuse and group

1. **Score.** The query is encoded by CLIP and MiniLM. Each index returns its top
   200 matches. Scores are spread onto a per-second timeline (0 where not retrieved).
2. **Align.** Speech scores belong to segments, not seconds. Each frame takes the
   score of the segment covering its timestamp. Frames in silence get 0, because
   voice-activity filtering leaves gaps.
3. **Calibrate.** Each modality is min-max normalized to [0, 1] on its own. CLIP
   similarities cluster near 0.2–0.3 while text similarities spread wider, so
   fusing raw values would let one modality dominate.
4. **Fuse.** `score = α · picture + (1 − α) · speech`, with α = 0.6 by default and
   adjustable live.
5. **Group.** Seconds scoring at least the cutoff (0.5) are merged when they are
   less than 2 s apart. A moment's score is the **max** of its seconds. Single-second
   hits are padded to 1 s. Without grouping, the top ten results would be ten
   consecutive seconds of one scene.

A video with no speech is valid input: the speech score is simply 0 everywhere.

## 3. The application

- **Backend:** FastAPI. Search, byte-range video serving (needed for seeking),
  thumbnails, upload with background indexing and per-stage progress, remove.
- **Frontend:** Next.js (React, TypeScript, plain CSS).
- **Behavior:** the full video plays until you search. A search replaces it with the
  best clip only, with a bounded timeline that stops at the clip's end. Previous /
  Next and the result list move between clips, and "Show full video" returns.
- **Relevance trace:** every second of the video is plotted, split into picture
  (blue) and speech (amber), with numbered result brackets and the cutoff line.
  Dragging the weight slider re-ranks results live.
- **Upload:** drag in a video, watch seven steps complete, then search it.
- **CLI:** `scripts/ingest.py` and `scripts/query.py` for terminal use.

## 4. Measured behavior

Measured on an Apple M4 laptop, CPU only:

| Item | Result |
|---|---|
| Index a 10:47 video | about 76 s (Whisper 44 s, CLIP 13 s, frames 12 s, text and index 7 s) |
| Upload and index a 3:48 video | 22 s end to end |
| Re-ingest a finished video | under 1 s |
| Search latency | about 25–120 ms |
| Automated tests | 44 passing (fusion, grouping, index, pipeline, API, upload, byte ranges, benchmark labels, features, cross-validation) |

Qualitative check on a public-domain NASA news episode: "baseball first pitch"
returns the 10:16 ceremonial pitch first; "Mars rover robotic arm" returns the
1:25 shot of the rover arm; "astronauts with Elmo" lands at about 2:40, where the
narration says it. Moving the slider to speech-only or picture-only changes the
ranking as expected.

## 5. Regression and classification analysis

**Goal.** Replace the hand-set fusion weight with models learned from labeled examples.
The encoders stay frozen; six small scikit-learn models are trained on features computed
from them. One row is one (query, second of video).

- **Data:** 9 public-domain NASA episodes (about 45 min) and 86 hand-written queries, each
  with the time interval of its true answer (`benchmark/queries.json`). Intervals were
  derived from transcripts, so labels lean toward speech. 28,900 rows; 8.3% are positive.
- **Features (31):** 15 original (raw and normalized picture and speech similarity, per-video
  z-scores and ranks, +/-2 s smoothed scores, frame-to-frame change, speech present, position,
  query length, the hand-set fused score) plus 16 context features (+/-5 s and +/-10 s windows,
  distance in seconds to each score's peak and the score relative to that peak, local rank,
  scene cuts nearby, overlap between query words and spoken words).
- **Targets:** regression, graded relevance (1 inside the true interval, decaying with
  distance); classification, is the second inside the true interval.
- **Models:** regression: Ridge, Random forest, Gradient boosting. Classification: Logistic
  regression, k-nearest neighbours, Gradient boosting. Each is compared with a mean/prior baseline.
- **Protocol:** leave-one-video-out (every prediction comes from models that never saw that
  video). Hyper-parameters, the width of the temporal smoothing applied to each model's output,
  and the F1 threshold are all chosen on the training videos only.

| Regression (mean over 9 held-out videos) | RMSE | MAE | R² |
|---|---|---|---|
| Mean baseline | 0.319 | 0.204 | -0.03 |
| Ridge | 0.198 | 0.129 | 0.59 |
| Random forest | 0.154 | 0.069 | **0.74** |
| Gradient boosting | 0.154 | 0.070 | **0.74** |

| Classification (mean over 9 held-out videos) | Precision | Recall | F1 | Balanced acc. | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Prior baseline | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.10 |
| Logistic regression | 0.73 | 0.81 | **0.76** | 0.89 | 0.98 | **0.85** |
| k-nearest neighbours | 0.69 | 0.82 | 0.74 | 0.89 | 0.98 | 0.81 |
| Gradient boosting | 0.71 | 0.81 | 0.75 | 0.88 | 0.98 | 0.83 |

Pooled over all held-out seconds, classification accuracy is 95.7% (logistic), 95.3% (gradient
boosting) and 95.1% (kNN). Accuracy alone is not the headline metric: always answering "not in
the moment" already scores 91.7% because only 8.3% of seconds are positive.

**What improved the models** (all numbers pooled, held-out videos; the first rows were re-run
from scratch and reproduce the earlier results):

| Variant | Regression R² (RF) | Logistic F1 | Logistic PR-AUC | Logistic Hit@1 |
|---|---|---|---|---|
| Original 15 features | 0.39 | 0.64 | 0.70 | 0.91 |
| + context features | 0.70 | 0.75 | 0.82 | 0.92 |
| + temporal smoothing of outputs | 0.72 | 0.75 | 0.82 | 0.93 |

Almost all of the gain comes from the context features; smoothing adds a little. For regression
every fold picked the widest smoothing width offered (15 s), so a wider option may help slightly.

Ranking seconds of each held-out query (86 queries; Hit@k = a true second is among the
top k; 95% bootstrap interval over queries in brackets):

| Method | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| Random | 0.08 (0.07–0.10) | 0.37 | 0.23 |
| Picture only | 0.60 (0.50–0.71) | 0.85 | 0.71 |
| Speech only | 0.90 (0.83–0.95) | 0.92 | 0.91 |
| Hand-set fusion (α = 0.6) | 0.87 (0.80–0.94) | 0.97 | 0.91 |
| Ridge regression | 0.93 (0.87–0.98) | 0.94 | 0.94 |
| Random forest regression | 0.92 (0.85–0.97) | 0.94 | 0.93 |
| Gradient boosting regression | 0.91 (0.84–0.97) | 0.94 | 0.92 |
| Logistic regression | 0.93 (0.87–0.98) | 0.94 | 0.94 |
| k-nearest neighbours | 0.91 (0.84–0.97) | 0.94 | 0.93 |
| Gradient boosting classifier | 0.92 (0.86–0.98) | 0.94 | 0.93 |

**Findings.**
1. All six models clearly beat their baselines. With context features, regression reaches R² of
   0.74 and classification F1 of 0.76, PR-AUC 0.85 and balanced accuracy 0.89.
2. Context matters more than the model: distance to the speech peak, wider windows and per-video
   z-scores are the most important features; raw similarity is least. Random forest and gradient
   boosting tie; the linear model is clearly worse for regression but about equal for classification.
3. Learned rankers put a true second first in 91 to 93% of queries versus 87% for hand-set fusion.
   The direction is consistent but the 95% intervals (about +/-0.07) overlap, so the improvement
   is **not statistically established** on 86 queries.
4. Speech-only matches fusion partly because labels were written from transcripts.
5. Held-out scores keep improving as training videos are added (logistic PR-AUC 0.79 with one
   video, 0.85 with six), with diminishing returns, so more labeled videos would likely help.

19 charts (class balance, feature distributions, correlations, predicted vs actual, residuals,
confusion matrices, ROC, precision-recall, importances, learning curves, thresholds, retrieval
comparison, alpha sweep, per-video results, and the improvement ablation) are in
`results/ml/figures/`; tables are in `results/ml/`; `notebooks/analysis.ipynb` walks through them.

## 6. Evaluation status

The harness is implemented (`eval/`): MSR-VTT clip retrieval (Recall@1/5/10,
median rank; mean vs max frame pooling), Charades moment retrieval (R@1 at
IoU 0.5 and 0.7, mean IoU), and ablations (picture-only, speech-only, an α sweep,
pooling, ViT-B/32 vs ViT-L/14) with the α-sweep plot. It is unit-tested on
synthetic data.

**It has not been run on MSR-VTT or Charades**, so this report contains no
results on those benchmarks (those datasets must be obtained under their own licenses).
Section 5 reports retrieval results on our own labeled benchmark instead.

## 7. Limitations and next steps

- **Top result is always about 1.0.** Min-max normalization is relative to each
  query, so the best moment always scores 1.0, even when nothing truly matches.
  A fix would normalize against fixed similarity floors, or add an absolute
  relevance check.
- **Only the top 200 frames per modality are scored**, so very long videos
  (thousands of frames) are covered sparsely.
- **One frame per second** can miss fast events and rounds clip edges to whole seconds.
- **CPU only on Apple Silicon.** The code uses a GPU only when CUDA is present.
- **Speech quality limits speech search.** Whisper `base` mishears names and jargon.
- **Small, biased labeled set.** The supervised analysis uses 86 queries over 9 videos from one
  content domain (NASA news), with labels derived from transcripts. Confidence intervals are wide (R² varies by about +/-0.13 across videos)
  and the labels favour speech. A benchmark labeled by watching the picture, more videos, and
  other domains (lectures, sports, silent footage) are the next steps.
- **Learned ranking beats hand-set fusion by a few points, but not significantly** on this data
  (section 5). The models show the engineered context features carry strong signal.
- **Not run on MSR-VTT or Charades yet** (section 6).

## 8. Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # also needs ffmpeg on PATH
python scripts/fetch_demo.py              # optional demo video
uvicorn server.main:app --port 8000
cd web && npm install && npm run dev      # http://localhost:3000

# regression / classification analysis (about 20 minutes on a laptop; folds run in parallel)
python scripts/fetch_benchmark.py         # needs yt-dlp; 7 more episodes
python scripts/ingest.py --dir data/raw
python scripts/run_analysis.py            # tables and 19 charts in results/ml/
pytest                                    # 44 tests
```

Code: `cmvs/` (pipeline), `server/` (API), `web/` (frontend), `analysis/` (regression and
classification), `eval/` (benchmarks), `benchmark/` (labels), `notebooks/`, `tests/`. Settings
live in `config.yaml`.
