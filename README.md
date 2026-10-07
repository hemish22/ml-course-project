# Cross-Modal Video Search

Searches inside videos using frozen CLIP frame embeddings and Whisper transcript
segments. Visual and transcript scores are independently calibrated, fused on a
frame timeline, and grouped into timestamped moments. This project contains no
training, fine-tuning, or LoRA workflows.

## Setup

Use Python 3.10 on the GPU machine, ensure `ffmpeg` and `ffprobe` are on
`PATH`, then install the project requirements:

```bash
pip install -r requirements.txt
```

Set all paths, models, inference settings, and retrieval thresholds in
`config.yaml`. The first actual inference downloads the configured model weights.

## Run

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
python scripts/run_analysis.py               # features -> 6 models -> tables -> 18 charts
```

Labels are in `benchmark/queries.json`. The run takes about 10 minutes
(leave-one-video-out with hyper-parameter search); `--step plots` redraws charts only.

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
