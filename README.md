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

```bash
python scripts/ingest.py --video data/raw/lecture.mp4
python scripts/query.py --q "person writing on the whiteboard" --video lecture --alpha 0.7
streamlit run app.py
```

Ingestion is resumable: rerun the first command to reuse complete artifacts, or
add `--force` to rebuild them.

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
