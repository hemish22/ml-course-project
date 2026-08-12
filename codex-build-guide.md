# Build Guide for Codex — Cross-Modal Video Search

This document is the complete specification. Hand it to Codex whole for context, then issue the tasks in Section 10 **one at a time**. Do not ask for the entire system in a single prompt — agents produce better code against narrow, testable tasks.

---

## 1. What Is Being Built

A search engine that operates *inside* video files. The user submits a natural-language query; the system returns ranked, timestamped moments from an indexed video library.

Two modalities are indexed independently and fused at query time:

- **Visual** — frames sampled at 1 fps, embedded with CLIP into a shared image-text space.
- **Transcript** — speech transcribed by Whisper into timestamped segments, embedded with a sentence encoder.

A query is embedded by both encoders, matched against both indices, and the two score streams are fused on a common timeline before contiguous high-scoring frames are grouped into moments.

**Explicit non-goals.** No model training. No fine-tuning. No custom architecture. All encoders are used frozen and off the shelf. The engineering value is in the pipeline, the timeline alignment, the fusion, and the evaluation harness.

---

## 2. Environment

Target **Python 3.10**. Assume a CUDA GPU may or may not be present — every module must fall back to CPU without code changes.

```
torch>=2.1
open_clip_torch>=2.24
faster-whisper>=1.0
faiss-cpu>=1.7.4
sentence-transformers>=2.5
numpy>=1.24
pillow>=10.0
streamlit>=1.31
pandas>=2.0
matplotlib>=3.7
tqdm>=4.66
scenedetect>=0.6.2
pyyaml>=6.0
pytest>=7.4
```

System binary required: **ffmpeg** on PATH. The code calls it via `subprocess`, not via a Python wrapper — wrappers add a dependency and obscure error messages.

Models, downloaded on first use:

| Purpose | Model |
|---|---|
| Image + query encoding | `open_clip` `ViT-B-32`, pretrained `laion2b_s34b_b79k` |
| Speech recognition | `faster-whisper` `base` (configurable to `small`) |
| Transcript + query encoding | `sentence-transformers/all-MiniLM-L6-v2` |

---

## 3. Repository Structure

```
cross-modal-video-search/
├── AGENTS.md
├── README.md
├── requirements.txt
├── config.yaml
├── app.py                      # Streamlit entry point
├── cmvs/
│   ├── __init__.py
│   ├── config.py               # config loading, dataclass
│   ├── extract.py              # ffmpeg: frames + audio
│   ├── encode_visual.py        # CLIP frame embeddings
│   ├── transcribe.py           # Whisper -> segments
│   ├── encode_text.py          # sentence embeddings for segments
│   ├── index.py                # FAISS build / load / search
│   ├── fuse.py                 # timeline alignment + score fusion
│   ├── group.py                # frames -> moments
│   ├── search.py               # top-level query API
│   ├── pipeline.py             # orchestration: ingest one video
│   └── utils.py                # io, logging, device, timing
├── eval/
│   ├── msrvtt.py               # clip retrieval benchmark
│   ├── charades.py             # moment retrieval benchmark
│   ├── ablations.py            # alpha sweep, pooling, encoder swap
│   └── plots.py
├── scripts/
│   ├── ingest.py               # CLI: index a video
│   └── query.py                # CLI: search from terminal
├── tests/
└── data/                       # gitignored
    ├── raw/
    ├── processed/
    └── index/
```

---

## 4. On-Disk Data Contracts

These schemas are fixed. Every module reads and writes exactly these shapes.

### `data/processed/{video_id}/frames.jsonl`
One JSON object per line, ordered by `frame_idx`:
```json
{"frame_idx": 0, "timestamp": 0.0, "path": "data/processed/vid001/frames/000000.jpg"}
```

### `data/processed/{video_id}/transcript.jsonl`
```json
{"seg_idx": 0, "start": 0.0, "end": 4.12, "text": "today we look at gradient descent"}
```

### `data/index/{video_id}/`
```
visual.faiss           # IndexFlatIP, dim 512, L2-normalised vectors
visual_meta.json       # {"frame_idx": [...], "timestamp": [...], "path": [...]}
transcript.faiss       # IndexFlatIP, dim 384
transcript_meta.json   # {"seg_idx": [...], "start": [...], "end": [...], "text": [...]}
meta.json              # {"video_id", "source_path", "duration", "fps_sampled",
                       #  "n_frames", "n_segments", "clip_model", "whisper_model",
                       #  "text_model", "built_at"}
```

**Row order in the FAISS index must match array order in the meta JSON.** Every lookup depends on this. Assert it at build time.

### Search result object
```python
@dataclass
class Moment:
    video_id: str
    start: float            # seconds
    end: float              # seconds
    score: float            # fused, 0-1
    visual_score: float
    transcript_score: float
    thumbnail_path: str     # frame nearest the peak score
    transcript_text: str    # concatenated overlapping segments, may be ""
```

---

## 5. Configuration

`config.yaml`, loaded once into a frozen dataclass by `cmvs/config.py`. No magic numbers anywhere else in the codebase.

```yaml
paths:
  raw: data/raw
  processed: data/processed
  index: data/index

extract:
  fps: 1.0
  frame_width: 336          # resize longest side; keeps disk small
  jpeg_quality: 88
  use_scene_detect: false
  audio_sample_rate: 16000

models:
  clip_name: ViT-B-32
  clip_pretrained: laion2b_s34b_b79k
  whisper_size: base
  whisper_compute_type: int8   # "float16" when CUDA is available
  text_model: sentence-transformers/all-MiniLM-L6-v2

encode:
  batch_size: 64

search:
  alpha: 0.6                # weight on visual; (1-alpha) on transcript
  top_k_frames: 200         # candidate frames pulled from FAISS before fusion
  score_threshold: 0.28     # min fused score to enter a moment
  merge_gap_s: 2.0          # frames closer than this join one moment
  min_moment_s: 1.0
  max_results: 10
```

---

## 6. Module Specifications

Write type hints on every public function. Keep modules import-light — no module should import Streamlit.

### `cmvs/utils.py`
```python
def get_device() -> str                       # "cuda" if available else "cpu"
def slugify(name: str) -> str                 # source filename -> video_id
def read_jsonl(path: Path) -> list[dict]
def write_jsonl(path: Path, rows: Iterable[dict]) -> None
def l2_normalize(x: np.ndarray) -> np.ndarray # row-wise, float32, eps-guarded
@contextmanager
def timed(label: str)                         # logs elapsed seconds
```

### `cmvs/extract.py`
```python
def extract_frames(video_path: Path, out_dir: Path, cfg) -> list[dict]
def extract_audio(video_path: Path, out_wav: Path, cfg) -> Path
def probe_duration(video_path: Path) -> float
```
- Frames: one `ffmpeg` call with `-vf fps={fps},scale='min({w},iw)':-2`, output `%06d.jpg`.
- Timestamp for frame *n* (zero-indexed) is `n / fps`. ffmpeg's `fps` filter emits frames on that grid, so this is exact enough for 1 fps. Do not attempt per-frame PTS parsing.
- Audio: `-ac 1 -ar 16000 -vn` to WAV.
- `probe_duration` shells out to `ffprobe`. If it fails, fall back to `n_frames / fps`.
- Raise a clear `RuntimeError` naming `ffmpeg` if the binary is missing — this is the single most common setup failure.

### `cmvs/encode_visual.py`
```python
def load_clip(cfg) -> tuple[nn.Module, Callable, Any]   # model, preprocess, tokenizer
def encode_images(paths: list[str], cfg) -> np.ndarray  # (N, 512) float32, normalized
def encode_query_visual(text: str, cfg) -> np.ndarray   # (1, 512) float32, normalized
```
- Batch by `cfg.encode.batch_size`, wrap in `torch.no_grad()`, `tqdm` over batches.
- Normalize inside these functions, once. Downstream code must be able to assume unit vectors.
- Cache the loaded model in a module-level singleton — reloading CLIP per call makes the UI unusable.

### `cmvs/transcribe.py`
```python
def transcribe(wav_path: Path, cfg) -> list[dict]   # frames.jsonl-style segment rows
```
- `WhisperModel(size, device=..., compute_type=...)`, `vad_filter=True`.
- Return segment-level rows only. Word timestamps are not needed and slow things down.
- If the audio track is absent or silent, return `[]`. Every downstream module must handle a video with zero transcript segments — that is a normal case, not an error.

### `cmvs/encode_text.py`
```python
def encode_segments(texts: list[str], cfg) -> np.ndarray    # (N, 384) normalized
def encode_query_text(text: str, cfg) -> np.ndarray         # (1, 384) normalized
```
Same singleton-caching rule.

### `cmvs/index.py`
```python
def build_index(vectors: np.ndarray) -> faiss.Index          # IndexFlatIP
def save_index(index, meta: dict, dir: Path, kind: str) -> None
def load_index(dir: Path, kind: str) -> tuple[faiss.Index, dict]
def search(index, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]
```
- `IndexFlatIP` on normalized vectors makes inner product equal cosine similarity. Exact search is correct here — the corpus is thousands of vectors, not billions. Do not add IVF or HNSW.
- Guard `k > index.ntotal`; FAISS returns `-1` ids for the overflow and unguarded code will index out of range.

### `cmvs/fuse.py`
This is the module that makes the project multimodal. Get it right.

```python
def align_transcript_to_frames(
    frame_timestamps: np.ndarray,     # (F,)
    seg_starts: np.ndarray,           # (S,)
    seg_ends: np.ndarray,             # (S,)
    seg_scores: np.ndarray,           # (S,)
) -> np.ndarray:                      # (F,) transcript score per frame
```
For each frame timestamp, find the segment whose `[start, end)` interval contains it and take that segment's score. Frames in silence get `0.0`. Use `np.searchsorted` on `seg_starts`, then verify the candidate's `end` actually covers the timestamp — do not assume segments are contiguous, because VAD filtering removes silence and leaves gaps.

```python
def minmax_normalize(scores: np.ndarray) -> np.ndarray
def fuse_scores(visual: np.ndarray, transcript: np.ndarray, alpha: float) -> np.ndarray
```
Normalize each modality independently to `[0, 1]` **before** weighting — raw CLIP cosine similarities cluster around 0.2–0.3 while sentence-embedding similarities spread much wider, so unnormalized fusion is dominated by whichever modality has the larger variance. If a modality's scores are all-constant (or the array is empty), return zeros rather than dividing by zero.

`fuse_scores` returns `alpha * visual + (1 - alpha) * transcript`.

### `cmvs/group.py`
```python
def group_frames_into_moments(
    timestamps: np.ndarray,
    fused: np.ndarray,
    visual: np.ndarray,
    transcript: np.ndarray,
    cfg,
) -> list[dict]
```
1. Keep frames with `fused >= cfg.search.score_threshold`.
2. Sort by timestamp; start a new moment when the gap to the previous kept frame exceeds `cfg.search.merge_gap_s`.
3. A moment's `score` is the **max** fused score of its frames, not the mean — mean punishes long moments that contain one perfect match.
4. Drop moments shorter than `min_moment_s`, but pad them to that length first rather than discarding a strong single-frame hit: expand symmetrically around the peak.
5. Sort by score descending, truncate to `max_results`.

Without this step the top ten results are ten consecutive frames of one scene. Make sure the tests cover that case.

### `cmvs/search.py`
```python
def search_video(query: str, video_id: str, cfg, alpha: float | None = None) -> list[Moment]
def search_library(query: str, cfg, alpha: float | None = None) -> list[Moment]
```
Flow for one video: encode query twice → FAISS search both indices for `top_k_frames` → scatter sparse FAISS scores into dense per-frame and per-segment arrays (unretrieved entries are 0) → align transcript to frames → normalize → fuse → group.

`alpha` overrides config, so the UI slider and the ablation harness can both drive it without rewriting config.

`search_library` runs `search_video` across every indexed video and merges results by score.

### `cmvs/pipeline.py`
```python
def ingest(video_path: Path, cfg, force: bool = False) -> str   # returns video_id
```
Stages: extract frames → extract audio → encode frames → transcribe → encode segments → build both indices → write `meta.json`.

**Every stage must be resumable.** Check whether its output already exists and skip unless `force=True`. Colab disconnects mid-job constantly, and a pipeline that restarts from zero each time will burn the project's schedule.

---

## 7. CLI

```bash
python scripts/ingest.py --video data/raw/lecture.mp4
python scripts/ingest.py --dir data/raw --force
python scripts/query.py --q "person writing on the whiteboard" --video lecture --alpha 0.7
```

`query.py` prints a table: rank, `mm:ss–mm:ss`, fused score, visual score, transcript score, and the transcript snippet truncated to 60 characters.

---

## 8. Streamlit App (`app.py`)

Single page, three regions.

**Sidebar** — video selector (reads `data/index/*/meta.json`), alpha slider `0.0–1.0` default from config, results count, score threshold, and a small caption showing the active model names.

**Main, top** — a text input and a Search button. On submit, call `search_video`.

**Main, results** — one row per moment:
- Thumbnail image (the peak frame).
- `mm:ss – mm:ss` as a header.
- Two horizontal bars showing the visual and transcript contributions to that result, drawn with `st.progress` or a small matplotlib bar. **This is the single most important UI element for the demo** — it makes the fusion visible instead of theoretical.
- The overlapping transcript text, italicised, or "(no speech)".
- A button that seeks the embedded `st.video` player via its `start_time` argument.

**Caching.** Decorate index loading and model loading with `@st.cache_resource`. Without it, every widget interaction reloads CLIP and the app becomes unusable. Do not cache search results themselves — the alpha slider must produce a visible live change.

---

## 9. Evaluation Harness

### `eval/msrvtt.py`
Clip-level retrieval on the **1K-A test split** (1000 clips, first caption per clip as the query).

Procedure: ingest each clip, mean-pool its frame embeddings into one clip vector, L2-normalize, stack into a matrix. Encode all 1000 queries. Compute the full similarity matrix. For each query, find the rank of its true clip.

Report **Recall@1, Recall@5, Recall@10, Median Rank**. Write `results/msrvtt.json`.

Also implement max-pooling as an alternative aggregation, selectable by flag — that comparison is one row of the ablation table.

### `eval/charades.py`
Moment retrieval. For each `(query, video, ground-truth start/end)` triple, run `search_video` and take the top-1 moment. Compute IoU against ground truth.

Report **R@1 IoU≥0.5**, **R@1 IoU≥0.7**, **mean IoU**. Write `results/charades.json`.

### `eval/ablations.py`
Runs and tabulates:

| Row | Configuration |
|---|---|
| Visual only | `alpha=1.0` |
| Transcript only | `alpha=0.0` |
| Fused | `alpha` swept `0.0 → 1.0` step `0.1`, best reported |
| Pooling | mean vs max frame aggregation |
| Encoder | `ViT-B-32` vs `ViT-L-14` |

Emits `results/ablations.csv` and, via `eval/plots.py`, a line chart of metric against alpha. That chart is the headline figure of the report — make it clean, labelled, and saved at 200 dpi.

Add a per-query-type breakdown: bucket queries by whether their ground-truth moment contains speech, and report each bucket separately. This is what makes the fusion result interpretable rather than a single averaged number.

---

## 10. Task Sequence for Codex

Issue these as separate prompts. Each has an acceptance test — do not move to the next task until it passes.

**Task 1 — Scaffold.** Create the directory tree, `requirements.txt`, `config.yaml`, `cmvs/config.py`, `cmvs/utils.py`, and a `.gitignore` covering `data/`, `results/`, `__pycache__`, `*.faiss`.
*Accept:* `python -c "from cmvs.config import load_config; print(load_config())"` prints a populated config object.

**Task 2 — Extraction.** Implement `cmvs/extract.py`.
*Accept:* running it on a 60-second test clip produces ~60 JPEGs, a valid `frames.jsonl` with monotonically increasing timestamps, and a 16 kHz mono WAV.

**Task 3 — Visual encoding.** Implement `cmvs/encode_visual.py`.
*Accept:* embeddings have shape `(N, 512)`, dtype `float32`, and every row has L2 norm within `1e-5` of `1.0`.

**Task 4 — Transcription and text encoding.** Implement `cmvs/transcribe.py` and `cmvs/encode_text.py`.
*Accept:* segments have non-decreasing `start` values and `end > start` for all rows; a silent input returns `[]` without raising.

**Task 5 — Indexing.** Implement `cmvs/index.py`.
*Accept:* build an index, save it, reload it, and confirm searching with a stored vector returns that vector's own row at rank 0 with score ≈ 1.0.

**Task 6 — Fusion and grouping.** Implement `cmvs/fuse.py` and `cmvs/group.py` with unit tests on synthetic arrays.
*Accept:* `align_transcript_to_frames` returns 0.0 for a frame in a transcript gap; `group_frames_into_moments` collapses 10 consecutive above-threshold frames into exactly 1 moment, and splits into 2 when a gap larger than `merge_gap_s` sits between them.

**Task 7 — Pipeline and search.** Implement `cmvs/pipeline.py`, `cmvs/search.py`, and both CLI scripts.
*Accept:* ingest a real video, then query it and get plausible moments; re-running ingest without `--force` completes in under a second by skipping every stage.

**Task 8 — Streamlit app.** Implement `app.py` per Section 8.
*Accept:* the app loads, search returns thumbnails, and dragging the alpha slider visibly reorders results without reloading models.

**Task 9 — Evaluation.** Implement `eval/msrvtt.py` and `eval/charades.py`.
*Accept:* both write valid JSON with all required metrics on a small subset (50 clips) before being run at full scale.

**Task 10 — Ablations and plots.** Implement `eval/ablations.py` and `eval/plots.py`.
*Accept:* `results/ablations.csv` contains one row per configuration, and the alpha plot renders with axis labels and a legend.

---

## 11. Known Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| All results score ~0.25 and ranking looks random | Fusing unnormalized similarities | Min-max normalize each modality before weighting |
| Top 10 results are consecutive seconds of one scene | Grouping not applied | Check `merge_gap_s` and that grouping runs before truncation |
| `IndexError` on lookup after FAISS search | `k > ntotal`, FAISS returned `-1` ids | Clamp `k`, filter `-1` before indexing meta arrays |
| Streamlit unusable, multi-second lag per click | Models reloading each rerun | `@st.cache_resource` on loaders |
| Ingest restarts from scratch after a disconnect | Stages not resumable | Existence checks per stage |
| Transcript scores always 0 | Alignment searching a gap left by VAD | Verify `end` covers the timestamp, don't assume contiguity |
| CUDA OOM while encoding | Batch too large for T4 | Drop `batch_size` to 32; encode in shards, write incrementally |
| `ffmpeg not found` | Binary missing on PATH | `apt-get install ffmpeg` — surface this as a named error |

---

## 12. Definition of Done

- `python scripts/ingest.py --dir data/raw` indexes every video without manual intervention.
- `streamlit run app.py` serves working search with visible per-modality score bars.
- `results/msrvtt.json`, `results/charades.json`, `results/ablations.csv` all exist and are populated.
- The alpha plot is saved and readable at slide resolution.
- `pytest` passes.
- `README.md` documents setup, the three commands, and the headline numbers.
