# Needle: Cross-Modal Video Search

## 1. What it is

Needle finds moments inside a video from a plain-language query. You type
"baseball first pitch" or "astronauts with Elmo", and it returns ranked, timestamped
clips. It matches the query against both **what is shown** (the pictures) and
**what is said** (the speech), then combines the two.

No model is trained or fine-tuned. Three pretrained, frozen models are used. The
work is in the pipeline, the way the two signals are aligned in time and fused,
and the app built on top.

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
| Automated tests | 36 passing (fusion, grouping, index, pipeline, API, upload, ranges) |

Qualitative check on a public-domain NASA news episode: "baseball first pitch"
returns the 10:16 ceremonial pitch first; "Mars rover robotic arm" returns the
1:25 shot of the rover arm; "astronauts with Elmo" lands at about 2:40, where the
narration says it. Moving the slider to speech-only or picture-only changes the
ranking as expected.

## 5. Evaluation status

The harness is implemented (`eval/`): MSR-VTT clip retrieval (Recall@1/5/10,
median rank; mean vs max frame pooling), Charades moment retrieval (R@1 at
IoU 0.5 and 0.7, mean IoU), and ablations (picture-only, speech-only, an α sweep,
pooling, ViT-B/32 vs ViT-L/14) with the α-sweep plot. It is unit-tested on
synthetic data.

**It has not been run on MSR-VTT or Charades**, so this report contains no
benchmark numbers. Those datasets must be obtained under their own licenses.

## 6. Limitations and next steps

- **Top result is always about 1.0.** Min-max normalization is relative to each
  query, so the best moment always scores 1.0, even when nothing truly matches.
  A fix would normalize against fixed similarity floors, or add an absolute
  relevance check.
- **Only the top 200 frames per modality are scored**, so very long videos
  (thousands of frames) are covered sparsely.
- **One frame per second** can miss fast events and rounds clip edges to whole seconds.
- **CPU only on Apple Silicon.** The code uses a GPU only when CUDA is present.
- **Speech quality limits speech search.** Whisper `base` mishears names and jargon.
- **Not run on the benchmarks yet** (section 5). That is the main next step.

## 7. Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # also needs ffmpeg on PATH
python scripts/fetch_demo.py              # optional demo video
uvicorn server.main:app --port 8000
cd web && npm install && npm run dev      # http://localhost:3000
```

Code: `cmvs/` (pipeline), `server/` (API), `web/` (frontend), `eval/` (benchmarks),
`tests/`. Settings live in `config.yaml`.
