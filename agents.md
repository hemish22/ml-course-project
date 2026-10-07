# Agent Instructions

## Project
Cross-modal video search: CLIP frame embeddings + Whisper transcript embeddings,
fused at query time, returning timestamped moments. The pretrained encoders (CLIP,
Whisper, MiniLM) are never trained. The `analysis/` package may train small
scikit-learn regression/classification models on features extracted from them.

## Rules
- Python 3.10. Type hints on all public functions.
- Never hardcode paths, thresholds, or model names — read them from `config.yaml`.
- All embeddings are L2-normalized at creation. Downstream code assumes unit vectors.
- FAISS index row order must match the corresponding meta.json array order. Assert it.
- Every pipeline stage is resumable: skip if output exists unless `force=True`.
- A video with no speech is valid input. Never raise on an empty transcript.
- Cache loaded models in module-level singletons.
- Run `pytest` before declaring a task complete.

## Do Not
- Do not add training loops, fine-tuning, or LoRA to the CLIP/Whisper/MiniLM encoders.
- Trained models live only in `analysis/` (scikit-learn). `cmvs/` stays free of training code.
- Do not swap FAISS for IVF/HNSW — the corpus is small and exact search is correct.
- Do not import Streamlit inside the `cmvs/` package.
- Do not add new dependencies without stating why.