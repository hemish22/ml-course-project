"""Streamlit interface for cross-modal video moment search."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import streamlit as st

from cmvs.config import Config, load_config
from cmvs.encode_text import _load_text_model
from cmvs.encode_visual import load_clip
from cmvs.index import load_index
import cmvs.search as search_module


@st.cache_resource
def _cached_index(index_dir: str, kind: str) -> tuple[Any, dict[str, Any]]:
    """Cache one FAISS index resource for Streamlit reruns."""
    return load_index(Path(index_dir), kind)


@st.cache_resource
def _cached_models(cfg: Config) -> None:
    """Initialize and retain both query encoders across Streamlit reruns."""
    load_clip(cfg)
    _load_text_model(cfg)


def _cached_search_index(directory: Path, kind: str) -> tuple[Any, dict[str, Any]]:
    """Adapt the Streamlit index cache to the package search loader signature."""
    return _cached_index(str(directory), kind)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as ``mm:ss`` for display."""
    minutes, remaining = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remaining:02d}"


def _available_videos(cfg: Config) -> dict[str, dict[str, Any]]:
    """Read metadata for completed index directories."""
    if not cfg.paths.index.exists():
        return {}
    videos: dict[str, dict[str, Any]] = {}
    for index_dir in sorted(cfg.paths.index.iterdir()):
        meta_path = index_dir / "meta.json"
        if index_dir.is_dir() and meta_path.exists():
            with meta_path.open(encoding="utf-8") as meta_file:
                videos[index_dir.name] = json.load(meta_file)
    return videos


def main() -> None:
    """Render the video selector, query controls, and fused moment results."""
    st.set_page_config(page_title="Cross-Modal Video Search", layout="wide")
    cfg = load_config()
    videos = _available_videos(cfg)
    st.title("Cross-Modal Video Search")
    if not videos:
        st.info("No indexed videos found. Run scripts/ingest.py on the target machine first.")
        return

    with st.sidebar:
        video_id = st.selectbox("Video", list(videos))
        alpha = st.slider("Visual weight", 0.0, 1.0, cfg.search.alpha, 0.05)
        max_results = st.number_input("Results", min_value=1, value=cfg.search.max_results, step=1)
        score_threshold = st.slider(
            "Score threshold", 0.0, 1.0, cfg.search.score_threshold, 0.01
        )
        st.caption(
            f"CLIP: {cfg.models.clip_name}\n\n"
            f"Whisper: {cfg.models.whisper_size}\n\n"
            f"Text: {cfg.models.text_model}"
        )

    query = st.text_input("Search inside this video", placeholder="person writing on a whiteboard")
    if not st.button("Search") or not query.strip():
        return

    runtime_cfg = replace(
        cfg,
        search=replace(
            cfg.search,
            score_threshold=float(score_threshold),
            max_results=int(max_results),
        ),
    )
    _cached_models(runtime_cfg)
    search_module.load_index = _cached_search_index
    results = search_module.search_video(query, video_id, runtime_cfg, alpha=alpha)
    source_path = videos[video_id]["source_path"]
    if not results:
        st.warning("No moments passed the current score threshold.")
        return

    for result in results:
        thumbnail, detail = st.columns([1, 3])
        with thumbnail:
            st.image(result.thumbnail_path, use_container_width=True)
        with detail:
            st.subheader(f"{_format_timestamp(result.start)} – {_format_timestamp(result.end)}")
            st.caption(f"Fused score: {result.score:.3f}")
            st.write("Visual contribution")
            st.progress(min(1.0, alpha * result.visual_score))
            st.write("Transcript contribution")
            st.progress(min(1.0, (1.0 - alpha) * result.transcript_score))
            st.caption(f"*{result.transcript_text or '(no speech)'}*")
            if st.button("Play from this moment", key=f"play-{result.video_id}-{result.start}"):
                st.video(source_path, start_time=result.start)


if __name__ == "__main__":
    main()
