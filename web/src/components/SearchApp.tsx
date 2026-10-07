"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  API_URL,
  fetchConfig,
  fetchSearch,
  fetchTranscript,
  fetchVideos,
  mediaUrl,
  removeVideo,
  type AppConfig,
  type SearchResponse,
  type Segment,
  type VideoInfo,
} from "@/lib/api";
import { timecode } from "@/lib/format";
import { DEMOS } from "@/lib/demos";
import Player, { type PlayerHandle, type Range } from "./Player";
import Results from "./Results";
import Trace from "./Trace";
import TranscriptPanel from "./TranscriptPanel";
import UploadDialog from "./UploadDialog";
import styles from "./SearchApp.module.css";

type Tab = "results" | "transcript";

export default function SearchApp() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [draft, setDraft] = useState("");
  const [activeQuery, setActiveQuery] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const [alpha, setAlpha] = useState(0.6);
  const [threshold, setThreshold] = useState(0.28);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const [clip, setClip] = useState<Range | null>(null);
  const [clipKey, setClipKey] = useState(0);
  const [time, setTime] = useState(0);
  const [tab, setTab] = useState<Tab>("results");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<Segment[]>([]);

  const playerRef = useRef<PlayerHandle>(null);
  const pickTopOnNextResponse = useRef(false);

  useEffect(() => {
    Promise.all([fetchConfig(), fetchVideos()])
      .then(([appConfig, list]) => {
        setConfig(appConfig);
        setAlpha(appConfig.alpha);
        setThreshold(appConfig.score_threshold);
        setVideos(list);
        setVideoId(list[0]?.id ?? null);
      })
      .catch((error: Error) => setLoadError(error.message));
  }, []);

  const video = useMemo(() => videos.find((item) => item.id === videoId) ?? null, [videos, videoId]);

  useEffect(() => {
    if (!videoId) return;
    fetchTranscript(videoId).then(setTranscript).catch(() => setTranscript([]));
  }, [videoId]);

  // Search whenever the query, video or a weighting control changes. Sliders re-rank live.
  useEffect(() => {
    if (!activeQuery || !videoId || !config) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      setSearching(true);
      setSearchError(null);
      fetchSearch(
        { video: videoId, q: activeQuery, alpha, threshold, limit: config.max_results },
        controller.signal,
      )
        .then((result) => {
          setResponse(result);
          setSearching(false);
          setSubmitting(false);
          if (pickTopOnNextResponse.current) {
            pickTopOnNextResponse.current = false;
            const top = result.moments[0];
            setClip(top ? { start: top.clip_start, end: top.clip_end } : null);
            setClipKey((key) => key + 1);
            setTab("results");
          }
        })
        .catch((error: Error) => {
          if (error.name === "AbortError") return;
          setSearching(false);
          setSubmitting(false);
          setSearchError(error.message);
        });
    }, 160);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [activeQuery, videoId, config, alpha, threshold, nonce]);

  const submit = useCallback(
    (query: string) => {
      const trimmed = query.trim();
      if (!trimmed) return;
      setDraft(trimmed);
      setActiveQuery(trimmed);
      pickTopOnNextResponse.current = true;
      setSubmitting(true);
      setNonce((value) => value + 1);
    },
    [],
  );

  const selectMoment = useCallback(
    (index: number) => {
      const moment = response?.moments[index];
      if (!moment) return;
      setClip({ start: moment.clip_start, end: moment.clip_end });
      setClipKey((key) => key + 1);
    },
    [response],
  );

  const showFullVideo = useCallback(() => {
    setClip(null);
  }, []);

  const seekFull = useCallback((seconds: number) => {
    setClip(null);
    // Wait one frame so the player drops its clip bounds before seeking.
    requestAnimationFrame(() => playerRef.current?.seek(seconds, true));
  }, []);

  const changeVideo = useCallback((id: string) => {
    setVideoId(id);
    setResponse(null);
    setClip(null);
    setActiveQuery(null);
    setSearchError(null);
    setTime(0);
    setNotice(null);
  }, []);

  const onIndexed = useCallback(
    (newId: string) => {
      fetchVideos()
        .then((list) => {
          setVideos(list);
          changeVideo(newId);
        })
        .catch((error: Error) => setNotice(error.message));
    },
    [changeVideo],
  );

  const onRemove = useCallback(() => {
    if (!video) return;
    const label = DEMOS[video.id]?.title ?? video.title;
    if (!window.confirm(`Remove "${label}" from the library? The original file stays on disk.`)) return;
    removeVideo(video.id)
      .then(() => fetchVideos())
      .then((list) => {
        setVideos(list);
        changeVideo(list[0]?.id ?? "");
        if (list.length === 0) setVideoId(null);
      })
      .catch((error: Error) => setNotice(error.message));
  }, [video, changeVideo]);

  const moments = response?.moments ?? [];
  const activeIndex = clip ? moments.findIndex((moment) => moment.clip_start === clip.start) : -1;
  const suggestions = videoId ? (DEMOS[videoId]?.suggestions ?? []) : [];
  const title = video ? (DEMOS[video.id]?.title ?? video.title) : "";
  const noMatches = !!response && !searching && moments.length === 0;

  const header = (
    <header className={styles.header}>
      <div className={styles.brand}>
        <Mark />
        <span className={styles.wordmark}>Needle</span>
      </div>
      <div className={styles.headerRight}>
        {videos.length > 1 ? (
          <select
            className={styles.videoSelect}
            value={videoId ?? ""}
            onChange={(event) => changeVideo(event.target.value)}
            aria-label="Video"
          >
            {videos.map((item) => (
              <option key={item.id} value={item.id}>
                {DEMOS[item.id]?.title ?? item.title}
              </option>
            ))}
          </select>
        ) : (
          <span className={styles.videoTitle}>{title}</span>
        )}
        {config && (
          <span className={styles.models} title={`Text model: ${config.models.text}`}>
            CLIP {config.models.clip}, Whisper {config.models.whisper}
          </span>
        )}
        {video && (
          <button className={styles.ghost} onClick={onRemove}>
            Remove
          </button>
        )}
        {config && (
          <button className={styles.solid} onClick={() => setUploadOpen(true)}>
            Upload video
          </button>
        )}
      </div>
    </header>
  );

  if (loadError) {
    return (
      <main className={styles.page}>
        {header}
        <div className={styles.notice} role="alert">
          <h1>Can’t reach the search server</h1>
          <p>
            The page tried <code>{API_URL}</code> and got: {loadError}.
          </p>
          <p>
            Start it from the project folder with <code>.venv/bin/python -m uvicorn server.main:app --port 8000</code>,
            then reload.
          </p>
        </div>
      </main>
    );
  }

  if (config && videos.length === 0) {
    return (
      <main className={styles.page}>
        {header}
        <div className={styles.notice}>
          <h1>Add your first video</h1>
          <p>Upload a video to make it searchable by what is shown and what is said.</p>
          <button className={styles.solid} onClick={() => setUploadOpen(true)}>
            Upload video
          </button>
        </div>
        <UploadDialog open={uploadOpen} onClose={() => setUploadOpen(false)} onIndexed={onIndexed} />
      </main>
    );
  }

  return (
    <main className={styles.page}>
      {header}
      {notice && (
        <p className={styles.banner} role="alert">
          {notice}
        </p>
      )}
      <UploadDialog open={uploadOpen} onClose={() => setUploadOpen(false)} onIndexed={onIndexed} />

      <form
        className={styles.searchRow}
        onSubmit={(event) => {
          event.preventDefault();
          submit(draft);
        }}
      >
        <input
          className={styles.searchInput}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Describe what you see or what is said"
          aria-label="Search query"
          autoComplete="off"
          spellCheck={false}
        />
        <button className={styles.searchButton} type="submit" disabled={!draft.trim() || !video}>
          {submitting ? "Searching…" : "Search"}
        </button>
      </form>
      {suggestions.length > 0 && (
        <div className={styles.suggestions}>
          <span className={styles.suggestLead}>Try</span>
          {suggestions.map((suggestion) => (
            <button key={suggestion} className={styles.chip} onClick={() => submit(suggestion)}>
              {suggestion}
            </button>
          ))}
        </div>
      )}

      <div className={styles.layout}>
        <section className={styles.main} aria-label="Player">
          <div className={styles.modeBar}>
            {clip ? (
              <>
                <span className={styles.modeLabel}>
                  <strong>Clip</strong>
                  <span className="time">
                    {" "}
                    {timecode(clip.start)}–{timecode(clip.end)}
                  </span>
                  {activeIndex >= 0 && (
                    <span className={styles.modeSub}>
                      {" "}
                      result {activeIndex + 1} of {moments.length}
                    </span>
                  )}
                </span>
                <span className={styles.modeActions}>
                  {activeIndex >= 0 && (
                    <>
                      <button
                        className={styles.ghost}
                        onClick={() => selectMoment(activeIndex - 1)}
                        disabled={activeIndex <= 0}
                      >
                        Previous
                      </button>
                      <button
                        className={styles.ghost}
                        onClick={() => selectMoment(activeIndex + 1)}
                        disabled={activeIndex >= moments.length - 1}
                      >
                        Next
                      </button>
                    </>
                  )}
                  <button className={styles.solid} onClick={showFullVideo}>
                    Show full video
                  </button>
                </span>
              </>
            ) : (
              <span className={styles.modeLabel}>
                <strong>Full video</strong>
                <span className="time"> {video ? timecode(video.duration) : ""}</span>
                {noMatches && <span className={styles.modeSub}> no moments matched, try fewer words or a lower cutoff</span>}
              </span>
            )}
          </div>

          {video && (
            <Player
              key={video.id}
              ref={playerRef}
              src={mediaUrl(video.video_url)}
              poster={mediaUrl(video.poster_url)}
              duration={video.duration}
              range={clip}
              rangeKey={clipKey}
              onTime={setTime}
            />
          )}

          {video && (
            <Trace
              duration={video.duration}
              step={1 / video.fps}
              trace={response?.trace ?? null}
              moments={moments}
              threshold={threshold}
              activeStart={clip?.start ?? null}
              activeEnd={clip?.end ?? null}
              currentTime={time}
              onSeek={seekFull}
              onSelect={selectMoment}
            />
          )}
        </section>

        <aside className={styles.side}>
          <div className={styles.weighting}>
            <div className={styles.weightHead}>
              <label htmlFor="alpha">Weight</label>
              <span className={`${styles.legend}`}>
                <i className={styles.dotPicture} /> Picture {Math.round(alpha * 100)}%
                <i className={styles.dotSpeech} /> Speech {Math.round((1 - alpha) * 100)}%
              </span>
            </div>
            <input
              id="alpha"
              className={styles.alpha}
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={1 - alpha}
              onChange={(event) => setAlpha(1 - Number(event.target.value))}
              disabled={video ? !video.has_speech : false}
              aria-label="Balance between picture and speech"
              
            />
            <details className={styles.tuning}>
              <summary>Cutoff</summary>
              <div className={styles.tuningBody}>
                <input
                  type="range"
                  min={0}
                  max={0.95}
                  step={0.01}
                  value={threshold}
                  onChange={(event) => setThreshold(Number(event.target.value))}
                  aria-label="Score cutoff"
                />
                <span className="time">{threshold.toFixed(2)}</span>
              </div>
            </details>
          </div>

          <div className={styles.tabs} role="tablist">
            <button
              role="tab"
              aria-selected={tab === "results"}
              className={tab === "results" ? styles.tabActive : styles.tab}
              onClick={() => setTab("results")}
            >
              Results{moments.length > 0 ? ` (${moments.length})` : ""}
            </button>
            <button
              role="tab"
              aria-selected={tab === "transcript"}
              className={tab === "transcript" ? styles.tabActive : styles.tab}
              onClick={() => setTab("transcript")}
            >
              Transcript
            </button>
          </div>

          <div className={styles.panel}>
            {tab === "results" ? (
              <Results
                moments={moments}
                alpha={alpha}
                activeStart={clip?.start ?? null}
                searching={searching}
                error={searchError}
                hasQuery={activeQuery !== null}
                onSelect={selectMoment}
              />
            ) : (
              <TranscriptPanel
                segments={transcript}
                time={time}
                hasSpeech={video?.has_speech ?? true}
                onSeek={seekFull}
              />
            )}
          </div>
        </aside>
      </div>
    </main>
  );
}

function Mark() {
  return (
    <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden>
      <rect x="11.5" y="2" width="3" height="22" rx="1.5" fill="#141821" />
      <circle cx="13" cy="7" r="5" fill="#2b55f0" />
      <circle cx="13" cy="7" r="2" fill="#f0a400" />
    </svg>
  );
}
