"use client";

import { useCallback, useEffect, useImperativeHandle, useRef, useState, forwardRef } from "react";
import { timecode } from "@/lib/format";
import styles from "./Player.module.css";

export type PlayerHandle = {
  seek: (seconds: number, play?: boolean) => void;
};

export type Range = { start: number; end: number };

type Props = {
  src: string;
  poster: string;
  duration: number;
  /** When set, only this range is playable: the scrubber, clock and playback are bounded to it. */
  range: Range | null;
  /** Fires about 15 times a second with the absolute time in the source video. */
  onTime: (seconds: number) => void;
  /** Changes whenever a clip is (re)selected, so picking the same clip again restarts it. */
  rangeKey: number;
};

const Player = forwardRef<PlayerHandle, Props>(function Player(
  { src, poster, duration, range, onTime, rangeKey },
  handle,
) {
  const stageRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [absolute, setAbsolute] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [ended, setEnded] = useState(false);
  const rangeRef = useRef<Range | null>(range);
  const onTimeRef = useRef(onTime);

  useEffect(() => {
    rangeRef.current = range;
    onTimeRef.current = onTime;
  });

  const lo = range ? range.start : 0;
  const hi = range ? range.end : duration;
  const span = Math.max(0.001, hi - lo);
  const position = Math.min(1, Math.max(0, (absolute - lo) / span));

  const seek = useCallback((seconds: number, play?: boolean) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = seconds;
    setAbsolute(seconds);
    setEnded(false);
    if (play) void video.play().catch(() => setPlaying(false));
  }, []);

  useImperativeHandle(handle, () => ({ seek }), [seek]);

  // Jump to the start of a newly selected clip and play it.
  const handledKey = useRef(rangeKey);
  useEffect(() => {
    if (handledKey.current === rangeKey) return;
    handledKey.current = rangeKey;
    const bounds = rangeRef.current;
    if (bounds) seek(bounds.start, true);
  }, [rangeKey, seek]);

  // Smooth clock while playing, and the clip boundary.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let frame = 0;
    let lastReport = 0;
    const tick = (now: number) => {
      const current = video.currentTime;
      const bounds = rangeRef.current;
      if (bounds && current >= bounds.end) {
        video.pause();
        video.currentTime = bounds.end;
        setAbsolute(bounds.end);
        setEnded(true);
        setPlaying(false);
        onTimeRef.current(bounds.end);
        return;
      }
      setAbsolute(current);
      if (now - lastReport > 66) {
        lastReport = now;
        onTimeRef.current(current);
      }
      frame = requestAnimationFrame(tick);
    };
    const onPlay = () => {
      setPlaying(true);
      setEnded(false);
      frame = requestAnimationFrame(tick);
    };
    const onPause = () => {
      setPlaying(false);
      cancelAnimationFrame(frame);
      setAbsolute(video.currentTime);
      onTimeRef.current(video.currentTime);
    };
    const onFullEnd = () => setEnded(true);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("ended", onFullEnd);
    return () => {
      cancelAnimationFrame(frame);
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("ended", onFullEnd);
    };
  }, []);

  const toggle = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      const bounds = rangeRef.current;
      if (bounds && video.currentTime >= bounds.end - 0.05) video.currentTime = bounds.start;
      if (!bounds && video.ended) video.currentTime = 0;
      void video.play().catch(() => setPlaying(false));
    } else {
      video.pause();
    }
  }, []);

  const scrub = (event: React.ChangeEvent<HTMLInputElement>) => {
    seek(lo + (Number(event.target.value) / 1000) * span);
  };

  const fullscreen = () => {
    const stage = stageRef.current;
    if (!stage) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void stage.requestFullscreen();
  };

  const onKey = (event: React.KeyboardEvent) => {
    // Buttons and the scrubber handle Space/arrows themselves; handling them here too would double-fire.
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement) return;
    if (event.key === " " || event.key === "k") {
      event.preventDefault();
      toggle();
    } else if (event.key === "ArrowLeft") {
      seek(Math.max(lo, absolute - 5));
    } else if (event.key === "ArrowRight") {
      seek(Math.min(hi, absolute + 5));
    } else if (event.key === "m") {
      setMuted((value) => !value);
    } else if (event.key === "f") {
      fullscreen();
    }
  };

  return (
    <div className={styles.stage} ref={stageRef} onKeyDown={onKey} tabIndex={0} aria-label="Video player">
      <video
        ref={videoRef}
        className={styles.video}
        src={src}
        poster={poster}
        muted={muted}
        playsInline
        preload="auto"
        onClick={toggle}
      />
      {!playing && (
        <button className={styles.bigPlay} onClick={toggle} aria-label={ended ? "Replay" : "Play"}>
          {ended ? <ReplayIcon /> : <PlayIcon />}
        </button>
      )}
      <div className={styles.controls}>
        <button className={styles.iconButton} onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
          {playing ? <PauseIcon /> : <PlayIcon />}
        </button>
        <span className={`${styles.clock} time`}>
          {timecode(absolute - lo)}
          <span className={styles.clockTotal}> / {timecode(span)}</span>
        </span>
        <input
          className={styles.scrubber}
          type="range"
          min={0}
          max={1000}
          value={Math.round(position * 1000)}
          onChange={scrub}
          aria-label={range ? "Position in clip" : "Position in video"}
          style={{ ["--played" as string]: `${position * 100}%` }}
        />
        <button
          className={styles.iconButton}
          onClick={() => setMuted((value) => !value)}
          aria-label={muted ? "Unmute" : "Mute"}
          aria-pressed={muted}
        >
          {muted ? <MutedIcon /> : <VolumeIcon />}
        </button>
        <button className={styles.iconButton} onClick={fullscreen} aria-label="Fullscreen">
          <FullscreenIcon />
        </button>
      </div>
    </div>
  );
});

export default Player;

const icon = { width: 18, height: 18, viewBox: "0 0 24 24", fill: "currentColor", "aria-hidden": true } as const;
const PlayIcon = () => (
  <svg {...icon}>
    <path d="M7 4.5v15a1 1 0 0 0 1.5.86l12.5-7.5a1 1 0 0 0 0-1.72L8.5 3.64A1 1 0 0 0 7 4.5Z" />
  </svg>
);
const PauseIcon = () => (
  <svg {...icon}>
    <rect x="6" y="4" width="4.5" height="16" rx="1" />
    <rect x="13.5" y="4" width="4.5" height="16" rx="1" />
  </svg>
);
const ReplayIcon = () => (
  <svg {...icon}>
    <path d="M12 5V2L7 6l5 4V7a6 6 0 1 1-6 6H4a8 8 0 1 0 8-8Z" />
  </svg>
);
const VolumeIcon = () => (
  <svg {...icon}>
    <path d="M3 9v6h4l5 4V5L7 9H3Zm13.5 3a4.5 4.5 0 0 0-2.5-4v8a4.5 4.5 0 0 0 2.5-4Z" />
  </svg>
);
const MutedIcon = () => (
  <svg {...icon}>
    <path d="M3 9v6h4l5 4V5L7 9H3Zm14.6 3 2.4-2.4-1.4-1.4-2.4 2.4-2.4-2.4-1.4 1.4 2.4 2.4-2.4 2.4 1.4 1.4 2.4-2.4 2.4 2.4 1.4-1.4-2.4-2.4Z" />
  </svg>
);
const FullscreenIcon = () => (
  <svg {...icon}>
    <path d="M4 4h6v2H6v4H4V4Zm10 0h6v6h-2V6h-4V4ZM4 14h2v4h4v2H4v-6Zm14 0h2v6h-6v-2h4v-4Z" />
  </svg>
);
