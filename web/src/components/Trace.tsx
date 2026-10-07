"use client";

import { useMemo, useState } from "react";
import type { Moment, Trace as TraceData } from "@/lib/api";
import { timecode } from "@/lib/format";
import styles from "./Trace.module.css";

type Props = {
  duration: number;
  step: number;
  trace: TraceData | null;
  moments: Moment[];
  threshold: number;
  activeStart: number | null;
  activeEnd: number | null;
  currentTime: number;
  onSeek: (seconds: number) => void;
  onSelect: (index: number) => void;
};

const WIDTH = 1000;
const HEIGHT = 100;

function barPaths(trace: TraceData, duration: number, step: number) {
  const unit = WIDTH / duration;
  const width = Math.max(0.5, step * unit);
  let picture = "";
  let speech = "";
  for (let i = 0; i < trace.t.length; i++) {
    const x = (trace.t[i] * unit).toFixed(2);
    const pictureHeight = Math.min(1, trace.visual[i]) * HEIGHT;
    const speechHeight = Math.min(1 - Math.min(1, trace.visual[i]), trace.transcript[i]) * HEIGHT;
    if (pictureHeight > 0.2) {
      picture += `M${x} ${HEIGHT}h${width.toFixed(2)}v${(-pictureHeight).toFixed(2)}h${(-width).toFixed(2)}z`;
    }
    if (speechHeight > 0.2) {
      speech += `M${x} ${(HEIGHT - pictureHeight).toFixed(2)}h${width.toFixed(2)}v${(-speechHeight).toFixed(2)}h${(-width).toFixed(2)}z`;
    }
  }
  return { picture, speech };
}

export default function Trace({
  duration,
  step,
  trace,
  moments,
  threshold,
  activeStart,
  activeEnd,
  currentTime,
  onSeek,
  onSelect,
}: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const paths = useMemo(() => (trace ? barPaths(trace, duration, step) : null), [trace, duration, step]);
  const ticks = useMemo(() => [0, 0.25, 0.5, 0.75, 1].map((f) => f * duration), [duration]);

  const toTime = (event: React.PointerEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    return Math.min(duration, Math.max(0, ((event.clientX - box.left) / box.width) * duration));
  };

  return (
    <div className={styles.wrap}>
      <div
        className={styles.trace}
        onPointerMove={(event) => setHover(toTime(event))}
        onPointerLeave={() => setHover(null)}
        onClick={(event) => onSeek(toTime(event as unknown as React.PointerEvent<HTMLDivElement>))}
        role="img"
        aria-label={
          trace
            ? `Relevance across the video with ${moments.length} matching moments`
            : "Video timeline"
        }
      >
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" className={styles.svg}>
          {paths && <path d={paths.picture} className={styles.picture} />}
          {paths && <path d={paths.speech} className={styles.speech} />}
        </svg>
        {trace && (
          <div className={styles.threshold} style={{ bottom: `${threshold * 100}%` }}>
            <span>cutoff</span>
          </div>
        )}
        {moments.map((moment, index) => (
          <button
            key={moment.clip_start}
            className={`${styles.bracket} ${moment.clip_start === activeStart ? styles.bracketActive : ""}`}
            style={{
              left: `${(moment.clip_start / duration) * 100}%`,
              width: `max(14px, ${((moment.clip_end - moment.clip_start) / duration) * 100}%)`,
            }}
            onClick={(event) => {
              event.stopPropagation();
              onSelect(index);
            }}
            aria-label={`Result ${index + 1}, ${timecode(moment.clip_start)} to ${timecode(moment.clip_end)}`}
          >
            <span className={styles.rank}>{index + 1}</span>
          </button>
        ))}
        <div className={styles.played} style={{ width: `${(currentTime / duration) * 100}%` }} />
        {activeStart !== null && activeEnd !== null && currentTime >= activeStart && (
          <div
            className={styles.clipProgress}
            style={{
              left: `${(activeStart / duration) * 100}%`,
              width: `${((Math.min(currentTime, activeEnd) - activeStart) / duration) * 100}%`,
            }}
          />
        )}
        <div className={styles.playhead} style={{ left: `${(currentTime / duration) * 100}%` }} />
        {hover !== null && (
          <div className={`${styles.hover} time`} style={{ left: `${(hover / duration) * 100}%` }}>
            {timecode(hover)}
          </div>
        )}
      </div>
      <div className={`${styles.ticks} time`} aria-hidden>
        {ticks.map((tick) => (
          <span key={tick}>{timecode(tick)}</span>
        ))}
      </div>
    </div>
  );
}
