"use client";

import type { Moment } from "@/lib/api";
import { mediaUrl } from "@/lib/api";
import { timecode } from "@/lib/format";
import styles from "./Results.module.css";

type Props = {
  moments: Moment[];
  alpha: number;
  activeStart: number | null;
  searching: boolean;
  error: string | null;
  hasQuery: boolean;
  onSelect: (index: number) => void;
};

export default function Results({ moments, alpha, activeStart, searching, error, hasQuery, onSelect }: Props) {
  if (error) {
    return (
      <p className={styles.empty} role="alert">
        Search failed: {error}
      </p>
    );
  }
  if (!hasQuery) {
    return (
      <p className={styles.empty}>
        Matching moments appear here. Every second of the video is scored twice, once on what the picture
        shows and once on what is said, then the two are combined.
      </p>
    );
  }
  if (moments.length === 0) {
    return (
      <p className={styles.empty}>
        {searching ? "Searching…" : "Nothing scored above the cutoff. Lower the cutoff or describe it differently."}
      </p>
    );
  }
  return (
    <ol className={`${styles.list} ${searching ? styles.stale : ""}`}>
      {moments.map((moment, index) => {
        const picture = alpha * moment.visual;
        const speech = (1 - alpha) * moment.transcript;
        return (
          <li key={moment.clip_start}>
            <button
              className={`${styles.card} ${moment.clip_start === activeStart ? styles.cardActive : ""}`}
              onClick={() => onSelect(index)}
            >
              <span className={styles.thumb}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={mediaUrl(moment.thumbnail)} alt="" loading="lazy" />
                <span className={styles.rank}>{index + 1}</span>
              </span>
              <span className={styles.body}>
                <span className={styles.head}>
                  <span className={`${styles.range} time`}>
                    {timecode(moment.clip_start)}–{timecode(moment.clip_end)}
                  </span>
                  <span className={`${styles.score} time`}>{moment.score.toFixed(2)}</span>
                </span>
                <span
                  className={styles.bar}
                  role="img"
                  aria-label={`Picture ${Math.round(picture * 100)} percent, speech ${Math.round(speech * 100)} percent`}
                >
                  <i className={styles.barPicture} style={{ width: `${picture * 100}%` }} />
                  <i className={styles.barSpeech} style={{ width: `${speech * 100}%` }} />
                </span>
                <span className={moment.text ? styles.quote : styles.quoteNone}>
                  {moment.text || "No speech in this moment"}
                </span>
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
