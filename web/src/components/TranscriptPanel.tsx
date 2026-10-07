"use client";

import { useEffect, useRef } from "react";
import type { Segment } from "@/lib/api";
import { timecode } from "@/lib/format";
import styles from "./TranscriptPanel.module.css";

type Props = {
  segments: Segment[];
  time: number;
  hasSpeech: boolean;
  onSeek: (seconds: number) => void;
};

export default function TranscriptPanel({ segments, time, hasSpeech, onSeek }: Props) {
  const listRef = useRef<HTMLOListElement>(null);
  const current = segments.findIndex((segment) => time >= segment.start && time < segment.end);

  useEffect(() => {
    const list = listRef.current;
    const scroller = list?.parentElement;
    const item = current >= 0 ? list?.children[current] : null;
    if (!list || !scroller || !(item instanceof HTMLElement)) return;
    const top = item.offsetTop + list.offsetTop;
    if (top < scroller.scrollTop || top + item.offsetHeight > scroller.scrollTop + scroller.clientHeight) {
      scroller.scrollTo({ top: top - scroller.clientHeight / 3, behavior: "smooth" });
    }
  }, [current]);

  if (!hasSpeech || segments.length === 0) {
    return <p className={styles.empty}>This video has no speech, so only the picture is searched.</p>;
  }
  return (
    <ol className={styles.list} ref={listRef}>
      {segments.map((segment, index) => (
        <li key={segment.seg_idx}>
          <button
            className={index === current ? styles.rowActive : styles.row}
            onClick={() => onSeek(segment.start)}
          >
            <span className={`${styles.at} time`}>{timecode(segment.start)}</span>
            <span>{segment.text}</span>
          </button>
        </li>
      ))}
    </ol>
  );
}
