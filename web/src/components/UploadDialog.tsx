"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchJob, uploadVideo, type JobStage } from "@/lib/api";
import styles from "./UploadDialog.module.css";

type Props = {
  open: boolean;
  onClose: () => void;
  /** Called once the new video is searchable. */
  onIndexed: (videoId: string) => void;
};

type Step = "upload" | Exclude<JobStage, "queued" | "done">;

const STEPS: { id: Step; label: string }[] = [
  { id: "upload", label: "Uploading" },
  { id: "prepare", label: "Preparing the video" },
  { id: "frames", label: "Extracting one frame per second" },
  { id: "speech", label: "Transcribing speech" },
  { id: "picture", label: "Embedding the pictures" },
  { id: "text", label: "Embedding the speech" },
  { id: "finish", label: "Building the index" },
];

const ACCEPT = ".mp4,.mov,.m4v,.mkv,.webm,.avi,video/*";

export default function UploadDialog({ open, onClose, onIndexed }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [step, setStep] = useState<Step | null>(null);
  const [uploaded, setUploaded] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [finished, setFinished] = useState(false);

  const busy = step !== null && !error && !finished;

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setFile(null);
    setStep(null);
    setUploaded(0);
    setError(null);
    setFinished(false);
    setDragging(false);
  }, []);

  const close = useCallback(() => {
    if (busy && !window.confirm("Stop adding this video?")) return;
    reset();
    onClose();
  }, [busy, onClose, reset]);

  const start = useCallback(
    async (chosen: File) => {
      reset();
      setFile(chosen);
      setStep("upload");
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const { job_id, video_id } = await uploadVideo(chosen, setUploaded, controller.signal);
        for (;;) {
          if (controller.signal.aborted) return;
          const job = await fetchJob(job_id);
          if (job.state === "error") throw new Error(job.error ?? "Indexing failed.");
          if (job.state === "done") break;
          if (job.stage !== "queued") setStep(job.stage as Step);
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
        setFinished(true);
        onIndexed(video_id);
      } catch (caught) {
        if ((caught as Error).name === "AbortError") return;
        setError((caught as Error).message);
      }
    },
    [onIndexed, reset],
  );

  const pick = (list: FileList | null) => {
    const chosen = list?.[0];
    if (chosen) void start(chosen);
  };

  const stepIndex = step ? STEPS.findIndex((item) => item.id === step) : -1;

  return (
    <dialog
      ref={dialogRef}
      className={styles.dialog}
      onClose={() => {
        if (open) close();
      }}
      onCancel={(event) => {
        if (busy) event.preventDefault();
      }}
      aria-labelledby="upload-title"
    >
      <div className={styles.head}>
        <h2 id="upload-title">Add a video</h2>
        <button className={styles.x} onClick={close} aria-label="Close">
          ×
        </button>
      </div>

      {step === null ? (
        <>
          <div
            className={`${styles.drop} ${dragging ? styles.dropActive : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              pick(event.dataTransfer.files);
            }}
          >
            <p className={styles.dropTitle}>Drop a video here</p>
            <p className={styles.dropSub}>or</p>
            <button className={styles.choose} onClick={() => inputRef.current?.click()}>
              Choose a file
            </button>
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPT}
              hidden
              onChange={(event) => {
                pick(event.target.files);
                event.target.value = "";
              }}
            />
          </div>
          <p className={styles.note}>
            MP4, MOV, MKV, WEBM or AVI, up to 2 GB. Indexing runs on this computer and takes roughly 7 seconds
            for every minute of video.
          </p>
        </>
      ) : (
        <div>
          <p className={styles.fileName}>{file?.name}</p>
          <ol className={styles.steps}>
            {STEPS.map((item, index) => {
              const state = finished || index < stepIndex ? "done" : index === stepIndex ? (error ? "failed" : "active") : "todo";
              return (
                <li key={item.id} className={styles[state]}>
                  <span className={styles.marker} aria-hidden>
                    {state === "done" ? "✓" : state === "failed" ? "!" : ""}
                  </span>
                  <span>{item.label}</span>
                  {item.id === "upload" && state === "active" && (
                    <span className={`${styles.pct} time`}>{Math.round(uploaded * 100)}%</span>
                  )}
                </li>
              );
            })}
          </ol>
          {busy && (
            <div className={styles.bar} role="progressbar" aria-label="Indexing progress">
              <i
                style={{
                  width: `${stepIndex === 0 ? uploaded * 100 : ((stepIndex + 0.5) / STEPS.length) * 100}%`,
                }}
              />
            </div>
          )}
          {finished && <p className={styles.success}>Ready. Search it from the main page.</p>}
          {error && (
            <div className={styles.error} role="alert">
              <p>{error}</p>
              <button className={styles.choose} onClick={reset}>
                Try another file
              </button>
            </div>
          )}
          {finished && (
            <button
              className={styles.primary}
              onClick={() => {
                reset();
                onClose();
              }}
            >
              Start searching
            </button>
          )}
        </div>
      )}
    </dialog>
  );
}
