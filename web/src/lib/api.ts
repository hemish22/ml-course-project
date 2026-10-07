export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type VideoInfo = {
  id: string;
  title: string;
  duration: number;
  fps: number;
  n_frames: number;
  n_segments: number;
  has_speech: boolean;
  video_url: string;
  poster_url: string;
};

export type Moment = {
  start: number;
  end: number;
  clip_start: number;
  clip_end: number;
  score: number;
  visual: number;
  transcript: number;
  thumbnail: string;
  text: string;
};

export type Trace = {
  t: number[];
  visual: number[];
  transcript: number[];
};

export type SearchResponse = {
  query: string;
  alpha: number;
  moments: Moment[];
  trace: Trace;
};

export type Segment = { seg_idx: number; start: number; end: number; text: string };

export type AppConfig = {
  alpha: number;
  score_threshold: number;
  max_results: number;
  models: { clip: string; whisper: string; text: string };
};

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { signal });
  if (!response.ok) {
    const detail = await response.json().then((body) => body.detail).catch(() => null);
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const fetchConfig = () => getJson<AppConfig>("/api/config");
export const fetchVideos = () => getJson<VideoInfo[]>("/api/videos");
export const fetchTranscript = (videoId: string) =>
  getJson<Segment[]>(`/api/videos/${encodeURIComponent(videoId)}/transcript`);

export function fetchSearch(
  params: { video: string; q: string; alpha: number; threshold: number; limit: number },
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const query = new URLSearchParams({
    video: params.video,
    q: params.q,
    alpha: String(params.alpha),
    threshold: String(params.threshold),
    limit: String(params.limit),
  });
  return getJson<SearchResponse>(`/api/search?${query}`, signal);
}

export const mediaUrl = (path: string) => `${API_URL}${path}`;

export type JobStage = "queued" | "prepare" | "frames" | "speech" | "picture" | "text" | "finish" | "done";

export type Job = {
  id: string;
  video_id: string;
  state: "queued" | "running" | "done" | "error";
  stage: JobStage;
  error: string | null;
};

export const fetchJob = (jobId: string) => getJson<Job>(`/api/jobs/${encodeURIComponent(jobId)}`);

/** Upload a file as the raw request body; XHR is used because fetch cannot report upload progress. */
export function uploadVideo(
  file: File,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<{ job_id: string; video_id: string }> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${API_URL}/api/upload?filename=${encodeURIComponent(file.name)}`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    request.onload = () => {
      let body: { detail?: unknown; job_id?: string; video_id?: string } = {};
      try {
        body = JSON.parse(request.responseText);
      } catch {
        // fall through to the generic message
      }
      if (request.status >= 200 && request.status < 300 && body.job_id && body.video_id) {
        resolve({ job_id: body.job_id, video_id: body.video_id });
      } else {
        reject(new Error(typeof body.detail === "string" ? body.detail : `Upload failed (${request.status})`));
      }
    };
    request.onerror = () => reject(new Error(`Could not reach the server at ${API_URL}.`));
    signal?.addEventListener("abort", () => {
      request.abort();
      reject(new DOMException("Upload cancelled", "AbortError"));
    });
    request.send(file);
  });
}

export async function removeVideo(videoId: string): Promise<void> {
  const response = await fetch(`${API_URL}/api/videos/${encodeURIComponent(videoId)}`, { method: "DELETE" });
  if (!response.ok) {
    const detail = await response.json().then((body) => body.detail).catch(() => null);
    throw new Error(typeof detail === "string" ? detail : `Could not remove the video (${response.status})`);
  }
}
