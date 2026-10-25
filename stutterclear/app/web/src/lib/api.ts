// Talks to the Python app server (stutterclear/app/server.py).

export type Mode = "off" | "gentle" | "strong"
export type DetectorKind = "teacher" | "student" | "heuristic"

export interface Device {
  index: number
  name: string
}

export interface Devices {
  inputs: Device[]
  outputs: Device[]
  default_input: number
}

// Times are seconds on the input (microphone) timeline.
export interface Span {
  start: number
  end: number
  kind: string
  word?: string // the word removed, when cuts are placed on word boundaries
}

// One WebSocket message: everything new since the previous one (sent every 50 ms).
export interface Tick {
  t: number
  lag_ms: number
  level_db: number
  removed_s: number
  n_cuts: number
  detector: string
  detect_ms: number
  in_start: number // index of env_in[0] since the session started
  out_start: number
  env_in: number[] // 10 ms peak envelopes
  env_out: number[]
  cuts: Span[]
  events: Span[] // blocks and pauses, never cut
  caption?: { lines: string[]; partial: string } | null // what the listener hears, as text
  detector_note?: string | null // why the detector changed mid-session (say, the GPU dropped)
}

export interface CaptionWord {
  w: string
  t: number // start, seconds
}

export interface StartBody {
  mode: Mode
  detector: DetectorKind
  in_device: number | null
  out_device: number | null
  in_name: string | null // the server matches names, since Windows renumbers devices
  out_name: string | null
  save: boolean
}

export interface PlaybackInfo {
  available: boolean
  original_s: number
  clean_s: number
}

export interface Summary {
  mode_delay_ms: number
  removed_ms: number
  held_silence_ms: number
  pause_shortened_ms: number
  lag_ms: { p50: number; p95: number; max: number }
  cuts: { kind: string; start_s: number; end_s: number }[]
  saved_to?: string
}

// A failed request, worded for the visitor, with the HTTP status when the server answered at all.
export type ApiError = Error & { status?: number }

export const OFFLINE = "Can't reach the StutterClear server. Check your connection and try again."
const BUSY = "The server is busy or restarting. Try again in a moment."

const failure = (message: string, status?: number): ApiError => Object.assign(new Error(message), { status })

// The server's own sentence (`detail`) when it sent one; otherwise what a proxy's bare status means.
function problem(status: number, statusText: string, detail: unknown): ApiError {
  if (typeof detail === "string") return failure(detail, status)
  if (status === 413) return failure("That file is too large for this server.", status)
  if ([429, 502, 503, 504].includes(status)) return failure(BUSY, status)
  return failure(`${status} ${statusText}`, status)
}

async function json<T>(r: Promise<Response>): Promise<T> {
  let res: Response
  try {
    res = await r
  } catch {
    throw failure(OFFLINE) // fetch rejects only when no answer came back at all
  }
  if (!res.ok) {
    const detail = await res.json().then((b) => b?.detail).catch(() => null)
    throw problem(res.status, res.statusText, detail)
  }
  try {
    return (await res.json()) as T
  } catch {
    throw failure(BUSY, res.status) // a page from something in between (a proxy, a captive portal)
  }
}

// Whether trying the same request again later may help: no answer, or a server that was busy.
export const transient = (e: unknown) => {
  const status = (e as ApiError).status
  return status === undefined || status === 408 || status === 429 || status >= 500
}

export const getDevices = () => json<Devices>(fetch("/api/devices"))

// Measured on held-out speakers (scripts/eval_live.py), Gentle mode.
export interface StudentEval {
  removed_pct: number
  words_lost_pct: number
}

// A detector this machine may run, with its measured quality on held-out speakers (Gentle mode).
export interface DetectorInfo {
  kind: DetectorKind
  label: string
  available: boolean
  quality: StudentEval | null
  on_demand?: boolean // runs on the GPU worker, which may be asleep: usable when /api/gpu says so
  delay_ms?: { gentle: number; strong: number } // how far behind the speaker the listener hears
}

// The GPU worker the server wakes on demand (stutterclear/app/gpu.py). Present only when the
// server has one; the features it serves never change while the page is open.
export interface GpuInfo {
  where: string
  typical_s: [number, number] // how long waking usually takes
  features: string[] // "notes", "dictate", "video", "live"
  live: boolean
}

export interface Info {
  student: boolean
  student_eval: StudentEval | null
  notes: boolean
  captions: boolean
  rescue: boolean
  nemotron: boolean
  search: boolean
  model: string
  hosted: boolean
  detectors: DetectorInfo[]
  recommended: DetectorKind
  tts: boolean
  interview: boolean
  feedback: "browser" | "server"
  video?: boolean
  gpu?: GpuInfo
  video_max_mb?: number
  version?: string
}

export const getInfo = () => json<Info>(fetch("/api/info"))

export type GpuState = "off" | "waking" | "connecting" | "ready" | "busy" | "resting" | "unavailable"
// While waking or connecting, what it is doing; while resting, whose share of the hours is used up.
export type GpuStep = "laptop" | "asking" | "queued" | "starting" | "loading" | "reconnecting" | "week" | "day" | null
export type WakeReason = "voice-note" | "dictate" | "video" | "live" | "button"

// The GPU worker right now. Times are seconds; `until` is ISO 8601; `reason` is a plain sentence.
export interface GpuStatus {
  state: GpuState
  step: GpuStep
  now: number
  since: number
  elapsed_s: number | null // since the wake began, while waking or connecting
  typical_s: [number, number]
  where: string | null
  ahead: number // recordings queued or being cleaned
  notes: boolean
  standby: boolean // the team's computer stands by: a wake takes seconds
  live: { available: boolean; delay_ms: { gentle: number; strong: number } | null; reason: string | null }
  sleeps_in_s: number | null
  until: string | null
  reason: string | null
  retry_s: number | null
  can_wake: boolean
  week?: { used_h: number; cap_h: number } | null
}

export const getGpu = () => json<GpuStatus>(fetch("/api/gpu", { cache: "no-store" }))
export const wakeGpu = (reason: WakeReason) => send<GpuStatus>("/api/gpu/wake", { reason })

// A random id for this page. On a public server a live session belongs to the page that started
// it: only that page gets its live updates and its listen-back.
export const PAGE_ID = Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("")
export const playbackUrl = (which: "clean" | "original", version: number) =>
  `/api/playback/${which}.wav?v=${version}&page=${PAGE_ID}`

export const getPlayback = () => json<PlaybackInfo>(fetch(`/api/playback?page=${PAGE_ID}`))

export const startSession = (body: StartBody) =>
  json<{ ok: boolean; detector: string; mode: Mode }>(
    fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
  )

export const stopSession = () =>
  json<{ ok: boolean; summary: Summary | null; playback?: PlaybackInfo }>(fetch("/api/stop", { method: "POST" }))

// "Try a real recording" (UCLASS excerpts): "natural" and "thorough" are the offline model with two
// cutting styles, "live" the real-time pipeline.
export type DemoEngine = "natural" | "thorough" | "live"

export interface DemoSample {
  id: string
  title: string
  who: string
  duration_s: number
  engines: DemoEngine[]
}

// What was said, word by word, with what the cuts removed, and the cleaned text (offline engines).
export interface Transcript {
  words: { w: string; removed: boolean; label: "fluent" | "repeat" | "cutoff" | "filler" }[]
  clean: string
  said: string
}

export interface DemoTimeline {
  id: string
  engine: DemoEngine
  detector: string
  label: string
  lag_ms: number
  detect_ms: number
  original_s: number
  clean_s: number
  removed_s: number
  env_in: number[] // 10 ms peaks of the original
  env_out: number[] // 10 ms peaks of the cleaned audio
  cuts: Span[]
  events: Span[]
  captions?: { original: CaptionWord[]; clean: CaptionWord[] } | null
  text?: Transcript | null
  // Nemotron's review of each proposed cut (offline engines, when a key is set)
  guard?: { model: string; checked: number; kept: { word: string; context: string; reason: string }[]; error?: string } | null
}

export const getDemos = () => json<{ samples: DemoSample[]; credit: string }>(fetch("/api/demo"))
// captions: false skips the word captions (Dictate reads the transcript, not the timed words).
export const getDemo = (id: string, engine: DemoEngine, opts?: { captions?: boolean }) =>
  json<DemoTimeline>(fetch(`/api/demo/${id}?engine=${engine}${opts?.captions === false ? "&captions=0" : ""}`))
export const demoAudioUrl = (id: string, which: "original" | "clean", engine: DemoEngine) =>
  `/api/demo/${id}/${which}.wav?engine=${engine}`

export const sayIt = async (text: string) => {
  const r = await fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
  if (!r.ok) {
    const detail = await r.json().then((b) => b?.detail).catch(() => null)
    throw new Error(typeof detail === "string" ? detail : `${r.status} ${r.statusText}`)
  }
  return { wav: await r.blob(), label: r.headers.get("X-Voice-Label") ?? "Synthetic voice" }
}

// Say it in the call: the server mixes the word into what the listener hears (device sessions only).
export const sayInCall = (text: string) =>
  send<{ ok: boolean; seconds: number; label: string }>("/api/say", { text, page: PAGE_ID })

// Say it here: play the synthetic word in this browser.
export async function playSaid(text: string) {
  const r = await sayIt(text)
  const url = URL.createObjectURL(r.wav)
  const audio = new Audio(url)
  audio.onended = () => URL.revokeObjectURL(url)
  await audio.play()
  return r.label
}

// Clean my video: upload in parts (with progress), then poll; the result plays from
// /api/video/{id}/clean.mp4.
export interface VideoStatus {
  id: string
  status: "queued" | "sound" | "cleaning" | "cutting" | "done" | "error"
  error: string | null
  engine: "natural" | "thorough"
  duration_s?: number | null
  clean_s?: number | null
  cuts?: Span[] | null
  text?: Transcript | null
  slow?: boolean
  note?: NoteStatus["status"] | null // while cleaning: where its soundtrack is (queued for the GPU, listening...)
  ahead?: number | null // recordings before it in the GPU's queue
}

const STALL_MS = 60_000 // a part whose upload makes no progress for this long has stalled (a dead mobile link)

// One part by XHR (fetch cannot report upload progress). It fails as "offline" once it stalls, so the
// caller's retries apply, instead of waiting for the server's hour-long request limit.
function putPart(url: string, part: Blob, onSent: (bytes: number) => void) {
  return new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    let watchdog = 0
    const watch = () => {
      window.clearTimeout(watchdog)
      watchdog = window.setTimeout(() => xhr.abort(), STALL_MS)
    }
    const settle = (fn: () => void) => {
      window.clearTimeout(watchdog)
      fn()
    }
    xhr.open("PUT", url)
    xhr.upload.onprogress = (e) => {
      watch()
      onSent(e.loaded)
    }
    xhr.onload = () =>
      settle(() => {
        if (xhr.status >= 200 && xhr.status < 300) return resolve()
        let detail: unknown = null
        try {
          detail = JSON.parse(xhr.responseText)?.detail
        } catch {
          // not JSON: the status says enough
        }
        reject(problem(xhr.status, xhr.statusText, detail))
      })
    xhr.onerror = xhr.onabort = xhr.ontimeout = () => settle(() => reject(failure(OFFLINE)))
    watch()
    xhr.send(part)
  })
}

// The file goes up in parts (Cloud Run takes at most 32 MiB in one request), in order. A part that
// fails for a passing reason is sent again, up to three more times: the server accepts a part it
// already has, in case only its answer was lost. An upload that fails for good is let go at once,
// so its place is free for the next video.
export async function uploadVideo(file: File, engine: string, onProgress: (fraction: number) => void) {
  const { upload, part_bytes, max_bytes } = await json<{ upload: string; part_bytes: number; max_bytes: number }>(
    fetch("/api/video/upload", { method: "POST" })
  )
  const at = `/api/video/upload/${encodeURIComponent(upload)}`
  try {
    if (file.size > max_bytes) throw failure("That video is too large for this server.", 413)
    const size = Math.max(1, part_bytes)
    for (let i = 0, sent = 0; i === 0 || sent < file.size; i++) {
      const part = file.slice(sent, sent + size)
      for (let attempt = 0; ; attempt++) {
        try {
          await putPart(`${at}?i=${i}`, part, (n) => onProgress((sent + n) / Math.max(1, file.size)))
          break
        } catch (e) {
          if (attempt >= 3 || !transient(e)) throw e
          await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt))
        }
      }
      sent += part.size
    }
    const ext = /\.[a-z0-9]{1,5}$/i.exec(file.name)?.[0].toLowerCase() ?? ".mp4"
    const clean = `/api/video?upload=${encodeURIComponent(upload)}&engine=${engine}&ext=${encodeURIComponent(ext)}`
    for (let attempt = 0; ; attempt++) {
      try {
        return await json<{ id: string }>(fetch(clean, { method: "POST" }))
      } catch (e) {
        // Another video is being cleaned: this one is here already, so it waits for its turn (about a
        // minute and a quarter) rather than being sent all over again.
        if (attempt >= 4 || (e as ApiError).status !== 429) throw e
        await new Promise((r) => setTimeout(r, 5000 * 2 ** attempt))
      }
    }
  } catch (e) {
    fetch(at, { method: "DELETE" }).catch(() => undefined)
    throw e
  }
}
export const getVideo = (id: string) => json<VideoStatus>(fetch(`/api/video/${id}`))

// The fluency journal (on the person's own computer, once started).
export interface JournalSummary {
  enabled: boolean
  stored: boolean // false on a public server, which keeps none
  entries: number
  minutes: number
  per_min: number | null
  this_week: number | null
  last_week: number | null
  days: { date: string; minutes: number; stutters: number; per_min: number | null }[]
  kinds: Record<string, number>
  words: [string, number][]
  onsets: [string, number][]
}
export const getJournal = () => json<JournalSummary>(fetch("/api/journal"))
export const startJournal = () => send<JournalSummary>("/api/journal/start", {})
export const forgetJournal = () => json<{ ok: boolean }>(fetch("/api/journal", { method: "DELETE" }))

export interface InterviewTurn {
  question: string
  answer: string
}

const send = <T,>(path: string, body: unknown) =>
  json<T>(fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }))

export const interviewStart = (body: { role: string; company: string; posting?: string }) =>
  send<{ card: string; question: string }>("/api/interview/start", body)
export const interviewNext = (body: { role: string; company: string; card: string; history: InterviewTurn[] }) =>
  send<{ question: string }>("/api/interview/next", body)
export const interviewSummary = (body: { role: string; company: string; history: InterviewTurn[] }) =>
  send<{ points: string[] }>("/api/interview/summary", body)

export interface FeedbackSummary {
  multipliers: Record<string, number> // per kind of stutter: above 1 more careful, below 1 bolder
  recent: number // review decisions counted for agreement (the last 50)
  agreement: number | null // share of them that kept the cut
  auto_ready: boolean // enough agreement to offer to stop asking
  entries: number
}

export const sendFeedback = (body: Record<string, unknown>) =>
  send<{ entry: Record<string, unknown>; stored: boolean } & Partial<FeedbackSummary>>("/api/feedback", body)
export const getFeedback = () => json<FeedbackSummary & { stored: boolean }>(fetch("/api/feedback"))
export const feedbackSummary = (entries: Record<string, unknown>[]) =>
  send<FeedbackSummary>("/api/feedback/summary", { entries })

// "Clean a voice note": the page uploads WAV, the server cleans it with the offline engines and the
// result plays through the demo player under its note id.
export interface NoteStatus {
  id: string
  status: "queued" | "listening" | "transcribing" | "done" | "error"
  error: string | null
  slow?: boolean // the GPU was full: transcribing on the CPU, which takes longer
  duration_s: number
  ahead?: number | null // recordings before it in the GPU's queue (a GPU worker elsewhere only)
}

export const createNote = (wav: Blob) =>
  json<{ id: string }>(fetch("/api/notes", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: wav }))
export const getNote = (id: string) => json<NoteStatus>(fetch(`/api/notes/${id}`))

export const askRescue = (said: string, context: string) =>
  json<{ suggestions: string[]; model: string }>(
    fetch("/api/rescue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ said, context }),
    })
  )

const post = <T,>(url: string, body: unknown) =>
  json<T>(fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }))

// The meaning guard on typed text: each word, whether it is a candidate cut, and Nemotron's verdict.
export type GuardWord = { w: string; candidate: boolean; kept: boolean; reason: string }
export const checkGuard = (text: string) => post<{ model: string; words: GuardWord[] }>("/api/guard", { text })

export const prepareContext = (situation: string, current: string) =>
  post<{ card: string; sources: string[]; searched: boolean }>("/api/context", { situation, current })

export interface InsightsBody {
  kinds: Record<string, number>
  duration_s: number
  removed_s: number
  words: string[]
  transcript: string
}
export const getInsights = (body: InsightsBody) => post<{ points: string[] }>("/api/insights", body)

export const KIND_LABEL: Record<string, string> = {
  sound_rep: "Sound repeat",
  word_rep: "Word repeat",
  repetition: "Repeat",
  prolongation: "Stretched sound",
  interjection: "Filler",
  block: "Block",
  pause: "Pause",
}
