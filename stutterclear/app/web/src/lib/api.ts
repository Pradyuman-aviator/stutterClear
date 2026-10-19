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

async function json<T>(r: Promise<Response>): Promise<T> {
  const res = await r
  if (!res.ok) {
    const detail = await res.json().then((b) => b?.detail).catch(() => null)
    throw new Error(typeof detail === "string" ? detail : `${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
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
}

export const getInfo = () =>
  json<{
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
  }>(fetch("/api/info"))

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
export const getDemo = (id: string, engine: DemoEngine) => json<DemoTimeline>(fetch(`/api/demo/${id}?engine=${engine}`))
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
