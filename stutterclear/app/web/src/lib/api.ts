// Talks to the Python app server (stutterclear/app/server.py).

export type Mode = "off" | "gentle" | "strong"
export type DetectorKind = "student" | "heuristic"

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

export const getInfo = () => json<{ student: boolean; student_eval: StudentEval | null }>(fetch("/api/info"))

export const getPlayback = () => json<PlaybackInfo>(fetch("/api/playback"))

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
}

export const getDemos = () => json<{ samples: DemoSample[]; credit: string }>(fetch("/api/demo"))
export const getDemo = (id: string, engine: DemoEngine) => json<DemoTimeline>(fetch(`/api/demo/${id}?engine=${engine}`))
export const demoAudioUrl = (id: string, which: "original" | "clean", engine: DemoEngine) =>
  `/api/demo/${id}/${which}.wav?engine=${engine}`

export const KIND_LABEL: Record<string, string> = {
  sound_rep: "Sound repeat",
  word_rep: "Word repeat",
  repetition: "Repeat",
  prolongation: "Stretched sound",
  interjection: "Filler",
  block: "Block",
  pause: "Pause",
}
