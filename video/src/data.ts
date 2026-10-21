// The film's data comes from scripts/export_video_data.py (real pipeline output) and its script from
// storyboard.json. The timeline sizes every scene to its narration, so a new voice re-times the film.
import board from "../storyboard.json"

export interface Narration {
  id: string
  text: string
  file: string
  seconds: number
}

export interface Word {
  w: string
  removed: boolean
  label: string // fluent | repeat | cutoff | filler
  s: number
  e: number
}

export interface Cut {
  start: number
  end: number
  kind: string // word_rep | sound_rep | interjection | pause
  words: string
}

export interface Data {
  clip: {
    before_s: number
    after_s: number
    env_before: number[] // peak per 10 ms
    env_after: number[]
    cuts: Cut[] // on the original clock, in order
    words: Word[] // CrisperWhisper's verbatim words, on the original clock
    after_words: { w: string; s: number; e: number }[] // the words that remain, on the cleaned clock
    clean: string
    said: string
    guard: { model: string; checked: number; kept: { word: string; context: string; reason: string }[]; error?: string } | null
    probs: number[][] // [frame][class], 20 ms frames, the teacher's output
    classes: string[]
    frame_ms: number
    removed_pct: number
    lost_pct: number
    credit: string
  }
  // the meaning guard's real exchange on a typed sentence: the marked request and Nemotron's JSON reply
  guard: { model: string; request: string; reply: string; agreed: string; words: { w: string; candidate: boolean; kept: boolean; reason: string }[] }
  rescue: { said: string; onset: string; context: string; suggestions: string[] }
  numbers: {
    live_strong_removed: number
    live_strong_lost: number
    live_gentle_removed: number
    live_gentle_lost: number
    rec_thorough_removed: number
    rec_thorough_lost: number
    speakers: number
    f1_sep28k: number
  }
  model: string
  narration: Narration[]
  voice_label: string
  // "Say it": the rescued word in a synthetic voice (NVIDIA Magpie with a key, else a labelled placeholder)
  say: { word: string; file: string; seconds: number; label: string }
}

export type Step = { line: string } | { audio: AudioKey } | { gap: number }
export type AudioKey = "before" | "after" | "say"
export type TransitionKind = "fade" | "slide" | "slide-up" | "wipe" | "push" | "iris" | "flip" | "clock" | "zoom" | "none"

export interface BoardScene {
  id: string
  title: string
  steps: Step[]
  lead?: number // seconds before the first step
  tail?: number // seconds after the last step
  transition?: TransitionKind // how this scene comes in
  nvidia?: NvKey[] // the NVIDIA tracker's lit pills during this scene (absent: tracker hidden)
  on_screen_text?: string[]
  visuals?: string
}

export type NvKey = "gpu" | "nemo" | "nemotron" | "magpie"
export const NV_ITEMS: { key: NvKey; label: string }[] = [
  { key: "gpu", label: "GPU · CUDA" },
  { key: "nemo", label: "NeMo FastConformer" },
  { key: "nemotron", label: "Nemotron 3 Super" },
  { key: "magpie", label: "Magpie TTS" },
]

export const BOARD = board as unknown as { title: string; scenes: BoardScene[] }
export const SCENE_IDS = BOARD.scenes.map((s) => s.id)

export const FPS = 30
export const TRANSITION = 20 // frames two scenes overlap
const LEAD_S = 0.6
const TAIL_S = 1.0

export const sec = (s: number) => Math.round(s * FPS)

export interface Cue {
  kind: "line" | "audio"
  id: string // narration id (scene-k), or "before" / "after"
  k: number // index among the scene's lines; -1 for clip audio
  at: number // frames from the scene's start
  frames: number
  text?: string
  file: string
}

export interface Scene {
  id: string
  title: string
  index: number
  from: number // first frame in the film (scenes overlap by TRANSITION)
  frames: number
  cues: Cue[]
  transition: TransitionKind
  nvidia?: NvKey[]
}

export interface SceneProps {
  d: Data
  scene: Scene
}

export function timeline(d: Data): Scene[] {
  const byId = new Map(d.narration.map((n) => [n.id, n]))
  const scenes: Scene[] = []
  let from = 0
  BOARD.scenes.forEach((b, index) => {
    const transition: TransitionKind = index === 0 ? "none" : b.transition ?? "fade"
    if (transition !== "none") from -= TRANSITION
    let t = b.lead ?? LEAD_S
    const cues: Cue[] = []
    let k = 0
    for (const st of b.steps) {
      if ("gap" in st) t += st.gap
      else if ("line" in st) {
        const id = `${b.id}-${k}`
        const n = byId.get(id)
        if (!n || n.text !== st.line) throw new Error(`Narration ${id} is missing or out of date: run scripts/export_video_data.py`)
        cues.push({ kind: "line", id, k, at: sec(t), frames: sec(n.seconds), text: n.text, file: n.file })
        k++
        t += n.seconds
      } else {
        const s = st.audio === "before" ? d.clip.before_s : st.audio === "after" ? d.clip.after_s : d.say.seconds
        const file = st.audio === "say" ? d.say.file : `generated/${st.audio}.wav`
        cues.push({ kind: "audio", id: st.audio, k: -1, at: sec(t), frames: sec(s), file })
        t += s
      }
    }
    const frames = sec(t + (b.tail ?? TAIL_S))
    scenes.push({ id: b.id, title: b.title, index, from: Math.max(0, from), frames, cues, transition, nvidia: b.nvidia })
    from = Math.max(0, from) + frames
  })
  return scenes
}

export const filmFrames = (scenes: Scene[]) => {
  const last = scenes[scenes.length - 1]
  return last.from + last.frames
}

// The k-th narration line of a scene (0-based), and when the clip audio plays in it.
export const lines = (s: Scene) => s.cues.filter((c) => c.kind === "line")
export const lineCue = (s: Scene, k: number): Cue =>
  lines(s)[k] ?? { kind: "line", id: `${s.id}-${k}`, k, at: 0, frames: 0, file: "" }
export const clipCue = (s: Scene, which: AudioKey): Cue | undefined =>
  s.cues.find((c) => c.kind === "audio" && c.id === which)

// Where speech plays in the film (narration and the real clip), for ducking the music.
export interface Span {
  a: number
  b: number
  kind: "line" | "audio"
}
export const speechSpans = (scenes: Scene[]): Span[] =>
  scenes.flatMap((s) => s.cues.map((c) => ({ a: s.from + c.at, b: s.from + c.at + c.frames, kind: c.kind })))

const MUSIC = 0.3 // between lines
const UNDER_LINE = 0.1
const UNDER_CLIP = 0.025 // the real voice must be heard clearly
const RAMP = 14

export function musicVolume(f: number, spans: Span[], total: number): number {
  let v = MUSIC
  for (const sp of spans) {
    const d = f < sp.a ? sp.a - f : f > sp.b ? f - sp.b : 0
    if (d >= RAMP) continue
    const low = sp.kind === "audio" ? UNDER_CLIP : UNDER_LINE
    v = Math.min(v, low + (MUSIC - low) * (d / RAMP))
  }
  const fadeIn = Math.min(1, f / sec(1.5))
  const fadeOut = Math.min(1, Math.max(0, (total - f) / sec(4)))
  return v * fadeIn * fadeOut
}
