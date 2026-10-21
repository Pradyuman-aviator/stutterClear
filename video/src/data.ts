// The video's data comes from scripts/export_video_data.py (real pipeline output); the timeline
// sizes every scene to its narration, so a new voice re-times the whole video by itself.

export interface Narration {
  id: string
  text: string
  file: string
  seconds: number
}

export interface Word {
  w: string
  removed: boolean
  label: string
  s: number
  e: number
}

export interface Data {
  clip: {
    before_s: number
    after_s: number
    env_before: number[]
    env_after: number[]
    cuts: { start: number; end: number; kind: string }[]
    words: Word[]
    clean: string
    credit: string
  }
  guard: { model: string; words: { w: string; candidate: boolean; kept: boolean; reason: string }[] }
  rescue: { said: string; onset: string; context: string; suggestions: string[] }
  numbers: {
    live_strong_removed: number
    live_strong_lost: number
    live_gentle_removed: number
    live_gentle_lost: number
    rec_thorough_removed: number
    rec_thorough_lost: number
    speakers: number
  }
  model: string
  narration: Narration[]
  voice_label: string
}

export const FPS = 30
const LEAD_S = 0.4 // quiet before the first line of a scene

type Step = { line: string } | { audio: "before" | "after" } | { gap: number }

// Scene id, what plays in it, and the pause after it.
const PLAN: [string, Step[], number][] = [
  ["hook", [{ line: "hook" }], 1.0],
  ["intro", [{ line: "intro" }], 1.0],
  ["beforeafter", [{ line: "before" }, { gap: 0.3 }, { audio: "before" }, { gap: 0.7 }, { line: "after" }, { gap: 0.3 }, { audio: "after" }], 1.2],
  ["how", [{ line: "how" }], 1.2],
  ["guard", [{ line: "guard" }], 1.6],
  ["calls", [{ line: "calls" }], 1.0],
  ["rescue", [{ line: "rescue" }], 1.6],
  ["app", [{ line: "app" }], 1.2],
  ["numbers", [{ line: "numbers" }], 1.4],
  ["end", [{ line: "end" }], 3.5],
]

export interface Cue {
  kind: "line" | "audio"
  id: string // narration id, or "before" / "after"
  at: number // frames from the scene's start
  frames: number
  text?: string
  file: string
}

export interface Scene {
  id: string
  from: number
  frames: number
  cues: Cue[]
}

export const sec = (s: number) => Math.round(s * FPS)

export function timeline(d: Data): Scene[] {
  const line = (id: string) => {
    const n = d.narration.find((x) => x.id === id)
    if (!n) throw new Error(`no narration line ${id}`)
    return n
  }
  const scenes: Scene[] = []
  let from = 0
  for (const [id, steps, tail] of PLAN) {
    let t = LEAD_S
    const cues: Cue[] = []
    for (const st of steps) {
      if ("gap" in st) t += st.gap
      else if ("line" in st) {
        const n = line(st.line)
        cues.push({ kind: "line", id: n.id, at: sec(t), frames: sec(n.seconds), text: n.text, file: n.file })
        t += n.seconds
      } else {
        const s = st.audio === "before" ? d.clip.before_s : d.clip.after_s
        cues.push({ kind: "audio", id: st.audio, at: sec(t), frames: sec(s), file: `generated/${st.audio}.wav` })
        t += s
      }
    }
    const frames = sec(t + tail)
    scenes.push({ id, from, frames, cues })
    from += frames
  }
  return scenes
}
