import { useCallback, useEffect, useRef, useState, type RefObject } from "react"
import { demoAudioUrl, getDemo, type DemoEngine, type DemoTimeline } from "@/lib/api"
import type { Store } from "@/lib/store"
import { loudness, PLAYBACK_DELAY_S, STEP_S, type Caption, type LiveData, type Stats } from "./use-live"

export type Hear = "clean" | "original" | "heard" // heard: the live engine's call version

export interface DemoState {
  id: string
  engine: DemoEngine
  hear: Hear
  timeline: DemoTimeline | null // null while loading
  playing: boolean
  t: number // seconds, updated a few times a second for the time display
  error?: string
}

// Plays a sample and feeds the same LiveData the mic would, frame by frame from the audio clock,
// so the Fluency Mirror, orb, status and stats all behave exactly as in a live session.
// Captions up to time t, grouped into lines at pauses (as the live captioner does at endpoints).
function captionAt(words: { w: string; t: number }[], t: number): Caption {
  const lines: string[][] = [[]]
  let last = -1
  for (const w of words) {
    if (w.t > t) break
    if ((last >= 0 && w.t - last > 0.9) || lines[lines.length - 1].length >= 12) lines.push([])
    lines[lines.length - 1].push(w.w)
    last = w.t
  }
  const text = lines.map((l) => l.join(" ")).filter(Boolean)
  return { lines: text.slice(0, -1), partial: text[text.length - 1] ?? "" }
}

// Timelines of the sample clips opened so far, so a sample plays again while the server is
// unreachable (the browser keeps its audio for an hour). Voice notes are left out: the server
// forgets them, and so does the page.
const sampleTimelines = new Map<string, DemoTimeline>()
const NO_AUDIO = "The audio could not be loaded. Check your connection and try again."
const pauseAll = (tracks: Partial<Record<Hear, HTMLAudioElement>> | null) =>
  Object.values(tracks ?? {}).forEach((a) => a?.pause())

export function useDemo(
  data: RefObject<LiveData>,
  stats: Store<Stats | null>,
  reset: () => void,
  caption?: Store<Caption | null>
) {
  const [state, setState] = useState<DemoState | null>(null)
  const audio = useRef<Partial<Record<Hear, HTMLAudioElement>> | null>(null)
  const timeline = useRef<DemoTimeline | null>(null)
  const hear = useRef<Hear>("clean")
  const cursor = useRef({ cut: 0, event: 0, lastStats: 0 })
  const progress = useRef<SVGCircleElement | null>(null)
  const loadId = useRef(0)

  const current = () => audio.current?.[hear.current] ?? null

  const close = useCallback(() => {
    loadId.current++
    pauseAll(audio.current)
    audio.current = null
    timeline.current = null
    reset()
    setState(null)
  }, [reset])

  const play = useCallback(() => {
    const a = current()
    if (!a) return
    if (a.ended) a.currentTime = 0
    a.play()
      .then(() => setState((s) => s && { ...s, playing: true }))
      .catch(() => setState((s) => s && { ...s, playing: false }))
  }, [])

  const pause = useCallback(() => {
    current()?.pause()
    setState((s) => s && { ...s, playing: false })
  }, [])

  const load = useCallback(
    async (id: string, engine: DemoEngine, keep?: { hear: Hear; t: number; playing: boolean }) => {
      const my = ++loadId.current
      pauseAll(audio.current)
      hear.current = keep?.hear ?? "clean"
      setState({ id, engine, hear: hear.current, timeline: null, playing: false, t: 0 })
      try {
        const key = `${id}/${engine}`
        const tl = sampleTimelines.get(key) ?? (await getDemo(id, engine))
        if (!id.startsWith("note-")) sampleTimelines.set(key, tl)
        if (my !== loadId.current) return
        const make = (which: Hear) => {
          const a = new Audio(demoAudioUrl(id, which, engine))
          a.preload = "auto"
          a.addEventListener("ended", () => {
            if (audio.current?.[hear.current] === a) setState((s) => s && { ...s, playing: false })
          })
          a.addEventListener("error", () => {
            if (audio.current?.[hear.current] === a) setState((s) => s && { ...s, playing: false, error: NO_AUDIO })
          })
          return a
        }
        // Only the live engine has the call's own version (with its pauses): elsewhere, the clean voice.
        if (hear.current === "heard" && tl.heard_s == null) hear.current = "clean"
        audio.current = { original: make("original"), clean: make("clean"), ...(tl.heard_s != null ? { heard: make("heard") } : {}) }
        timeline.current = tl
        cursor.current = { cut: 0, event: 0, lastStats: 0 }
        reset()
        const a = audio.current[hear.current]!
        if (keep?.t) a.currentTime = keep.t
        setState({ id, engine, hear: hear.current, timeline: tl, playing: false, t: keep?.t ?? 0 })
        if (keep?.playing ?? true) {
          await a.play().catch(() => undefined)
          if (my === loadId.current) setState((s) => s && { ...s, playing: !a.paused })
        }
      } catch (e) {
        if (my === loadId.current) setState((s) => s && { ...s, error: (e as Error).message })
      }
    },
    [reset]
  )

  // Switch between hearing the original and the cleaned voice without losing your place.
  const setHear = useCallback((h: Hear) => {
    const tracks = audio.current
    if (!tracks || h === hear.current) return
    const from = tracks[hear.current]
    const to = tracks[h]
    if (!from || !to) return
    data.current.envOut = [] // the mirror redraws the output this track plays
    const wasPlaying = !from.paused
    from.pause()
    to.currentTime = Math.min(from.currentTime, Number.isFinite(to.duration) ? to.duration : from.currentTime)
    hear.current = h
    setState((s) => s && { ...s, hear: h })
    if (wasPlaying) to.play().catch(() => undefined)
  }, [])

  const setEngine = useCallback(
    (engine: DemoEngine) => {
      const s = state
      if (!s || engine === s.engine) return
      const a = current()
      load(s.id, engine, { hear: hear.current, t: a?.currentTime ?? 0, playing: !!a && !a.paused })
    },
    [state, load]
  )

  // Drive the page from the audio clock.
  const active = state !== null
  useEffect(() => {
    if (!active) return
    let raf = 0
    let lastT = -1
    const frame = (now: number) => {
      const tl = timeline.current
      const a = current()
      if (tl && a) {
        const t = a.currentTime
        const d = data.current
        const envOut = hear.current === "heard" && tl.env_heard ? tl.env_heard : tl.env_out
        const nIn = Math.min(tl.env_in.length, Math.floor(t / STEP_S))
        const nOut = Math.min(envOut.length, Math.floor(t / STEP_S))
        if (nIn < d.envIn.length || nOut < d.envOut.length) {
          // Went backwards (restart or a switch): rebuild from the start.
          d.envIn = []
          d.envOut = []
          d.cuts = []
          d.events = []
          cursor.current.cut = 0
          cursor.current.event = 0
        }
        for (let i = d.envIn.length; i < nIn; i++) d.envIn.push(tl.env_in[i])
        for (let i = d.envOut.length; i < nOut; i++) d.envOut.push(envOut[i])
        const c = cursor.current
        // A cut is shown once it has fully passed, as the live detector decides after the fact.
        while (c.cut < tl.cuts.length && tl.cuts[c.cut].end <= t) {
          const cut = tl.cuts[c.cut++]
          d.cuts.push(cut)
          d.lastCutAt = now
          d.lastCutKind = cut.kind
        }
        while (c.event < tl.events.length && tl.events[c.event].end <= t) {
          const ev = tl.events[c.event++]
          d.events.push(ev)
          if (ev.kind === "block") d.lastBlockAt = now
        }
        d.clockOffset = t - now / 1000 + PLAYBACK_DELAY_S // the mirror's clock = the audio clock
        d.receivedAt = now
        d.level = a.paused ? 0 : loudness(tl.env_in[Math.max(0, nIn - 1)] ?? 0)
        d.outLevel = a.paused ? 0 : loudness(envOut[Math.max(0, nOut - 1)] ?? 0)
        d.heardSound = true
        d.lastSoundAt = now
        if (now - c.lastStats > 150) {
          c.lastStats = now
          const removed = d.cuts.reduce((sum, x) => sum + x.end - x.start, 0)
          stats.set({ lag_ms: tl.lag_ms, n_cuts: d.cuts.length, removed_s: Math.round(removed * 10) / 10,
                      detect_ms: tl.detect_ms, detector: tl.detector })
        }
        const dur = Number.isFinite(a.duration) ? a.duration : tl.original_s
        if (progress.current) progress.current.style.strokeDashoffset = String(1 - Math.min(1, t / dur))
        if (Math.abs(t - lastT) >= 0.25 || (a.ended && t !== lastT)) {
          lastT = t
          setState((s) => s && { ...s, t })
          const words = tl.captions?.[hear.current]
          if (caption && words) caption.set(captionAt(words, t))
        }
      }
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [active, data, stats, caption])

  // Leaving the page or unmounting: stop the audio.
  useEffect(() => () => pauseAll(audio.current), [])

  return { state, load, play, pause, setHear, setEngine, close, progress }
}
