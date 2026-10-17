import { useEffect, useRef, useState } from "react"
import type { Span, Tick } from "@/lib/api"
import { createStore } from "@/lib/store"

export const STEP_S = 0.01 // one envelope point per 10 ms
const KEEP = 3000 // points kept per lane (30 s)

// Mutable live state. Animation loops read it every frame, so it lives in a ref and never
// triggers React renders on its own.
export interface LiveData {
  envIn: number[]
  envOut: number[]
  inOffset: number // points dropped from the front of envIn, so envIn[i] is at (inOffset + i) * STEP_S
  outOffset: number
  cuts: Span[]
  events: Span[]
  receivedAt: number // performance.now() of the last message
  clockOffset: number | null // data time minus performance.now(), in s, smoothed over messages
  level: number // mic peak of the last message, 0..1 on a perceptual scale
  outLevel: number
  lastCutAt: number // performance.now() of the newest cut, -Infinity if none
  lastCutKind: string
  lastBlockAt: number
  lastSoundAt: number // performance.now() when the mic last carried anything above -80 dB
  heardSound: boolean // anything at all this session (some mics send pure zeros until you speak)
  cutCount: number
}

export interface Caption {
  lines: string[] // finished lines, oldest first
  partial: string // the line being spoken
}

export interface Stats {
  lag_ms: number
  n_cuts: number
  removed_s: number
  detect_ms: number
  detector: string
}

const fresh = (): LiveData => ({
  envIn: [], envOut: [], inOffset: 0, outOffset: 0, cuts: [], events: [], receivedAt: 0, clockOffset: null,
  level: 0, outLevel: 0, lastCutAt: -Infinity, lastCutKind: "", lastBlockAt: -Infinity, lastSoundAt: performance.now(), heardSound: false, cutCount: 0,
})

// Peak (0..1 linear) to a 0..1 loudness that moves nicely: -60 dB is silent, -6 dB is full.
export function loudness(peak: number) {
  const db = 20 * Math.log10(Math.max(peak, 1e-6))
  return Math.min(1, Math.max(0, (db + 60) / 54))
}

function trim(env: number[], offset: number): [number[], number] {
  if (env.length <= 2 * KEEP) return [env, offset]
  const drop = env.length - KEEP
  return [env.slice(drop), offset + drop]
}

// Data time to draw at `now`: a steady clock a little behind the newest data, so the waveform
// scrolls at exactly real time and never waits for the next message.
export const PLAYBACK_DELAY_S = 0.15
export function liveClock(d: LiveData, now: number) {
  return d.clockOffset === null ? 0 : now / 1000 + d.clockOffset - PLAYBACK_DELAY_S
}

export function useLive() {
  const data = useRef<LiveData>(fresh())
  const [stats] = useState(() => createStore<Stats | null>(null))
  const [caption] = useState(() => createStore<Caption | null>(null))
  const [connected, setConnected] = useState(false)
  const [serverRunning, setServerRunning] = useState(false)

  useEffect(() => {
    let ws: WebSocket | null = null
    let retry: number | undefined
    let closed = false
    let lastStats = 0
    let running = false

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws"
      ws = new WebSocket(`${proto}://${location.host}/ws`)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!closed) retry = window.setTimeout(connect, 1000)
      }
      ws.onmessage = (m) => {
        const msg = JSON.parse(m.data)
        if (msg.idle) {
          if (running) {
            running = false
            setServerRunning(false)
          }
          return
        }
        if (!running) {
          running = true
          setServerRunning(true)
        }
        const d = data.current
        const tick = msg as Tick
        const now = performance.now()
        // Joined mid-session, or missed messages: restart the lane at the server's index so the
        // waveform and the cut times stay on the same clock.
        if (tick.in_start !== d.inOffset + d.envIn.length) {
          d.envIn = []
          d.inOffset = tick.in_start
        }
        if (tick.out_start !== d.outOffset + d.envOut.length) {
          d.envOut = []
          d.outOffset = tick.out_start
        }
        d.envIn.push(...tick.env_in)
        d.envOut.push(...tick.env_out)
        ;[d.envIn, d.inOffset] = trim(d.envIn, d.inOffset)
        ;[d.envOut, d.outOffset] = trim(d.envOut, d.outOffset)
        if (tick.cuts.length) {
          d.cuts.push(...tick.cuts)
          d.lastCutAt = now
          d.lastCutKind = tick.cuts[tick.cuts.length - 1].kind
          d.cutCount += tick.cuts.length
        }
        if (tick.events.length) {
          d.events.push(...tick.events)
          if (tick.events.some((e) => e.kind === "block")) d.lastBlockAt = now
        }
        const horizon = (d.inOffset + d.envIn.length) * STEP_S - 40
        if (d.cuts.length > 200) d.cuts = d.cuts.filter((c) => c.end > horizon)
        if (d.events.length > 200) d.events = d.events.filter((c) => c.end > horizon)
        const peak = Math.max(0, ...tick.env_in)
        if (peak > 1e-4) {
          d.lastSoundAt = now
          d.heardSound = true
        }
        d.level = loudness(peak)
        d.outLevel = loudness(Math.max(0, ...tick.env_out))
        d.receivedAt = now
        const est = (d.inOffset + d.envIn.length) * STEP_S - now / 1000
        if (d.clockOffset === null || Math.abs(est - d.clockOffset) > 0.3) d.clockOffset = est
        else d.clockOffset += (est - d.clockOffset) * 0.05
        const cap = tick.caption
        if (cap) {
          const cur = caption.get()
          if (!cur || cur.partial !== cap.partial || cur.lines.join("|") !== cap.lines.join("|")) caption.set(cap)
        }
        if (now - lastStats > 100) {
          lastStats = now
          stats.set({ lag_ms: tick.lag_ms, n_cuts: tick.n_cuts, removed_s: tick.removed_s,
                     detect_ms: tick.detect_ms, detector: tick.detector })
        }
      }
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(retry)
      ws?.close()
    }
  }, [stats, caption])

  const reset = () => {
    data.current = fresh()
    stats.set(null)
    caption.set(null)
  }

  return { data, stats, caption, connected, serverRunning, reset }
}
