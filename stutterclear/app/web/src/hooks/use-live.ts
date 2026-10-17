import { useEffect, useRef, useState } from "react"
import type { Span, Tick } from "@/lib/api"

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
  level: number // mic peak of the last message, 0..1 on a perceptual scale
  outLevel: number
  lastCutAt: number // performance.now() of the newest cut, -Infinity if none
  lastCutKind: string
  lastBlockAt: number
  cutCount: number
}

export interface Stats {
  lag_ms: number
  n_cuts: number
  removed_s: number
  detect_ms: number
  detector: string
}

const fresh = (): LiveData => ({
  envIn: [], envOut: [], inOffset: 0, outOffset: 0, cuts: [], events: [], receivedAt: 0,
  level: 0, outLevel: 0, lastCutAt: -Infinity, lastCutKind: "", lastBlockAt: -Infinity, cutCount: 0,
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

export function useLive() {
  const data = useRef<LiveData>(fresh())
  const [stats, setStats] = useState<Stats | null>(null)
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
        d.level = loudness(Math.max(0, ...tick.env_in))
        d.outLevel = loudness(Math.max(0, ...tick.env_out))
        d.receivedAt = now
        if (now - lastStats > 100) {
          lastStats = now
          setStats({ lag_ms: tick.lag_ms, n_cuts: tick.n_cuts, removed_s: tick.removed_s,
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
  }, [])

  const reset = () => {
    data.current = fresh()
    setStats(null)
  }

  return { data, stats, connected, serverRunning, reset }
}
