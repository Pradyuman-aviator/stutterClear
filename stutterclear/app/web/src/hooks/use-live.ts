import { useCallback, useEffect, useRef, useState } from "react"
import { PAGE_ID, type Span, type Tick } from "@/lib/api"
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

const WATCHDOG_MS = 4000 // a live session sends ~30 messages a second: this long without one, the socket is dead
const HIDDEN_MS = 5 * 60_000 // a page hidden this long lets go of its socket until it is shown again
const AWAY_MS = 30 * 60_000 // no pointer, key or touch for this long, and no session: pause (see `pauseWhenAway`)

// The page's live connection (/ws). It reconnects with backoff, at once when the network or the
// visitor comes back, and lets go of the socket when nobody is looking, since an open socket keeps
// a cloud server running (and billed). `pauseWhenAway`: also pause after 30 minutes without any
// pointer, key or touch, on a public server (a page on this computer, like an OBS overlay, stays).
export function useLive(pauseWhenAway = false) {
  const data = useRef<LiveData>(fresh())
  const [stats] = useState(() => createStore<Stats | null>(null))
  const [caption] = useState(() => createStore<Caption | null>(null))
  const [notice] = useState(() => createStore<string | null>(null)) // the session's detector_note, once per change
  const [connected, setConnected] = useState(false)
  const [serverRunning, setServerRunning] = useState(false)
  const [everConnected, setEverConnected] = useState(false)
  const [paused, setPaused] = useState(false)
  const [retryAt, setRetryAt] = useState<number | null>(null) // performance.now() of the next attempt
  const [downAt, setDownAt] = useState(0) // performance.now() when the connection was lost (0: never up)
  const [refused, setRefused] = useState<string | null>(null) // the server turned this page away, and why
  const retry = useRef<() => void>(() => undefined)
  const away = useRef(pauseWhenAway)
  useEffect(() => {
    away.current = pauseWhenAway
  }, [pauseWhenAway])

  useEffect(() => {
    let ws: WebSocket | null = null
    let open = false // ws has opened
    let timer = 0
    let closed = false
    let failures = 0 // attempts since a message last arrived
    let lastStats = 0
    let running = false // the server says a session is live (ticks, not IDLE)
    let ticking = false // ticks arrive on this socket: the watchdog is armed
    let lastMsg = 0
    let lastActive = performance.now()
    let hiddenSince = document.hidden ? performance.now() : 0
    let resting = false // let go of on purpose (hidden, or away): no retries until the visitor is back
    let turnedAway = false // the last close was the server's "too many pages from your network"

    const lost = () => {
      if (open) setDownAt(performance.now())
      open = false
      ticking = false
      setConnected(false)
    }
    // Let go of the socket at once, without waiting for a closing handshake that may never come.
    const drop = () => {
      const sock = ws
      if (!sock) return
      ws = null
      sock.onopen = sock.onclose = sock.onmessage = null
      sock.close()
      lost()
    }
    // 1, 2, 4... up to 30 s, each +-20 %, so the pages of a restarting server do not all come back at once.
    const later = () => {
      if (closed || resting) return
      const wait = Math.min(30000, 1000 * 2 ** failures) * (0.8 + Math.random() * 0.4)
      failures += 1
      timer = window.setTimeout(connect, wait)
      setRetryAt(performance.now() + wait)
    }
    const rest = (isAway: boolean) => {
      resting = true
      window.clearTimeout(timer)
      setRetryAt(null)
      drop()
      if (isAway) setPaused(true)
    }
    const resume = () => {
      lastActive = performance.now()
      if (resting) {
        resting = false
        setPaused(false)
        setDownAt(performance.now()) // the time away is not an outage: no banner unless this fails
      }
      if (ws?.readyState !== WebSocket.OPEN) {
        drop()
        connect()
      }
    }

    const connect = () => {
      window.clearTimeout(timer)
      setRetryAt(null)
      if (closed || resting || ws) return
      const proto = location.protocol === "https:" ? "wss" : "ws"
      const sock = new WebSocket(`${proto}://${location.host}/ws?page=${PAGE_ID}`)
      ws = sock
      sock.onopen = () => {
        open = true
        setConnected(true)
        setEverConnected(true)
      }
      sock.onclose = (e) => {
        if (ws !== sock) return
        ws = null
        lost()
        // The server closes a page's socket after 30 minutes (4000): come back only for someone there.
        if (e.code === 4000) return performance.now() - lastActive < AWAY_MS ? connect() : rest(true)
        if (e.code === 4429) {
          // Too many pages from this network: say so (it is not an outage), and look again now and then
          // (another page closing frees a place), not every second.
          turnedAway = true
          setRefused(e.reason || "Too many StutterClear pages are open from your network.")
          failures = Math.max(failures, 4)
        }
        later()
      }
      sock.onmessage = (m) => {
        failures = 0
        lastMsg = performance.now()
        if (turnedAway) {
          turnedAway = false
          setRefused(null)
        }
        let msg
        try {
          msg = JSON.parse(m.data)
        } catch {
          return
        }
        if (msg.idle) {
          ticking = false
          notice.set(null)
          if (running) {
            running = false
            setServerRunning(false)
          }
          return
        }
        ticking = true
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
        if (tick.detector_note) notice.set(tick.detector_note)
      }
    }

    const check = window.setInterval(() => {
      const now = performance.now()
      if (ws && ticking && now - lastMsg > WATCHDOG_MS) {
        drop() // a live session went quiet: the connection died without saying so
        connect()
      } else if (!resting && hiddenSince && now - hiddenSince > HIDDEN_MS) rest(false)
      else if (!resting && away.current && !running && now - lastActive > AWAY_MS) rest(true)
    }, 1000)
    const onVisibility = () => {
      if (document.hidden) hiddenSince = performance.now()
      else {
        hiddenSince = 0
        resume()
      }
    }
    // Any pointer, key or touch counts as someone there, and brings a paused page back.
    const onActivity = () => {
      lastActive = performance.now()
      if (resting && !document.hidden) resume()
    }
    const onOnline = () => !resting && resume()
    const activity = ["pointerdown", "pointermove", "keydown", "wheel", "touchstart"] as const
    activity.forEach((t) => addEventListener(t, onActivity, { passive: true }))
    document.addEventListener("visibilitychange", onVisibility)
    addEventListener("online", onOnline)
    retry.current = resume
    connect()
    return () => {
      closed = true
      window.clearTimeout(timer)
      window.clearInterval(check)
      activity.forEach((t) => removeEventListener(t, onActivity))
      document.removeEventListener("visibilitychange", onVisibility)
      removeEventListener("online", onOnline)
      retry.current = () => undefined
      drop()
    }
  }, [stats, caption, notice])

  const reset = () => {
    data.current = fresh()
    stats.set(null)
    caption.set(null)
    notice.set(null)
  }
  const retryNow = useCallback(() => retry.current(), [])

  return { data, stats, caption, notice, connected, serverRunning, reset, retryAt, retryNow, everConnected, paused, downAt,
           refused }
}
