import { useCallback, useEffect, useRef, useState } from "react"
import { getGpu, transient, wakeGpu, type ApiError, type GpuStatus, type WakeReason } from "@/lib/api"

const FAST_MS = 2000 // while the GPU wakes or works, and for a minute after the page needed it
const SLOW_MS = 30000
const WATCH_MS = 60000

export interface Gpu {
  status: GpuStatus | null
  at: number // performance.now() when `status` arrived: its clocks count on from there
  error: { message: string; retry: boolean } | null // why the visitor's own Wake press failed; retry: worth pressing again
  wake: (reason: WakeReason, opts?: { quiet?: boolean }) => Promise<void>
  refresh: () => void
}

// Whether a recording needs the GPU woken: it is asleep, or offline but allowed to try again.
export const wakeable = (s: GpuStatus | null) => !!s && (s.state === "off" || (s.state === "unavailable" && s.can_wake))

// Record and upload wait while the GPU rests (its free hours are used up) or cannot be woken, unless
// the team's computer stands by: it serves them whatever Kaggle's state.
export const blocked = (s: GpuStatus | null) =>
  !!s && !s.standby && (s.state === "resting" || (s.state === "unavailable" && !s.can_wake))

// "2–4 min" from typical_s [120, 240]
export function typical([a, b]: [number, number], unit = "min") {
  const lo = Math.max(1, Math.round(a / 60))
  const hi = Math.max(lo, Math.round(b / 60))
  return lo === hi ? `${lo} ${unit}` : `${lo}–${hi} ${unit}`
}

// What a recording that waits for the GPU shows: what it waits for, or its place in line.
// `remote`: a GPU worker elsewhere cleans it (else the models on this computer).
export function queuedText(s: GpuStatus | null, ahead: number | null | undefined, remote: boolean) {
  const n = ahead ?? 0
  if (s && (s.state === "waking" || s.state === "connecting"))
    return n ? `Waiting for the GPU to wake, ${n} ${n === 1 ? "recording" : "recordings"} ahead of yours…` : "Waiting for the GPU to wake…"
  if (n === 1) return "The GPU is cleaning another recording; yours is next."
  if (n > 1) return `In line: ${n} recordings ahead of yours.`
  return remote ? "Sending it to the GPU…" : "Waiting for the models…"
}

// The GPU worker the server wakes on demand (/api/gpu): polled every 2 s while it wakes or works
// and for a minute after the page needed it, every 30 s otherwise, never while the page is hidden
// or `paused` (the visitor is away). Nothing here wakes it by itself: wake() runs only on a
// visitor's action (a Wake button, Record once the mic is allowed, a file once it passed its
// checks). A quiet wake shows no refusal: the upload right after it gets the same answer and
// explains it where the visitor is looking.
export function useGpu(enabled: boolean, paused = false): Gpu {
  const [status, setStatus] = useState<GpuStatus | null>(null)
  const [at, setAt] = useState(0)
  const [error, setError] = useState<Gpu["error"]>(null)
  const latest = useRef<GpuStatus | null>(null)
  const watchUntil = useRef(0)
  const poll = useRef<() => void>(() => undefined) // poll now
  const reschedule = useRef<() => void>(() => undefined) // poll next at the pace the state now needs

  const adopt = useCallback((s: GpuStatus) => {
    if (s.state !== latest.current?.state) setError(null) // the refusal was about the state before
    latest.current = s
    setStatus(s)
    setAt(performance.now())
  }, [])

  useEffect(() => {
    if (!enabled || paused) return
    let alive = true
    let timer = 0
    let inflight = false
    let again = false
    const schedule = () => {
      window.clearTimeout(timer)
      timer = window.setTimeout(run, nextPoll(latest.current, watchUntil.current))
    }
    const run = async () => {
      window.clearTimeout(timer)
      if (!alive || document.hidden) return // showing the page again polls at once
      if (inflight) {
        again = true // asked again mid-request: look once more when it returns
        return
      }
      inflight = true
      try {
        const s = await getGpu()
        if (alive) adopt(s)
      } catch {
        // keep showing the last state: the next poll may get through
      }
      inflight = false
      if (!alive) return
      if (again) {
        again = false
        return run()
      }
      schedule()
    }
    const onVisible = () => {
      if (!document.hidden) run()
    }
    poll.current = run
    reschedule.current = () => {
      if (alive && !inflight && !document.hidden) schedule() // a poll in flight schedules itself when it returns
    }
    document.addEventListener("visibilitychange", onVisible)
    run()
    return () => {
      alive = false
      window.clearTimeout(timer)
      poll.current = reschedule.current = () => undefined
      document.removeEventListener("visibilitychange", onVisible)
    }
  }, [enabled, paused, adopt])

  const refresh = useCallback(() => poll.current(), [])

  const wake = useCallback(
    async (reason: WakeReason, opts?: { quiet?: boolean }) => {
      watchUntil.current = performance.now() + WATCH_MS
      if (opts?.quiet && !wakeable(latest.current)) return poll.current() // awake already, or waking: just look again
      setError(null)
      try {
        adopt(await wakeGpu(reason))
        reschedule.current() // waking now: follow it every 2 s, not at the asleep pace
      } catch (e) {
        // A limit ("try again in 40 minutes") is not worth pressing again; a lost connection is.
        if (!opts?.quiet) setError({ message: (e as Error).message, retry: transient(e) && (e as ApiError).status !== 429 })
        poll.current()
      }
    },
    [adopt]
  )

  return { status: enabled ? status : null, at, error, wake, refresh }
}

// Fast while something is happening; at the moment a countdown ends (the idle sleep, a retry
// allowed again) rather than up to 30 s later; slow otherwise.
function nextPoll(s: GpuStatus | null, watchUntil: number) {
  if (performance.now() < watchUntil || (s && (s.state === "waking" || s.state === "connecting" || s.state === "busy")))
    return FAST_MS
  const due = s?.state === "ready" ? s.sleeps_in_s : s?.state === "unavailable" ? s.retry_s : null
  return due != null ? Math.min(SLOW_MS, Math.max(FAST_MS, due * 1000 + 1500)) : SLOW_MS
}
