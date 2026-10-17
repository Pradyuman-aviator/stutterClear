import { useEffect, useRef, type RefObject } from "react"
import type { LiveData } from "./use-live"

// Smoothed 0..1 signals that every animated part of the page shares (orb, background, waveform).
export interface Signals {
  level: number // mic loudness, fast attack, slow release
  outLevel: number // what the listener hears
  live: number // eases 0 -> 1 when a session starts
  cut: number // 1 right after a stutter is removed, decays
  block: number // 1 right after a block or long pause, decays
}

export function useSignals(data: RefObject<LiveData>, running: boolean) {
  const signals = useRef<Signals>({ level: 0, outLevel: 0, live: 0, cut: 0, block: 0 })
  const runningRef = useRef(running)

  useEffect(() => {
    runningRef.current = running
  }, [running])

  useEffect(() => {
    let raf = 0
    let last = performance.now()
    const follow = (cur: number, target: number, dt: number) =>
      cur + (target - cur) * (1 - Math.exp(-dt * (target > cur ? 18 : 5)))

    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const d = data.current
      const s = signals.current
      const fresh = now - d.receivedAt < 250
      s.level = follow(s.level, fresh ? d.level : 0, dt)
      s.outLevel = follow(s.outLevel, fresh ? d.outLevel : 0, dt)
      s.live += ((runningRef.current ? 1 : 0) - s.live) * (1 - Math.exp(-dt * 3))
      s.cut = Math.exp(-(now - d.lastCutAt) / 500)
      s.block = Math.exp(-(now - d.lastBlockAt) / 1200)
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [data])

  return signals
}
