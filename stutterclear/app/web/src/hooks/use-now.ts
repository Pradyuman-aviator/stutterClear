import { useEffect, useState } from "react"

// performance.now() for labels that count time ("waking · 1:23", "retrying in 4 s"), refreshed
// every `ms` while `active`. It never reads the clock during render: until the first tick it is
// `from`, the moment the value being counted from arrived. Ticks fall on the beat of `from`, so
// two labels counting from the same moment (the header pill and a card) always agree.
export function useNow(active: boolean, from: number, ms = 1000) {
  const [now, setNow] = useState(from)
  useEffect(() => {
    if (!active) return
    let id = 0
    const tick = () => {
      const t = performance.now()
      setNow(t)
      id = window.setTimeout(tick, ms - ((t - from) % ms) + 5)
    }
    id = window.setTimeout(tick, 0)
    return () => window.clearTimeout(id)
  }, [active, from, ms])
  return Math.max(now, from)
}

// 83 s -> "1:23"
export const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`

// How far behind the speaker the listener hears: 250 -> "250 ms", 840 -> "0.85 s".
export const behind = (ms: number) => (ms < 400 ? `${Math.round(ms / 10) * 10} ms` : `${+(Math.round(ms / 50) * 0.05).toFixed(2)} s`)
