import { useEffect, useState } from "react"
import type { Transition } from "motion/react"

// One motion language for the whole app. Springs are close to critically damped: they glide in
// and settle without wobbling, which reads as smooth rather than bouncy.
export const SPRING = {
  // Default for layout, pills and things that move across the screen.
  glide: { type: "spring", stiffness: 260, damping: 32, mass: 1 } satisfies Transition,
  // Hover and press feedback: quick, still no overshoot.
  press: { type: "spring", stiffness: 520, damping: 36, mass: 0.7 } satisfies Transition,
  // Entrances.
  enter: { type: "spring", stiffness: 120, damping: 22, mass: 1 } satisfies Transition,
}

// Spring configs for useSpring() followers (tilt, magnet, parallax).
export const FOLLOW = {
  tilt: { stiffness: 120, damping: 22, mass: 1 },
  magnet: { stiffness: 200, damping: 24, mass: 0.8 },
  parallax: { stiffness: 70, damping: 20, mass: 1 },
}

export const EASE_OUT = [0.22, 1, 0.36, 1] as const

// True on a mouse or trackpad. Touch screens skip hover-only effects (tilt, magnet, parallax).
export function useFinePointer() {
  const query = "(hover: hover) and (pointer: fine)"
  const [fine, setFine] = useState(() => typeof window !== "undefined" && matchMedia(query).matches)
  useEffect(() => {
    const mq = matchMedia(query)
    const on = () => setFine(mq.matches)
    mq.addEventListener("change", on)
    return () => mq.removeEventListener("change", on)
  }, [])
  return fine
}

// Runs fn at most once per frame with the latest pointer event: high-rate mice send 500-1000
// events a second, and layout reads in each of them cost frames.
export function rafThrottle<E>(fn: (e: E) => void) {
  let last: E | null = null
  let id = 0
  const run = () => {
    id = 0
    if (last) fn(last)
  }
  const handler = (e: E) => {
    last = e
    if (!id) id = requestAnimationFrame(run)
  }
  handler.cancel = () => cancelAnimationFrame(id)
  return handler
}
