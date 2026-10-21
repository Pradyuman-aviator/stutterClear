import { AbsoluteFill, interpolate, interpolateColors, useCurrentFrame } from "remotion"

import type { NvKey } from "../data"
import { NV_ITEMS } from "../data"
import { C, MONO, SANS } from "../theme"
import { clamp, springAt } from "./motion"

// The NVIDIA parts of the system, top right, drawn by the film: each pill lights in NVIDIA green
// from the scene that introduces it. `fresh` pills light up at frame `at`.
export function NvTracker({ lit, fresh = [], at = 0, opacity = 1 }: { lit: NvKey[]; fresh?: NvKey[]; at?: number; opacity?: number }) {
  const f = useCurrentFrame()
  return (
    <AbsoluteFill style={{ pointerEvents: "none", opacity }}>
      <div style={{ position: "absolute", right: 56, top: 48, display: "flex", alignItems: "center", gap: 11 }}>
        <span style={{ fontFamily: MONO, fontSize: 15, letterSpacing: 3, color: C.faint, marginRight: 6 }}>NVIDIA</span>
        {NV_ITEMS.map(({ key, label }) => {
          const isFresh = fresh.includes(key)
          const p = lit.includes(key) ? (isFresh ? Math.min(1, springAt(f, at)) : 1) : 0
          const flash = isFresh ? Math.max(0, 1 - Math.abs(f - at - 8) / 20) : 0
          return (
            <span
              key={key}
              style={{
                fontFamily: MONO,
                fontSize: 15,
                fontWeight: p > 0.5 ? 700 : 400,
                padding: "5px 11px",
                borderRadius: 999,
                color: interpolateColors(p, [0, 1], ["rgba(245,247,251,0.4)", "#0b0f02"]),
                background: interpolateColors(p, [0, 1], ["rgba(255,255,255,0.04)", C.nv]),
                border: `1px solid ${interpolateColors(p, [0, 1], ["rgba(255,255,255,0.09)", C.nv])}`,
                boxShadow: p > 0 ? `0 0 ${10 + 34 * flash}px rgba(118,185,0,${0.3 * p + 0.5 * flash})` : undefined,
                transform: `scale(${1 + 0.07 * flash})`,
                display: "inline-block",
              }}
            >
              {label}
            </span>
          )
        })}
      </div>
    </AbsoluteFill>
  )
}

// The narration, burned in: the film works muted too.
export function Subtitle({ text, frames }: { text: string; frames: number }) {
  const f = useCurrentFrame()
  const o = interpolate(f, [0, 6, frames - 5, frames], [0, 1, 1, 0], clamp)
  const y = interpolate(f, [0, 8], [10, 0], clamp)
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 52, opacity: o, pointerEvents: "none" }}>
      <div
        style={{
          maxWidth: 1480,
          padding: "13px 30px 15px",
          borderRadius: 18,
          background: "rgba(3,5,10,0.74)",
          border: "1px solid rgba(255,255,255,0.06)",
          color: C.text,
          fontFamily: SANS,
          fontWeight: 500,
          fontSize: 34,
          lineHeight: 1.36,
          textAlign: "center",
          transform: `translateY(${y}px)`,
          boxShadow: "0 18px 50px -20px rgba(0,0,0,0.9)",
        }}
      >
        {text}
      </div>
    </AbsoluteFill>
  )
}

// Shown while the real recording plays: whose voice it is and who to thank.
export function Credit({ text, frames }: { text: string; frames: number }) {
  const f = useCurrentFrame()
  const o = interpolate(f, [0, 10, frames - 8, frames], [0, 1, 1, 0], clamp)
  return (
    <AbsoluteFill style={{ pointerEvents: "none", opacity: o }}>
      <div
        style={{
          position: "absolute",
          left: 56,
          top: 48,
          maxWidth: 760,
          padding: "14px 20px",
          borderRadius: 16,
          background: "rgba(3,5,10,0.7)",
          border: `1px solid ${C.border}`,
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        <span style={{ fontFamily: MONO, fontSize: 17, letterSpacing: 3, color: C.teal }}>REAL RECORDING</span>
        <span style={{ fontFamily: SANS, fontSize: 19, lineHeight: 1.4, color: C.dim }}>{text}</span>
      </div>
    </AbsoluteFill>
  )
}

// A small fixed note at the top right (e.g. that the narrator's voice is synthetic).
export function Note({ text, frames, top = 52 }: { text: string; frames: number; top?: number }) {
  const f = useCurrentFrame()
  const o = interpolate(f, [0, 10, frames - 10, frames], [0, 1, 1, 0], clamp)
  return (
    <AbsoluteFill style={{ pointerEvents: "none", opacity: o }}>
      <div style={{ position: "absolute", right: 56, top, fontFamily: MONO, fontSize: 17, color: C.faint, letterSpacing: 1 }}>{text}</div>
    </AbsoluteFill>
  )
}
