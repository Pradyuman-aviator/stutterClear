import type { CSSProperties, ReactNode } from "react"
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion"

import { C, SANS, SERIF } from "./theme"

// Slow drifting glows behind everything, so the frame never sits still.
export function Background() {
  const f = useCurrentFrame()
  const a = Math.sin(f / 140) * 120
  const b = Math.cos(f / 170) * 140
  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(900px 700px at ${420 + a}px ${260 + b}px, rgba(125,211,252,0.13), transparent 70%),
          radial-gradient(1000px 800px at ${1540 - b}px ${820 + a}px, rgba(196,181,253,0.11), transparent 70%),
          ${C.bg}`,
      }}
    />
  )
}

// A scene fades in and out at its edges.
export function SceneFade({ frames, children }: { frames: number; children: ReactNode }) {
  const f = useCurrentFrame()
  const o = interpolate(f, [0, 12, frames - 12, frames], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
  return <AbsoluteFill style={{ opacity: o, fontFamily: SANS, color: C.text }}>{children}</AbsoluteFill>
}

// Rises in with a spring, `delay` frames after its scene starts.
export function Rise({ delay = 0, children, style }: { delay?: number; children: ReactNode; style?: CSSProperties }) {
  const f = useCurrentFrame()
  const { fps } = useVideoConfig()
  const p = spring({ frame: f - delay, fps, config: { damping: 200 } })
  return <div style={{ opacity: p, transform: `translateY(${(1 - p) * 26}px)`, ...style }}>{children}</div>
}

export function Logo({ size = 120 }: { size?: number }) {
  return (
    <div style={{ fontFamily: SERIF, fontSize: size, lineHeight: 1, letterSpacing: -1 }}>
      Stutter
      <span style={{ background: `linear-gradient(90deg, ${C.sky}, ${C.teal})`, WebkitBackgroundClip: "text", color: "transparent" }}>
        Clear
      </span>
    </div>
  )
}

export function Card({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div
      style={{
        background: C.panel,
        border: `1px solid ${C.border}`,
        borderRadius: 28,
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.08), 0 30px 80px -30px rgba(0,0,0,0.7)",
        ...style,
      }}
    >
      {children}
    </div>
  )
}

export function Pill({ children, color = C.dim, style }: { children: ReactNode; color?: string; style?: CSSProperties }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 10,
        padding: "10px 22px",
        borderRadius: 999,
        border: `1px solid ${C.border}`,
        background: "rgba(255,255,255,0.05)",
        color,
        fontSize: 28,
        ...style,
      }}
    >
      {children}
    </span>
  )
}

// The narration, as burned-in subtitles: the video works muted too.
export function Subtitle({ text, frames }: { text: string; frames: number }) {
  const f = useCurrentFrame()
  const o = interpolate(f, [0, 6, frames - 4, frames], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 46, opacity: o, fontFamily: SANS }}>
      <div
        style={{
          maxWidth: 1500,
          padding: "14px 30px",
          borderRadius: 18,
          background: "rgba(3,5,10,0.72)",
          color: C.text,
          fontSize: 34,
          lineHeight: 1.35,
          textAlign: "center",
        }}
      >
        {text}
      </div>
    </AbsoluteFill>
  )
}

// A waveform from 10 ms peaks: bars inside a cut are rose, the rest `color`; bars after the
// playhead (progress 0..1) are dimmed.
export function Wave({
  env,
  seconds,
  width,
  height,
  color,
  cuts = [],
  progress = 1,
  bar = 6,
}: {
  env: number[]
  seconds: number
  width: number
  height: number
  color: string
  cuts?: { start: number; end: number }[]
  progress?: number
  bar?: number
}) {
  const n = Math.floor(width / bar)
  const peak = Math.max(1e-4, ...env)
  const bars = Array.from({ length: n }, (_, i) => {
    const a = Math.floor((i / n) * env.length)
    const b = Math.max(a + 1, Math.floor(((i + 1) / n) * env.length))
    const v = Math.max(...env.slice(a, b)) / peak
    const t = ((i + 0.5) / n) * seconds
    return { h: Math.max(3, Math.sqrt(v) * height), cut: cuts.some((c) => t >= c.start && t < c.end), done: (i + 0.5) / n <= progress }
  })
  return (
    <div style={{ position: "relative", width, height, display: "flex", alignItems: "center", gap: bar * 0.35 }}>
      {bars.map((b, i) => (
        <div
          key={i}
          style={{
            width: bar * 0.65,
            height: b.h,
            borderRadius: 3,
            background: b.cut ? C.rose : color,
            opacity: b.done ? 1 : 0.32,
          }}
        />
      ))}
      {progress > 0 && progress < 1 && (
        <div style={{ position: "absolute", left: progress * width, top: -10, bottom: -10, width: 3, borderRadius: 2, background: C.text, boxShadow: `0 0 18px ${C.sky}` }} />
      )}
    </div>
  )
}
