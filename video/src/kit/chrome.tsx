import type { CSSProperties, ReactNode } from "react"
import { Img, staticFile, useCurrentFrame } from "remotion"

import { C, MONO, SANS, SERIF } from "../theme"

export function Card({ children, style, glow }: { children: ReactNode; style?: CSSProperties; glow?: string }) {
  return (
    <div
      style={{
        background: "linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.025))",
        border: `1px solid ${C.border}`,
        borderRadius: 28,
        boxShadow: `inset 0 1px 0 rgba(255,255,255,0.08), 0 30px 80px -30px rgba(0,0,0,0.75)${glow ? `, 0 0 60px -10px ${glow}` : ""}`,
        position: "relative",
        overflow: "hidden",
        ...style,
      }}
    >
      {children}
    </div>
  )
}

export function Pill({ children, color = C.dim, border = C.border, style }: { children: ReactNode; color?: string; border?: string; style?: CSSProperties }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 10,
        padding: "10px 22px",
        borderRadius: 999,
        border: `1px solid ${border}`,
        background: "rgba(255,255,255,0.05)",
        color,
        fontSize: 26,
        fontFamily: SANS,
        ...style,
      }}
    >
      {children}
    </span>
  )
}

// A small mono label above a title: "02 · HOW IT HEARS".
export function Eyebrow({ children, color = C.sky, style }: { children: ReactNode; color?: string; style?: CSSProperties }) {
  return (
    <div style={{ fontFamily: MONO, fontSize: 22, letterSpacing: 4, textTransform: "uppercase", color, ...style }}>{children}</div>
  )
}

// A big serif statement.
export function Title({ children, size = 96, style }: { children: ReactNode; size?: number; style?: CSSProperties }) {
  return <div style={{ fontFamily: SERIF, fontSize: size, lineHeight: 1.04, letterSpacing: -1, color: C.text, ...style }}>{children}</div>
}

export function Logo({ size = 120, style }: { size?: number; style?: CSSProperties }) {
  return (
    <div style={{ fontFamily: SERIF, fontSize: size, lineHeight: 1, letterSpacing: -1, color: C.text, ...style }}>
      Stutter
      <span style={{ background: `linear-gradient(90deg, ${C.sky}, ${C.teal})`, WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>Clear</span>
    </div>
  )
}

// Alex (fictional): an abstract avatar, never a face. `level` (0..1, e.g. envAt of his clip at the
// playhead) makes the rings breathe with his voice; `struggle` (0..1) tints them amber.
export function Avatar({
  size = 220,
  initial = "A",
  level = 0,
  struggle = 0,
  color = C.sky,
  style,
}: {
  size?: number
  initial?: string
  level?: number
  struggle?: number
  color?: string
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  const ring = struggle > 0.5 ? C.amber : color
  return (
    <div style={{ position: "relative", width: size, height: size, ...style }}>
      {[0, 1, 2].map((i) => {
        const s = 1 + 0.12 * (i + 1) + level * 0.22 * (i + 1) + 0.015 * Math.sin(f / 9 + i)
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              inset: 0,
              borderRadius: "50%",
              border: `2px solid ${ring}`,
              opacity: (0.42 - i * 0.12) * (0.35 + 0.65 * Math.max(level, 0.25)),
              transform: `scale(${s})`,
            }}
          />
        )
      })}
      <div
        style={{
          position: "absolute",
          inset: 0,
          borderRadius: "50%",
          background: `radial-gradient(circle at 35% 30%, ${color}55, rgba(196,181,253,0.18) 55%, rgba(10,14,26,0.95) 100%)`,
          border: `1.5px solid ${C.border2}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: SERIF,
          fontSize: size * 0.46,
          color: C.text,
          boxShadow: `0 0 ${50 + level * 60}px -10px ${ring}`,
        }}
      >
        {initial}
      </div>
    </div>
  )
}

// A desktop video-call window.
export function CallWindow({
  title = "Video call",
  width = 1180,
  height = 700,
  live = true,
  children,
  style,
}: {
  title?: string
  width?: number
  height?: number
  live?: boolean
  children: ReactNode
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  return (
    <Card style={{ width, height, padding: 22, display: "flex", flexDirection: "column", gap: 18, boxSizing: "border-box", ...style }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, fontFamily: SANS, fontSize: 22, color: C.dim }}>
        <span style={{ width: 13, height: 13, borderRadius: 7, background: C.rose }} />
        <span style={{ width: 13, height: 13, borderRadius: 7, background: C.amber }} />
        <span style={{ width: 13, height: 13, borderRadius: 7, background: C.green }} />
        <span style={{ marginLeft: 14 }}>{title}</span>
        {live && (
          <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8, fontFamily: MONO, fontSize: 18, color: C.rose }}>
            <span style={{ width: 10, height: 10, borderRadius: 5, background: C.rose, opacity: Math.floor(f / 20) % 2 ? 0.35 : 1 }} />
            LIVE
          </span>
        )}
      </div>
      <div style={{ position: "relative", flex: 1, display: "flex", gap: 18 }}>{children}</div>
    </Card>
  )
}

// A participant tile in a call.
export function Tile({ name, children, speaking = 0, style }: { name: string; children?: ReactNode; speaking?: number; style?: CSSProperties }) {
  return (
    <div
      style={{
        position: "relative",
        flex: 1,
        borderRadius: 20,
        background: "linear-gradient(160deg, #182033, #0d121d)",
        border: `2px solid ${speaking > 0.3 ? C.teal : "transparent"}`,
        boxShadow: speaking > 0.3 ? `0 0 30px -6px ${C.teal}` : undefined,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
        ...style,
      }}
    >
      {children}
      <div style={{ position: "absolute", left: 18, bottom: 14, fontFamily: SANS, fontSize: 22, color: C.dim }}>{name}</div>
    </div>
  )
}

// A screenshot of the app in a browser frame, optionally tilted in 3D.
export function ScreenFrame({
  src,
  width = 1200,
  tiltX = 0,
  tiltY = 0,
  url = "localhost:8000",
  style,
}: {
  src: string // under public/, e.g. "generated/screens/calls.png"
  width?: number
  tiltX?: number
  tiltY?: number
  url?: string
  style?: CSSProperties
}) {
  return (
    <div style={{ perspective: 2200, ...style }}>
      <div
        style={{
          width,
          borderRadius: 18,
          overflow: "hidden",
          border: `1px solid ${C.border2}`,
          background: "#0b0f19",
          transform: `rotateX(${tiltX}deg) rotateY(${tiltY}deg)`,
          boxShadow: "0 50px 120px -40px rgba(0,0,0,0.9), 0 0 0 1px rgba(255,255,255,0.03)",
        }}
      >
        <div style={{ height: 40, display: "flex", alignItems: "center", gap: 8, padding: "0 16px", background: "rgba(255,255,255,0.04)", borderBottom: `1px solid ${C.border}` }}>
          <span style={{ width: 11, height: 11, borderRadius: 6, background: "rgba(255,255,255,0.18)" }} />
          <span style={{ width: 11, height: 11, borderRadius: 6, background: "rgba(255,255,255,0.18)" }} />
          <span style={{ width: 11, height: 11, borderRadius: 6, background: "rgba(255,255,255,0.18)" }} />
          <span style={{ marginLeft: 16, fontFamily: MONO, fontSize: 15, color: C.faint }}>{url}</span>
        </div>
        <Img src={staticFile(src)} style={{ width, display: "block" }} />
      </div>
    </div>
  )
}
