import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion"

import type { Data, Scene } from "../data"
import { sec } from "../data"
import { C } from "../theme"
import { Card, Pill, Rise } from "../ui"

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const

function Tile({ name, initial, speaking }: { name: string; initial: string; speaking: boolean }) {
  return (
    <div style={{ flex: 1, height: 330, borderRadius: 22, background: "linear-gradient(160deg, #1b2236, #10141f)",
      border: `2px solid ${speaking ? C.teal : "transparent"}`, display: "flex", alignItems: "center", justifyContent: "center",
      position: "relative" }}>
      <div style={{ width: 120, height: 120, borderRadius: 60, background: "rgba(255,255,255,0.08)", display: "flex",
        alignItems: "center", justifyContent: "center", fontSize: 50, color: C.dim }}>{initial}</div>
      <div style={{ position: "absolute", left: 18, bottom: 14, fontSize: 24, color: C.dim }}>{name}</div>
    </div>
  )
}

// A video call: the cleaned voice goes in through a virtual microphone; captions write themselves.
export function Calls({ d }: { d: Data; scene: Scene }) {
  const f = useCurrentFrame()
  const typed = Math.floor(interpolate(f, [sec(1.2), sec(7.5)], [0, d.clip.clean.length], clamp))
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", paddingBottom: 150 }}>
      <div style={{ display: "flex", gap: 46, alignItems: "center" }}>
        <Rise delay={2}>
          <Card style={{ width: 1080, padding: 26, display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ display: "flex", gap: 10, alignItems: "center", fontSize: 24, color: C.dim }}>
              <span style={{ width: 14, height: 14, borderRadius: 7, background: C.rose }} />
              <span style={{ width: 14, height: 14, borderRadius: 7, background: C.amber }} />
              <span style={{ width: 14, height: 14, borderRadius: 7, background: C.green }} />
              <span style={{ marginLeft: 14 }}>Weekly sync · video call</span>
            </div>
            <div style={{ display: "flex", gap: 18 }}>
              <Tile name="You" initial="A" speaking={f % 40 < 30} />
              <Tile name="Your manager" initial="M" speaking={false} />
            </div>
            <div style={{ minHeight: 92, padding: "14px 22px", borderRadius: 16, background: "rgba(0,0,0,0.45)", fontSize: 28, lineHeight: 1.45 }}>
              {d.clip.clean.slice(0, typed)}
              <span style={{ opacity: f % 30 < 15 ? 1 : 0, color: C.teal }}>▍</span>
            </div>
          </Card>
        </Rise>
        <div style={{ display: "flex", flexDirection: "column", gap: 26, width: 560 }}>
          <Rise delay={sec(1.4)}>
            <Card style={{ padding: 24, display: "flex", flexDirection: "column", gap: 8 }}>
              <span style={{ fontSize: 22, color: C.faint }}>Microphone</span>
              <span style={{ fontSize: 30, fontWeight: 600 }}>StutterClear (virtual cable)</span>
            </Card>
          </Rise>
          <Rise delay={sec(2.6)} style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <Pill>Zoom</Pill>
            <Pill>Google Meet</Pill>
            <Pill>Microsoft Teams</Pill>
          </Rise>
          <Rise delay={sec(4.2)}>
            <div style={{ fontSize: 28, color: C.dim, lineHeight: 1.45 }}>
              Live captions: <span style={{ color: C.text }}>NVIDIA streaming FastConformer</span>, on your computer
            </div>
          </Rise>
        </div>
      </div>
    </AbsoluteFill>
  )
}

// Stuck on a word: Nemotron's real suggestions, and Say it on a tap.
export function Rescue({ d }: { d: Data; scene: Scene }) {
  const f = useCurrentFrame()
  const { fps } = useVideoConfig()
  const text = d.rescue.said
  const typed = Math.floor(interpolate(f, [sec(0.4), sec(3.2)], [0, text.length], clamp))
  const stuck = f > sec(3.4)
  const chips = spring({ frame: f - sec(4.6), fps, config: { damping: 18 } })
  const tap = spring({ frame: f - sec(7.4), fps, config: { damping: 12 } })
  const pick = d.rescue.suggestions[0] ?? ""
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 50, paddingBottom: 170 }}>
      <Rise delay={2}>
        <div style={{ fontSize: 24, color: C.faint, maxWidth: 1400, textAlign: "center" }}>Context card: {d.rescue.context}</div>
      </Rise>
      <div style={{ fontSize: 56, fontWeight: 600, maxWidth: 1600, textAlign: "center", lineHeight: 1.3 }}>
        {text.slice(0, typed)}
        {stuck && <span style={{ color: C.amber }}> {d.rescue.onset}… {d.rescue.onset}…</span>}
        <span style={{ opacity: f % 30 < 15 ? 1 : 0, color: C.sky }}>▍</span>
      </div>
      <div style={{ display: "flex", gap: 22, opacity: chips, transform: `scale(${0.85 + 0.15 * chips})` }}>
        {d.rescue.suggestions.map((s, i) => (
          <div key={s} style={{
            fontSize: 40, padding: "14px 34px", borderRadius: 999,
            background: i === 0 && tap > 0.3 ? "rgba(196,181,253,0.35)" : "rgba(196,181,253,0.14)",
            border: `2px solid ${i === 0 && tap > 0.3 ? C.violet : `${C.violet}55`}`,
            color: C.text,
            transform: `scale(${i === 0 ? 1 + 0.08 * Math.sin(Math.min(1, tap) * Math.PI) : 1})`,
          }}>
            {s}
          </div>
        ))}
      </div>
      <div style={{ opacity: tap, display: "flex", alignItems: "center", gap: 18, fontSize: 36 }}>
        <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke={C.teal} strokeWidth="2" strokeLinecap="round">
          <path d="M11 5 6 9H2v6h4l5 4V5z" />
          <path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14" />
        </svg>
        <span>Say it: “{pick}”</span>
        <span style={{ fontSize: 24, color: C.faint }}>synthetic voice · only on your tap</span>
      </div>
    </AbsoluteFill>
  )
}
