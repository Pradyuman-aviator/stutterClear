import { AbsoluteFill, Freeze, interpolate, OffthreadVideo, Sequence, staticFile, useCurrentFrame } from "remotion"

import type { SceneProps } from "../data"
import { clipCue, FPS, lineCue } from "../data"
import { clamp, ease, easeOut, Eyebrow, GlowOrb, KineticText, Particles, ramp, springAt } from "../kit"
import { C, MONO, SANS, SERIF } from "../theme"

// The real app, recorded (scripts/capture_demo.py) and cut to ~30 s: the live session cleaning Alex's
// real recording on the GPU, its summary, the Recordings tab with Thorough, and the meaning guard
// answering from Nemotron. The footage plays in a browser frame and the view inside it zooms to
// what matters, like a product screencast.

const FW = 1280 // the frame's screen, 16:9
const FH = (FW * 9) / 16
const FX = (1920 - FW) / 2
const FY = 254 // below the credit card that shows while his voice plays
const BAR = 38

// Where to look in each piece of the cut (capture pixels, 1920x1080) and how close.
const VIEWS: { x: number; y: number; zoom: number }[] = [
  { x: 560, y: 330, zoom: 1.75 }, // Start listening
  { x: 640, y: 600, zoom: 1.3 }, // the live session: status and Fluency Mirror
  { x: 960, y: 1000, zoom: 2.2 }, // its summary (bottom of the page)
  { x: 520, y: 600, zoom: 2.0 }, // Recordings, Thorough (the page scrolled the player to the middle)
  { x: 1180, y: 790, zoom: 2.0 }, // the meaning guard, Check (at the foot of the page)
  { x: 470, y: 780, zoom: 2.2 }, // Nemotron's answer under it
]

const CHAPTERS = [
  { label: "Live call", parts: [0, 1] },
  { label: "Summary", parts: [2] },
  { label: "Recordings", parts: [3] },
  { label: "Meaning guard", parts: [4, 5] },
]

const XFADE = 4

function Footage({ src, from, len, view, prev, fadeIn }: { src: string; from: number; len: number; view: (typeof VIEWS)[number]; prev?: (typeof VIEWS)[number]; fadeIn: boolean }) {
  const f = useCurrentFrame()
  // ease in from the previous piece's framing, then drift slowly closer
  const p = prev ? ramp(f, 0, 16, ease) : 1
  const z0 = prev ? prev.zoom : view.zoom
  const x0 = prev ? prev.x : view.x
  const y0 = prev ? prev.y : view.y
  const drift = 1 + 0.03 * (f / Math.max(1, len))
  const zoom = (z0 + (view.zoom - z0) * p) * drift
  const cx = x0 + (view.x - x0) * p
  const cy = y0 + (view.y - y0) * p
  // keep the zoomed capture covering the screen
  const s = (FW / 1920) * zoom
  const half = { w: FW / 2 / s, h: FH / 2 / s }
  const x = Math.min(1920 - half.w, Math.max(half.w, cx))
  const y = Math.min(1080 - half.h, Math.max(half.h, cy))
  const o = fadeIn ? interpolate(f, [0, XFADE], [0, 1], clamp) : 1
  return (
    <AbsoluteFill style={{ opacity: o }}>
      <div style={{ position: "absolute", left: 0, top: 0, width: 1920, height: 1080, transformOrigin: "0 0", transform: `translate(${FW / 2}px, ${FH / 2}px) scale(${s}) translate(${-x}px, ${-y}px)` }}>
        <OffthreadVideo src={staticFile(src)} muted trimBefore={from} style={{ width: 1920, height: 1080 }} />
      </div>
    </AbsoluteFill>
  )
}

export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const demo = d.demo
  const l0 = lineCue(scene, 0)
  const cue = clipCue(scene, "demo")
  if (!demo || !cue) {
    return (
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", fontFamily: SANS, fontSize: 36, color: C.dim }}>
        Record the demo first: python scripts/capture_demo.py
      </AbsoluteFill>
    )
  }

  // the cut, end to end from the demo audio's start
  let at = cue.at
  const parts = demo.cut.map((c, i) => {
    const len = Math.round((c.to - c.from) * FPS)
    const p = { i, at, len, from: Math.round(c.from * FPS), label: c.label }
    at += len
    return p
  })
  const live = f >= cue.at
  const cur = parts.reduce((k, p) => (f >= p.at ? p.i : k), 0)

  // the window arrives with the line; the title hands its place to the credit as the demo starts
  const win = springAt(f, 4)
  const title = 1 - ramp(f, cue.at - 14, cue.at + 2)
  const tag = ramp(f, cue.at - 6, cue.at + 10, easeOut)

  return (
    <AbsoluteFill>
      <Particles count={18} color={C.teal} seed="demo" opacity={0.3} />
      <GlowOrb x={960} y={FY + FH / 2} r={820} color={C.teal} opacity={0.1} />

      {/* the title, while the narrator introduces the demo */}
      {title > 0.01 && (
        <div style={{ position: "absolute", left: 0, right: 0, top: 104, display: "flex", flexDirection: "column", alignItems: "center", gap: 10, opacity: title, transform: `translateY(${(1 - title) * -16}px)` }}>
          <Eyebrow color={C.teal}>Live demo · the real app · this laptop</Eyebrow>
          <KineticText text="Not a mock-up." at={l0.at + 4} per={4} style={{ fontFamily: SERIF, fontSize: 64, lineHeight: 1, color: C.text }} />
        </div>
      )}

      {/* the browser window */}
      <div
        style={{
          position: "absolute",
          left: FX,
          top: FY - BAR,
          width: FW,
          height: FH + BAR,
          borderRadius: 20,
          overflow: "hidden",
          border: `1px solid ${C.border2}`,
          background: "#070a12",
          boxShadow: `0 60px 140px -50px rgba(0,0,0,0.95), 0 0 80px -30px ${C.teal}`,
          opacity: Math.min(1, win),
          transform: `translateY(${(1 - Math.min(1, win)) * 40}px) scale(${0.96 + 0.04 * Math.min(1, win)})`,
        }}
      >
        <div style={{ height: BAR, display: "flex", alignItems: "center", gap: 8, padding: "0 16px", background: "rgba(255,255,255,0.045)", borderBottom: `1px solid ${C.border}` }}>
          <span style={{ width: 11, height: 11, borderRadius: 6, background: C.rose }} />
          <span style={{ width: 11, height: 11, borderRadius: 6, background: C.amber }} />
          <span style={{ width: 11, height: 11, borderRadius: 6, background: C.green }} />
          <span style={{ marginLeft: 16, fontFamily: MONO, fontSize: 16, color: C.faint }}>127.0.0.1:8766 · StutterClear</span>
          <span
            style={{
              marginLeft: "auto",
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontFamily: MONO,
              fontSize: 15,
              letterSpacing: 2,
              color: C.teal,
              opacity: tag,
            }}
          >
            <span style={{ width: 9, height: 9, borderRadius: 5, background: C.rose, opacity: Math.floor(f / 15) % 2 ? 0.4 : 1 }} />
            SCREEN RECORDING · REAL APP · NOT SIMULATED
          </span>
        </div>
        <div style={{ position: "relative", width: FW, height: FH, overflow: "hidden" }}>
          {!live && (
            <Freeze frame={0}>
              <Footage src={demo.video} from={parts[0].from} len={parts[0].len} view={VIEWS[0]} fadeIn={false} />
            </Freeze>
          )}
          {parts.map((p) => (
            <Sequence key={p.i} from={p.at} durationInFrames={p.len + (p.i < parts.length - 1 ? XFADE : 0)} layout="none">
              <Footage src={demo.video} from={p.from} len={p.len} view={VIEWS[p.i] ?? VIEWS[VIEWS.length - 1]} prev={p.i ? VIEWS[p.i - 1] : undefined} fadeIn={p.i > 0} />
            </Sequence>
          ))}
          {/* what this piece shows */}
          {live && (
            <div
              key={cur}
              style={{
                position: "absolute",
                right: 22,
                top: 18,
                padding: "10px 18px",
                borderRadius: 12,
                background: "rgba(5,7,13,0.82)",
                border: `1px solid ${C.teal}66`,
                fontFamily: SANS,
                fontWeight: 600,
                fontSize: 24,
                color: C.text,
                opacity: Math.min(1, springAt(f, parts[cur].at + 3)),
                transform: `translateY(${(1 - Math.min(1, springAt(f, parts[cur].at + 3))) * 12}px)`,
              }}
            >
              {parts[cur].label}
            </div>
          )}
        </div>
      </div>

      {/* chapters under the window */}
      {live && (
        <div style={{ position: "absolute", left: FX, top: FY + FH + 20, width: FW, display: "flex", gap: 14, opacity: ramp(f, cue.at, cue.at + 12, easeOut) }}>
          {CHAPTERS.map((ch) => {
            const ps = ch.parts.map((i) => parts[i]).filter(Boolean)
            if (!ps.length) return null
            const a = ps[0].at
            const b = ps[ps.length - 1].at + ps[ps.length - 1].len
            const fill = interpolate(f, [a, b], [0, 1], clamp)
            const on = f >= a && f < b
            return (
              <div key={ch.label} style={{ flex: b - a, display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ height: 5, borderRadius: 3, background: "rgba(255,255,255,0.1)", overflow: "hidden" }}>
                  <div style={{ width: `${fill * 100}%`, height: "100%", background: C.teal, boxShadow: on ? `0 0 12px ${C.teal}` : undefined }} />
                </div>
                <span style={{ fontFamily: MONO, fontSize: 18, letterSpacing: 1, color: on ? C.text : C.dim }}>{ch.label}</span>
              </div>
            )
          })}
        </div>
      )}
    </AbsoluteFill>
  )
}
