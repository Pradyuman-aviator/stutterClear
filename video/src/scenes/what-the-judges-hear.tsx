import type { CSSProperties, ReactNode } from "react"
import { AbsoluteFill, interpolate, interpolateColors, useCurrentFrame } from "remotion"

import type { Cut, SceneProps } from "../data"
import { clipCue, lineCue } from "../data"
import { Avatar, Camera, easeIn, easeOut, envAt, GlowOrb, Icon, KineticText, LightSweep, Particles, ramp, Sfx, SNAPPY, springAt, Wave, waveX } from "../kit"
import { C, JP, MONO, SANS, SERIF } from "../theme"

// What the judges hear: Alex shares last night's take in the call. Its stutters snap out of the
// recording (0:19 → 0:08), the clean take plays with its captions while the judges listen, then the
// verbatim and the clean line sit side by side at equal weight: which one plays is his choice.

// ---- layout (stage px) ----
const PL = { x: 120, y: 200, w: 1224, h: 688 } // the shared video player
const VID = { x: 24, y: 64, w: 1176, h: 404 } // the video frame inside the player
const SCRUB_Y = 480 // the scrub area inside the player
const RING = { cx: 112, cy: 584 - SCRUB_Y, r: 66 } // inside the scrub area
const WAVE = { x: 236, y: 22, w: 900, h: 96 } // inside the scrub area
const COL = { x: 1392, w: 408, h: 160, gap: 16 } // the call's tiles, as a column
const GRID = { w: 600, h: 300, gap: 24, cx: 960, cy: 520 } // the call's tiles before sharing

const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(Math.max(0, s) % 60)).padStart(2, "0")}`
const mix = (a: number, b: number, p: number) => a + (b - a) * p
const c01 = (v: number) => Math.min(1, Math.max(0, v))

export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const c = d.clip
  const L0 = lineCue(scene, 0)
  const L1 = lineCue(scene, 1)
  const after = clipCue(scene, "after") ?? { at: L0.at + L0.frames + 15, frames: Math.round(c.after_s * 30) }
  const l0 = (u: number) => L0.at + u * L0.frames
  const l1 = (u: number) => L1.at + u * L1.frames

  // ---- beats ----
  const shareAt = l0(0.1) // "Then he plays…": Alex starts sharing
  const openAt = l0(0.16) // the player opens, the tiles step aside
  const waveAt = l0(0.36) // the take appears, its cuts tinted
  const hearAt = l0(0.86) // "…what the judges hear": the judges put their headphones on, on "hear"
  const c0 = l0(0.46) // the cuts start to snap out…
  const c1 = after.at - 12 // …and are all gone before the clean take plays
  const morphAt = c1 // the collapsed take becomes the clean take's own waveform, done as it starts
  const playEnd = after.at + after.frames
  const outAt = L1.at - 22 // the call steps back, crossing into the comparison
  const cmpAt = L1.at - 10 // both versions arrive together, on "Both versions"
  const flipA = l1(0.6) // "Which one you hear…"
  const flipB = l1(0.9) // "…is up to him."

  // ---- the cuts snapping out ----
  const n = c.cuts.length
  const step = n > 1 ? (c1 - 12 - c0) / (n - 1) : 0
  const cutStart = (i: number) => c0 + i * step
  const collapse = c.cuts.map((_, i) => ramp(f, cutStart(i), cutStart(i) + 12, easeOut))
  const done = collapse.reduce((a, b) => a + b, 0) / Math.max(1, n)
  const stutters = c.cuts.filter((k) => k.kind !== "pause").length
  const pauses = n - stutters
  // the counts tick up as each cut snaps out
  const stuttersNow = c.cuts.filter((k, i) => k.kind !== "pause" && collapse[i] > 0.5).length
  const pausesNow = c.cuts.filter((k, i) => k.kind === "pause" && collapse[i] > 0.5).length
  const firstOf = (pause: boolean) => Math.max(0, c.cuts.findIndex((k) => (k.kind === "pause") === pause))
  const stutterLabelAt = cutStart(firstOf(false)) + 4
  const pauseLabelAt = cutStart(firstOf(true)) + 4
  const removed = c.cuts.reduce((a, k, i) => a + (k.end - k.start) * collapse[i], 0)
  // the take's length while it shrinks (the cuts on the real clock), landing on the clean take's own length
  const lengthNow = mix(c.before_s - removed, c.after_s, ramp(f, c1 - 10, c1))
  const play = ramp(f, after.at, playEnd, (t) => t)
  const tNow = play * c.after_s
  const playing = f >= after.at && f < playEnd
  const lvl = playing ? Math.min(1, envAt(c.env_after, tNow, tNow + 0.12) * 1.3) : 0

  // ---- the call ----
  const callIn = ramp(f, 0, 14, easeOut)
  const callOut = ramp(f, outAt, outAt + 14, easeIn)
  const pOpen = springAt(f, openAt + 10, SNAPPY)
  const TILE_ORDER = [2, 0, 3, 1] // Judge 1 and Judge 3 (the right-hand tiles) step aside first
  const pTile = (i: number) => springAt(f, openAt - 14 + TILE_ORDER[i] * 3)
  // the player shows only once Judge 2 (bottom-left, the last to move) has crossed the centre
  const playerGate = ramp(pTile(2), 0.5, 0.8)

  // ---- the comparison ----
  const pickP = springAt(f, flipA) - springAt(f, flipB) // 0: StutterClear, 1: As it is

  const snaps = c.cuts
    .map((k, i) => ({ k, i }))
    .filter(({ k }) => k.kind !== "pause")
    .filter((_, j) => j % 3 === 0)
    .map(({ i }) => cutStart(i) + 6)
    .filter((at) => at < after.at - 12)

  return (
    <AbsoluteFill>
      <Particles count={18} color={C.teal} seed="judges" opacity={0.35} rise={0.3} />
      <Camera
        shots={[
          { at: 0, x: 960, y: 548, zoom: 0.97 },
          { at: openAt, x: 960, y: 540, zoom: 1, dur: 40 },
          { at: after.at - 6, x: 960, y: 540, zoom: 1.015, dur: Math.max(30, after.frames) },
          { at: outAt, x: 960, y: 540, zoom: 1, dur: 26 },
          { at: L1.at + 14, x: 960, y: 540, zoom: 1.045, dur: Math.max(40, scene.frames - L1.at - 14) },
        ]}
        drift={4}
        seed="judges-cam"
      >
        {/* ================= the call ================= */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            opacity: callIn * (1 - callOut),
            transform: `translateY(${-callOut * 24}px) scale(${1 - callOut * 0.03})`,
            transformOrigin: "960px 540px",
          }}
        >
          <GlowOrb x={PL.x + PL.w / 2} y={PL.y + PL.h / 2} r={760} color="rgba(125,211,252,0.5)" opacity={0.22 * pOpen + 0.18 * lvl} />

          {/* the player */}
          <div
            style={{
              position: "absolute",
              left: PL.x,
              top: PL.y,
              width: PL.w,
              height: PL.h,
              opacity: Math.min(1, pOpen * 1.5) * playerGate,
              transform: `scale(${0.86 + 0.14 * pOpen})`,
              transformOrigin: "50% 50%",
              borderRadius: 30,
              background: "linear-gradient(160deg, rgba(255,255,255,0.085), rgba(255,255,255,0.025))",
              border: `1px solid ${C.border2}`,
              boxShadow: `inset 0 1px 0 rgba(255,255,255,0.1), 0 50px 120px -40px rgba(0,0,0,0.9), 0 0 ${60 + 50 * lvl}px -18px rgba(94,234,212,${0.25 + 0.4 * lvl})`,
              overflow: "hidden",
            }}
          >
            <PlayerBar f={f} at={openAt + 12} />
            <VideoFrame f={f} openAt={openAt} after={after} words={c.after_words} />
            <div style={{ position: "absolute", left: 0, top: SCRUB_Y, width: PL.w, height: PL.h - SCRUB_Y }}>
              <div style={{ position: "absolute", left: 24, right: 24, top: 0, height: 1, background: C.border }} />
              <Ring f={f} at={waveAt} lengthNow={lengthNow} before={c.before_s} afterS={c.after_s} play={play} playing={f >= after.at} />
              <div style={{ position: "absolute", left: WAVE.x, top: WAVE.y, width: WAVE.w, height: WAVE.h }}>
                {/* the original length, as a faint track */}
                <div
                  style={{
                    position: "absolute",
                    left: 0,
                    top: WAVE.h / 2 - 1,
                    width: WAVE.w,
                    height: 2,
                    borderRadius: 1,
                    background: "rgba(255,255,255,0.09)",
                    opacity: ramp(f, waveAt, waveAt + 12),
                  }}
                />
                <div
                  style={{
                    position: "absolute",
                    left: WAVE.w + 10,
                    top: WAVE.h / 2 - 12,
                    fontFamily: MONO,
                    fontSize: 18,
                    color: C.dim,
                    opacity: ramp(f, waveAt + 10, waveAt + 22),
                  }}
                >
                  {fmt(c.before_s)}
                </div>
                <div
                  style={{
                    position: "absolute",
                    left: (WAVE.w * c.after_s) / c.before_s + 22,
                    top: WAVE.h / 2 - 12,
                    fontFamily: MONO,
                    fontSize: 18,
                    color: C.teal,
                    opacity: ramp(f, morphAt + 4, morphAt + 16),
                  }}
                >
                  {fmt(c.after_s)}
                </div>
                <div style={{ opacity: 1 - ramp(f, morphAt, morphAt + 12), position: "absolute", inset: 0 }}>
                  <TakeWave
                    f={f}
                    env={c.env_before}
                    seconds={c.before_s}
                    width={WAVE.w}
                    height={WAVE.h}
                    cuts={c.cuts}
                    tint={ramp(f, waveAt + 6, waveAt + 20)}
                    collapse={collapse}
                    reveal={ramp(f, waveAt, waveAt + 16, easeOut)}
                    seamAt={c.cuts.map((_, i) => cutStart(i) + 10)}
                  />
                </div>
                <div style={{ opacity: ramp(f, morphAt, morphAt + 12), position: "absolute", inset: 0 }}>
                  <Wave
                    env={c.env_after}
                    seconds={c.after_s}
                    width={(WAVE.w * c.after_s) / c.before_s}
                    height={WAVE.h}
                    color={C.teal}
                    progress={f >= after.at ? play : undefined}
                    dimAfter
                    glow={4 + 6 * lvl}
                  />
                </div>
              </div>
              {/* 0:19 → 0:08 · 9 stutters cut · 9 long pauses shortened */}
              <div
                style={{
                  position: "absolute",
                  left: WAVE.x,
                  top: WAVE.y + WAVE.h + 26,
                  display: "flex",
                  alignItems: "baseline",
                  gap: 18,
                  fontFamily: SANS,
                  fontSize: 28,
                  color: C.dim,
                  whiteSpace: "nowrap",
                }}
              >
                <span style={{ ...fadeUp(f, c0 + 4), fontWeight: 700, fontSize: 32, color: C.text, fontVariantNumeric: "tabular-nums" }}>
                  {fmt(c.before_s)} <span style={{ color: C.teal }}>→</span> {fmt(lengthNow)}
                </span>
                <span style={fadeUp(f, stutterLabelAt)}>·</span>
                <span style={{ ...fadeUp(f, stutterLabelAt), color: C.rose, fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
                  {f >= c1 ? stutters : stuttersNow} stutters cut
                </span>
                <span style={fadeUp(f, pauseLabelAt)}>·</span>
                <span style={{ ...fadeUp(f, pauseLabelAt), fontWeight: 500, fontVariantNumeric: "tabular-nums" }}>
                  {f >= c1 ? pauses : pausesNow} long pauses shortened
                </span>
              </div>
            </div>
            <LightSweep at={openAt + 10} dur={34} color="rgba(255,255,255,0.14)" />
            <LightSweep at={after.at - 4} dur={30} color="rgba(94,234,212,0.16)" />
          </div>

          {/* the call's tiles: a grid, then a column beside the shared screen */}
          {[1, 2, 3, 0].map((i) => {
            const gx = GRID.cx - GRID.w - GRID.gap / 2 + (i % 2) * (GRID.w + GRID.gap)
            const gy = GRID.cy - GRID.h - GRID.gap / 2 + Math.floor(i / 2) * (GRID.h + GRID.gap)
            const cy = PL.y + i * (COL.h + COL.gap)
            const pt = pTile(i)
            const r = { x: mix(gx, COL.x, pt), y: mix(gy, cy, pt), w: mix(GRID.w, COL.w, pt), h: mix(GRID.h, COL.h, pt) }
            const enter = springAt(f, 2 + i * 3)
            return i === 0 ? (
              <AlexTile key={i} f={f} r={r} enter={enter} shareAt={shareAt} settled={c01((pt - 0.8) / 0.15)} />
            ) : (
              <JudgeTile key={i} f={f} r={r} enter={enter} n={i} hearAt={hearAt + (i - 1) * 4} lvl={lvl} />
            )
          })}
        </div>

        {/* ================= both versions ================= */}
        <Compare f={f} at={cmpAt} pick={pickP} said={c.said} clean={c.clean} flips={[flipA, flipB]} />
      </Camera>

      {snaps.map((at, i) => (
        <Sfx key={`s${i}`} name="snap" at={at} volume={0.24} />
      ))}
      <Sfx name="pop" at={flipA} volume={0.22} />
      <Sfx name="pop" at={flipB} volume={0.22} />
    </AbsoluteFill>
  )
}

function fadeUp(f: number, at: number): CSSProperties {
  const p = springAt(f, at)
  return { opacity: Math.min(1, p * 1.3), transform: `translateY(${(1 - p) * 14}px)`, display: "inline-block" }
}

// ---------------------------------------------------------------- the player

function PlayerBar({ f, at }: { f: number; at: number }) {
  const parts: { t: string; c: string; w?: number }[] = [
    { t: "Last night's take", c: C.text, w: 700 },
    { t: "Thorough", c: C.teal },
    { t: "checked by Nemotron", c: C.violet },
    { t: "joined with room tone", c: C.dim },
  ]
  return (
    <div style={{ position: "absolute", left: 30, right: 30, top: 0, height: 64, display: "flex", alignItems: "center", gap: 14, fontFamily: MONO, fontSize: 21 }}>
      <div
        style={{
          ...fadeUp(f, at),
          width: 34,
          height: 34,
          borderRadius: 10,
          background: "rgba(94,234,212,0.16)",
          color: C.teal,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Icon name="video" size={22} />
      </div>
      {parts.map((p, i) => (
        <span key={i} style={{ ...fadeUp(f, at + 3 + i * 4), color: p.c, fontWeight: p.w ?? 500, whiteSpace: "nowrap" }}>
          {i > 0 && <span style={{ color: C.faint, marginRight: 14 }}>·</span>}
          {p.t}
        </span>
      ))}
    </div>
  )
}

// Alex's pitch video: his slide, and (while the clean take plays) its captions, word by word.
function VideoFrame({ f, openAt, after, words }: { f: number; openAt: number; after: { at: number; frames: number }; words: { w: string; s: number; e: number }[] }) {
  const cap = springAt(f, after.at - 6)
  const lift = springAt(f, after.at - 16)
  const tNow = (f - after.at) / 30
  return (
    <div
      style={{
        position: "absolute",
        left: VID.x,
        top: VID.y,
        width: VID.w,
        height: VID.h,
        borderRadius: 20,
        overflow: "hidden",
        background: "radial-gradient(120% 140% at 20% 10%, #13213a 0%, #0a1222 55%, #070b15 100%)",
        border: `1px solid ${C.border}`,
      }}
    >
      {/* rails of light, like the platform's edge */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 0,
          bottom: 0,
          background: "repeating-linear-gradient(100deg, rgba(125,211,252,0.035) 0 2px, transparent 2px 90px)",
          opacity: 0.8,
        }}
      />
      {/* the slide */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          gap: 64,
          padding: "0 80px",
          transform: `translateY(${-lift * 52}px) scale(${1 - lift * 0.24})`,
          transformOrigin: "50% 0%",
        }}
      >
        <Sign f={f} at={openAt + 14} />
        <div style={{ ...fadeUp(f, openAt + 20), fontFamily: SERIF, fontSize: 70, lineHeight: 1.02, letterSpacing: -1, color: C.text, maxWidth: 560 }}>
          Why are train signs <span style={{ fontStyle: "italic", color: C.sky }}>hard to read?</span>
        </div>
      </div>
      {/* captions of the clean take */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          bottom: 0,
          height: 200,
          padding: "0 44px 24px",
          boxSizing: "border-box",
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-end",
          gap: 10,
          background: "linear-gradient(180deg, transparent, rgba(3,6,12,0.82) 38%)",
          opacity: Math.min(1, cap * 1.4),
          transform: `translateY(${(1 - cap) * 30}px)`,
        }}
      >
        <div style={{ fontFamily: MONO, fontSize: 20, letterSpacing: 4, color: C.teal }}>
          WHAT ALEX MEANS <span style={{ color: C.dim, letterSpacing: 1 }}>· captions of the clean take</span>
        </div>
        <div style={{ fontFamily: SERIF, fontSize: 50, lineHeight: 1.08, letterSpacing: -0.5, color: C.text, minHeight: 108 }}>
          {words.map((w, i) => {
            const at = after.at + w.s * 30
            const p = springAt(f, at, SNAPPY)
            if (f < at) return null
            const current = tNow >= w.s && tNow < (words[i + 1]?.s ?? w.e + 0.4)
            return (
              <span
                key={i}
                style={{
                  display: "inline-block",
                  whiteSpace: "pre",
                  opacity: Math.min(1, p * 1.5),
                  transform: `translateY(${(1 - p) * 16}px)`,
                  filter: `blur(${(1 - Math.min(1, p)) * 5}px)`,
                  color: current ? C.teal : C.text,
                  textShadow: current ? `0 0 24px rgba(94,234,212,0.6)` : undefined,
                }}
              >
                {w.w}
                {i < words.length - 1 ? " " : ""}
              </span>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// The station sign on Alex's slide: 東京 over Tōkyō.
function Sign({ f, at }: { f: number; at: number }) {
  const p = springAt(f, at, SNAPPY)
  return (
    <div
      style={{
        flexShrink: 0,
        width: 360,
        height: 214,
        borderRadius: 18,
        background: "linear-gradient(180deg, #f5f7fb, #dfe6f0)",
        boxShadow: "0 30px 60px -24px rgba(0,0,0,0.8), 0 0 50px -10px rgba(125,211,252,0.35)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        overflow: "hidden",
        opacity: Math.min(1, p * 1.4),
        transform: `perspective(900px) rotateY(${(1 - p) * -24}deg) translateY(${(1 - p) * 20}px)`,
      }}
    >
      <div style={{ fontFamily: JP, fontWeight: 700, fontSize: 104, lineHeight: 1, color: "#0b1220", letterSpacing: 6 }}>東京</div>
      <div style={{ fontFamily: SANS, fontWeight: 600, fontSize: 38, color: "#24324a", marginTop: 6 }}>Tōkyō</div>
      <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 14, background: `linear-gradient(90deg, ${C.sky}, ${C.teal})` }} />
    </div>
  )
}

// Alex's take, cuts and all: stutter cuts tinted rose, long pauses grey; each cut shrinks out of the
// line (`collapse`), the rest closing up, and the seam it leaves flashes teal.
function TakeWave({
  f,
  env,
  seconds,
  width,
  height,
  cuts,
  tint,
  collapse,
  reveal,
  seamAt,
}: {
  f: number
  env: number[]
  seconds: number
  width: number
  height: number
  cuts: Cut[]
  tint: number
  collapse: number[]
  reveal: number
  seamAt: number[]
}) {
  const barS = 0.05
  const n = Math.ceil(seconds / barS)
  const slot = width / n
  const grey = "rgba(245,247,251,0.55)"
  const bars: ReactNode[] = []
  let x = 0
  for (let i = 0; i < n; i++) {
    const t0 = i * barS
    const tc = Math.min(seconds, t0 + barS / 2)
    const ci = cuts.findIndex((k) => tc >= k.start && tc < k.end)
    const k = ci >= 0 ? 1 - c01(collapse[ci] ?? 0) : 1
    const v = envAt(env, t0, t0 + barS)
    const shown = c01((reveal * n - i) / 10)
    const h = Math.max(3, Math.sqrt(v) * height) * (0.3 + 0.7 * k)
    const isPause = ci >= 0 && cuts[ci].kind === "pause"
    const col = ci < 0 ? C.sky : interpolateColors(tint, [0, 1], [C.sky, isPause ? grey : C.rose])
    if (slot * k > 0.05 && shown > 0.01)
      bars.push(
        <rect
          key={i}
          x={x + slot * k * 0.18}
          y={(height - h) / 2}
          width={Math.max(0.4, slot * k * 0.64)}
          height={h}
          rx={Math.min(slot * k * 0.32, 3)}
          fill={col}
          opacity={shown * (ci >= 0 ? 0.3 + 0.7 * k : 1)}
        />,
      )
    x += slot * k
  }
  return (
    <svg width={width} height={height} style={{ position: "absolute", left: 0, top: 0, overflow: "visible", filter: `drop-shadow(0 0 5px rgba(125,211,252,0.55))` }}>
      {/* the cut regions, as soft bands */}
      {cuts.map((k, i) => {
        const a = waveX(k.start, seconds, width, cuts, collapse)
        const w = (k.end - k.start) * (width / seconds) * (1 - c01(collapse[i]))
        if (w < 0.3) return null
        return (
          <rect
            key={`b${i}`}
            x={a}
            y={-8}
            width={w}
            height={height + 16}
            rx={6}
            fill={k.kind === "pause" ? "rgba(245,247,251,0.07)" : "rgba(251,113,133,0.13)"}
            stroke={k.kind === "pause" ? "rgba(245,247,251,0.14)" : "rgba(251,113,133,0.35)"}
            opacity={tint * reveal}
          />
        )
      })}
      {bars}
      {/* the seams flash teal as each cut closes */}
      {cuts.map((k, i) => {
        const fl = Math.max(0, 1 - Math.abs(f - seamAt[i]) / 9)
        if (fl <= 0) return null
        const sx = waveX(k.start, seconds, width, cuts, collapse)
        return <rect key={`s${i}`} x={sx - 1.5} y={-14} width={3} height={height + 28} rx={1.5} fill={C.teal} opacity={fl} style={{ filter: `drop-shadow(0 0 8px ${C.teal})` }} />
      })}
    </svg>
  )
}

// The listener's ring: a ghost ring is the take as recorded (0:19); the lit arc is what the judges
// sit through (0:08), and fills as it plays.
function Ring({ f, at, lengthNow, before, afterS, play, playing }: { f: number; at: number; lengthNow: number; before: number; afterS: number; play: number; playing: boolean }) {
  const p = springAt(f, at)
  const { r, cx, cy } = RING
  const circ = 2 * Math.PI * r
  const frac = lengthNow / before
  const fill = (afterS / before) * play
  const size = r * 2 + 40
  return (
    <div style={{ position: "absolute", left: cx - size / 2, top: cy - size / 2, width: size, height: size, opacity: Math.min(1, p * 1.4), transform: `scale(${0.8 + 0.2 * p})` }}>
      <svg width={size} height={size} style={{ position: "absolute", inset: 0, overflow: "visible" }}>
        <g transform={`translate(${size / 2} ${size / 2}) rotate(-90)`}>
          <circle r={r} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth={10} />
          <circle r={r} fill="none" stroke={C.sky} strokeOpacity={0.45} strokeWidth={10} strokeLinecap="round" strokeDasharray={`${circ * frac} ${circ}`} />
          {fill > 0.001 && (
            <circle
              r={r}
              fill="none"
              stroke={C.teal}
              strokeWidth={10}
              strokeLinecap="round"
              strokeDasharray={`${circ * fill} ${circ}`}
              style={{ filter: `drop-shadow(0 0 8px ${C.teal})` }}
            />
          )}
        </g>
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 2 }}>
        <div style={{ fontFamily: SANS, fontWeight: 700, fontSize: 36, color: C.text, fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>
          {playing ? fmt(play * afterS) : fmt(lengthNow)}
        </div>
        <div style={{ fontFamily: MONO, fontSize: 18, color: C.dim }}>{playing ? `of ${fmt(before)}` : "length"}</div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- the tiles

type Rect = { x: number; y: number; w: number; h: number }

function TileBox({ r, enter, border, glow, children }: { r: Rect; enter: number; border: string; glow?: string; children: ReactNode }) {
  return (
    <div
      style={{
        position: "absolute",
        left: r.x,
        top: r.y,
        width: r.w,
        height: r.h,
        borderRadius: 22,
        background: "linear-gradient(160deg, #182238, #0c111c)",
        border: `2px solid ${border}`,
        boxShadow: `0 24px 60px -28px rgba(0,0,0,0.9)${glow ? `, 0 0 34px -6px ${glow}` : ""}`,
        overflow: "hidden",
        opacity: Math.min(1, enter * 1.4),
        transform: `scale(${0.92 + 0.08 * enter})`,
      }}
    >
      {children}
    </div>
  )
}

function AlexTile({ f, r, enter, shareAt, settled }: { f: number; r: Rect; enter: number; shareAt: number; settled: number }) {
  const border = springAt(f, shareAt)
  const s = border * settled // the label switches once the tile has settled in the column
  const av = r.h * 0.4
  return (
    <TileBox r={r} enter={enter} border={interpolateColors(border, [0, 1], ["rgba(255,255,255,0.08)", "rgba(125,211,252,0.55)"])}>
      <div style={{ position: "absolute", left: r.w / 2 - av / 2, top: r.h * 0.42 - av / 2 }}>
        <Avatar size={av} level={0.25} />
      </div>
      <div
        style={{
          position: "absolute",
          left: 18,
          right: 14,
          bottom: 12,
          display: "flex",
          alignItems: "center",
          gap: 10,
          fontFamily: SANS,
          fontSize: 22,
          color: C.text,
          whiteSpace: "nowrap",
          overflow: "hidden",
          height: 30,
        }}
      >
        <span style={{ opacity: 1 - c01(s * 2.5), position: "absolute", left: 0 }}>Alex</span>
        <span style={{ opacity: c01((s - 0.4) / 0.6), display: "flex", alignItems: "center", gap: 10, minWidth: 0, transform: `translateY(${(1 - s) * 10}px)` }}>
          <span style={{ color: C.sky, display: "flex", flexShrink: 0 }}>
            <Icon name="video" size={22} />
          </span>
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", minWidth: 0 }}>Alex is sharing his screen</span>
        </span>
      </div>
    </TileBox>
  )
}

function JudgeTile({ f, r, enter, n, hearAt, lvl }: { f: number; r: Rect; enter: number; n: number; hearAt: number; lvl: number }) {
  const hear = springAt(f, hearAt, SNAPPY)
  const pulse = Math.max(0, 1 - Math.abs(f - hearAt - 6) / 16)
  const glowA = Math.max(0.35 * c01(hear) * 0.4, lvl, pulse * 0.8)
  const border = `rgba(94,234,212,${(0.08 + 0.85 * glowA).toFixed(3)})`
  const av = r.h * 0.4
  const initials = ["R", "M", "S"][n - 1]
  return (
    <TileBox r={r} enter={enter} border={border} glow={glowA > 0.2 ? `rgba(94,234,212,${(glowA * 0.8).toFixed(3)})` : undefined}>
      <div
        style={{
          position: "absolute",
          left: r.w / 2 - av / 2,
          top: r.h * 0.42 - av / 2,
          width: av,
          height: av,
          borderRadius: "50%",
          background: "radial-gradient(circle at 35% 30%, rgba(245,247,251,0.18), rgba(245,247,251,0.05) 60%, rgba(10,14,26,0.9))",
          border: `1.5px solid ${C.border2}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: SANS,
          fontWeight: 600,
          fontSize: av * 0.4,
          color: C.dim,
          boxShadow: `0 0 0 ${4 + 10 * lvl}px rgba(94,234,212,${(0.12 * lvl).toFixed(3)})`,
        }}
      >
        {initials}
      </div>
      <div style={{ position: "absolute", left: 18, bottom: 12, fontFamily: SANS, fontSize: 22, color: C.dim }}>Judge {n}</div>
      <div
        style={{
          position: "absolute",
          right: 14,
          top: 12,
          width: 40,
          height: 40,
          borderRadius: 12,
          background: "rgba(94,234,212,0.14)",
          color: C.teal,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          opacity: Math.min(1, hear * 1.4),
          transform: `scale(${0.5 + 0.5 * hear})`,
        }}
      >
        <Icon name="headphones" size={24} />
      </div>
    </TileBox>
  )
}

// ---------------------------------------------------------------- both versions

function Compare({ f, at, pick, said, clean, flips }: { f: number; at: number; pick: number; said: string; clean: string; flips: [number, number] }) {
  const inP = ramp(f, at - 2, at + 12, easeOut)
  if (inP <= 0) return null
  const flash = Math.max(...flips.map((fl) => Math.max(0, 1 - Math.abs(f - fl - 4) / 12)))
  const head = springAt(f, at)
  const rows = [
    { label: "As it is", text: said, at: at + 4, on: c01(pick) },
    { label: "StutterClear", text: clean, at: at + 8, on: c01(1 - pick) },
  ]
  return (
    <div style={{ position: "absolute", inset: 0, opacity: inP }}>
      <GlowOrb x={960} y={540} r={860} color="rgba(94,234,212,0.4)" opacity={0.15} />
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 196,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          gap: 28,
          opacity: Math.min(1, head * 1.3),
          transform: `translateY(${(1 - head) * 18}px)`,
        }}
      >
        <Avatar size={62} level={0.3 + 0.5 * flash} />
        <Toggle pick={pick} flash={flash} />
      </div>
      <div style={{ position: "absolute", left: 230, top: 304, width: 1460, display: "flex", flexDirection: "column", gap: 24 }}>
        {rows.map((r, i) => {
          const e = springAt(f, r.at - 4)
          const glow = r.on
          return (
            <div
              key={r.label}
              style={{
                position: "relative",
                boxSizing: "border-box",
                padding: "28px 116px 32px 44px",
                borderRadius: 28,
                background: "linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02))",
                border: `1.5px solid ${interpolateColors(glow, [0, 1], ["rgba(255,255,255,0.1)", "rgba(94,234,212,0.7)"])}`,
                boxShadow: `inset 0 1px 0 rgba(255,255,255,0.08), 0 30px 80px -34px rgba(0,0,0,0.85), 0 0 ${50 * glow}px -12px rgba(94,234,212,${(0.55 * glow).toFixed(3)})`,
                opacity: Math.min(1, e * 1.4),
                transform: `translateY(${(1 - e) * 26}px)`,
                overflow: "hidden",
              }}
            >
              <div style={{ fontFamily: MONO, fontSize: 22, letterSpacing: 4, textTransform: "uppercase", color: C.teal, marginBottom: 14 }}>{r.label}</div>
              <KineticText text={r.text} at={r.at} per={0.7} effect="blur" style={{ fontFamily: SERIF, fontSize: 54, lineHeight: 1.08, letterSpacing: -0.5, color: C.text }} />
              {/* what plays: a speaker on the chosen version (both lines stay equally bright) */}
              <div
                style={{
                  position: "absolute",
                  right: 30,
                  top: 28,
                  width: 62,
                  height: 62,
                  borderRadius: 18,
                  background: "rgba(94,234,212,0.14)",
                  border: "1.5px solid rgba(94,234,212,0.6)",
                  color: C.teal,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  opacity: glow * ramp(f, at + 8, at + 20),
                  transform: `scale(${0.6 + 0.4 * glow})`,
                }}
              >
                <Icon name="speaker" size={32} />
              </div>
              <LightSweep at={r.at + 2 + i * 4} dur={40} color="rgba(255,255,255,0.09)" />
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Toggle({ pick, flash }: { pick: number; flash: number }) {
  const segW = 270
  const h = 72
  return (
    <div
      style={{
        position: "relative",
        width: segW * 2 + 12,
        height: h,
        borderRadius: 999,
        background: "rgba(255,255,255,0.05)",
        border: `1px solid ${C.border2}`,
        boxShadow: `inset 0 1px 0 rgba(255,255,255,0.08), 0 0 ${30 * flash}px rgba(94,234,212,${0.5 * flash})`,
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 6,
          left: 6 + (1 - pick) * segW,
          width: segW,
          height: h - 12,
          borderRadius: 999,
          background: "linear-gradient(135deg, rgba(125,211,252,0.35), rgba(94,234,212,0.35))",
          border: `1px solid rgba(94,234,212,0.7)`,
          boxShadow: "0 0 24px -4px rgba(94,234,212,0.6)",
        }}
      />
      {["As it is", "StutterClear"].map((t, i) => {
        const on = i === 0 ? pick : 1 - pick
        return (
          <div
            key={t}
            style={{
              position: "absolute",
              top: 0,
              left: 6 + i * segW,
              width: segW,
              height: h,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontFamily: SANS,
              fontWeight: 600,
              fontSize: 30,
              color: interpolateColors(c01(on), [0, 1], [C.dim, C.text]),
            }}
          >
            {t}
          </div>
        )
      })}
    </div>
  )
}
