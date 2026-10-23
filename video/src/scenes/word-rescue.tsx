import type { CSSProperties, ReactNode } from "react"
import { AbsoluteFill, interpolate, interpolateColors, useCurrentFrame } from "remotion"
import { getLength, getPointAtLength } from "@remotion/paths"
import { noise2D } from "@remotion/noise"

import type { Cue, SceneProps } from "../data"
import { clipCue, lineCue } from "../data"
import {
  Avatar,
  Block,
  BOUNCY,
  CallWindow,
  Camera,
  clamp,
  Diagram,
  easeIn,
  easeOut,
  FlowPath,
  GlowOrb,
  Icon,
  LightSweep,
  Logo,
  NvTag,
  Particles,
  ramp,
  Sfx,
  SNAPPY,
  springAt,
  Tile,
} from "../kit"
import { C, MONO, SANS, SERIF } from "../theme"

// When a word won't come.
// The call fills the left of the stage: Alex's tile, three judges and the SHARED live caption strip
// (NeMo). Under it, Alex's PRIVATE StutterClear panel: the stuck sound shows only there. Right of a
// dashed "text only" line, the cloud: Nemotron (word rescue) and Magpie (Say it).

// Stage layout (1920x1080 stage coordinates).
const CALL = { x: 96, y: 108, w: 1160, h: 476 }
const PANEL = { x: 96, y: 620, w: 1160, h: 228 }
const BOUNDARY_X = 1298
const CLOUD_X = 1340
const CLOUD_W = 484
const NEMO = { x: CLOUD_X, y: 196, w: CLOUD_W, h: 262 }
const MAGPIE = { x: CLOUD_X, y: 556, w: CLOUD_W, h: 176 }
const GAP_Y = 602 // the channel between the call window and the panel
// The rescue row (chips and the Say it button) in stage coordinates.
const ROW_Y = 726
const ROW_H = 50
const CHIP_W = [134, 142, 168]
const CHIP_X = [184, 330, 484]
const SAY = { x: 668, y: ROW_Y, w: 140, h: ROW_H }
const SAY_C = { x: SAY.x + SAY.w / 2, y: SAY.y + SAY.h / 2 }

// Paths
const P_CAPTION = `M 1236 506 C 1292 506, 1286 330, ${CLOUD_X} 330`
const P_CONTEXT = `M 1234 756 C 1296 756, 1278 440, ${CLOUD_X} 440`
// Nemotron's three suggestions come back along the channel, each stopping above its own chip.
const pBack = (x: number) => `M ${CLOUD_X} 420 C 1290 420, 1306 ${GAP_Y}, 1240 ${GAP_Y} L ${x} ${GAP_Y}`
const P_BACK = pBack(CHIP_X[0] + CHIP_W[0] / 2)
const P_MAGPIE = `M ${SAY_C.x} ${SAY.y - 4} L ${SAY_C.x} ${GAP_Y + 24} Q ${SAY_C.x} ${GAP_Y} ${SAY_C.x + 24} ${GAP_Y} L 1236 ${GAP_Y} C 1290 ${GAP_Y}, 1290 644, ${CLOUD_X} 644`
const P_MAGPIE_BACK = `M ${CLOUD_X} 660 C 1290 660, 1290 ${GAP_Y + 12}, 1236 ${GAP_Y + 12} L ${SAY_C.x + 36} ${GAP_Y + 12} Q ${SAY_C.x + 12} ${GAP_Y + 12} ${SAY_C.x + 12} ${GAP_Y + 36} L ${SAY_C.x + 12} ${SAY.y - 4}`

// The frame where a phrase is said inside a narration line (by its place in the text).
const phraseAt = (c: Cue, phrase: string, fallback: number) => {
  const t = c.text ?? ""
  const i = t.toLowerCase().indexOf(phrase.toLowerCase())
  const u = i >= 0 && t.length ? i / t.length : fallback
  return Math.round(c.at + u * c.frames)
}

// A pill of text (or a dot of audio) that travels a path once, from t0 over dur frames, and is
// absorbed where it arrives (it stays full size until it reaches the block's edge, then shrinks over
// its last few frames) instead of parking.
function Packet({ d, t0, dur, label, color = C.violet, dot = false, absorb = true }: { d: string; t0: number; dur: number; label?: string; color?: string; dot?: boolean; absorb?: boolean }) {
  const f = useCurrentFrame()
  if (f < t0 || f > t0 + dur + 8) return null
  const len = getLength(d)
  const u = interpolate(f, [t0, t0 + dur], [0, 1], { ...clamp, easing: easeOut })
  const p = getPointAtLength(d, u * len)
  if (!p) return null
  const o = Math.min(1, (f - t0) / 4) * (1 - ramp(f, t0 + dur, t0 + dur + 8, easeIn))
  const k = absorb ? 1 - 0.7 * ramp(f, t0 + dur - 5, t0 + dur + 3, easeIn) : 1
  if (dot) {
    // the label rides on its own dark pill (so the paths never strike through it) and leaves before
    // the dot lands, so it never sits on the rescue card
    const lw = 24 + (label ?? "").length * 12
    return (
      <g opacity={o}>
        <circle cx={p.x} cy={p.y} r={16} fill={color} opacity={0.18} />
        <circle cx={p.x} cy={p.y} r={7} fill={color} style={{ filter: `drop-shadow(0 0 8px ${color})` }} />
        {label && (
          <g opacity={1 - ramp(u, 0.65, 0.9)}>
            <rect x={p.x - lw / 2} y={p.y - 32 - 26} width={lw} height={32} rx={14} fill="rgba(14,12,32,0.95)" stroke={color} strokeWidth={1.6} />
            <text x={p.x} y={p.y - 32 - 26 + 23} textAnchor="middle" fontFamily={MONO} fontSize={20} fill={C.text}>
              {label}
            </text>
          </g>
        )}
      </g>
    )
  }
  const text = label ?? ""
  const w = 30 + text.length * 12.2
  return (
    <g opacity={o} transform={`translate(${p.x} ${p.y}) scale(${k})`}>
      <rect x={-w / 2} y={-19} width={w} height={38} rx={19} fill="rgba(14,12,32,0.95)" stroke={color} strokeWidth={1.8} style={{ filter: `drop-shadow(0 0 10px ${color}88)` }} />
      <text x={0} y={7} textAnchor="middle" fontFamily={MONO} fontSize={20} fill={C.text}>
        {text}
      </text>
    </g>
  )
}

// A judge: a tile with initials, never a face.
function Judge({ initial }: { initial: string }) {
  return (
    <Tile name="Judge" style={{ flex: 1 }}>
      <div
        style={{
          width: 48,
          height: 48,
          borderRadius: 24,
          marginLeft: 120,
          background: "linear-gradient(160deg, rgba(255,255,255,0.14), rgba(255,255,255,0.04))",
          border: `1px solid ${C.border2}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: SANS,
          fontWeight: 600,
          fontSize: 22,
          color: C.dim,
        }}
      >
        {initial}
      </div>
    </Tile>
  )
}

function Abs({ x, y, w, h, children, style }: { x: number; y: number; w?: number; h?: number; children?: ReactNode; style?: CSSProperties }) {
  return <div style={{ position: "absolute", left: x, top: y, width: w, height: h, ...style }}>{children}</div>
}

export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const L0 = lineCue(scene, 0)
  const L1 = lineCue(scene, 1)
  const L2 = lineCue(scene, 2)
  const sayCue = clipCue(scene, "say")
  const sayAt = sayCue?.at ?? L2.at + L2.frames + 12
  const sayLen = sayCue?.frames ?? 18

  // --- beats -----------------------------------------------------------------------------------
  const stuckAt = L0.at + 2 // "Then a word won't come."
  const captionAt = phraseAt(L1, "the caption", 0.15)
  const contextAt = phraseAt(L1, "context card", 0.35)
  const asTextAt = phraseAt(L1, "as text", 0.55)
  const suggestsAt = phraseAt(L1, "suggests", 0.7)
  const rescueAt = Math.max(suggestsAt + 18, L1.at + L1.frames - 10) // the card springs up at the end of line 1
  const himselfAt = phraseAt(L2, "himself", 0.15)
  const tapAt = phraseAt(L2, "tap it", 0.33) + 4
  const magpieAt = phraseAt(L2, "Magpie", 0.5)
  const syntheticAt = phraseAt(L2, "synthetic", 0.8)
  const completeAt = sayAt + Math.round(sayLen * 0.4) // NeMo captions the word as it is heard

  // --- the shared caption: NeMo bursts every 480 ms, stopping after "Roman" -----------------------
  const words = d.rescue.said.split(" ")
  const BURST = 14.4
  const lastBurst = L0.at + 1
  let hidden = 0
  for (let j = 0; j < 4; j++) if (f < lastBurst - j * BURST) hidden += 2
  const shownWords = words.slice(0, Math.max(0, words.length - hidden))
  const newestFrom = Math.max(0, words.length - hidden - 2)
  const burstIdx = [0, 1, 2, 3].find((j) => f >= lastBurst - j * BURST) ?? 4
  const burstFrame = lastBurst - burstIdx * BURST
  const completeP = springAt(f, completeAt, SNAPPY)
  const caretOn = Math.floor(f / 14) % 2 === 0

  // --- Alex ------------------------------------------------------------------------------------------
  const struggle = ramp(f, stuckAt, stuckAt + 10) * (1 - ramp(f, sayAt, sayAt + 12))
  const talk = 0.32 + 0.3 * Math.abs(noise2D("alex", f / 5, 0))
  const level = interpolate(struggle, [0, 1], [f < stuckAt ? talk : 0.22 + 0.12 * Math.abs(noise2D("alex2", f / 9, 1)), 0.1])

  // --- status in the private panel -------------------------------------------------------------------
  const statusAmber = f >= stuckAt + 4 && f < sayAt + 4
  const statusSwap = statusAmber ? springAt(f, stuckAt + 4, BOUNCY) : springAt(f, sayAt + 4, BOUNCY)
  const statusColor = statusAmber ? C.amber : C.sky
  const statusText = statusAmber ? "Pause kept. Take your time." : "Listening…"

  // --- the cloud side ---------------------------------------------------------------------------
  const boundaryDraw = ramp(f, L1.at - 4, L1.at + 22, easeOut)
  const nemoAppear = springAt(f, L1.at - 2) // solid as its name is spoken
  const nemoActive = ramp(f, captionAt + 20, captionAt + 30) * (1 - ramp(f, rescueAt + 10, rescueAt + 30))
  const textPulse = Math.max(0, 1 - Math.abs(f - asTextAt - 6) / 16)
  const magpieAppear = springAt(f, magpieAt - 3)
  const toMagpie = { t0: tapAt + 4, dur: 30 }
  const fromMagpie = { t0: sayAt - 22, dur: 20 }
  const magpieActive = ramp(f, toMagpie.t0 + toMagpie.dur - 4, toMagpie.t0 + toMagpie.dur + 6) * (1 - ramp(f, sayAt + sayLen, sayAt + sayLen + 16))

  // --- the rescue card --------------------------------------------------------------------------
  const stuckIn = springAt(f, stuckAt, SNAPPY)
  const stuckOut = ramp(f, rescueAt - 12, rescueAt, easeIn)
  const rescueIn = springAt(f, rescueAt + 2, SNAPPY)
  const chipIn = [0, 1, 2].map((i) => springAt(f, rescueAt + 8 + i * 5, BOUNCY))
  const pick = d.rescue.suggestions.indexOf(d.say.word)
  const glowLetters = ramp(f, himselfAt - 2, himselfAt + 10, easeOut)
  const sayBtnIn = springAt(f, rescueAt + 22, SNAPPY)
  const press = Math.max(0, 1 - Math.abs(f - tapAt - 2) / 4)
  const tapped = f >= tapAt
  const playing = f >= sayAt && f < sayAt + sayLen + 6
  const labelBright = ramp(f, syntheticAt - 4, syntheticAt + 8)

  // --- context card -----------------------------------------------------------------------------
  const ctxLit = ramp(f, contextAt - 4, contextAt + 8) * (1 - ramp(f, rescueAt + 14, rescueAt + 34))
  const capLit = ramp(f, captionAt - 4, captionAt + 8) * (1 - ramp(f, contextAt + 8, contextAt + 24))

  // --- cursor -----------------------------------------------------------------------------------
  const curIn = ramp(f, tapAt - 22, tapAt - 2, easeOut)
  const curOut = ramp(f, tapAt + 18, tapAt + 32, easeIn)
  const cur = {
    x: interpolate(curIn, [0, 1], [1010, SAY.x + SAY.w - 8]) + curOut * 60,
    y: interpolate(curIn, [0, 1], [872, SAY.y + 14]) - curOut * 50,
  }

  // --- camera -----------------------------------------------------------------------------------
  const shots = [
    { at: 0, x: 690, y: 510, zoom: 1.05 },
    // line 1: pull back fast so Nemotron is fully in frame when it is named, then a slow push
    { at: L1.at - 16, x: 960, y: 520, zoom: 1.0, dur: 22 },
    { at: L1.at + 18, x: 975, y: 515, zoom: 1.035, dur: Math.max(40, himselfAt - 6 - (L1.at + 18)) },
    // "He can say it himself": the rescue row, large
    { at: himselfAt - 6, x: 650, y: 745, zoom: 1.42, dur: 28 },
    // "NVIDIA Magpie": pull back with the word as it travels to Magpie; the audio comes back in frame
    // (y 540 keeps the call window's top edge below the NVIDIA tracker strip)
    { at: Math.min(magpieAt - 8, tapAt + 6), x: 960, y: 540, zoom: 1.03, dur: 24 },
    // the payoff: a push onto the call and its shared caption (the cloud leaves the frame). Centred
    // high so the empty stage above the call window sits under the NVIDIA tracker strip.
    { at: sayAt - 4, x: CALL.x + CALL.w / 2, y: 410, zoom: 1.45, dur: 36 },
  ]

  const panelIn = springAt(f, 2)
  const cloudDim = 1 - ramp(f, completeAt, completeAt + 24)
  const restDim = 1 - 0.55 * ramp(f, completeAt + 8, completeAt + 32)
  const chipLabel = (i: number) => d.rescue.suggestions[i] ?? ""

  return (
    <AbsoluteFill>
      <Camera shots={shots} drift={5} seed="rescue">
        {/* light behind the focal points */}
        <GlowOrb x={330} y={756} r={300} color={C.amber} opacity={0.22 * struggle * (1 - stuckOut)} />
        <GlowOrb x={CLOUD_X + CLOUD_W / 2} y={330} r={360} color={C.violet} opacity={0.08 + 0.2 * nemoAppear * (0.4 + 0.6 * nemoActive)} />
        <GlowOrb x={460} y={740} r={340} color={C.violet} opacity={0.16 * rescueIn} />
        <GlowOrb x={660} y={510} r={420} color={C.sky} opacity={0.1 + 0.12 * completeP} />
        <Particles count={18} color={C.violet} seed="nemo" x={CLOUD_X - 20} y={150} w={CLOUD_W + 40} h={620} opacity={0.4 * nemoAppear} rise={0.35} />
        <Particles count={8} color={C.amber} seed="stuck" x={140} y={680} w={560} h={170} opacity={0.45 * struggle * (1 - stuckOut)} rise={0.2} />

        {/* the call */}
        <Abs x={CALL.x} y={CALL.y}>
          <CallWindow title="Demo day · video call" width={CALL.w} height={CALL.h}>
            <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={{ flex: 1, display: "flex", gap: 14 }}>
                <Tile name="Alex" speaking={1} style={{ flex: 2.75 }}>
                  <Avatar size={118} level={level} struggle={struggle} />
                </Tile>
                <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 10 }}>
                  <Judge initial="M" />
                  <Judge initial="R" />
                  <Judge initial="S" />
                </div>
              </div>
              {/* the SHARED caption strip */}
              <div
                style={{
                  position: "relative",
                  height: 128,
                  borderRadius: 16,
                  padding: "12px 22px",
                  boxSizing: "border-box",
                  background: "rgba(3,5,10,0.62)",
                  border: `1.5px solid ${interpolateColors(capLit, [0, 1], ["rgba(255,255,255,0.09)", C.violet])}`,
                  boxShadow: capLit > 0.05 ? `0 0 ${40 * capLit}px -6px ${C.violet}` : undefined,
                  overflow: "hidden",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 12, fontFamily: MONO, fontSize: 20, letterSpacing: 3, color: C.dim }}>
                  <Icon name="captions" size={22} color={C.teal} />
                  LIVE CAPTIONS (SHARED)
                  <NvTag label="NVIDIA NeMo" style={{ fontSize: 18, letterSpacing: 1, marginLeft: 4 }} />
                </div>
                <div style={{ marginTop: 6, fontFamily: SANS, fontSize: 28, fontWeight: 500, lineHeight: 1.3, color: C.text }}>
                  {shownWords.map((w, i) => {
                    const fresh = i >= newestFrom
                    const p = fresh ? springAt(f, burstFrame, SNAPPY) : 1
                    return (
                      <span key={i} style={{ display: "inline-block", whiteSpace: "pre", opacity: Math.min(1, p * 1.4), transform: `translateY(${(1 - p) * 10}px)` }}>
                        {w + " "}
                      </span>
                    )
                  })}
                  {f >= completeAt && (
                    <span
                      style={{
                        display: "inline-block",
                        whiteSpace: "pre",
                        opacity: Math.min(1, completeP * 1.4),
                        transform: `translateY(${(1 - completeP) * 10}px) scale(${1 + 0.4 * completeP})`,
                        transformOrigin: "0 100%",
                      }}
                    >
                      <span style={{ position: "relative", display: "inline-block", color: C.violet, textShadow: `0 0 24px ${C.violet}` }}>
                        {d.say.word}
                        <span
                          style={{
                            position: "absolute",
                            left: 0,
                            bottom: 1,
                            height: 3,
                            borderRadius: 2,
                            width: "100%",
                            background: C.violet,
                            boxShadow: `0 0 10px ${C.violet}`,
                            transformOrigin: "0 50%",
                            transform: `scaleX(${ramp(f, completeAt + 4, completeAt + 16, easeOut)})`,
                          }}
                        />
                      </span>
                      .
                    </span>
                  )}
                  {f < completeAt && <span style={{ color: C.teal, opacity: caretOn ? 0.9 : 0 }}>▍</span>}
                </div>
                <LightSweep at={completeAt + 2} dur={34} color="rgba(196,181,253,0.22)" />
              </div>
            </div>
          </CallWindow>
        </Abs>

        {/* Alex's PRIVATE panel, the rescue row and the context card (they step back at the end) */}
        <div style={{ position: "absolute", inset: 0, opacity: restDim }}>
        <Abs
          x={PANEL.x}
          y={PANEL.y}
          w={PANEL.w}
          h={PANEL.h}
          style={{
            opacity: Math.min(1, panelIn * 1.3),
            transform: `translateY(${(1 - panelIn) * 30}px)`,
            borderRadius: 26,
            background: "linear-gradient(160deg, rgba(22,28,48,0.9), rgba(10,14,26,0.86))",
            border: `1px solid ${C.border2}`,
            boxShadow: `inset 0 1px 0 rgba(255,255,255,0.1), 0 40px 90px -30px rgba(0,0,0,0.9)${rescueIn > 0.05 ? `, 0 0 ${70 * rescueIn}px -24px ${C.violet}` : ""}`,
            overflow: "hidden",
          }}
        >
          {/* header */}
          <div style={{ position: "absolute", left: 24, right: 24, top: 16, height: 34, display: "flex", alignItems: "center", gap: 16 }}>
            <Logo size={30} />
            <span style={{ width: 1, height: 22, background: C.border2 }} />
            <Icon name="lock" size={20} color={C.dim} />
            <span style={{ fontFamily: MONO, fontSize: 20, letterSpacing: 3, color: C.dim }}>ONLY ALEX SEES THIS</span>
            <span
              style={{
                marginLeft: "auto",
                display: "inline-flex",
                alignItems: "center",
                gap: 10,
                padding: "5px 16px",
                borderRadius: 999,
                fontFamily: SANS,
                fontSize: 24,
                fontWeight: 600,
                color: statusAmber ? C.amber : C.text,
                background: `${statusColor}1c`,
                border: `1px solid ${statusColor}66`,
                transform: `scale(${f < stuckAt + 4 ? 1 : 0.92 + 0.08 * statusSwap})`,
                opacity: f < stuckAt + 4 ? 1 : Math.min(1, Math.max(0.3, statusSwap * 1.4)),
              }}
            >
              <span style={{ width: 10, height: 10, borderRadius: 5, background: statusColor, boxShadow: `0 0 10px ${statusColor}`, opacity: 0.55 + 0.45 * Math.abs(Math.sin(f / 9)) }} />
              {statusText}
            </span>
          </div>

          {/* left: the stuck sound, then the word rescue */}
          {stuckOut < 1 && (
            <div
              style={{
                position: "absolute",
                left: 30,
                top: 62,
                width: 680,
                height: 150,
                display: "flex",
                alignItems: "baseline",
                gap: 22,
                opacity: (1 - stuckOut) * Math.min(1, 0.001 + stuckIn * 1.3),
                transform: `translateY(${(1 - stuckIn) * 18 - stuckOut * 24}px)`,
              }}
            >
              <span style={{ fontFamily: SERIF, fontSize: 76, color: C.dim, letterSpacing: -1, lineHeight: 1.8 }}>…in Roman</span>
              <span style={{ fontFamily: SERIF, fontStyle: "italic", fontSize: 132, color: C.amber, letterSpacing: -2, lineHeight: 1, textShadow: `0 0 40px ${C.amber}66` }}>
                {d.rescue.onset}–
              </span>
              <span style={{ fontFamily: SERIF, fontSize: 110, color: C.amber, opacity: caretOn ? 0.8 : 0.1, marginLeft: -16 }}>|</span>
            </div>
          )}
          {rescueIn > 0.01 && (
            <div
              style={{
                position: "absolute",
                left: 16,
                top: 56,
                width: 712,
                height: 160,
                borderRadius: 18,
                border: `1px solid ${C.violet}55`,
                background: "linear-gradient(160deg, rgba(196,181,253,0.1), rgba(196,181,253,0.03))",
                opacity: Math.min(1, rescueIn * 1.3),
                transform: `translateY(${(1 - rescueIn) * 26}px) scale(${0.96 + 0.04 * rescueIn})`,
                overflow: "hidden",
              }}
            >
              <div style={{ position: "absolute", left: 22, top: 12, display: "flex", alignItems: "center", gap: 10, fontFamily: MONO, fontSize: 20, letterSpacing: 3, color: C.violet }}>
                <Icon name="sparkles" size={20} color={C.violet} />
                WORD RESCUE · NEMOTRON
              </div>
              <LightSweep at={rescueAt + 4} dur={36} color="rgba(196,181,253,0.25)" />
            </div>
          )}
        </Abs>

        {/* the rescue row, in stage coordinates so the cursor and paths can find it */}
        {rescueIn > 0.01 && (
          <>
            <Abs x={124} y={ROW_Y - 6} style={{ fontFamily: SERIF, fontStyle: "italic", fontSize: 60, lineHeight: 1, color: C.amber, opacity: Math.min(1, rescueIn) * (1 - 0.5 * ramp(f, sayAt, sayAt + 12)) }}>
              {d.rescue.onset}…
            </Abs>
            {[0, 1, 2].map((i) => {
              const p = chipIn[i]
              const isPick = i === pick
              const g = isPick ? glowLetters : 0
              const dimOthers = !isPick ? 1 - 0.45 * glowLetters : 1
              return (
                <Abs
                  key={i}
                  x={CHIP_X[i]}
                  y={ROW_Y}
                  w={CHIP_W[i]}
                  h={ROW_H}
                  style={{
                    borderRadius: 999,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 8,
                    fontFamily: SANS,
                    fontSize: 28,
                    fontWeight: 600,
                    color: C.text,
                    background: interpolateColors(g, [0, 1], ["rgba(255,255,255,0.06)", "rgba(196,181,253,0.24)"]),
                    border: `1.5px solid ${interpolateColors(g, [0, 1], ["rgba(196,181,253,0.4)", C.violet])}`,
                    boxShadow: g > 0.02 ? `0 0 ${46 * g}px -4px ${C.violet}` : undefined,
                    opacity: Math.min(1, p * 1.5) * dimOthers,
                    transform: `scale(${(0.5 + 0.5 * p) * (1 + 0.06 * g)})`,
                  }}
                >
                  {isPick && tapped && <Icon name="check" size={22} color={C.violet} stroke={3} />}
                  {chipLabel(i)}
                </Abs>
              )
            })}
            {/* Say it */}
            <Abs
              x={SAY.x}
              y={SAY.y}
              w={SAY.w}
              h={SAY.h}
              style={{
                borderRadius: 999,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 10,
                fontFamily: SANS,
                fontSize: 28,
                fontWeight: 700,
                color: tapped ? "#140f2b" : C.text,
                background: tapped ? C.violet : "rgba(196,181,253,0.14)",
                border: `1.5px solid ${C.violet}`,
                boxShadow: `0 0 ${tapped ? 44 : 14 * glowLetters}px -4px ${C.violet}`,
                opacity: Math.min(1, sayBtnIn * 1.4),
                transform: `scale(${(0.6 + 0.4 * sayBtnIn) * (1 - 0.08 * press)})`,
              }}
            >
              {playing ? (
                <span style={{ display: "inline-flex", alignItems: "center", gap: 3, height: 26 }}>
                  {[0, 1, 2, 3, 4].map((b) => {
                    const env = Math.sin(Math.PI * Math.min(1, Math.max(0, (f - sayAt) / sayLen)))
                    const h = 6 + 20 * env * (0.5 + 0.5 * Math.abs(Math.sin(f / 2.2 + b * 1.7)))
                    return <span key={b} style={{ width: 4, height: h, borderRadius: 2, background: "#140f2b" }} />
                  })}
                </span>
              ) : (
                <Icon name="speaker" size={24} color={tapped ? "#140f2b" : C.violet} />
              )}
              Say it
            </Abs>
            {/* the honest label of the synthetic voice */}
            <Abs
              x={124}
              y={ROW_Y + ROW_H + 14}
              w={690}
              style={{
                fontFamily: SANS,
                fontSize: 20,
                lineHeight: 1.3,
                color: interpolateColors(labelBright, [0, 1], ["rgba(245,247,251,0.66)", C.text]),
                opacity: Math.min(1, sayBtnIn * 1.4),
                whiteSpace: "nowrap",
              }}
            >
              <span style={{ color: C.violet }}>●</span> {d.say.label}
            </Abs>
          </>
        )}

        {/* context card */}
        <Abs
          x={PANEL.x + 740}
          y={PANEL.y + 66}
          w={396}
          h={150}
          style={{
            opacity: Math.min(1, panelIn * 1.3),
            borderRadius: 18,
            padding: "14px 20px",
            boxSizing: "border-box",
            background: interpolateColors(ctxLit, [0, 1], ["rgba(255,255,255,0.04)", "rgba(196,181,253,0.1)"]),
            border: `1.5px solid ${interpolateColors(ctxLit, [0, 1], ["rgba(255,255,255,0.12)", C.violet])}`,
            boxShadow: ctxLit > 0.05 ? `0 0 ${44 * ctxLit}px -8px ${C.violet}` : undefined,
            overflow: "hidden",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, fontFamily: MONO, fontSize: 20, letterSpacing: 3, color: interpolateColors(ctxLit, [0, 1], ["rgba(245,247,251,0.66)", C.violet]) }}>
            <Icon name="text" size={20} />
            CONTEXT CARD
          </div>
          <div style={{ marginTop: 8, fontFamily: SANS, fontSize: 23, lineHeight: 1.28, color: interpolateColors(ctxLit, [0, 1], ["rgba(245,247,251,0.7)", C.text]) }}>{d.rescue.context}</div>
          <LightSweep at={contextAt} dur={30} color="rgba(196,181,253,0.22)" />
        </Abs>
        </div>

        {/* the cloud (it steps back while the camera settles on the finished caption) */}
        <div style={{ position: "absolute", inset: 0, opacity: cloudDim }}>
        <Block
          x={NEMO.x}
          y={NEMO.y}
          w={NEMO.w}
          h={NEMO.h}
          title="Nemotron 3 Super"
          model="nvidia/nemotron-3-super-120b-a12b"
          where="Nebius Token Factory"
          icon="sparkles"
          accent={C.violet}
          nv
          appear={nemoAppear}
          active={Math.max(nemoActive, 0.001)}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 6, fontFamily: MONO, fontSize: 19 }}>
            <span style={{ color: C.dim, opacity: ramp(f, captionAt + 24, captionAt + 34) }}>
              reads: caption{f >= contextAt + 24 ? " + context card" : ""}
            </span>
            <span style={{ color: C.violet, opacity: ramp(f, rescueAt - 14, rescueAt - 4) }}>→ {d.rescue.suggestions.join(" · ")}</span>
          </div>
        </Block>
        <Block
          x={MAGPIE.x}
          y={MAGPIE.y}
          w={MAGPIE.w}
          h={MAGPIE.h}
          title="Magpie TTS"
          model="build.nvidia.com"
          icon="speaker"
          accent={C.violet}
          nv
          where="synthetic voice · only on tap"
          appear={magpieAppear}
          active={magpieActive}
        />
        <Abs
          x={CLOUD_X}
          y={128}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            fontFamily: MONO,
            fontSize: 20,
            letterSpacing: 3,
            color: C.violet,
            // it steps out before the push to the rescue row (it would sit under the NVIDIA tracker)
            // and comes back on the "Magpie" pull-back
            opacity: ramp(f, L1.at, L1.at + 12, easeOut) * Math.min(1, 1 - ramp(f, himselfAt - 10, himselfAt - 2) + ramp(f, magpieAt - 4, magpieAt + 8)),
            transform: `scale(${1 + 0.08 * textPulse})`,
            transformOrigin: "0 50%",
            textShadow: textPulse > 0.05 ? `0 0 ${18 * textPulse}px ${C.violet}` : undefined,
          }}
        >
          <Icon name="text" size={22} />
          TEXT ONLY · NEVER AUDIO
        </Abs>

        </div>

        <Diagram>
          {/* the line between the laptop and the cloud */}
          <line
            x1={BOUNDARY_X}
            x2={BOUNDARY_X}
            y1={118}
            y2={118 + (848 - 118) * boundaryDraw}
            stroke={C.violet}
            strokeOpacity={(0.55 + 0.4 * textPulse) * cloudDim}
            strokeWidth={2 + 1.5 * textPulse}
            strokeDasharray="3 11"
            strokeLinecap="round"
          />
          {/* the paths fade back once what they carried has arrived, so the frame stays clean */}
          <g opacity={(1 - 0.8 * ramp(f, rescueAt + 16, rescueAt + 40)) * cloudDim}>
            <FlowPath d={P_CAPTION} draw={ramp(f, captionAt - 6, captionAt + 10)} color={C.violet} width={2.5} />
            <FlowPath d={P_CONTEXT} draw={ramp(f, contextAt - 6, contextAt + 12)} color={C.violet} width={2.5} />
            <FlowPath d={P_BACK} draw={ramp(f, rescueAt - 26, rescueAt - 8)} color={C.violet} width={2.5} />
          </g>
          <g opacity={1 - ramp(f, completeAt, completeAt + 20)}>
            <FlowPath d={P_MAGPIE} draw={ramp(f, tapAt, tapAt + 16)} color={C.violet} width={2.5} />
            {f >= fromMagpie.t0 - 6 && <FlowPath d={P_MAGPIE_BACK} draw={ramp(f, fromMagpie.t0 - 6, fromMagpie.t0 + 8)} color={C.violet} width={2} dashed />}
          </g>
          <Packet d={P_CAPTION} t0={captionAt + 2} dur={26} label={`…${words.slice(-2).join(" ")}`} />
          <Packet d={P_CONTEXT} t0={contextAt + 4} dur={28} label="context card" />
          {[0, 1, 2].map((i) => (
            <Packet key={i} d={pBack(CHIP_X[i] + CHIP_W[i] / 2)} t0={rescueAt - 22 + i * 4} dur={24} label={chipLabel(i)} absorb={false} />
          ))}
          <Packet d={P_MAGPIE} t0={toMagpie.t0} dur={toMagpie.dur} label={d.say.word} />
          {/* text went out; what comes back from Magpie is the one word, spoken */}
          <Packet d={P_MAGPIE_BACK} t0={fromMagpie.t0} dur={fromMagpie.dur} dot label="the word, as audio" />
        </Diagram>

        {/* the tap */}
        {curIn > 0 && curOut < 1 && (
          <>
            {f >= tapAt && f < tapAt + 22 && (
              <div
                style={{
                  position: "absolute",
                  left: SAY_C.x - 60,
                  top: SAY_C.y - 60,
                  width: 120,
                  height: 120,
                  borderRadius: 60,
                  border: `2px solid ${C.violet}`,
                  opacity: 1 - ramp(f, tapAt, tapAt + 22),
                  transform: `scale(${0.3 + 1.1 * ramp(f, tapAt, tapAt + 22, easeOut)})`,
                }}
              />
            )}
            <svg
              width={40}
              height={44}
              viewBox="0 0 24 26"
              style={{
                position: "absolute",
                left: cur.x,
                top: cur.y,
                opacity: curIn * (1 - curOut),
                transform: `scale(${1 - 0.15 * press})`,
                transformOrigin: "0 0",
                filter: "drop-shadow(0 6px 12px rgba(0,0,0,0.7))",
              }}
            >
              <path d="M2 1 L2 20 L7 15.5 L10.5 23.5 L14 22 L10.6 14.2 L17.5 14.2 Z" fill="#fff" stroke="#0b0f19" strokeWidth={1.4} strokeLinejoin="round" />
            </svg>
          </>
        )}
      </Camera>

      {[0, 1, 2].map((i) => (
        <Sfx key={i} name="pop" at={rescueAt + 8 + i * 5} volume={0.28} />
      ))}
    </AbsoluteFill>
  )
}
