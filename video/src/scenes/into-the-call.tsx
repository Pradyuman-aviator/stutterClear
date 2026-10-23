import type { CSSProperties, ReactNode } from "react"
import { useId } from "react"
import { AbsoluteFill, interpolate, interpolateColors, random, useCurrentFrame } from "remotion"
import { noise2D } from "@remotion/noise"

import type { Cue, SceneProps } from "../data"
import { lineCue } from "../data"
import {
  Avatar,
  Block,
  Camera,
  CallWindow,
  clamp,
  Diagram,
  ease,
  easeIn,
  easeOut,
  Eyebrow,
  FlowPath,
  GlowOrb,
  Icon,
  KineticText,
  LightSweep,
  NvTag,
  Particles,
  Pill,
  ramp,
  Rise,
  Ribbon,
  Sfx,
  SNAPPY,
  springAt,
  Tag,
  Tile,
  envAt,
} from "../kit"
import { C, MONO, SANS, SERIF } from "../theme"

// Out to the call. Act A (stage x 0..1920): the cut logic on one magnified, schematic score lane. A
// detector head writes new audio frames in and scores them; a play head trails it (the delay
// buffer); a cut opens when the score crosses "open" and closes under "close"; close cuts merge; a
// block is reported, never cut; the cut frames fall out of the stream before the play head reaches
// them, and the join is crossfaded and filled with room tone. Act B (stage x 1920..3840): the
// cleaned voice goes into a virtual microphone, out to the call apps and into the call, where NVIDIA
// NeMo captions what the listener hears.

// ---------- Act A geometry (stage px) ----------
const X0 = 300 // plot left
const PW = 1340 // plot width
const LANE_TOP = 250
const LANE_H = 190
const OPEN = 0.62
const CLOSE = 0.36
const STREAM_Y = 652 // centre line of the frame stream
const STREAM_H = 104
const N_TILES = 134 // 20 ms frames, schematic
const MERGE_GAP = 0.05
const CARD_L = 112 // the lane card, inside the safe margin after the push-in
const CARD_W = 1696
const TEXT_A = 120 // Act A eyebrow and headline (x)
const TOP_A = 84 // Act A eyebrow top: >= 60 px clear of the frame top at the push-in's 1.015 zoom, drift included
const TEXT_B = 130 // Act B eyebrow, headline and footnote (the end shot zooms 1.025)

const gauss = (u: number, c: number, w: number, a: number) => a * Math.exp(-(((u - c) / w) ** 2))
// The magnified lane: a schematic stutter score (repetitions), not real values.
const score = (u: number) =>
  0.1 + 0.03 * Math.sin(u * 47) + 0.02 * Math.sin(u * 113 + 1) + gauss(u, 0.255, 0.035, 0.85) + gauss(u, 0.49, 0.02, 0.8) + gauss(u, 0.555, 0.02, 0.8) + gauss(u, 0.885, 0.03, 0.78)
// A block, from its own class: reported, never cut.
const blockScore = (u: number) => 0.06 + gauss(u, 0.71, 0.04, 0.82)

type Seg = { s: number; e: number }
// Hysteresis: open when the score crosses OPEN, close when it falls under CLOSE.
function detect(fn: (u: number) => number): Seg[] {
  const out: Seg[] = []
  const n = 2000
  let open: number | null = null
  for (let i = 0; i <= n; i++) {
    const u = i / n
    const v = fn(u)
    if (open === null && v > OPEN) open = u
    else if (open !== null && v < CLOSE) {
      out.push({ s: open, e: u })
      open = null
    }
  }
  return out
}
const RAW = detect(score) // four raw cuts: A, B, C, D
// close cuts merge: [A], [B+C], [D]
const CUTS: { s: number; e: number; parts: number[] }[] = []
RAW.forEach((r, i) => {
  const last = CUTS[CUTS.length - 1]
  if (last && r.s - last.e < MERGE_GAP) {
    last.e = r.e
    last.parts.push(i)
  } else CUTS.push({ s: r.s, e: r.e, parts: [i] })
})
const MERGED = CUTS.findIndex((c) => c.parts.length > 1)
const BLOCK = detect(blockScore)[0] ?? { s: 0.68, e: 0.74 }
const CUT_LEN = CUTS.map((c) => c.e - c.s)
const H_END = 1 - CUT_LEN.reduce((a, b) => a + b, 0) // the heard length, after the cuts
const H_START = CUTS.map((c, i) => c.s - CUT_LEN.slice(0, i).reduce((a, b) => a + b, 0))

// Tile loudness: a schematic voice envelope (syllables), quieter through the block.
const TILES = Array.from({ length: N_TILES }, (_, i) => {
  const u = (i + 0.5) / N_TILES
  const syl = 0.5 + 0.5 * Math.sin(u * 62 + 1.4 * Math.sin(u * 11))
  let a = 0.28 + 0.72 * syl * (0.65 + 0.35 * random(`tile${i}`))
  if (u > BLOCK.s && u < BLOCK.e) a = 0.12 + 0.06 * random(`blk${i}`)
  return { u, a }
})

const lanePath = (fn: (u: number) => number, from = 0, to = 1, closeBase = false) => {
  const n = 240
  let d = ""
  for (let i = 0; i <= n; i++) {
    const u = from + ((to - from) * i) / n
    const x = X0 + u * PW
    const y = LANE_TOP + LANE_H * (1 - Math.min(1, fn(u)))
    d += `${i ? "L" : "M"} ${x.toFixed(1)} ${y.toFixed(1)} `
  }
  if (closeBase) d += `L ${(X0 + to * PW).toFixed(1)} ${LANE_TOP + LANE_H} L ${(X0 + from * PW).toFixed(1)} ${LANE_TOP + LANE_H} Z`
  return d
}
const SCORE_D = lanePath(score)
const BLOCK_D = lanePath(blockScore, BLOCK.s - 0.08, BLOCK.e + 0.08)
const BLOCK_FILL = lanePath(blockScore, BLOCK.s - 0.08, BLOCK.e + 0.08, true)
const CUT_FILLS = RAW.map((r) => lanePath(score, r.s, r.e, true))
const yOf = (v: number) => LANE_TOP + LANE_H * (1 - v)

// ---------- Act B (local px inside the right half of the stage) ----------
const ROW_Y = 560
const MIC = { x: 290, y: ROW_Y - 95, w: 360, h: 190 }
const APPS = ["Zoom", "Google Meet", "Microsoft Teams"]
const APP_X = 730
const APP_W = 236
const APP_Y = [ROW_Y - 140, ROW_Y, ROW_Y + 140]
const CALL = { x: 1010, y: 214, w: 780, h: 580 }
const CHUNK = 14.4 // frames in a 480 ms NeMo chunk
const JUDGES = ["MK", "RS", "DT"]
const RIB_X = X0 + H_END * PW + 10 // the cleaned stream ends here; the ribbon carries it on to the mic
const RIB_W = 1920 + MIC.x - RIB_X
const RIB_ROT = (Math.atan2(ROW_Y - STREAM_Y, 1920 + MIC.x - RIB_X) * 180) / Math.PI

// The frame where a phrase of a narration line is spoken (by its share of the line's characters).
const phraseAt = (c: Cue, phrase: string, fallback: number) => {
  const t = c.text ?? ""
  const i = t.toLowerCase().indexOf(phrase.toLowerCase())
  const u = i >= 0 && t.length ? i / t.length : fallback
  return Math.round(c.at + u * c.frames)
}

function Abs({ x, y, children, style }: { x: number; y: number; children: ReactNode; style?: CSSProperties }) {
  return <div style={{ position: "absolute", left: x, top: y, ...style }}>{children}</div>
}

export default function Scene({ d, scene }: SceneProps) {
  const uid = `itc${useId().replace(/[^a-zA-Z0-9]/g, "")}`
  const f = useCurrentFrame()
  const L0 = lineCue(scene, 0)
  const L1 = lineCue(scene, 1)
  const a0 = (p: number) => L0.at + p * L0.frames
  const a1 = (p: number) => L1.at + p * L1.frames

  // ---------- Act A timing ----------
  // The skip lands on the word "skipped": the merged (double) cut lifts and falls away there.
  const skippedAt = phraseAt(L0, "skipped", 0.71)
  const scanA = a0(0.04)
  const scanB = a0(0.8)
  const scanU = interpolate(f, [scanA, scanB], [0, 1], clamp)
  const scanAt = (u: number) => scanA + u * (scanB - scanA)
  const playB = a0(0.97)
  // the play head's start, solved so the merged cut's fall (cutAt - 12 .. cutAt - 1) brackets the word
  const rM = MERGED >= 0 ? H_START[MERGED] / H_END : 0.5
  const playA = Math.min(a0(0.6), Math.max(a0(0.4), (skippedAt + 8 - rM * playB) / (1 - rM)))
  const h = interpolate(f, [playA, playB], [0, H_END], clamp)
  const cutAt = H_START.map((hs) => playA + (hs / H_END) * (playB - playA))
  const lift = cutAt.map((t) => ramp(f, t - 22, t - 12, easeOut))
  const fall = cutAt.map((t) => ramp(f, t - 12, t - 1, ease))
  const merge = MERGED >= 0 ? ramp(f, scanAt(CUTS[MERGED].e) + 4, scanAt(CUTS[MERGED].e) + 16, ease) : 0
  const blockOn = springAt(f, scanAt(BLOCK.s) + 2, SNAPPY)

  // x of an original position u in the stream, with the cuts that have collapsed taken out
  const removedBefore = (u: number, skip = -1) =>
    CUTS.reduce((acc, c, i) => (i === skip ? acc : acc + fall[i] * Math.min(c.e - c.s, Math.max(0, u - c.s))), 0)
  const sx = (u: number, skip = -1) => X0 + (u - removedBefore(u, skip)) * PW
  const playX = X0 + h * PW
  const scanX = X0 + scanU * PW

  // which cut covers u (bracketed so far)
  const bracketOf = (u: number) => {
    for (let i = 0; i < CUTS.length; i++) {
      const c = CUTS[i]
      if (u < c.s || u > c.e) continue
      if (c.parts.length > 1) {
        const inPart = c.parts.some((p) => u >= RAW[p].s && u <= RAW[p].e)
        if (!inPart && merge < 0.5) return -1
      }
      if (u <= scanU) return i
    }
    return -1
  }

  // ---------- Camera ----------
  const pan = L1.at - 16
  const shots = [
    { at: 0, x: 960, y: 540, zoom: 0.985 },
    { at: 1, x: 960, y: 548, zoom: 1.015, dur: Math.max(60, pan - 1) },
    { at: pan, x: 2880, y: 540, zoom: 1, dur: 36 },
    { at: pan + 36, x: 2880, y: 528, zoom: 1.025, dur: Math.max(60, scene.frames - pan - 36) },
  ]

  // ---------- Act B timing ----------
  const micIn = springAt(f, a1(0.03), SNAPPY)
  const ribbon = ramp(f, pan - 6, pan + 30, ease)
  const fanDraw = ramp(f, a1(0.16), a1(0.32), easeOut)
  const appIn = APPS.map((_, i) => springAt(f, a1(0.2) + i * 4, SNAPPY))
  const callGhost = ramp(f, a1(0.08), a1(0.08) + 14, easeOut) // the call waits, faint, while the mic fans out
  const callIn = springAt(f, a1(0.34))
  const fanIn = ramp(f, a1(0.36), a1(0.46), easeOut)
  const speaking = ramp(f, a1(0.48), a1(0.52))
  const nemoIn = springAt(f, a1(0.54), SNAPPY)
  const capAt = a1(0.6)
  const capWords = d.rescue.said.split(" ")
  const capEnd = capWords.findIndex((w) => /[,.]$/.test(w))
  const capMax = capEnd >= 0 ? capEnd + 1 : Math.min(capWords.length, 11)
  const bursts = f < capAt ? 0 : Math.floor((f - capAt) / CHUNK) + 1
  const shown = Math.min(capMax, bursts * 2)
  const voiceT = Math.max(0, (f - a1(0.45)) / 30) % d.clip.after_s
  const level = speaking * envAt(d.clip.env_after, voiceT, voiceT + 0.12)

  const ribEdge = Math.min(1, ribbon * 1.06) * 100
  const ribMask = `linear-gradient(to right, black ${Math.max(0, ribEdge - 7).toFixed(1)}%, transparent ${ribEdge.toFixed(1)}%)`

  const headA2 = skippedAt
  const headA1Out = headA2 - 12 // the old headline is gone (10-frame exit) before the new one starts

  return (
    <AbsoluteFill>
      <Camera shots={shots} drift={5} seed="itc">
        {/* ================= ACT A: the cut logic ================= */}
        <GlowOrb x={900} y={520} r={760} color="rgba(125,211,252,0.10)" opacity={1} />
        <GlowOrb x={sx(CUTS[0]?.s ?? 0.25)} y={STREAM_Y} r={260} color="rgba(251,113,133,0.16)" opacity={lift[0] * (1 - 0.6 * fall[0])} />
        <Particles count={12} color={C.sky} seed="itcA" x={96} y={200} w={1728} h={640} opacity={0.35} rise={0.3} />

        <Abs x={TEXT_A} y={TOP_A}>
          <Eyebrow>Cut logic · live · schematic</Eyebrow>
        </Abs>
        <div style={{ position: "absolute", left: TEXT_A, top: TOP_A + 36, width: 1200, height: 80 }}>
          <Headline at={L0.at + 2} out={headA1Out}>
            <KineticText text="The score crosses. A cut opens." at={L0.at + 2} per={3} highlight={{ cut: C.rose }} style={{ fontFamily: SERIF, fontSize: 68, letterSpacing: -1, lineHeight: 1.05, color: C.text }} />
          </Headline>
          <Headline at={headA2}>
            <KineticText text="Skipped before anyone hears it." at={headA2} per={3} highlight={{ skipped: C.rose }} style={{ fontFamily: SERIF, fontSize: 68, letterSpacing: -1, lineHeight: 1.05, color: C.text }} />
          </Headline>
        </div>

        {/* the lane card */}
        <LaneCard f={f} />
        <Abs x={136} y={LANE_TOP + 38} style={{ width: 160, opacity: springAt(f, 6) }}>
          <div style={{ fontFamily: SANS, fontSize: 26, fontWeight: 600, color: C.text, lineHeight: 1.15 }}>Stutter score</div>
          <div style={{ fontFamily: MONO, fontSize: 20, color: C.dim, marginTop: 8 }}>
            repeats<br />
            <span style={{ color: C.amber }}>+ block</span>
          </div>
          <Tag style={{ marginTop: 12, display: "inline-block", fontSize: 18 }}>schematic</Tag>
        </Abs>
        <Abs x={136} y={STREAM_Y - 46} style={{ width: 160, opacity: springAt(f, 10) }}>
          <div style={{ fontFamily: SANS, fontSize: 26, fontWeight: 600, color: C.sky, lineHeight: 1.15 }}>His voice</div>
          <div style={{ fontFamily: MONO, fontSize: 20, color: C.dim, marginTop: 8 }}>audio frames</div>
        </Abs>

        <svg width={1920} height={1080} style={{ position: "absolute", left: 0, top: 0, overflow: "visible" }}>
          <defs>
            <clipPath id={`${uid}s`}>
              <rect x={X0 - 2} y={LANE_TOP - 20} width={Math.max(0, scanX - X0 + 2)} height={LANE_H + 40} />
            </clipPath>
            <linearGradient id={`${uid}c`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={C.rose} stopOpacity={0.55} />
              <stop offset="100%" stopColor={C.rose} stopOpacity={0.04} />
            </linearGradient>
            <linearGradient id={`${uid}b`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={C.amber} stopOpacity={0.4} />
              <stop offset="100%" stopColor={C.amber} stopOpacity={0.03} />
            </linearGradient>
          </defs>

          {/* thresholds */}
          {[
            { v: OPEN, label: "open", color: C.rose, at: 8 },
            { v: CLOSE, label: "close", color: C.dim, at: 12 },
          ].map((t) => {
            const p = ramp(f, t.at, t.at + 18, easeOut)
            return (
              <g key={t.label} opacity={p}>
                <line x1={X0} x2={X0 + PW * p} y1={yOf(t.v)} y2={yOf(t.v)} stroke={t.color} strokeOpacity={0.8} strokeWidth={2} strokeDasharray="8 8" />
                <text x={X0 + PW + 16} y={yOf(t.v) + 7} fontFamily={MONO} fontSize={21} fontWeight={700} fill={t.color}>
                  {t.label}
                </text>
              </g>
            )
          })}

          {/* the scored curve, written in as the detector reaches it */}
          <g clipPath={`url(#${uid}s)`}>
            {CUT_FILLS.map((p, i) => (
              <path key={i} d={p} fill={`url(#${uid}c)`} />
            ))}
            <path d={BLOCK_FILL} fill={`url(#${uid}b)`} />
            <path d={BLOCK_D} fill="none" stroke={C.amber} strokeWidth={2.4} style={{ filter: `drop-shadow(0 0 6px ${C.amber})` }} />
            <path d={SCORE_D} fill="none" stroke={C.text} strokeOpacity={0.9} strokeWidth={2.6} style={{ filter: `drop-shadow(0 0 7px ${C.sky})` }} />
          </g>
          {/* open / close markers on the lane, as each happens */}
          {RAW.map((r, i) => {
            const po = springAt(f, scanAt(r.s), SNAPPY)
            const pc = springAt(f, scanAt(r.e), SNAPPY)
            return (
              <g key={i}>
                {f >= scanAt(r.s) && <circle cx={X0 + r.s * PW} cy={yOf(OPEN)} r={7 * Math.min(1.2, po)} fill={C.rose} style={{ filter: `drop-shadow(0 0 8px ${C.rose})` }} />}
                {f >= scanAt(r.s) && f < scanAt(r.s) + 14 && (
                  <circle cx={X0 + r.s * PW} cy={yOf(OPEN)} r={7 + 30 * ramp(f, scanAt(r.s), scanAt(r.s) + 14)} fill="none" stroke={C.rose} strokeWidth={2} opacity={1 - ramp(f, scanAt(r.s), scanAt(r.s) + 14)} />
                )}
                {f >= scanAt(r.e) && <circle cx={X0 + r.e * PW} cy={yOf(CLOSE)} r={6 * Math.min(1.2, pc)} fill="none" stroke={C.text} strokeWidth={2.4} />}
              </g>
            )
          })}
          {f >= scanAt(BLOCK.s) && (
            <text x={X0 + ((BLOCK.s + BLOCK.e) / 2) * PW} y={yOf(0.88) - 8} textAnchor="middle" fontFamily={MONO} fontSize={20} fontWeight={700} fill={C.amber} opacity={Math.min(1, blockOn)}>
              block
            </text>
          )}

          {/* detector head */}
          {f > scanA - 6 && scanU < 1 && (
            <g opacity={ramp(f, scanA - 6, scanA + 4) * (1 - ramp(f, scanB - 6, scanB + 6))}>
              <line x1={scanX} x2={scanX} y1={LANE_TOP - 6} y2={STREAM_Y + STREAM_H / 2 + 8} stroke={C.sky} strokeWidth={2.5} style={{ filter: `drop-shadow(0 0 10px ${C.sky})` }} />
              <circle cx={scanX} cy={yOf(Math.min(1, score(scanU)))} r={6} fill={C.sky} style={{ filter: `drop-shadow(0 0 8px ${C.sky})` }} />
            </g>
          )}

          {/* brackets over the stream: a cut opens (rose) and stays open until the score closes it */}
          {RAW.map((r, i) => {
            if (f < scanAt(r.s)) return null
            const ci = CUTS.findIndex((c) => c.parts.includes(i))
            const c = CUTS[ci]
            let s = r.s
            let e = Math.min(r.e, scanU)
            if (c.parts.length > 1) {
              const first = c.parts[0] === i
              if (first) e = e + merge * (RAW[c.parts[1]].s - r.e) * (scanU >= r.e ? 1 : 0)
              else s = s - merge * (r.s - RAW[c.parts[0]].e)
            }
            const o = (1 - fall[ci]) * Math.min(1, springAt(f, scanAt(r.s), SNAPPY) * 1.4)
            const x1 = sx(s)
            const x2 = Math.max(x1 + 2, sx(e))
            const y = STREAM_Y - STREAM_H / 2 - 18 - lift[ci] * 22 + fall[ci] * 40
            return (
              <path
                key={i}
                d={`M ${x1} ${y + 14} V ${y} H ${x2} V ${y + 14}`}
                fill="none"
                stroke={C.rose}
                strokeWidth={3.2}
                strokeLinejoin="round"
                opacity={o}
                style={{ filter: `drop-shadow(0 0 8px ${C.rose})` }}
              />
            )
          })}

          {/* the block: marked, not cut */}
          {f >= scanAt(BLOCK.s) && (
            <rect
              x={sx(BLOCK.s) - 6}
              y={STREAM_Y - STREAM_H / 2 - 10}
              width={(BLOCK.e - BLOCK.s) * PW + 12}
              height={STREAM_H + 20}
              rx={12}
              fill="rgba(252,211,77,0.06)"
              stroke={C.amber}
              strokeWidth={2}
              strokeDasharray="6 7"
              opacity={Math.min(1, blockOn)}
            />
          )}

          {/* the frame stream */}
          {TILES.map((t, i) => {
            if (t.u > scanU) return null
            const ci = CUTS.findIndex((c) => t.u >= c.s && t.u <= c.e)
            const inBracket = bracketOf(t.u) >= 0
            const born = Math.min(1, (scanU - t.u) * 60)
            const slot = (PW / N_TILES) * 0.66
            const hh = Math.max(6, t.a * STREAM_H) * (0.4 + 0.6 * born)
            if (ci >= 0 && fall[ci] > 0) {
              // lifts out and falls away before the play head reaches it
              const x = sx(t.u, ci)
              const fl = fall[ci]
              const rot = (random(`rot${i}`) - 0.5) * 70 * fl
              const dy = -lift[ci] * 22 + fl * fl * 150
              return (
                <rect
                  key={i}
                  x={x - slot / 2}
                  y={STREAM_Y - hh / 2 + dy}
                  width={slot}
                  height={hh}
                  rx={3}
                  fill={C.rose}
                  opacity={(1 - fl) * 0.95}
                  transform={`rotate(${rot} ${x} ${STREAM_Y + dy})`}
                />
              )
            }
            const x = sx(t.u)
            const played = x < playX
            const lifted = ci >= 0 ? lift[ci] : 0
            return (
              <rect
                key={i}
                x={x - slot / 2}
                y={STREAM_Y - hh / 2 - lifted * 22}
                width={slot}
                height={hh}
                rx={3}
                fill={inBracket ? C.rose : played ? C.teal : C.sky}
                opacity={inBracket ? 0.95 : played ? 1 : 0.5}
                style={played && !inBracket ? { filter: `drop-shadow(0 0 4px ${C.teal})` } : undefined}
              />
            )
          })}

          {/* joins: a crossfade, filled with the room's own background sound */}
          {CUTS.map((c, i) => {
            if (fall[i] < 0.6) return null
            const x = sx(c.s)
            const o = ramp(f, cutAt[i] - 4, cutAt[i] + 6)
            return (
              <g key={i} opacity={o}>
                <line x1={x - 13} x2={x + 13} y1={STREAM_Y + 30} y2={STREAM_Y - 30} stroke={C.teal} strokeWidth={2} strokeOpacity={0.9} />
                <line x1={x - 13} x2={x + 13} y1={STREAM_Y - 30} y2={STREAM_Y + 30} stroke={C.sky} strokeWidth={2} strokeOpacity={0.9} />
                {Array.from({ length: 9 }, (_, k) => {
                  const yy = STREAM_Y - 44 + k * 11
                  const fl = 0.5 + 0.5 * noise2D(`rt${i}`, k * 0.9, f / 7)
                  return <circle key={k} cx={x + noise2D(`rx${i}`, k, f / 11) * 9} cy={yy} r={1.6 + fl * 1.4} fill="#a5c4cf" opacity={0.25 + 0.6 * fl} />
                })}
              </g>
            )
          })}

          {/* play head: what the listener hears, half a second behind */}
          {f > playA - 14 && (
            <g opacity={ramp(f, playA - 14, playA) * (1 - ramp(f, pan + 10, pan + 24))}>
              <line x1={playX} x2={playX} y1={STREAM_Y - STREAM_H / 2 - 30} y2={STREAM_Y + STREAM_H / 2 + 22} stroke={C.teal} strokeWidth={3} style={{ filter: `drop-shadow(0 0 10px ${C.teal})` }} />
              <circle cx={playX} cy={STREAM_Y + STREAM_H / 2 + 22} r={6} fill={C.teal} />
            </g>
          )}
        </svg>

        {/* labels pinned to the stream */}
        {MERGED >= 0 && (
          <PinLabel x={sx((CUTS[MERGED].s + CUTS[MERGED].e) / 2)} y={STREAM_Y - STREAM_H / 2 - 84} color={C.rose} p={merge * (1 - fall[MERGED])}>
            close cuts merge
          </PinLabel>
        )}
        <PinLabel x={sx((BLOCK.s + BLOCK.e) / 2)} y={STREAM_Y + STREAM_H / 2 + 32} color={C.amber} p={Math.min(1, blockOn) * (1 - ramp(f, pan, pan + 14))}>
          block: reported, never cut
        </PinLabel>
        <PinLabel x={scanX} y={LANE_TOP - 36} color={C.sky} p={ramp(f, scanA, scanA + 10) * (1 - ramp(f, a0(0.36), a0(0.44)))}>
          detector · newest audio
        </PinLabel>
        <PinLabel x={playX} y={STREAM_Y + STREAM_H / 2 + 40} color={C.teal} p={ramp(f, playA - 10, playA + 4) * (1 - ramp(f, playA + 18, playA + 28))}>
          playback · ≈0.5 s behind
        </PinLabel>

        {/* what happens at a cut, as it happens */}
        <Abs x={X0} y={826} style={{ display: "flex", gap: 18 }}>
          <Chip at={skippedAt - 2} out={pan + 6} color={C.rose} icon="scissors">
            skipped before playback
          </Chip>
          <Chip at={skippedAt + 6} out={pan + 6} color={C.teal} glyph="x">
            crossfade
          </Chip>
          <Chip at={skippedAt + 14} out={pan + 6} color="#a5c4cf" icon="wave">
            room tone, never silence
          </Chip>
        </Abs>
        {cutAt.map((t, i) => (
          <Sfx key={i} name="snap" at={t - 12} volume={0.42} />
        ))}

        {/* ================= the cleaned voice, on to the call ================= */}
        <div style={{ position: "absolute", left: RIB_X, top: STREAM_Y - 70, width: RIB_W, height: 140, transform: `rotate(${RIB_ROT}deg)`, transformOrigin: "0 50%" }}>
          {/* a soft leading edge with a glowing head, so the voice travels instead of ending in a hard cut */}
          <div style={{ position: "absolute", inset: 0, ...(ribbon < 1 ? { WebkitMaskImage: ribMask, maskImage: ribMask } : {}) }}>
            <Ribbon env={d.clip.env_after} seconds={d.clip.after_s} width={RIB_W} height={140} color={C.teal} color2={C.sky} calm={0.7} progress={Math.min(1, ribbon * 1.06)} speed={1.2} />
          </div>
          {ribbon > 0 && ribbon < 1 && (
            <div
              style={{
                position: "absolute",
                left: ribbon * RIB_W - 34,
                top: 70 - 34,
                width: 68,
                height: 68,
                borderRadius: 34,
                background: "radial-gradient(circle, rgba(204,251,241,0.95) 0%, rgba(94,234,212,0.55) 18%, rgba(94,234,212,0.12) 48%, rgba(94,234,212,0) 70%)",
                opacity: ramp(ribbon, 0, 0.06) * (1 - ramp(ribbon, 0.9, 1)),
              }}
            />
          )}
        </div>

        {/* ================= ACT B ================= */}
        <div style={{ position: "absolute", left: 1920, top: 0, width: 1920, height: 1080 }}>
          <GlowOrb x={CALL.x + CALL.w / 2} y={CALL.y + CALL.h / 2} r={620} color="rgba(94,234,212,0.13)" opacity={Math.min(1, callIn)} />
          <Particles count={12} color={C.teal} seed="itcB" x={200} y={260} w={1600} h={600} opacity={0.35} rise={0.3} />

          <Abs x={TEXT_B} y={70}>
            <Rise at={a1(0.0)}>
              <Eyebrow color={C.teal}>Out to the call</Eyebrow>
            </Rise>
          </Abs>
          <Abs x={TEXT_B} y={106} style={{ width: 860 }}>
            <KineticText text="His voice, into the call." at={a1(0.02)} per={3} highlight={{ call: C.teal }} style={{ fontFamily: SERIF, fontSize: 68, letterSpacing: -1, lineHeight: 1.05, color: C.text }} />
          </Abs>

          {/* the mic is the focal point while it is alone */}
          <GlowOrb x={MIC.x + MIC.w / 2} y={ROW_Y} r={300} color="rgba(94,234,212,0.14)" opacity={Math.min(1, micIn) * (0.7 + 0.3 * ramp(f, a1(0.1), a1(0.18))) * (1 - 0.45 * Math.min(1, callIn))} />

          <Diagram>
            {APPS.map((_, i) => (
              <FlowPath
                key={`o${i}`}
                d={`M ${MIC.x + MIC.w} ${ROW_Y} C ${MIC.x + MIC.w + 50} ${ROW_Y}, ${APP_X - 50} ${APP_Y[i]}, ${APP_X} ${APP_Y[i]}`}
                draw={fanDraw}
                flow={fanDraw}
                color={C.teal}
                kind="frame"
                packets={2}
                period={44}
              />
            ))}
            {APPS.map((_, i) => (
              <FlowPath
                key={`i${i}`}
                d={`M ${APP_X + APP_W} ${APP_Y[i]} C ${APP_X + APP_W + 30} ${APP_Y[i]}, ${CALL.x - 30} ${ROW_Y}, ${CALL.x} ${ROW_Y}`}
                draw={fanIn}
                flow={fanIn}
                color={C.teal}
                packets={2}
                period={40}
              />
            ))}
          </Diagram>

          <Block
            x={MIC.x}
            y={MIC.y}
            w={MIC.w}
            h={MIC.h}
            title="Virtual microphone"
            model="CABLE Input → CABLE Output"
            icon="mic"
            accent={C.teal}
            appear={Math.min(1, micIn)}
            active={ramp(f, a1(0.1), a1(0.18))}
          >
            <div style={{ marginTop: "auto" }}>
              <Tag style={{ fontSize: 18 }}>e.g. VB-CABLE</Tag>
            </div>
          </Block>

          {APPS.map((a, i) => (
            <Abs key={a} x={APP_X} y={APP_Y[i] - 28} style={{ opacity: Math.min(1, appIn[i] * 1.5), transform: `translateX(${(1 - appIn[i]) * -30}px) scale(${0.85 + 0.15 * appIn[i]})` }}>
              <Pill color={C.text} border={C.border2} style={{ width: APP_W, boxSizing: "border-box", justifyContent: "center", fontSize: 24, fontWeight: 600, padding: "11px 0", background: "rgba(10,16,28,0.85)" }}>
                {a}
              </Pill>
            </Abs>
          ))}
          <Abs x={APP_X} y={APP_Y[2] + 40} style={{ width: APP_W, textAlign: "center", fontFamily: MONO, fontSize: 19, color: C.dim, opacity: ramp(f, a1(0.3), a1(0.4)) }}>
            mic: CABLE Output
          </Abs>

          {/* the call */}
          <Abs
            x={CALL.x}
            y={CALL.y}
            style={{
              opacity: Math.max(0.25 * callGhost, Math.min(1, callIn * 1.4)),
              transform: `translateX(${(1 - callIn) * 70}px)`,
              filter: callGhost > 0 && callIn < 0.98 ? `blur(${(12 * (1 - Math.min(1, callIn))).toFixed(1)}px)` : undefined,
            }}
          >
            <CallWindow title="Demo day · video call" width={CALL.w} height={CALL.h}>
              <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 14 }}>
                <Tile name="Alex" speaking={speaking} style={{ flex: "none", height: 262 }}>
                  <Avatar size={128} level={level} />
                </Tile>
                <div
                  style={{
                    height: 118,
                    borderRadius: 14,
                    background: "rgba(3,5,10,0.78)",
                    border: `1px solid ${C.border}`,
                    padding: "10px 18px",
                    boxSizing: "border-box",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "flex-end",
                    overflow: "hidden",
                    position: "relative",
                  }}
                >
                  <div style={{ position: "absolute", left: 18, top: 8, display: "flex", alignItems: "center", gap: 8, fontFamily: MONO, fontSize: 18, color: C.dim, letterSpacing: 1 }}>
                    <Icon name="captions" size={20} /> LIVE CAPTIONS
                  </div>
                  <div style={{ fontFamily: SANS, fontSize: 26, fontWeight: 500, lineHeight: 1.3, color: C.text, maxHeight: 34 * 2, overflow: "hidden", display: "flex", flexWrap: "wrap", alignContent: "flex-end" }}>
                    {capWords.slice(0, shown).map((w, i) => {
                      // a NeMo chunk lands whole: no invisible placeholders, so the caret sits on the text
                      const born = capAt + Math.floor(i / 2) * CHUNK
                      return (
                        <span key={i} style={{ marginRight: 7, color: interpolateColors(f, [born, born + 10], [C.teal, C.text]) }}>
                          {w}
                        </span>
                      )
                    })}
                    {f >= capAt && <span style={{ color: C.teal, opacity: Math.floor(f / 12) % 2 ? 0.2 : 1 }}>▍</span>}
                  </div>
                </div>
                <div style={{ opacity: Math.min(1, nemoIn * 1.4), transform: `translateY(${(1 - nemoIn) * 12}px)`, display: "flex", flexDirection: "column", gap: 6 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <NvTag style={{ fontSize: 18, boxShadow: `0 0 ${20 * (1 - ramp(f, a1(0.6), a1(0.9)))}px ${C.nv}` }} />
                    <span style={{ fontFamily: MONO, fontSize: 20, color: C.nv, fontWeight: 700 }}>NeMo streaming FastConformer</span>
                  </div>
                  <span style={{ fontFamily: MONO, fontSize: 20, color: C.dim }}>480 ms chunks · int8 ONNX · CPU</span>
                </div>
              </div>
              <div style={{ width: 208, display: "flex", flexDirection: "column", gap: 14 }}>
                {JUDGES.map((j) => (
                  <Tile key={j} name="Judge">
                    <div style={{ width: 70, height: 70, borderRadius: 35, background: "rgba(255,255,255,0.06)", border: `1.5px solid ${C.border2}`, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: SERIF, fontSize: 34, color: C.dim, marginTop: -18 }}>
                      {j}
                    </div>
                    <div style={{ position: "absolute", right: 14, bottom: 12, color: C.teal, opacity: speaking * (0.55 + 0.45 * level) }}>
                      <Icon name="headphones" size={22} />
                    </div>
                  </Tile>
                ))}
              </div>
            </CallWindow>
            <div style={{ position: "absolute", inset: 0, borderRadius: 28, overflow: "hidden", pointerEvents: "none" }}>
              <LightSweep at={a1(0.4)} dur={36} color="rgba(255,255,255,0.12)" />
            </div>
          </Abs>

          {/* privacy footnote */}
          <Abs x={TEXT_B} y={800} style={{ width: 860, display: "flex", gap: 14, alignItems: "flex-start" }}>
            <Rise at={a1(0.66)} y={16} style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
              <span style={{ color: C.teal, marginTop: 3 }}>
                <Icon name="lock" size={26} />
              </span>
              <span style={{ fontFamily: SANS, fontSize: 24, lineHeight: 1.4, color: C.dim }}>
                Audio stays on this laptop.
                <br />
                Nothing is recorded unless you switch saving on.
              </span>
            </Rise>
          </Abs>
        </div>
      </Camera>
    </AbsoluteFill>
  )
}

// The glass card behind the magnified lane and the stream; it opens from a thin strip.
function LaneCard({ f }: { f: number }) {
  const p = springAt(f, 0)
  return (
    <div
      style={{
        position: "absolute",
        left: CARD_L,
        top: LANE_TOP - 26,
        width: CARD_W,
        height: STREAM_Y + STREAM_H / 2 + 102 - (LANE_TOP - 26),
        borderRadius: 28,
        background: "linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.018))",
        border: `1px solid ${C.border}`,
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.07), 0 40px 90px -40px rgba(0,0,0,0.85)",
        opacity: Math.min(1, p * 1.3),
        transform: `scaleY(${0.4 + 0.6 * p})`,
        transformOrigin: "50% 40%",
      }}
    >
      {/* divider between the lane and the stream */}
      <div style={{ position: "absolute", left: 28, right: 28, top: LANE_H + 52, height: 1, background: C.border }} />
      {/* plot backing */}
      <div style={{ position: "absolute", left: X0 - CARD_L, top: 26, width: PW, height: LANE_H, borderRadius: 10, background: "rgba(255,255,255,0.03)" }} />
    </div>
  )
}

function Headline({ at, out, children }: { at: number; out?: number; children: ReactNode }) {
  const f = useCurrentFrame()
  if (f < at - 1) return null
  const q = out === undefined ? 0 : ramp(f, out, out + 10, easeIn)
  if (q >= 1) return null
  return <div style={{ position: "absolute", left: 0, top: 0, opacity: 1 - q, transform: `translateY(${-q * 18}px)`, filter: q ? `blur(${q * 6}px)` : undefined }}>{children}</div>
}

// A small label centred on x, pinned to something on the stream.
function PinLabel({ x, y, color, p, children }: { x: number; y: number; color: string; p: number; children: ReactNode }) {
  if (p <= 0.01) return null
  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        transform: `translate(-50%, ${(1 - Math.min(1, p)) * 10}px)`,
        opacity: Math.min(1, p),
        fontFamily: MONO,
        fontSize: 21,
        fontWeight: 700,
        color,
        whiteSpace: "nowrap",
        padding: "3px 10px",
        borderRadius: 8,
        background: "rgba(5,7,13,0.72)",
      }}
    >
      {children}
    </div>
  )
}

function Chip({ at, out, color, icon, glyph, children }: { at: number; out: number; color: string; icon?: "scissors" | "wave"; glyph?: "x"; children: ReactNode }) {
  const f = useCurrentFrame()
  const p = springAt(f, at, SNAPPY)
  const q = ramp(f, out, out + 12, easeIn)
  if (f < at) return null
  return (
    <div style={{ opacity: Math.min(1, p * 1.5) * (1 - q), transform: `translateY(${(1 - p) * 16 - q * 10}px) scale(${0.9 + 0.1 * Math.min(1, p)})` }}>
      <Pill color={color} border={`${color}66`} style={{ fontFamily: MONO, fontSize: 22, fontWeight: 700, padding: "10px 20px", background: "rgba(8,12,22,0.85)", boxShadow: `0 0 26px -8px ${color}` }}>
        {icon && <Icon name={icon} size={22} />}
        {glyph && (
          <svg width={22} height={22} viewBox="0 0 22 22">
            <line x1={2} y1={18} x2={20} y2={4} stroke={C.teal} strokeWidth={2.2} />
            <line x1={2} y1={4} x2={20} y2={18} stroke={C.sky} strokeWidth={2.2} />
          </svg>
        )}
        {children}
      </Pill>
    </div>
  )
}
