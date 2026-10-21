import type { CSSProperties, ReactNode } from "react"
import { getLength, getPointAtLength } from "@remotion/paths"
import { AbsoluteFill, interpolate, interpolateColors, random, useCurrentFrame } from "remotion"

import type { SceneProps } from "../data"
import { lineCue } from "../data"
import {
  BOUNCY,
  Camera,
  Counter,
  Diagram,
  easeIn,
  easeOut,
  FlowPath,
  GlowOrb,
  Icon,
  KineticText,
  LightSweep,
  NvTag,
  Particles,
  ramp,
  Sfx,
  SNAPPY,
  springAt,
  Tag,
} from "../kit"
import type { IconName } from "../kit"
import { C, MONO, SANS, SERIF } from "../theme"

// Some repeats are meant: the meaning guard. A typed test sentence with three marked repeats goes,
// as text only, to NVIDIA Nemotron 3 Super on Nebius; its real reply keeps "that that" (grammar) and
// "very very" (emphasis) and lets the extra "I" go.

// Layout (stage px)
const LX = 120 // left column: the laptop
const LW = 960
const DIV = 1160 // the divider
const RX = 1224 // right column: Nebius Token Factory (right edge 1808, inside the 96 px margin)
const RW = 584
const TOP = 124 // the column captions
const SENT_TOP = 176
const FS = 44 // sentence size
const ROW_GAP = 60 // room for the "cut" mark and the reasons between rows
const REQ_TOP = 606
const REQ_W = 720
const REP_TOP = 744
const CARD_TOP = 168
const CARD_H = 516
const TAKE_TOP = 716
const GATE_Y = 524
const DIV_TOP = TOP - 10
const DIV_BOT = 860

// Packets: the marked request goes out into the card's empty band (between the spec lines and the
// status rule); the keep list comes back out of the card's bottom edge (text both ways).
const OUT_PATH = `M ${LX + REQ_W + 12} ${REQ_TOP + 62} C ${LX + REQ_W + 170} ${REQ_TOP + 62}, ${DIV - 40} ${CARD_TOP + 424}, ${RX - 6} ${CARD_TOP + 416}`
const BACK_PATH = `M ${RX + 44} ${CARD_TOP + CARD_H + 4} C ${RX + 20} ${CARD_TOP + CARD_H + 80}, ${DIV - 10} ${REP_TOP + 64}, ${LX + LW + 10} ${REP_TOP + 64}`

// Where a path crosses the divider, so the divider lights just there while packets pass.
function crossY(d: string, x: number, fallback: number): number {
  const len = getLength(d)
  let prev = getPointAtLength(d, 0)
  for (let i = 1; i <= 120; i++) {
    const p = getPointAtLength(d, (len * i) / 120)
    if (!p || !prev) return fallback
    if ((prev.x - x) * (p.x - x) <= 0 && p.x !== prev.x) return prev.y + ((p.y - prev.y) * (x - prev.x)) / (p.x - prev.x)
    prev = p
  }
  return fallback
}
const OUT_CROSS = crossY(OUT_PATH, DIV, CARD_TOP + 430)
const BACK_CROSS = crossY(BACK_PATH, DIV, REP_TOP + 40)

type Group = { idx: number[]; pair: boolean }

// Word groups (a repeat and its twin stay together) laid into rows by an estimated width, so the
// sentence never re-wraps while a word leaves it.
function layoutRows(words: { w: string; candidate: boolean }[]): Group[][] {
  const groups: Group[] = []
  for (let i = 0; i < words.length; i++) {
    const next = words[i + 1]
    const clean = (s: string) => s.toLowerCase().replace(/[^a-z']/g, "")
    if (words[i].candidate && next && clean(next.w) === clean(words[i].w)) {
      groups.push({ idx: [i, i + 1], pair: true })
      i++
    } else groups.push({ idx: [i], pair: false })
  }
  const wordW = (w: string) => w.length * FS * 0.6 + 32
  const groupW = (g: Group) => g.idx.reduce((s, i) => s + wordW(words[i].w), 0) + (g.idx.length - 1) * 10
  const rows: Group[][] = [[]]
  let x = 0
  for (const g of groups) {
    const w = groupW(g)
    if (x > 0 && x + 14 + w > LW) {
      rows.push([])
      x = 0
    }
    rows[rows.length - 1].push(g)
    x += (x > 0 ? 14 : 0) + w
  }
  return rows
}

// The request, split so the marked candidates ("[3:that]") can be coloured.
function requestSegments(req: string): { t: string; c: string; b?: boolean }[] {
  const out: { t: string; c: string; b?: boolean }[] = []
  const re = /\[\d+:[^\]]+\]/g
  let last = 0
  for (const m of req.matchAll(re)) {
    if (m.index! > last) out.push({ t: req.slice(last, m.index), c: C.text })
    out.push({ t: m[0], c: C.rose, b: true })
    last = m.index! + m[0].length
  }
  if (last < req.length) out.push({ t: req.slice(last), c: C.text })
  return out
}

// The reply as two lines of JSON, keep entries in green (or the raw reply if it is not JSON).
function replyLines(reply: string): { t: string; c: string; id?: number }[][] {
  try {
    const keep = (JSON.parse(reply) as { keep: { id: number; reason: string }[] }).keep
    if (!Array.isArray(keep) || !keep.length) throw new Error("no keep")
    const entry = (k: { id: number; reason: string }) => `{"id": ${k.id}, "reason": ${JSON.stringify(k.reason)}}`
    return keep.map((k, i) => [
      { t: i === 0 ? `{"keep": [` : "          ", c: C.dim },
      { t: entry(k), c: C.green, id: k.id },
      { t: i < keep.length - 1 ? "," : "]}", c: C.dim },
    ])
  } catch {
    return [[{ t: reply, c: C.green }]]
  }
}

// Coloured text revealed `n` characters at a time.
function Reveal({ segs, n, mark }: { segs: { t: string; c: string; b?: boolean; id?: number }[]; n: number; mark?: (s: { id?: number }) => number }) {
  let left = n
  return (
    <>
      {segs.map((s, i) => {
        const shown = s.t.slice(0, Math.max(0, left))
        left -= s.t.length
        if (!shown) return null
        const m = mark ? mark(s) : 0
        return (
          <span
            key={i}
            style={{
              color: s.c,
              fontWeight: s.b ? 700 : undefined,
              whiteSpace: "pre-wrap",
              background: m > 0 ? `rgba(110,231,183,${0.16 * m})` : undefined,
              borderRadius: 6,
              boxShadow: m > 0 ? `0 0 ${18 * m}px rgba(110,231,183,${0.25 * m})` : undefined,
            }}
          >
            {shown}
          </span>
        )
      })}
    </>
  )
}

function Caption({ icon, color, children, p, style }: { icon: IconName; color: string; children: ReactNode; p: number; style?: CSSProperties }) {
  return (
    <div
      style={{
        position: "absolute",
        display: "flex",
        alignItems: "center",
        gap: 12,
        fontFamily: MONO,
        fontSize: 21,
        letterSpacing: 4,
        textTransform: "uppercase",
        color,
        opacity: Math.min(1, p),
        transform: `translateY(${(1 - p) * 16}px)`,
        ...style,
      }}
    >
      <Icon name={icon} size={24} color={color} />
      {children}
    </div>
  )
}

// 120 dots on a slowly turning sphere; 12 of them lit (12B of 120B parameters active per token).
// While Nemotron reasons, the lit dozen moves from token to token.
function ParamOrb({ size, routing, appear }: { size: number; routing: number; appear: number }) {
  const f = useCurrentFrame()
  const N = 120
  const token = routing > 0.05 ? Math.floor(f / 5) : -1
  const keys = Array.from({ length: N }, (_, i) => ({ i, k: random(`moe${token}-${i}`) }))
  const lit = new Set(keys.sort((a, b) => b.k - a.k).slice(0, 12).map((x) => x.i))
  const rot = f * 0.012
  const R = size * 0.42
  const pts = Array.from({ length: N }, (_, i) => {
    const y = 1 - (2 * (i + 0.5)) / N
    const r = Math.sqrt(1 - y * y)
    const th = i * 2.399963 + rot
    const x = Math.cos(th) * r
    const z = Math.sin(th) * r
    const tilt = 0.35
    const y2 = y * Math.cos(tilt) - z * Math.sin(tilt)
    const z2 = y * Math.sin(tilt) + z * Math.cos(tilt)
    return { i, x: size / 2 + x * R, y: size / 2 + y2 * R, z: z2 }
  }).sort((a, b) => a.z - b.z)
  return (
    <svg width={size} height={size} style={{ overflow: "visible" }}>
      <defs>
        <radialGradient id="mg-orb">
          <stop offset="0%" stopColor="rgba(196,181,253,0.22)" />
          <stop offset="100%" stopColor="rgba(196,181,253,0)" />
        </radialGradient>
      </defs>
      <circle cx={size / 2} cy={size / 2} r={R * 1.25} fill="url(#mg-orb)" opacity={0.6 + 0.4 * routing} />
      {pts.map((p) => {
        const on = lit.has(p.i)
        const depth = 0.5 + 0.5 * p.z
        const a = Math.min(1, Math.max(0, appear * 1.4 - (1 - depth) * 0.4))
        return (
          <circle
            key={p.i}
            cx={p.x}
            cy={p.y}
            r={on ? 4.2 + depth * 1.6 : 2 + depth * 1.4}
            fill={on ? "#ede9fe" : C.violet}
            opacity={a * (on ? 1 : 0.18 + 0.32 * depth)}
            style={on ? { filter: "drop-shadow(0 0 6px #c4b5fd) drop-shadow(0 0 12px #a78bfa)" } : undefined}
          />
        )
      })}
    </svg>
  )
}

export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const g = d.guard
  const words = g.words
  const L0 = lineCue(scene, 0)
  const L1 = lineCue(scene, 1)
  const L2 = lineCue(scene, 2)
  const at1 = (u: number) => L1.at + u * L1.frames
  const at2 = (u: number) => L2.at + u * L2.frames

  // ---- beats
  const markAt = L0.at + 0.3 * L0.frames // "repeat": the marked candidates light up
  // The title leaves (fade and lift) before the sentence glides up into its place.
  const titleOut = ramp(f, L1.at - 14, L1.at + 2, easeOut)
  const split = ramp(f, L1.at, L1.at + 30, easeOut) // the sentence moves to the laptop side
  const divDraw = ramp(f, L1.at + 6, L1.at + 40, easeOut)
  const cardIn = springAt(f, L1.at + 4)
  const nebiusAt = at1(0.24) // "on Nebius"
  const reqAt = at1(0.4) // "reads each cut in its sentence"
  const sendAt = at1(0.58)
  const textOnlyAt = at1(0.8) // "as text only"
  const thinkA = at1(0.62)
  const replyAt = at1(0.86)
  const thinkB = replyAt + 2
  const kept = words.map((w, i) => ({ ...w, i })).filter((w) => w.candidate && w.kept)
  const cutW = words.map((w, i) => ({ ...w, i })).filter((w) => w.candidate && !w.kept)
  const keepTimes = [0.04, 0.42]
  const keepAt = new Map(kept.map((w, j) => [w.i, at2(keepTimes[j] ?? 0.04 + (0.5 * j) / Math.max(1, kept.length - 1))]))
  const cutAt = at2(0.64)
  const bothAt = at2(0.82) // "Both stay"
  // Right after the extra "I" leaves (and before "Both stay"), so the chip is settled and read for
  // ~1.8 s before the outgoing transition (the scene's tail after "Both stay" is too short for it).
  const takeAt = cutAt + 4
  // Line 2: the sentence is the focal point; the cloud side and the request step back.
  const dimL2 = ramp(f, L2.at + 10, L2.at + 30, easeOut)
  // The twin of each kept repeat ("that [that]") turns green a beat after it.
  const twinKeepAt = new Map<number, number>()
  kept.forEach((w) => {
    const next = words[w.i + 1]
    if (next && !next.candidate) twinKeepAt.set(w.i + 1, (keepAt.get(w.i) ?? 0) + 4)
  })

  const routing = ramp(f, thinkA, thinkA + 8) * (1 - ramp(f, thinkB, thinkB + 12))
  const rows = layoutRows(words)
  const reqSegs = requestSegments(g.request)
  const reqN = Math.max(0, (f - reqAt) * 3.2)
  const rep = replyLines(g.reply)
  const repLen = rep.map((l) => l.reduce((s, x) => s + x.t.length, 0))
  const repN = (li: number) => Math.max(0, (f - replyAt - 6 - li * 8) * 6)
  const markEntry = (s: { id?: number }) => (s.id === undefined ? 0 : ramp(f, (keepAt.get(s.id) ?? 1e9) - 2, (keepAt.get(s.id) ?? 1e9) + 10))

  // The sentence starts centred and large, then settles on the laptop side.
  const sentH = 46 + rows.length * (FS * 1.25 + 18) + (rows.length - 1) * ROW_GAP
  const cx = LX + LW / 2
  const cy = SENT_TOP + sentH / 2
  const s0 = 1.2
  const sx = interpolate(split, [0, 1], [960 - cx, 0])
  const sy = interpolate(split, [0, 1], [600 - cy, 0])
  const sc = interpolate(split, [0, 1], [s0, 1])

  const outFlow = ramp(f, sendAt, sendAt + 6) * (1 - ramp(f, thinkB - 16, thinkB - 4))
  const backFlow = ramp(f, replyAt - 8, replyAt) * (1 - ramp(f, L2.at + 14, L2.at + 26))
  const pathFade = 1 - 0.65 * ramp(f, L2.at + 20, L2.at + 40)
  const pulse = 0.6 + 0.4 * Math.sin(f / 4)
  const gateGlow = Math.max(outFlow, backFlow) * pulse + Math.max(0, 1 - Math.abs(f - textOnlyAt - 6) / 16)

  return (
    <AbsoluteFill>
      <Camera
        shots={[
          { at: 0, zoom: 1.035 },
          { at: L1.at, zoom: 0.985, dur: 40 },
          // a slow push that keeps both columns whole: at 1.025 the laptop caption (x 120) and the
          // Nemotron card's text (to x ~1770) stay 96 px inside the frame, and the reply and
          // "On Alex's take" stay above the subtitles
          { at: L2.at + 4, x: 962, y: 540, zoom: 1.025, dur: 90 },
        ]}
        drift={5}
        seed="mg"
      >
        {/* atmosphere */}
        <GlowOrb x={interpolate(split, [0, 1], [960, 600])} y={interpolate(split, [0, 1], [600, 400])} r={620} color="rgba(125,211,252,0.16)" opacity={0.9} />
        <GlowOrb x={RX + RW / 2} y={CARD_TOP + 240} r={560} color="rgba(167,139,250,0.3)" opacity={(0.45 + 0.55 * routing) * Math.min(1, cardIn) * (1 - 0.6 * dimL2)} />
        <GlowOrb x={560} y={400} r={560} color="rgba(110,231,183,0.22)" opacity={ramp(f, bothAt - 2, bothAt + 8, easeOut) * (1 - 0.6 * ramp(f, bothAt + 10, bothAt + 40))} />
        <Particles count={12} color={C.sky} seed="mg-l" x={60} y={140} w={1040} h={720} size={[1.5, 3]} rise={0.3} opacity={0.4} />
        <Particles count={16} color={C.violet} seed="mg-r" x={1200} y={130} w={660} h={740} size={[1.5, 3.5]} rise={0.35} opacity={0.55 * Math.min(1, cardIn)} />

        {/* beat 0: the statement */}
        <div
          style={{
            position: "absolute",
            left: 0,
            width: 1920,
            top: 168,
            textAlign: "center",
            opacity: 1 - titleOut,
            transform: `translateY(${-60 * titleOut}px)`,
          }}
        >
          <KineticText text="Not every repeat is a stutter." at={0} per={2.4} effect="blur" style={{ fontFamily: SERIF, fontSize: 112, letterSpacing: -1, lineHeight: 1.04, color: C.text }} />
        </div>

        {/* the column captions */}
        <Caption icon="laptop" color={C.sky} p={springAt(f, L1.at + 10)} style={{ left: LX, top: TOP }}>
          on the laptop
        </Caption>
        <Caption icon="cloud" color={C.violet} p={springAt(f, nebiusAt) * (1 - 0.55 * dimL2)} style={{ left: RX, top: TOP }}>
          Nebius Token Factory
        </Caption>

        {/* the divider, with packets crossing it */}
        <Diagram>
          <defs>
            {/* user space: a bounding-box gradient draws nothing on a zero-width line */}
            <linearGradient id="mg-div" gradientUnits="userSpaceOnUse" x1={DIV} x2={DIV} y1={DIV_TOP} y2={DIV_BOT}>
              <stop offset="0%" stopColor={C.sky} stopOpacity={0} />
              <stop offset="12%" stopColor={C.violet} stopOpacity={0.6} />
              <stop offset="88%" stopColor={C.violet} stopOpacity={0.6} />
              <stop offset="100%" stopColor={C.violet} stopOpacity={0} />
            </linearGradient>
          </defs>
          <line x1={DIV} y1={DIV_TOP} x2={DIV} y2={DIV_TOP + (DIV_BOT - DIV_TOP) * divDraw} stroke="url(#mg-div)" strokeWidth={2.5} strokeDasharray="6 10" strokeDashoffset={-f * 0.5} />
          {/* the divider lights only where the packets cross it */}
          {[
            { y: OUT_CROSS, o: outFlow },
            { y: BACK_CROSS, o: backFlow },
          ].map((g, k) =>
            g.o > 0.01 ? (
              <line key={k} x1={DIV} y1={g.y - 60} x2={DIV} y2={g.y + 60} stroke={C.violet} strokeWidth={4} strokeLinecap="round" opacity={Math.min(1, g.o * pulse) * 0.9} style={{ filter: "drop-shadow(0 0 10px #c4b5fd)" }} />
            ) : null,
          )}
          <g opacity={pathFade}>
            <FlowPath d={OUT_PATH} draw={ramp(f, sendAt - 10, sendAt + 8, easeOut)} flow={outFlow} color={C.violet} width={2.5} kind="word" words={reqSegs.filter((s) => s.b).map((s) => s.t)} period={54} />
          </g>
          {/* the return path clears before "On Alex's take" lands over its end */}
          <g opacity={pathFade * (1 - ramp(f, takeAt - 10, takeAt + 2))}>
            <FlowPath d={BACK_PATH} draw={ramp(f, replyAt - 14, replyAt + 2, easeOut)} flow={backFlow} color={C.green} width={2.5} kind="word" words={kept.map((w) => `keep ${w.i}`)} period={44} />
          </g>
        </Diagram>

        {/* "text only · never audio": hung on the divider, above where the packets cross it */}
        <div
          style={{
            position: "absolute",
            right: 1920 - DIV + 18,
            top: GATE_Y - 24,
            display: "flex",
            alignItems: "center",
            opacity: ramp(f, L1.at + 24, L1.at + 38),
            transform: `translateX(${(1 - ramp(f, L1.at + 24, L1.at + 40, easeOut)) * 20}px) scale(${1 + 0.06 * Math.max(0, 1 - Math.abs(f - textOnlyAt - 6) / 12)})`,
            transformOrigin: "right center",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "9px 18px",
              borderRadius: 999,
              whiteSpace: "nowrap",
              background: "rgba(10,12,24,0.85)",
              border: `1.5px solid ${interpolateColors(Math.min(1, gateGlow), [0, 1], [C.border2, C.violet])}`,
              boxShadow: `0 0 ${30 * Math.min(1, gateGlow)}px -4px ${C.violet}`,
              fontFamily: MONO,
              fontSize: 20,
              color: C.violet,
            }}
          >
            <Icon name="text" size={22} color={C.violet} />
            text only · never audio
          </div>
          <div style={{ width: 18, height: 2, background: C.violet, opacity: 0.6 }} />
        </div>
        <div style={{ position: "absolute", left: DIV - 6, top: GATE_Y - 6, width: 12, height: 12, borderRadius: 6, background: C.violet, boxShadow: `0 0 14px ${C.violet}`, opacity: ramp(f, L1.at + 24, L1.at + 38) }} />

        {/* the test sentence */}
        <div
          style={{
            position: "absolute",
            left: LX,
            top: SENT_TOP,
            width: LW,
            transformOrigin: `${LW / 2}px ${sentH / 2}px`,
            transform: `translate(${sx}px, ${sy}px) scale(${sc})`,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 16, height: 34, opacity: ramp(f, 2, 14) }}>
            <Tag style={{ fontSize: 19 }}>a typed test sentence</Tag>
            <span style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: MONO, fontSize: 19, color: C.dim, opacity: ramp(f, markAt, markAt + 10) }}>
              <span style={{ width: 22, height: 16, borderRadius: 4, border: `2px dashed ${C.rose}` }} />
              marked for cutting
            </span>
          </div>
          <div style={{ height: 14 }} />
          {/* the sentence's light sweep on "Both stay" */}
          <div style={{ position: "relative" }}>
            {rows.map((row, ri) => (
              <div key={ri} style={{ display: "flex", gap: 14, alignItems: "flex-start", marginTop: ri ? ROW_GAP : 0, flexWrap: "nowrap" }}>
                {row.map((grp) => {
                  const kAt = grp.pair ? keepAt.get(grp.idx[0]) : undefined
                  const under = kAt !== undefined ? springAt(f, bothAt + (kept.findIndex((w) => w.i === grp.idx[0]) * 4), SNAPPY) : 0
                  const reason = kAt !== undefined ? springAt(f, kAt + 8) : 0
                  const cutHere = grp.idx.some((i) => words[i].candidate && !words[i].kept)
                  const sc = cutHere ? springAt(f, cutAt - 2, BOUNCY) : 0
                  // the label leaves with the word (same span as the chip's collapse)
                  const scOut = cutHere ? ramp(f, cutAt + 6, cutAt + 16, easeIn) : 0
                  return (
                    <div key={grp.idx[0]} style={{ position: "relative", display: "flex", gap: 10 }}>
                      {grp.idx.map((i) => (
                        <WordChip key={i} f={f} w={words[i].w} i={i} cand={words[i].candidate} markAt={markAt} keepAt={keepAt.get(i) ?? twinKeepAt.get(i)} cutAt={words[i].candidate && !words[i].kept ? cutAt : undefined} />
                      ))}
                      {cutHere && sc > 0.01 && scOut < 1 && (
                        <div style={{ position: "absolute", left: -2, top: "100%", marginTop: 12, opacity: Math.min(1, sc) * (1 - scOut), transform: `scale(${0.5 + 0.5 * sc}) translateY(${10 * scOut}px)`, transformOrigin: "left top", display: "flex", alignItems: "center", gap: 6, fontFamily: MONO, fontSize: 19, color: C.rose, whiteSpace: "nowrap" }}>
                          <Icon name="scissors" size={22} color={C.rose} />
                          cut
                        </div>
                      )}
                      {kAt !== undefined && (
                        <>
                          <div
                            style={{
                              position: "absolute",
                              left: 4,
                              right: 4,
                              bottom: -13,
                              height: 7,
                              borderRadius: 4,
                              background: C.green,
                              boxShadow: `0 0 14px ${C.green}`,
                              transformOrigin: "left center",
                              transform: `scaleX(${Math.max(0, Math.min(1.02, under))})`,
                              opacity: Math.min(1, under * 2),
                            }}
                          />
                          <div
                            style={{
                              position: "absolute",
                              left: "50%",
                              top: "100%",
                              marginTop: 18,
                              transform: `translate(-50%, ${(1 - reason) * -10}px)`,
                              opacity: Math.min(1, reason),
                              whiteSpace: "nowrap",
                              display: "flex",
                              alignItems: "center",
                              gap: 6,
                              fontFamily: MONO,
                              fontSize: 20,
                              color: C.green,
                            }}
                          >
                            <Icon name="check" size={20} color={C.green} stroke={3} />
                            {words[grp.idx[0]].reason}
                          </div>
                        </>
                      )}
                    </div>
                  )
                })}
              </div>
            ))}
          </div>
        </div>

        {/* the request: what Nemotron reads */}
        <GlassBox
          top={REQ_TOP}
          left={LX}
          width={REQ_W}
          accent={C.violet}
          p={springAt(f, reqAt - 6)}
          dim={0.7 * dimL2}
          label="request · text"
          icon="message"
        >
          <div style={{ fontFamily: MONO, fontSize: 21, lineHeight: 1.45, minHeight: 61 }}>
            <Reveal segs={reqSegs} n={reqN} />
          </div>
        </GlassBox>

        {/* the reply */}
        <GlassBox top={REP_TOP} left={LX} width={LW} accent={C.green} p={springAt(f, replyAt - 4)} label="Real Nemotron reply" note={/^\d+ of \d+$/.test(g.agreed ?? "") ? `same verdict in ${g.agreed} runs` : undefined} icon="check">
          <div style={{ fontFamily: MONO, fontSize: 21, lineHeight: 1.45, minHeight: 61 }}>
            {rep.map((line, li) => (
              <div key={li} style={{ whiteSpace: "pre" }}>
                <Reveal segs={line} n={Math.min(repLen[li], repN(li))} mark={markEntry} />
              </div>
            ))}
          </div>
        </GlassBox>

        {/* Nemotron */}
        <NemotronCard f={f} model={g.model} appear={cardIn} at={L1.at + 4} routing={routing} replied={f >= replyAt - 10} dim={0.55 * dimL2} />

        {/* on Alex's own take */}
        <div
          style={{
            position: "absolute",
            left: RX,
            top: TAKE_TOP,
            width: RW,
            opacity: Math.min(1, springAt(f, takeAt, SNAPPY) * 1.4),
            transform: `translateY(${(1 - Math.min(1, springAt(f, takeAt, SNAPPY))) * 24}px)`,
          }}
        >
          <div
            style={{
              position: "relative",
              overflow: "hidden",
              borderRadius: 22,
              padding: "18px 24px 20px",
              background: "linear-gradient(160deg, rgba(94,234,212,0.09), rgba(255,255,255,0.025))",
              border: `1px solid rgba(94,234,212,0.35)`,
              boxShadow: "0 24px 60px -24px rgba(0,0,0,0.8)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10, fontFamily: MONO, fontSize: 20, letterSpacing: 3, color: C.teal, textTransform: "uppercase" }}>
              <Icon name="wave" size={22} color={C.teal} />
              On Alex&rsquo;s take
            </div>
            <div style={{ marginTop: 10, fontFamily: SANS, fontSize: 25, fontWeight: 500, lineHeight: 1.36, color: C.text }}>
              <span style={{ whiteSpace: "nowrap" }}>
                Nemotron checked{" "}
                <Counter to={d.clip.guard?.checked ?? 0} at={takeAt + 2} dur={14} style={{ fontWeight: 700, color: C.teal }} /> words
              </span>
              <br />
              <span style={{ whiteSpace: "nowrap" }}>
                marked for cutting
                <span style={{ color: C.dim }}> · </span>
                {(d.clip.guard?.kept.length ?? 0) === 0 ? "all real stutters" : `kept ${d.clip.guard?.kept.length}`}
              </span>
            </div>
            <LightSweep at={takeAt + 6} dur={40} color="rgba(94,234,212,0.18)" />
          </div>
        </div>
      </Camera>

      <Sfx name="pop" at={reqAt - 4} volume={0.22} />
      {kept.map((w) => (
        <Sfx key={w.i} name="chime" at={(keepAt.get(w.i) ?? 0) + 1} volume={0.32} />
      ))}
      {cutW.length > 0 && <Sfx name="snap" at={cutAt + 6} volume={0.4} />}
      {/* "Both stay": the underlines land */}
      <Sfx name="chime" at={bothAt + 2} volume={0.26} />
    </AbsoluteFill>
  )
}

function WordChip({ f, w, i, cand, markAt, keepAt, cutAt }: { f: number; w: string; i: number; cand: boolean; markAt: number; keepAt?: number; cutAt?: number }) {
  const inP = springAt(f, 2 + i * 0.8, SNAPPY)
  const mark = cand ? springAt(f, markAt + (i % 5) * 2, BOUNCY) : 0
  const keep = keepAt !== undefined ? springAt(f, keepAt, SNAPPY) : 0
  const cutTint = cutAt !== undefined ? ramp(f, cutAt, cutAt + 6) : 0
  const gone = cutAt !== undefined ? ramp(f, cutAt + 6, cutAt + 20, easeIn) : 0
  const kc = Math.min(1, keep)
  const borderColor =
    keep > 0.5 ? C.green : cand ? interpolateColors(mark, [0, 1], ["rgba(255,255,255,0.12)", C.rose]) : "rgba(255,255,255,0.12)"
  const style = cand && keep <= 0.5 ? "dashed" : "solid"
  const flash = keepAt !== undefined ? Math.max(0, 1 - Math.abs(f - keepAt - 4) / 12) : 0
  return (
    <div
      style={{
        maxWidth: cutAt !== undefined ? `${(1 - gone) * 200}px` : undefined,
        marginRight: cutAt !== undefined ? -10 * gone : undefined,
        overflow: cutAt !== undefined ? "hidden" : undefined,
        opacity: Math.min(1, inP * 1.4) * (1 - gone),
        transform: `translateY(${(1 - inP) * 26 - gone * 18}px) scale(${1 + 0.1 * flash - 0.3 * gone})`,
        filter: `blur(${(1 - Math.min(1, inP)) * 6 + gone * 6}px)`,
      }}
    >
      <div
        style={{
          fontFamily: SANS,
          fontSize: FS,
          fontWeight: 600,
          lineHeight: 1.25,
          padding: "7px 14px",
          borderRadius: 14,
          whiteSpace: "nowrap",
          border: `2.5px ${style} ${cutTint > 0 ? interpolateColors(cutTint, [0, 1], [borderColor, C.rose]) : borderColor}`,
          background:
            kc > 0
              ? `rgba(110,231,183,${0.16 * kc})`
              : cutTint > 0
                ? `rgba(251,113,133,${0.16 * cutTint})`
                : cand
                  ? `rgba(251,113,133,${0.06 * mark})`
                  : "rgba(255,255,255,0.045)",
          color: kc > 0 ? interpolateColors(kc, [0, 1], [C.text, "#d1fae5"]) : cutTint > 0 ? interpolateColors(cutTint, [0, 1], [C.text, C.rose]) : C.text,
          boxShadow: `0 0 ${28 * (kc * 0.6 + flash)}px rgba(110,231,183,${0.45 * (kc * 0.5 + flash)})${cand && mark > 0.1 && kc < 0.5 ? `, 0 0 22px -6px rgba(251,113,133,${0.5 * mark})` : ""}`,
        }}
      >
        {w}
      </div>
    </div>
  )
}

function GlassBox({
  top,
  left,
  width,
  accent,
  p,
  dim = 0,
  label,
  note,
  icon,
  children,
}: {
  top: number
  left: number
  width: number
  accent: string
  p: number
  dim?: number
  label: string
  note?: string
  icon: IconName
  children: ReactNode
}) {
  const a = Math.min(1, p)
  return (
    <div
      style={{
        position: "absolute",
        top,
        left,
        width,
        boxSizing: "border-box",
        padding: "14px 24px 16px 28px",
        borderRadius: 18,
        background: "linear-gradient(160deg, rgba(255,255,255,0.065), rgba(255,255,255,0.02))",
        border: `1px solid ${C.border}`,
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.07), 0 24px 60px -26px rgba(0,0,0,0.85)",
        opacity: a * (1 - dim),
        transform: `translateY(${(1 - p) * 22}px)`,
        overflow: "hidden",
      }}
    >
      <div style={{ position: "absolute", left: 0, top: 14, bottom: 14, width: 4, borderRadius: 2, background: accent, boxShadow: `0 0 12px ${accent}` }} />
      <div style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: MONO, fontSize: 20, letterSpacing: 2, textTransform: "uppercase", color: accent, marginBottom: 6 }}>
        <Icon name={icon} size={20} color={accent} />
        {label}
        {note && <span style={{ fontSize: 19, letterSpacing: 0, textTransform: "none", color: C.dim, marginLeft: 6 }}>· {note}</span>}
      </div>
      {children}
    </div>
  )
}

function NemotronCard({ f, model, appear, at, routing, replied, dim = 0 }: { f: number; model: string; appear: number; at: number; routing: number; replied: boolean; dim?: number }) {
  const status = replied
    ? { text: "replied · JSON, text only", color: C.green }
    : routing > 0.05
      ? { text: "reading the request…", color: C.violet }
      : { text: "waiting for text", color: C.dim }
  // the frame comes in with its first line, so it is never an empty box
  const a = Math.min(1, appear) * Math.min(1, springAt(f, at + 2) * 1.3)
  const item = (k: number) => {
    const p = springAt(f, at + 2 + k * 3)
    return { opacity: Math.min(1, p), transform: `translateY(${(1 - p) * 14}px)` }
  }
  const pulse = routing * (0.7 + 0.3 * Math.sin(f / 5))
  const specs: [IconName, string][] = [
    ["layers", "hybrid Mamba-2 + LatentMoE + attention"],
    ["zap", "multi-token prediction · up to 1M tokens"],
  ]
  return (
    <div
      style={{
        position: "absolute",
        left: RX,
        top: CARD_TOP,
        width: RW,
        height: CARD_H,
        boxSizing: "border-box",
        padding: "24px 28px",
        borderRadius: 28,
        overflow: "hidden",
        background: "linear-gradient(160deg, rgba(196,181,253,0.12), rgba(255,255,255,0.03) 55%, rgba(196,181,253,0.05))",
        border: `1.5px solid ${interpolateColors(0.35 + 0.65 * pulse, [0, 1], ["rgba(196,181,253,0.3)", C.violet])}`,
        boxShadow: `inset 0 1px 0 rgba(255,255,255,0.1), 0 40px 90px -30px rgba(0,0,0,0.9), 0 0 ${30 + 60 * pulse}px -10px rgba(167,139,250,${0.35 + 0.4 * pulse})`,
        opacity: a * (1 - dim),
        transform: `translateY(${(1 - appear) * 40}px) scale(${0.96 + 0.04 * appear})`,
        fontFamily: SANS,
        color: C.text,
      }}
    >
      <LightSweep at={at + 8} dur={40} color="rgba(196,181,253,0.22)" />
      <div
        style={{
          position: "absolute",
          left: 28,
          right: 28,
          bottom: 20,
          paddingTop: 14,
          borderTop: `1px solid ${C.border}`,
          display: "flex",
          alignItems: "center",
          gap: 10,
          fontFamily: MONO,
          fontSize: 19,
          color: status.color,
          ...item(6),
        }}
      >
        <span style={{ width: 10, height: 10, borderRadius: 5, background: status.color, boxShadow: `0 0 10px ${status.color}`, opacity: routing > 0.05 ? 0.5 + 0.5 * Math.sin(f / 2.5) : 1 }} />
        {status.text}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, ...item(0) }}>
        <NvTag />
        <span style={{ fontFamily: MONO, fontSize: 18, color: C.dim, whiteSpace: "nowrap" }}>on Nebius · OpenAI-compatible API</span>
      </div>
      <div style={{ fontSize: 40, fontWeight: 650, letterSpacing: -0.5, marginTop: 12, lineHeight: 1.1, ...item(1) }}>NVIDIA Nemotron 3 Super</div>
      <div style={{ fontFamily: MONO, fontSize: 20, color: C.violet, marginTop: 8, ...item(2) }}>{model}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 18, marginTop: 10, ...item(3) }}>
        <div style={{ width: 176, height: 176, flexShrink: 0 }}>
          <ParamOrb size={176} routing={routing} appear={springAt(f, at + 16)} />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <span style={{ fontSize: 68, fontWeight: 700, color: "#ede9fe", letterSpacing: -2, textShadow: "0 0 30px rgba(196,181,253,0.6)" }}>12B</span>
            <span style={{ fontSize: 28, fontWeight: 500, color: C.dim }}>of 120B</span>
          </div>
          <div style={{ fontSize: 26, fontWeight: 500, color: C.text, lineHeight: 1.25 }}>
            parameters active
            <br />
            per token
          </div>
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 9, marginTop: 10 }}>
        {specs.map(([icon, t], k) => (
          <div key={t} style={{ display: "flex", alignItems: "center", gap: 12, fontFamily: MONO, fontSize: 20, color: C.dim, ...item(4 + k) }}>
            <Icon name={icon} size={20} color={C.violet} />
            {t}
          </div>
        ))}
      </div>
    </div>
  )
}

