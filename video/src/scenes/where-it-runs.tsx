import type { CSSProperties, ReactNode } from "react"
import { useMemo } from "react"
import { AbsoluteFill, useCurrentFrame } from "remotion"
import { evolvePath, getLength, getPointAtLength } from "@remotion/paths"

import type { SceneProps } from "../data"
import { lineCue } from "../data"
import {
  Camera,
  Diagram,
  easeOut,
  FlowPath,
  GlowOrb,
  Icon,
  LightSweep,
  Particles,
  ramp,
  Sfx,
  SNAPPY,
  springAt,
} from "../kit"
import type { IconName } from "../kit"
import { C, MONO, SANS, SERIF } from "../theme"

// Where it runs: the system in one picture. Alex's laptop on the left holds every piece that touches
// his voice (sky audio frames circulate there and never leave); a dashed "text only" border; the
// cloud on the right (Nemotron on Nebius Token Factory, Magpie on build.nvidia.com) receives words.

// ---- layout (stage px) -------------------------------------------------------------------------
// Everything sits inside x 113..1807 so the end push (zoom 1.02) keeps the 96 px safe margins.
const LAP = { x: 116, y: 112, w: 1130, h: 752 } // Alex's laptop
const CLOUD = { x: 1356, y: 112, w: 450, h: 752 } // the cloud zone
const BX = 1315 // the text-only border
// The recordings teacher (WavLM v2) is its own block: the brief only puts the LIVE detector on the
// RTX 4050 (the v2 checkpoint goes to the GPU only with 10 GB+ of VRAM, else the CPU), so no device here.
const TEACH = { x: 320, y: 250, w: 400, h: 104 }
const GPU = { x: 320, y: 392, w: 400, h: 420 }
const COLC = 800 // the right column of the laptop
const CW = { x: COLC, y: 250, w: 400, h: 110 }
const SPL = { x: COLC, y: 400, w: 400, h: 110 }
const NEMO = { x: COLC, y: 555, w: 400, h: 140 }
const VMIC = { x: COLC, y: 740, w: 400, h: 100 }
const NEB = { x: 1384, y: 190, w: 398, h: 392 }
const MAG = { x: 1384, y: 626, w: 398, h: 234 }
const NOTE_PORT = { x: 196, y: 302 }
const MIC_PORT = { x: 196, y: 530 }
const PORT_R = 42

// audio (sky frames): stays inside the laptop
const A_NOTE_TEACH = `M ${NOTE_PORT.x + PORT_R} ${NOTE_PORT.y} L ${TEACH.x} ${NOTE_PORT.y}`
const A_MIC_GPU = `M ${MIC_PORT.x + PORT_R} ${MIC_PORT.y} L ${GPU.x} ${MIC_PORT.y}`
const A_NOTE_CW = `M ${NOTE_PORT.x} ${NOTE_PORT.y - PORT_R} C ${NOTE_PORT.x} 230, 214 212, 262 212 L 740 212 C 778 212, 772 300, ${COLC} 300`
const A_TEACH_SPL = `M ${TEACH.x + TEACH.w} 330 C 764 330, 756 422, ${COLC} 422`
const A_GPU_SPL = `M ${GPU.x + GPU.w} 460 L ${COLC} 460`
const A_CW_SPL = `M 1000 ${CW.y + CW.h} L 1000 ${SPL.y}`
const A_SPL_NEMO = `M 1000 ${SPL.y + SPL.h} L 1000 ${NEMO.y}`
const A_SPL_VMIC = `M ${COLC} 498 C 752 498, 752 790, ${COLC} 790`
// text (violet words): crosses the border
const T_CW_NEB = `M ${COLC + 400} 300 C 1290 300, 1300 330, ${NEB.x} 330`
const T_NEMO_NEB = `M ${COLC + 400} 600 C 1290 600, 1300 520, ${NEB.x} 520`
// the tapped word leaves from Alex's own tap (the "Say it" button in his app), not from the captions:
// the caption stopped at "Roman", Nemotron suggested the word and Alex tapped it
const TAP = { x: 1036, y: 699, w: 160, h: 36 }
const T_NEMO_MAG = `M ${TAP.x + TAP.w} ${TAP.y + TAP.h / 2} C 1290 ${TAP.y + TAP.h / 2}, 1300 712, ${MAG.x} 712`
// the one thing that comes back: the tapped word, spoken by Magpie, into the call
const R_MAG_VMIC = `M ${MAG.x} 805 C 1300 805, 1290 790, ${COLC + 400} 790`

const roundedRect = (x: number, y: number, w: number, h: number, r: number) =>
  `M ${x + r} ${y} L ${x + w - r} ${y} Q ${x + w} ${y} ${x + w} ${y + r} L ${x + w} ${y + h - r} Q ${x + w} ${y + h} ${x + w - r} ${y + h} L ${x + r} ${y + h} Q ${x} ${y + h} ${x} ${y + h - r} L ${x} ${y + r} Q ${x} ${y} ${x + r} ${y}`

// ---- small pieces ------------------------------------------------------------------------------

// A glass panel that springs in (rise, unblur, settle) and glows in its accent while it works.
function Panel({
  box,
  at,
  accent,
  glow = 0,
  sweep = true,
  children,
  style,
}: {
  box: { x: number; y: number; w: number; h: number }
  at: number
  accent: string
  glow?: number
  sweep?: boolean
  children: ReactNode
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  const p = springAt(f, at, SNAPPY)
  const c = Math.min(1, p)
  if (f < at - 1) return null
  return (
    <div
      style={{
        position: "absolute",
        left: box.x,
        top: box.y,
        width: box.w,
        height: box.h,
        opacity: Math.min(1, p * 1.4),
        transform: `translateY(${(1 - p) * 34}px) scale(${0.92 + 0.08 * p})`,
        filter: c < 0.98 ? `blur(${(1 - c) * 10}px)` : undefined,
        borderRadius: 22,
        boxSizing: "border-box",
        background: "linear-gradient(160deg, rgba(255,255,255,0.09), rgba(255,255,255,0.025) 60%, rgba(255,255,255,0.04)), rgba(11,15,27,0.92)",
        border: `1.5px solid ${glow > 0.05 ? accent : C.border2}`,
        boxShadow: `0 0 ${18 + 46 * glow}px ${-4 + 4 * glow}px ${accent}${glow > 0.05 ? "70" : "00"}, inset 0 1px 0 rgba(255,255,255,0.12), 0 30px 70px -28px rgba(0,0,0,0.85)`,
        fontFamily: SANS,
        color: C.text,
        overflow: "hidden",
        ...style,
      }}
    >
      {children}
      {sweep && <LightSweep at={at + 4} dur={30} color="rgba(255,255,255,0.18)" />}
    </div>
  )
}

function Head({ icon, title, accent, sub, subColor = C.dim, size = 27 }: { icon: IconName; title: string; accent: string; sub?: string; subColor?: string; size?: number }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
      <div style={{ width: 44, height: 44, borderRadius: 12, background: `${accent}22`, border: `1px solid ${accent}44`, display: "flex", alignItems: "center", justifyContent: "center", color: accent, flexShrink: 0 }}>
        <Icon name={icon} size={26} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 3, minWidth: 0 }}>
        <div style={{ fontSize: size, fontWeight: 600, lineHeight: 1.1, letterSpacing: -0.2, whiteSpace: "nowrap" }}>{title}</div>
        {sub && <div style={{ fontFamily: MONO, fontSize: 19, color: subColor, lineHeight: 1.2, whiteSpace: "nowrap" }}>{sub}</div>}
      </div>
    </div>
  )
}

const NV = ({ style }: { style?: CSSProperties }) => (
  <span style={{ fontFamily: MONO, fontSize: 18, fontWeight: 700, letterSpacing: 1.5, color: "#0b0f02", background: C.nv, borderRadius: 6, padding: "3px 9px", ...style }}>NVIDIA</span>
)

const Chip = ({ children, color = C.dim, fill, at, style }: { children: ReactNode; color?: string; fill?: string; at?: number; style?: CSSProperties }) => {
  const f = useCurrentFrame()
  const p = at === undefined ? 1 : springAt(f, at, SNAPPY)
  return (
    <span
      style={{
        display: "inline-block",
        fontFamily: MONO,
        fontSize: 19,
        color,
        border: `1px solid ${fill ? fill : `${color}66`}`,
        background: fill ?? `${color}14`,
        borderRadius: 999,
        padding: "5px 13px",
        whiteSpace: "nowrap",
        opacity: Math.min(1, p * 1.5),
        transform: `scale(${0.7 + 0.3 * p})`,
        ...style,
      }}
    >
      {children}
    </span>
  )
}

// Where audio comes in: a microphone (live) or a voice note / video (recordings).
function Port({ x, y, icon, label, at, level }: { x: number; y: number; icon: IconName; label: string; at: number; level: number }) {
  const f = useCurrentFrame()
  const p = springAt(f, at, SNAPPY)
  if (f < at - 1) return null
  return (
    <div style={{ position: "absolute", left: x - 90, top: y - PORT_R, width: 180, display: "flex", flexDirection: "column", alignItems: "center", opacity: Math.min(1, p * 1.4), transform: `scale(${0.6 + 0.4 * p})` }}>
      <div style={{ position: "relative", width: PORT_R * 2, height: PORT_R * 2 }}>
        {[0, 1].map((i) => (
          <div
            key={i}
            style={{
              position: "absolute",
              inset: 0,
              borderRadius: "50%",
              border: `1.5px solid ${C.sky}`,
              opacity: (0.4 - i * 0.15) * (0.4 + 0.6 * level),
              transform: `scale(${1.18 + i * 0.2 + level * 0.25 * (i + 1)})`,
            }}
          />
        ))}
        <div
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: "50%",
            background: `radial-gradient(circle at 35% 30%, ${C.sky}44, rgba(10,14,26,0.95) 75%)`,
            border: `1.5px solid ${C.sky}88`,
            boxShadow: `0 0 ${24 + 30 * level}px -6px ${C.sky}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: C.sky,
          }}
        >
          <Icon name={icon} size={34} />
        </div>
      </div>
      <div style={{ marginTop: 16, fontFamily: MONO, fontSize: 19, color: C.dim, textAlign: "center", lineHeight: 1.3, whiteSpace: "pre-line" }}>{label}</div>
    </div>
  )
}

// One model inside the GPU card.
function ModelRow({ title, sub, at, active, style }: { title: string; sub: string; at: number; active: number; style?: CSSProperties }) {
  const f = useCurrentFrame()
  const p = springAt(f, at, SNAPPY)
  return (
    <div
      style={{
        position: "absolute",
        left: 20,
        right: 20,
        borderRadius: 16,
        padding: "13px 18px 13px 22px",
        background: `linear-gradient(100deg, rgba(118,185,0,${0.1 + 0.08 * active}), rgba(255,255,255,0.03))`,
        border: `1px solid rgba(118,185,0,${0.25 + 0.4 * active})`,
        opacity: Math.min(1, p * 1.4),
        transform: `translateX(${(1 - p) * -24}px)`,
        ...style,
      }}
    >
      <div style={{ position: "absolute", left: 0, top: 12, bottom: 12, width: 4, borderRadius: 2, background: C.nv, opacity: 0.5 + 0.5 * active, boxShadow: `0 0 ${12 * active}px ${C.nv}` }} />
      <div style={{ fontSize: 25, fontWeight: 600, lineHeight: 1.15 }}>{title}</div>
      <div style={{ fontFamily: MONO, fontSize: 19, color: C.dim, marginTop: 5 }}>{sub}</div>
    </div>
  )
}

// A schematic of the GPU at work: rows of cores that light in a wave as audio frames arrive.
function GpuCores({ at, level, top }: { at: number; level: number; top: number }) {
  const f = useCurrentFrame()
  const cols = 24
  const rows = 6
  const on = ramp(f, at, at + 16)
  return (
    <div style={{ position: "absolute", left: 22, top, display: "grid", gridTemplateColumns: `repeat(${cols}, 12px)`, gap: 3, opacity: on }}>
      {Array.from({ length: cols * rows }, (_, i) => {
        const c = i % cols
        const r = Math.floor(i / cols)
        const wave = 0.5 + 0.5 * Math.sin((c - f * 0.9) / 3.2 + r * 0.7)
        const v = Math.pow(wave, 3) * (0.45 + 0.55 * level)
        return <div key={i} style={{ width: 12, height: 12, borderRadius: 3, background: C.nv, opacity: 0.1 + 0.8 * v, boxShadow: v > 0.55 ? `0 0 8px ${C.nv}` : undefined }} />
      })}
    </div>
  )
}

// Words (text, never audio) travelling along a path as pills, one after another: each takes `travel`
// frames, a new one leaves every `every` frames (only one on the path at a time when every >= travel).
// `once` sends just the first. Drawn above the cards, fading in as it leaves and out as it docks.
function WordStream({ d, words, start, travel, every = travel, once = false, size = 22, color = C.violet, caption }: { d: string; words: string[]; start: number; travel: number; every?: number; once?: boolean; size?: number; color?: string; caption?: string }) {
  const f = useCurrentFrame()
  const len = useMemo(() => getLength(d), [d])
  const t = f - start
  if (t < 0 || !words.length) return null
  const sent = once ? 1 : Math.floor(t / every) + 1
  const pills: ReactNode[] = []
  for (let k = Math.max(0, sent - Math.ceil(travel / every) - 1); k < sent; k++) {
    const u = (t - k * every) / travel
    if (u < 0 || u > 1) continue
    const w = words[k % words.length]
    const wpx = 28 + w.length * size * 0.6
    const h = size + 18
    // the pill leaves with its left edge at the sending card and docks with its right edge just over
    // the receiving card's border, so it never covers either card's text
    const s0 = Math.min(len / 2, wpx / 2 - 8)
    const s1 = Math.max(len / 2, len - wpx / 2 + 16)
    const p = getPointAtLength(d, s0 + (0.5 * u + 0.5 * easeInOut(u)) * (s1 - s0))
    if (!p) continue
    const o = Math.min(1, u / 0.14, (1 - u) / 0.14)
    // the caption leaves before the pill docks, so it never runs over the receiving card
    const co = Math.max(0, Math.min(1, u / 0.14, (0.8 - u) / 0.12))
    pills.push(
      <g key={k} opacity={o} transform={`translate(${p.x} ${p.y}) scale(${0.85 + 0.15 * Math.min(1, u / 0.14)})`}>
        <rect x={-wpx / 2} y={-h / 2} width={wpx} height={h} rx={h / 2} fill="rgba(12,10,28,0.95)" stroke={color} strokeWidth={2} style={{ filter: `drop-shadow(0 0 10px ${color}99)` }} />
        <text x={0} y={size * 0.36} textAnchor="middle" fontFamily={MONO} fontSize={size} fontWeight={600} fill={C.text}>
          {w}
        </text>
        {caption && co > 0.001 && (
          <text opacity={co} x={0} y={-h / 2 - 10} textAnchor="middle" fontFamily={MONO} fontSize={19} fill={color} stroke="rgba(12,10,28,0.92)" strokeWidth={6} paintOrder="stroke">
            {caption}
          </text>
        )}
      </g>,
    )
  }
  return <g>{pills}</g>
}

const easeInOut = (u: number) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, u)))

// A small violet label on a dark plate, so the border's dashes don't run through its letters.
function PathTag({ x, y, text, on }: { x: number; y: number; text: string; on: number }) {
  if (on < 0.01) return null
  const w = 22 + text.length * 19 * 0.6
  return (
    <g opacity={on} transform={`translate(${x} ${y + (1 - on) * 8})`}>
      <rect x={-w / 2} y={-15} width={w} height={30} rx={8} fill="rgba(12,10,28,0.94)" stroke={C.violet} strokeOpacity={0.35} strokeWidth={1} />
      <text x={0} y={6.5} textAnchor="middle" fontFamily={MONO} fontSize={19} fill={C.violet}>
        {text}
      </text>
    </g>
  )
}

// The one thing that comes back: a single violet audio frame, Magpie's spoken word, into the call.
function ReturnPacket({ d, start, travel }: { d: string; start: number; travel: number }) {
  const f = useCurrentFrame()
  const len = useMemo(() => getLength(d), [d])
  const u = (f - start) / travel
  if (u < 0 || u > 1) return null
  const p = getPointAtLength(d, easeInOut(u) * len)
  if (!p) return null
  const o = Math.min(1, u / 0.12, (1 - u) / 0.12)
  return (
    <g opacity={o} transform={`translate(${p.x} ${p.y})`}>
      <circle r={26} fill={C.violet} opacity={0.18} />
      <rect x={-12} y={-12} width={24} height={24} rx={5} fill={C.violet} style={{ filter: `drop-shadow(0 0 12px ${C.violet})` }} />
      {[-6, 0, 6].map((x, i) => (
        <line key={i} x1={x} x2={x} y1={i === 1 ? -6 : -3} y2={i === 1 ? 6 : 3} stroke="#140f24" strokeWidth={2.4} strokeLinecap="round" />
      ))}
    </g>
  )
}

// ---- the scene ---------------------------------------------------------------------------------
export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const L = lineCue(scene, 0)
  // a beat a fraction of the way through the line (the Magpie narration says: "laptop" at ~0.385,
  // "Only" ~0.52, "text" ~0.555, "cloud" ~0.71, "Nemotron" ~0.80, "Magpie" ~0.92)
  const b = (p: number) => Math.round(L.at + p * L.frames)
  const end = L.at + L.frames

  // the beats
  const tLap = 0
  const tNote = b(0.0)
  const tMic = b(0.03)
  const tTeach = b(0.045)
  const tGpu = b(0.06)
  const tCw = b(0.17)
  const tSpl = b(0.22)
  const tNemo = b(0.27)
  const tVmic = b(0.32)
  const tLaptop = b(0.385) // "...on the laptop."
  const tPull = tLaptop - 4 // the whole laptop comes into view as it is named
  const tBorder = b(0.52) // "Only text"
  const tText = tBorder + 8 // the first words leave as "Only text goes..." is said
  const tCloud = b(0.705) // "...goes to the cloud"
  const tNeb = b(0.8) // "Nemotron"
  const tMag = b(0.92) // "Magpie"
  const tTap = tMag - 10 // the tapped word leaves for Magpie
  const TAP_TRAVEL = 24
  const tBack = tTap + TAP_TRAVEL // Magpie has it: the spoken word comes back
  const BACK_TRAVEL = 22 // lands with time to settle before the scene's closing transition
  const tArrive = tBack + BACK_TRAVEL

  // Alex's voice: the real clip's loudness, looped, drives the ports and the GPU glow
  const env = d.clip.env_before
  const lvl = (k: number) => {
    const i = Math.floor((f * 3.3 + k) % env.length)
    return Math.min(1, (env[i] ?? 0) * 1.6)
  }
  const voice = lvl(0)

  const draw = (at: number, dur = 14) => ramp(f, at, at + dur, easeOut)
  const flowOn = (at: number) => ramp(f, at, at + 10)

  // the laptop's frame draws itself, then fills
  const lapPath = roundedRect(LAP.x, LAP.y, LAP.w, LAP.h, 34)
  const lapDraw = ramp(f, tLap, tLap + 24, easeOut)
  const lapEv = evolvePath(lapDraw, lapPath)
  const lapFill = ramp(f, tLap + 6, tLap + 26)
  const lapLit = ramp(f, tLaptop, tLaptop + 10) * (1 - 0.6 * ramp(f, tLaptop + 30, tLaptop + 60))

  const cloudP = springAt(f, tCloud, SNAPPY)
  const cloudPath = roundedRect(CLOUD.x, CLOUD.y, CLOUD.w, CLOUD.h, 34)

  const borderDraw = ramp(f, tBorder, tBorder + 22, easeOut)
  const gateP = springAt(f, tBorder + 8, SNAPPY)

  const gpuActive = ramp(f, tGpu + 12, tGpu + 24) * (0.75 + 0.25 * voice)
  const nebActive = ramp(f, tNeb + 8, tNeb + 22)
  // empty slots in the cloud where the words dock until Nemotron and Magpie fill them
  const slotOn = ramp(f, tCloud, tCloud + 14)
  const nebSlot = slotOn * (1 - ramp(f, tNeb, tNeb + 10))
  const magSlot = slotOn * (1 - ramp(f, tMag, tMag + 10))
  // the virtual mic lights up when the spoken word arrives
  const vmicPulse = ramp(f, tArrive - 4, tArrive + 4) * (1 - 0.6 * ramp(f, tArrive + 14, tArrive + 34))
  // the hero push: a focus pull dims the left column, which the tighter frame crops
  const push = ramp(f, tBorder + 4, tBorder + 20) * (1 - ramp(f, tNeb - 6, tNeb + 14))
  const leftDim = 1 - 0.65 * push
  const tapP = springAt(f, tTap - 6, SNAPPY)

  // the guard sees only the words marked for cutting (text); word rescue the caption and the stuck sound
  const guardWords = d.clip.words.filter((w) => w.removed).map((w) => w.w).filter((w, i, a) => a.indexOf(w) === i).slice(0, 2)
  const said = d.rescue.said.replace(/[.,]/g, "").split(" ")
  const rescueWords = [said[said.length - 1], `${d.rescue.onset}–`]

  return (
    <AbsoluteFill>
      <Camera
        shots={[
          { at: 0, x: 690, y: 528, zoom: 1.05 },
          { at: tPull, x: 960, y: 525, zoom: 1.0, dur: 28 },
          // the hero shot: push in on the border while "Only text goes to the cloud" plays. Framed so
          // the teacher and GPU cards stay whole, their text >= 96 px from the left edge (the title and
          // the ports fade as they slide out), and the laptop's bottom edge stays above the subtitles
          // (stage y 864 -> ~899)
          { at: tBorder + 4, x: 1150, y: 528, zoom: 1.07, dur: 26 },
          // pull back to reveal Nemotron and Magpie
          { at: tNeb - 6, x: 960, y: 525, zoom: 1.0, dur: 30 },
          { at: end, x: 960, y: 522, zoom: 1.02, dur: Math.max(30, scene.frames - end) },
        ]}
        drift={5}
        seed="wir"
      >
        {/* light */}
        <GlowOrb x={GPU.x + GPU.w / 2} y={GPU.y + GPU.h / 2} r={420} color={C.nv} opacity={0.16 * ramp(f, tGpu, tGpu + 20) * (0.8 + 0.2 * voice)} />
        <GlowOrb x={620} y={480} r={620} color={C.sky} opacity={0.09 * lapFill} />
        <GlowOrb x={CLOUD.x + CLOUD.w / 2} y={430} r={460} color={C.violet} opacity={0.2 * Math.min(1, cloudP)} />

        {/* Alex's laptop */}
        <div
          style={{
            position: "absolute",
            left: LAP.x,
            top: LAP.y,
            width: LAP.w,
            height: LAP.h,
            borderRadius: 34,
            opacity: lapFill,
            background: `radial-gradient(900px 600px at 30% 20%, rgba(125,211,252,${0.07 + 0.05 * lapLit}), transparent 70%), linear-gradient(170deg, rgba(255,255,255,0.045), rgba(255,255,255,0.012))`,
            boxShadow: `inset 0 0 ${60 + 60 * lapLit}px rgba(125,211,252,${0.05 + 0.1 * lapLit}), 0 40px 100px -40px rgba(0,0,0,0.9)`,
            overflow: "hidden",
          }}
        >
          {/* a faint dot grid: the board everything sits on */}
          <div style={{ position: "absolute", inset: 0, backgroundImage: "radial-gradient(rgba(125,211,252,0.16) 1.2px, transparent 1.4px)", backgroundSize: "30px 30px", opacity: 0.55, maskImage: "radial-gradient(ellipse 80% 80% at 40% 45%, black, transparent)" }} />
          <LightSweep at={tLaptop} dur={34} color="rgba(125,211,252,0.16)" angle={22} />
        </div>
        {/* the title leaves the frame during the hero push: it fades as it slides out, never sliced */}
        <div style={{ position: "absolute", left: LAP.x + 40, top: LAP.y + 18, display: "flex", alignItems: "center", gap: 18, opacity: ramp(f, tLap + 4, tLap + 18) * (1 - push), transform: `translateY(${(1 - springAt(f, tLap + 4)) * 20}px)` }}>
          <Icon name="laptop" size={46} color={C.sky} />
          <div style={{ fontFamily: SERIF, fontSize: 60, lineHeight: 1, letterSpacing: -1, color: C.text }}>Alex's laptop</div>
        </div>
        <div
          style={{
            position: "absolute",
            right: 1920 - (LAP.x + LAP.w) + 40,
            top: LAP.y + 34,
            fontFamily: MONO,
            fontSize: 20,
            letterSpacing: 3,
            color: C.sky,
            opacity: Math.min(1, springAt(f, tLaptop)),
            transform: `translateY(${(1 - springAt(f, tLaptop)) * 14}px)`,
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          <span style={{ width: 10, height: 10, borderRadius: 5, background: C.sky, boxShadow: `0 0 12px ${C.sky}` }} />
          HIS VOICE STAYS HERE
        </div>

        {/* the cloud */}
        <div
          style={{
            position: "absolute",
            left: CLOUD.x,
            top: CLOUD.y,
            width: CLOUD.w,
            height: CLOUD.h,
            borderRadius: 34,
            opacity: Math.min(1, cloudP),
            background: "radial-gradient(500px 500px at 60% 30%, rgba(196,181,253,0.09), transparent 70%), rgba(196,181,253,0.025)",
          }}
        />
        <div style={{ position: "absolute", left: CLOUD.x + 30, top: CLOUD.y + 18, display: "flex", alignItems: "center", gap: 16, opacity: Math.min(1, cloudP * 1.3), transform: `translateY(${(1 - cloudP) * 20}px)` }}>
          <Icon name="cloud" size={44} color={C.violet} />
          <div style={{ fontFamily: SERIF, fontSize: 60, lineHeight: 1, letterSpacing: -1 }}>The cloud</div>
        </div>

        {/* paths and what travels on them */}
        <Diagram>
          {/* the laptop's frame */}
          <path d={lapPath} fill="none" stroke={C.sky} strokeOpacity={0.12 + 0.25 * lapLit} strokeWidth={8} strokeDasharray={lapEv.strokeDasharray} strokeDashoffset={lapEv.strokeDashoffset} />
          <path d={lapPath} fill="none" stroke={C.sky} strokeOpacity={0.55 + 0.35 * lapLit} strokeWidth={1.8} strokeDasharray={lapEv.strokeDasharray} strokeDashoffset={lapEv.strokeDashoffset} />
          {/* the cloud's frame */}
          <path d={cloudPath} fill="none" stroke={C.violet} strokeOpacity={0.5 * Math.min(1, cloudP)} strokeWidth={1.6} strokeDasharray="10 9" strokeDashoffset={-f * 0.4} />

          {/* audio frames: Alex's voice, inside the laptop only */}
          <FlowPath d={A_NOTE_TEACH} draw={draw(tTeach + 4)} flow={flowOn(tTeach + 14)} color={C.sky} kind="frame" packets={2} period={36} />
          <FlowPath d={A_MIC_GPU} draw={draw(tGpu + 6)} flow={flowOn(tGpu + 16)} color={C.sky} kind="frame" packets={2} period={32} />
          <FlowPath d={A_NOTE_CW} draw={draw(tCw - 6, 18)} flow={flowOn(tCw + 8)} color={C.sky} kind="frame" packets={6} period={110} />
          <FlowPath d={A_TEACH_SPL} draw={draw(tSpl - 2)} flow={flowOn(tSpl + 8)} color={C.sky} kind="frame" packets={2} period={40} />
          <FlowPath d={A_GPU_SPL} draw={draw(tSpl)} flow={flowOn(tSpl + 10)} color={C.sky} kind="frame" packets={2} period={34} />
          <FlowPath d={A_CW_SPL} draw={draw(tSpl + 2)} flow={flowOn(tSpl + 12)} color={C.sky} kind="frame" packets={1} period={30} />
          <FlowPath d={A_SPL_NEMO} draw={draw(tNemo)} flow={flowOn(tNemo + 10)} color={C.teal} kind="frame" packets={1} period={28} />
          <FlowPath d={A_SPL_VMIC} draw={draw(tVmic - 4, 18)} flow={flowOn(tVmic + 8)} color={C.teal} kind="frame" packets={4} period={80} />

          {/* the border: only text crosses it */}
          {borderDraw > 0.001 && (
            <>
          <line x1={BX} y1={LAP.y + 8} x2={BX} y2={LAP.y + 8 + (LAP.h - 16) * borderDraw} stroke={C.violet} strokeOpacity={0.18} strokeWidth={12} strokeLinecap="round" />
          <line x1={BX} y1={LAP.y + 8} x2={BX} y2={LAP.y + 8 + (LAP.h - 16) * borderDraw} stroke={C.violet} strokeWidth={2.4} strokeDasharray="3 11" strokeDashoffset={-f * 0.7} strokeLinecap="round" />
          {borderDraw < 1 && <circle cx={BX} cy={LAP.y + 8 + (LAP.h - 16) * borderDraw} r={7} fill={C.violet} style={{ filter: `drop-shadow(0 0 10px ${C.violet})` }} />}
            </>
          )}

          {/* the cloud's empty slots: where the words dock until Nemotron and Magpie arrive */}
          {nebSlot > 0.01 && <path d={roundedRect(NEB.x, NEB.y, NEB.w, NEB.h, 22)} fill="none" stroke={C.violet} strokeOpacity={0.25 * nebSlot} strokeWidth={1.6} strokeDasharray="6 8" />}
          {magSlot > 0.01 && <path d={roundedRect(MAG.x, MAG.y, MAG.w, MAG.h, 22)} fill="none" stroke={C.violet} strokeOpacity={0.25 * magSlot} strokeWidth={1.6} strokeDasharray="6 8" />}

          {/* word paths: text only, to Nemotron and Magpie (the pills ride above the cards, below) */}
          <FlowPath d={T_CW_NEB} draw={draw(tText, 18)} color={C.violet} width={2.5} />
          <FlowPath d={T_NEMO_NEB} draw={draw(tText + 4, 18)} color={C.violet} width={2.5} />
          <FlowPath d={T_NEMO_MAG} draw={draw(tTap - 6, 14)} color={C.violet} width={2.5} />
          {/* and the one thing that comes back: the word, spoken */}
          <FlowPath d={R_MAG_VMIC} draw={draw(tBack - 6, 12)} color={C.violet} width={3} dashed />
          {/* the spoken word lands in the call: a ring bursts from behind the virtual mic */}
          {f >= tArrive - 2 && f < tArrive + 18 && (
            <circle cx={VMIC.x + VMIC.w} cy={790} r={14 + 70 * ramp(f, tArrive - 2, tArrive + 18, easeOut)} fill="none" stroke={C.violet} strokeWidth={2.5} opacity={1 - ramp(f, tArrive - 2, tArrive + 18)} />
          )}
          {[330, 520].map((y) => (
            <circle key={y} cx={NEB.x} cy={y} r={6} fill={C.violet} opacity={slotOn * (0.6 + 0.4 * Math.sin(f / 6 + y))} style={{ filter: `drop-shadow(0 0 8px ${C.violet})` }} />
          ))}
        </Diagram>
        <GlowOrb x={NEB.x} y={330} r={110} color={C.violet} opacity={0.35 * slotOn} />
        <GlowOrb x={NEB.x} y={520} r={110} color={C.violet} opacity={0.35 * slotOn} />

        {/* the left column (inputs, teacher, GPU): dimmed by the focus pull during the hero push */}
        <div style={{ position: "absolute", inset: 0, opacity: leftDim }}>
        {/* inputs: they leave the frame during the hero push, fading as they slide out */}
        <div style={{ position: "absolute", inset: 0, opacity: 1 - push }}>
          <Port x={NOTE_PORT.x} y={NOTE_PORT.y} icon="video" label={"voice note\nor video"} at={tNote} level={lvl(40) * ramp(f, tNote, tNote + 20)} />
          <Port x={MIC_PORT.x} y={MIC_PORT.y} icon="mic" label={"mic · 16 kHz\nin a call"} at={tMic} level={voice * ramp(f, tMic, tMic + 20)} />
        </div>

        {/* the recordings teacher: its own block, no device named */}
        <Panel box={TEACH} at={tTeach} accent={C.sky} glow={0.35 * ramp(f, tTeach + 14, tTeach + 26) * lvl(40)} style={{ padding: "16px 22px" }}>
          <Head icon="wave" title="Recordings teacher" accent={C.sky} sub="WavLM v2 · every 20 ms" />
        </Panel>

        {/* the GPU: only the live model */}
        <Panel box={GPU} at={tGpu} accent={C.nv} glow={gpuActive}>
          <div style={{ padding: "20px 22px" }}>
            <Head icon="chip" title="NVIDIA GPU · CUDA" accent={C.nv} sub="GeForce RTX 4050" size={28} />
          </div>
          <ModelRow title="WavLM-Large · live" sub="316M · ~60 ms a window" at={tGpu + 6} active={ramp(f, tGpu + 14, tGpu + 24) * voice} style={{ top: 96 }} />
          <GpuCores at={tGpu + 10} level={voice} top={206} />
          <div style={{ position: "absolute", left: 22, top: 306, fontFamily: MONO, fontSize: 19, color: C.dim, opacity: ramp(f, tGpu + 12, tGpu + 22) }}>4 s window · every 160 ms</div>
          <div style={{ position: "absolute", left: 22, bottom: 20, display: "flex", gap: 10, alignItems: "center", opacity: ramp(f, tGpu + 12, tGpu + 22) }}>
            <NV />
            <span style={{ fontFamily: MONO, fontSize: 19, color: C.dim }}>on the laptop</span>
          </div>
        </Panel>
        </div>

        {/* the right column */}
        <Panel box={CW} at={tCw} accent={C.sky} style={{ padding: "16px 22px" }}>
          <Head icon="text" title="CrisperWhisper" accent={C.sky} sub="verbatim words · timings" />
        </Panel>
        <Panel box={SPL} at={tSpl} accent={C.sky} glow={0.35 * ramp(f, tSpl + 12, tSpl + 24) * voice} style={{ padding: "16px 22px" }}>
          <Head icon="scissors" title="Splicer" accent={C.sky} sub="room tone · call or file" />
        </Panel>
        <Panel box={NEMO} at={tNemo} accent={C.nv} style={{ padding: "16px 22px" }}>
          <Head icon="captions" title="NeMo captions" accent={C.nv} sub="FastConformer · int8 ONNX" subColor={C.nv} />
          <div style={{ display: "flex", gap: 10, marginTop: 14, alignItems: "center" }}>
            <NV />
            <span style={{ fontFamily: MONO, fontSize: 19, color: C.dim }}>on the CPU</span>
          </div>
        </Panel>
        <Panel box={VMIC} at={tVmic} accent={vmicPulse > 0.05 ? C.violet : C.teal} glow={Math.max(0.3 * ramp(f, tVmic + 12, tVmic + 24) * voice, 0.9 * vmicPulse)} style={{ padding: "16px 22px" }}>
          <Head icon="headphones" title="Virtual mic → the call" accent={C.teal} sub="Zoom · Meet · Teams" />
        </Panel>

        {/* Alex's tap: the "Say it" button in his app, where the tapped word starts */}
        {f >= tTap - 7 && (
          <div
            style={{
              position: "absolute",
              left: TAP.x,
              top: TAP.y,
              width: TAP.w,
              height: TAP.h,
              borderRadius: TAP.h / 2,
              boxSizing: "border-box",
              background: C.violet,
              boxShadow: `0 0 22px -2px ${C.violet}cc`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontFamily: MONO,
              fontSize: 19,
              fontWeight: 700,
              color: "#140f24",
              whiteSpace: "nowrap",
              opacity: Math.min(1, tapP * 1.5),
              transform: `scale(${0.6 + 0.4 * tapP})`,
            }}
          >
            Say it · tap
          </div>
        )}

        {/* Nemotron on Nebius Token Factory */}
        <Panel box={NEB} at={tNeb} accent={C.violet} glow={0.55 * nebActive * (0.8 + 0.2 * Math.sin(f / 7))} style={{ padding: "20px 22px" }}>
          <Head icon="cloud" title="Nebius Token Factory" accent={C.violet} size={27} />
          <div style={{ marginTop: 16, fontFamily: MONO, fontSize: 23, fontWeight: 700, color: C.nv }}>NVIDIA Nemotron 3 Super</div>
          <div style={{ marginTop: 4, fontFamily: MONO, fontSize: 19, color: C.dim }}>120B · 12B active per token</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 18 }}>
            {["meaning guard", "word rescue", "context card", "insights", "practice interview"].map((j, i) => (
              <Chip key={j} color={C.violet} at={tNeb + 6 + i * 4}>
                {j}
              </Chip>
            ))}
          </div>
          <div style={{ position: "absolute", left: 22, bottom: 20, display: "flex", gap: 10, alignItems: "center" }}>
            <NV />
            <span style={{ fontFamily: MONO, fontSize: 19, color: C.violet }}>text in · text out</span>
          </div>
        </Panel>

        {/* Magpie on build.nvidia.com */}
        <Panel box={MAG} at={tMag} accent={C.violet} glow={0.5 * ramp(f, tBack - 4, tBack + 4) * (1 - 0.6 * ramp(f, tBack + 24, tBack + 44))} style={{ padding: "20px 22px" }}>
          <Head icon="speaker" title="build.nvidia.com" accent={C.violet} size={28} />
          <div style={{ marginTop: 14, fontFamily: MONO, fontSize: 23, fontWeight: 700, color: C.nv }}>NVIDIA Magpie TTS</div>
          <div style={{ display: "flex", gap: 12, marginTop: 14, alignItems: "center" }}>
            <Chip color="#140f24" fill={C.violet} at={tMag + 6} style={{ fontWeight: 700 }}>
              Say it
            </Chip>
            <span style={{ fontFamily: MONO, fontSize: 19, color: C.dim }}>the tapped word, synthetic voice</span>
          </div>
        </Panel>


        {/* the words themselves, above the cards: text crossing the border */}
        <Diagram>
          <WordStream d={T_CW_NEB} words={guardWords} start={tText + 8} travel={44} every={48} />
          <WordStream d={T_NEMO_NEB} words={rescueWords} start={tText + 30} travel={44} every={48} />
          <WordStream d={T_NEMO_MAG} words={[d.say.word]} start={tTap} travel={TAP_TRAVEL} once caption="tapped word" />
          <ReturnPacket d={R_MAG_VMIC} start={tBack} travel={BACK_TRAVEL} />
          <PathTag x={1288} y={846} text="synthetic voice" on={ramp(f, tBack, tBack + 10)} />
        </Diagram>

        {/* what the shapes mean */}
        <div style={{ position: "absolute", left: LAP.x + 30, top: LAP.y + LAP.h - 104, display: "grid", gridTemplateColumns: "auto auto", alignItems: "center", columnGap: 12, rowGap: 12, fontFamily: MONO, fontSize: 19, color: C.dim, opacity: ramp(f, tBorder + 8, tBorder + 20) * (1 - push), transform: `translateY(${(1 - springAt(f, tBorder + 8)) * 12}px)` }}>
          <span style={{ justifySelf: "center", width: 14, height: 14, borderRadius: 3, background: C.sky, boxShadow: `0 0 8px ${C.sky}` }} />
          <span>audio</span>
          <span style={{ border: `1.5px solid ${C.violet}`, borderRadius: 999, padding: "1px 10px", color: C.text, background: "rgba(10,14,26,0.92)" }}>word</span>
          <span>text</span>
        </div>

        {/* the gate label on the border */}
        <div
          style={{
            position: "absolute",
            left: BX - 88,
            top: 418,
            width: 176,
            height: 46,
            borderRadius: 23,
            background: "rgba(12,10,28,0.92)",
            border: `1.5px solid ${C.violet}`,
            boxShadow: `0 0 30px -4px ${C.violet}aa`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            fontFamily: MONO,
            fontSize: 21,
            fontWeight: 700,
            letterSpacing: 1,
            color: C.violet,
            opacity: Math.min(1, gateP * 1.5),
            transform: `scale(${0.6 + 0.4 * gateP})`,
          }}
        >
          <Icon name="text" size={20} />
          text only
        </div>

        {/* a few motes: sky in the laptop, violet in the cloud */}
        <Particles count={16} color={C.sky} seed="wir-lap" x={LAP.x + 40} y={LAP.y + 80} w={LAP.w - 80} h={LAP.h - 120} opacity={0.35 * lapFill} rise={0.35} />
        <Particles count={10} color={C.violet} seed="wir-cloud" x={CLOUD.x + 20} y={CLOUD.y + 60} w={CLOUD.w - 40} h={CLOUD.h - 100} opacity={0.4 * Math.min(1, cloudP)} rise={0.3} />
      </Camera>

      {/* the push's crop edge: the cut-off left column fades out instead of being sliced */}
      {push > 0.01 && (
        <div style={{ position: "absolute", left: 0, top: 0, width: 260, height: 1080, opacity: push, background: "linear-gradient(90deg, rgba(6,9,18,0.85), rgba(6,9,18,0))", pointerEvents: "none" }} />
      )}

      <Sfx name="pop" at={tTap - 6} volume={0.18} />
      <Sfx name="pop" at={tNeb} volume={0.22} />
      <Sfx name="pop" at={tMag} volume={0.22} />
    </AbsoluteFill>
  )
}
