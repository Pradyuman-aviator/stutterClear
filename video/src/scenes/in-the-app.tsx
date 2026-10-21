import type { CSSProperties, ReactNode } from "react"
import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from "remotion"
import { noise2D } from "@remotion/noise"

import type { Cue, SceneProps } from "../data"
import { lineCue } from "../data"
import { Camera, clamp, Diagram, ease, easeIn, easeOut, FlowPath, GlowOrb, Icon, LightSweep, Particles, ramp, SNAPPY, SOFT, springAt } from "../kit"
import { C, MONO, SANS, SERIF } from "../theme"

// It's all in the app: a how-to on the real app. The Calls screenshot tilts in, its tab bar lifts out
// as the scene's navigation, the window pans to each control the narration names (mode, output device,
// the meeting app's microphone), then the tabs slide through as a coverflow of real screens.

const IMG_W = 1680
const IMG_H = 1050
const CHROME = 44
const URL = "127.0.0.1:8765"

// --- timing helpers -------------------------------------------------------------------------------

// The frame at which a word is said inside a narration line, by its share of the line's characters.
function wordAt(cue: Cue, word: string, lead = 0): number {
  const text = (cue.text ?? "").toLowerCase()
  const i = text.indexOf(word.toLowerCase())
  const share = i < 0 ? 0 : i / Math.max(1, text.length)
  return Math.round(cue.at + share * cue.frames + lead)
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t

// --- the app window -------------------------------------------------------------------------------

interface View {
  cx: number // image point at the viewport's centre
  cy: number
  s: number // image scale
}
interface Pose {
  x: number // window centre on the stage
  y: number
  w: number // viewport size (the window adds its chrome bar on top)
  h: number
  rx: number
  ry: number
  sc: number
  op: number
  view: View
}

const lerpView = (a: View, b: View, t: number): View => ({
  cx: lerp(a.cx, b.cx, t),
  cy: lerp(a.cy, b.cy, t),
  s: Math.exp(lerp(Math.log(a.s), Math.log(b.s), t)),
})
const lerpPose = (a: Pose, b: Pose, t: number): Pose => ({
  x: lerp(a.x, b.x, t),
  y: lerp(a.y, b.y, t),
  w: lerp(a.w, b.w, t),
  h: lerp(a.h, b.h, t),
  rx: lerp(a.rx, b.rx, t),
  ry: lerp(a.ry, b.ry, t),
  sc: lerp(a.sc, b.sc, t),
  op: lerp(a.op, b.op, t),
  view: lerpView(a.view, b.view, t),
})

// A box on the screenshot (image pixels) ringed in the window: drawn on with `p`, faded with `o`.
interface Ring {
  box: [number, number, number, number] // x0, y0, x1, y1 in image pixels
  p: number
  o: number
  color?: string
  label?: string
  labelBelow?: boolean
  labelRight?: boolean // align the label's right edge with the ring's (a ring near the window's right side)
  pulse?: number // 0..1 a single expanding echo
}

function toViewport(view: View, w: number, h: number, x: number, y: number) {
  return { x: (x - view.cx) * view.s + w / 2, y: (y - view.cy) * view.s + h / 2 }
}

// Where an image point lands on the stage, through the window's 3D transform (CSS perspective, then
// rotateX, rotateY, scale about the window's centre).
function projectToStage(pose: Pose, ix: number, iy: number) {
  const total = pose.h + CHROME
  const v = toViewport(pose.view, pose.w, pose.h, ix, iy)
  const px = (v.x - pose.w / 2) * pose.sc
  const py = (v.y + CHROME - total / 2) * pose.sc
  const ry = (pose.ry * Math.PI) / 180
  const rx = (pose.rx * Math.PI) / 180
  const x1 = px * Math.cos(ry)
  const z1 = -px * Math.sin(ry)
  const y2 = py * Math.cos(rx) - z1 * Math.sin(rx)
  const z2 = py * Math.sin(rx) + z1 * Math.cos(rx)
  const k = 2400 / (2400 - z2)
  return { x: pose.x + x1 * k, y: pose.y + y2 * k }
}

// A box of the screenshot (image pixels) painted over inside the window's viewport.
function ImgBox({
  view,
  w,
  h,
  box,
  style,
  children,
}: {
  view: View
  w: number
  h: number
  box: [number, number, number, number]
  style?: CSSProperties
  children?: ReactNode
}) {
  const a = toViewport(view, w, h, box[0], box[1])
  const b = toViewport(view, w, h, box[2], box[3])
  return (
    <div style={{ position: "absolute", left: a.x, top: a.y, width: b.x - a.x, height: b.y - a.y, overflow: "hidden", ...style }}>
      {children}
    </div>
  )
}

// The screenshots' own small tab row (the big rail above is the scene's navigation) and page colour.
const TAB_ROW: [number, number, number, number] = [28, 84, 478, 130]
const PAGE_BG = "rgb(7,9,16)"

function AppWindow({
  src,
  pose,
  rings = [],
  patches,
  overlay,
  sweepAt,
  dim = 0,
  z = 1,
}: {
  src: string
  pose: Pose
  rings?: Ring[]
  patches?: ReactNode
  overlay?: ReactNode
  sweepAt?: number
  dim?: number
  z?: number
}) {
  const { w, h, view } = pose
  if (pose.op <= 0.002) return null
  const total = h + CHROME
  return (
    <div
      style={{
        position: "absolute",
        left: pose.x - w / 2,
        top: pose.y - total / 2,
        width: w,
        height: total,
        opacity: pose.op,
        zIndex: z,
        transform: `perspective(2400px) rotateX(${pose.rx}deg) rotateY(${pose.ry}deg) scale(${pose.sc})`,
        transformOrigin: "50% 50%",
      }}
    >
      <div
        style={{
          width: w,
          height: total,
          borderRadius: 22,
          overflow: "hidden",
          border: `1px solid ${C.border2}`,
          background: "#070a12",
          boxShadow: `0 60px 140px -40px rgba(0,0,0,0.95), 0 0 0 1px rgba(255,255,255,0.03), 0 0 90px -30px rgba(125,211,252,${0.35 * (1 - dim)})`,
        }}
      >
        <div
          style={{
            height: CHROME,
            display: "flex",
            alignItems: "center",
            gap: 9,
            padding: "0 18px",
            background: "linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03))",
            borderBottom: `1px solid ${C.border}`,
          }}
        >
          {[0, 1, 2].map((i) => (
            <span key={i} style={{ width: 12, height: 12, borderRadius: 6, background: "rgba(255,255,255,0.2)" }} />
          ))}
          <span
            style={{
              marginLeft: 18,
              padding: "4px 16px",
              borderRadius: 999,
              background: "rgba(255,255,255,0.05)",
              fontFamily: MONO,
              fontSize: 18,
              color: C.dim,
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <Icon name="lock" size={16} color={C.green} />
            {URL}
          </span>
        </div>
        <div style={{ position: "relative", width: w, height: h, overflow: "hidden", background: "#05070d" }}>
          <Img
            src={staticFile(src)}
            style={{
              position: "absolute",
              left: w / 2 - view.cx * view.s,
              top: h / 2 - view.cy * view.s,
              width: IMG_W * view.s,
              height: IMG_H * view.s,
            }}
          />
          {patches}
          {rings.map((r, i) => (
            <RingMark key={i} ring={r} view={view} w={w} h={h} />
          ))}
          {overlay}
          {sweepAt !== undefined && <LightSweep at={sweepAt} dur={40} color="rgba(186,230,253,0.16)" />}
          {dim > 0.01 && <div style={{ position: "absolute", inset: 0, background: `rgba(3,5,10,${0.62 * dim})` }} />}
        </div>
      </div>
    </div>
  )
}

function RingMark({ ring, view, w, h }: { ring: Ring; view: View; w: number; h: number }) {
  if (ring.o <= 0.002 || ring.p <= 0.002) return null
  const pad = 9
  const a = toViewport(view, w, h, ring.box[0], ring.box[1])
  const b = toViewport(view, w, h, ring.box[2], ring.box[3])
  const x = a.x - pad
  const y = a.y - pad
  const rw = b.x - a.x + pad * 2
  const rh = b.y - a.y + pad * 2
  const color = ring.color ?? C.sky
  const pulse = ring.pulse ?? 0
  return (
    <div style={{ position: "absolute", inset: 0, opacity: ring.o, pointerEvents: "none" }}>
      <svg width={w} height={h} style={{ position: "absolute", left: 0, top: 0, overflow: "visible" }}>
        <rect x={x} y={y} width={rw} height={rh} rx={14} fill={color} fillOpacity={0.07 * Math.min(1, ring.p * 2)} />
        <rect
          x={x}
          y={y}
          width={rw}
          height={rh}
          rx={14}
          fill="none"
          stroke={color}
          strokeWidth={3}
          pathLength={1}
          strokeDasharray="1 1"
          strokeDashoffset={1 - Math.min(1, ring.p)}
          style={{ filter: `drop-shadow(0 0 8px ${color})` }}
        />
        {pulse > 0 && pulse < 1 && (
          <rect
            x={x - pulse * 16}
            y={y - pulse * 16}
            width={rw + pulse * 32}
            height={rh + pulse * 32}
            rx={14 + pulse * 10}
            fill="none"
            stroke={color}
            strokeWidth={2}
            opacity={0.7 * (1 - pulse)}
          />
        )}
      </svg>
      {ring.label && (
        <div
          style={{
            position: "absolute",
            ...(ring.labelRight ? { right: w - (x + rw) } : { left: x }),
            top: ring.labelBelow ? y + rh + 10 : y - 46,
            padding: "6px 14px",
            borderRadius: 10,
            background: "rgba(3,6,12,0.88)",
            border: `1px solid ${color}`,
            fontFamily: MONO,
            fontSize: 20,
            color,
            whiteSpace: "nowrap",
            opacity: Math.min(1, ring.p * 1.5),
            transform: `translateY(${(1 - Math.min(1, ring.p)) * 8}px)`,
            boxShadow: `0 0 24px -8px ${color}`,
          }}
        >
          {ring.label}
        </div>
      )}
    </div>
  )
}

// --- the tab bar, lifted out of the screenshot -------------------------------------------------------

const TABS = ["Calls", "Recordings", "Dictate", "Practice", "Journal"]
const TAB_W = [132, 210, 164, 176, 164]
const TAB_GAP = 8
const TAB_H = 58
const tabLeft = (i: number) => TAB_W.slice(0, i).reduce((s, v) => s + v + TAB_GAP, 0)
const RAIL_W = tabLeft(TABS.length) - TAB_GAP + 16

function TabRail({ active, style }: { active: number; style?: CSSProperties }) {
  const i0 = Math.max(0, Math.min(TABS.length - 1, Math.floor(active)))
  const i1 = Math.min(TABS.length - 1, i0 + 1)
  const t = active - i0
  const hl = lerp(tabLeft(i0), tabLeft(i1), t)
  const hw = lerp(TAB_W[i0], TAB_W[i1], t)
  return (
    <div
      style={{
        position: "absolute",
        width: RAIL_W,
        height: TAB_H + 16,
        borderRadius: 999,
        padding: 8,
        boxSizing: "border-box",
        background: "linear-gradient(180deg, rgb(20,27,42), rgb(10,14,24))",
        border: `1px solid ${C.border2}`,
        boxShadow: "0 30px 70px -30px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.08)",
        ...style,
      }}
    >
      <div
        style={{
          position: "absolute",
          left: 8 + hl,
          top: 8,
          width: hw,
          height: TAB_H,
          borderRadius: 999,
          background: "linear-gradient(180deg, rgba(125,211,252,0.26), rgba(94,234,212,0.12))",
          border: `1.5px solid ${C.sky}`,
          boxShadow: `0 0 30px -6px ${C.sky}`,
        }}
      />
      {TABS.map((name, i) => {
        const near = Math.max(0, 1 - Math.abs(active - i))
        return (
          <div
            key={name}
            style={{
              position: "absolute",
              left: 8 + tabLeft(i),
              top: 8,
              width: TAB_W[i],
              height: TAB_H,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontFamily: SANS,
              fontSize: 28,
              fontWeight: near > 0.5 ? 700 : 500,
              color: near > 0.5 ? C.text : C.dim,
            }}
          >
            {name}
          </div>
        )
      })}
    </div>
  )
}

// --- right-hand steps for calls ------------------------------------------------------------------

function StepRow({ n, title, at, out, f, children, color = C.sky }: { n: number; title: ReactNode; at: number; out: number; f: number; children?: ReactNode; color?: string }) {
  const p = springAt(f, at, SNAPPY)
  const q = ramp(f, out, out + 10, ease)
  const o = Math.min(1, p * 1.4) * (1 - q)
  if (o <= 0.002) return null
  return (
    <div style={{ display: "flex", gap: 22, opacity: o, transform: `translateX(${(1 - Math.min(1, p)) * 50}px) translateY(${-q * 18}px)` }}>
      <div
        style={{
          flex: "none",
          width: 56,
          height: 56,
          borderRadius: 28,
          border: `2px solid ${color}`,
          background: "rgba(125,211,252,0.1)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: MONO,
          fontSize: 26,
          fontWeight: 700,
          color,
          boxShadow: `0 0 26px -6px ${color}`,
          transform: `scale(${0.6 + 0.4 * p})`,
        }}
      >
        {n}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10, paddingTop: 6, minWidth: 0 }}>
        <div style={{ fontFamily: SANS, fontSize: 36, fontWeight: 650, color: C.text, lineHeight: 1.15, letterSpacing: -0.3 }}>{title}</div>
        {children}
      </div>
    </div>
  )
}

function Chip({ children, color = C.sky, p = 1, style }: { children: ReactNode; color?: string; p?: number; style?: CSSProperties }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 14px",
        borderRadius: 12,
        border: `1px solid ${color}66`,
        background: `${color}14`,
        fontFamily: MONO,
        fontSize: 22,
        color,
        opacity: Math.min(1, p * 1.5),
        transform: `scale(${0.7 + 0.3 * Math.min(1.05, p)})`,
        transformOrigin: "0 50%",
        ...style,
      }}
    >
      {children}
    </span>
  )
}

// A plain, unbranded mock of a meeting app's audio settings: the microphone is CABLE Output.
function MicCard({ f, at, valueAt, live }: { f: number; at: number; valueAt: number; live: number }) {
  const p = springAt(f, at, SNAPPY)
  const v = springAt(f, valueAt, SNAPPY)
  return (
    <div
      style={{
        width: 560,
        borderRadius: 22,
        padding: "20px 24px 22px",
        boxSizing: "border-box",
        background: "linear-gradient(160deg, rgba(255,255,255,0.08), rgba(255,255,255,0.025))",
        border: `1px solid ${C.border2}`,
        boxShadow: `0 30px 70px -30px rgba(0,0,0,0.9), 0 0 ${40 * live}px -12px ${C.teal}`,
        opacity: Math.min(1, p * 1.4),
        transform: `translateY(${(1 - Math.min(1, p)) * 30}px) scale(${0.94 + 0.06 * p})`,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, fontFamily: MONO, fontSize: 19, letterSpacing: 2, color: C.dim }}>
        <Icon name="headphones" size={22} color={C.dim} />
        MEETING APP · AUDIO SETTINGS
      </div>
      <div style={{ fontFamily: SANS, fontSize: 24, fontWeight: 600, color: C.dim, marginTop: 16 }}>Microphone</div>
      <div
        style={{
          marginTop: 10,
          height: 60,
          borderRadius: 14,
          border: `1.5px solid ${interpolateHex(v)}`,
          background: "rgba(3,6,12,0.6)",
          display: "flex",
          alignItems: "center",
          gap: 14,
          padding: "0 18px",
          position: "relative",
          overflow: "hidden",
          boxShadow: v > 0.5 ? `0 0 26px -8px ${C.teal}` : undefined,
        }}
      >
        <Icon name="mic" size={26} color={v > 0.5 ? C.teal : C.faint} />
        <div style={{ position: "relative", flex: 1, height: 40 }}>
          <span style={{ position: "absolute", top: 4, fontFamily: SANS, fontSize: 26, color: C.faint, opacity: 1 - Math.min(1, v * 1.5) }}>Default microphone</span>
          <span
            style={{
              position: "absolute",
              top: 2,
              fontFamily: SANS,
              fontSize: 28,
              fontWeight: 700,
              color: C.text,
              opacity: Math.min(1, v * 1.4),
              transform: `translateY(${(1 - Math.min(1, v)) * 22}px)`,
            }}
          >
            CABLE Output
          </span>
        </div>
        <svg width={18} height={12} viewBox="0 0 18 12">
          <path d="M2 2 L9 9 L16 2" fill="none" stroke={C.dim} strokeWidth={2.5} strokeLinecap="round" />
        </svg>
        <LightSweep at={valueAt} dur={30} color="rgba(94,234,212,0.25)" />
      </div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 5, height: 30, marginTop: 16 }}>
        {Array.from({ length: 30 }, (_, i) => {
          const n = 0.5 + 0.5 * noise2D("mic", i * 0.35, f / 7)
          const lvl = live * (0.15 + 0.85 * n * n)
          return <span key={i} style={{ width: 11, height: 4 + 26 * lvl, borderRadius: 3, background: C.teal, opacity: 0.25 + 0.75 * live }} />
        })}
      </div>
    </div>
  )
}

const interpolateHex = (v: number) => (v > 0.5 ? C.teal : C.border2)

// --- the scene --------------------------------------------------------------------------------------

interface CardDef {
  src: string
  title: string
  sub: string
  icon: "video" | "text" | "message" | "lock"
  view: View
  viewEnd: View
  ring: [number, number, number, number]
  ringLabel: string
  labelRight?: boolean
  color: string
}

const CARDS: CardDef[] = [
  {
    src: "generated/screens/recordings.png",
    title: "Recordings",
    sub: "Cleans voice notes and videos",
    icon: "video",
    // ends just under the Download row: the demo's own counts below it are a different take
    // (image y ≈ 82–482), and starts at the voice-note card's left edge so no line is sliced
    view: { cx: 560, cy: 298, s: 1.2 },
    viewEnd: { cx: 488, cy: 325, s: 1.4 },
    ring: [498, 438, 822, 466],
    ringLabel: "Download clean · Timestamps · Subtitles",
    labelRight: true,
    color: C.sky,
  },
  {
    src: "generated/screens/dictate.png",
    title: "Dictate",
    sub: "Types clean text, without the stutters",
    icon: "text",
    view: { cx: 520, cy: 250, s: 1.2 },
    viewEnd: { cx: 444, cy: 238, s: 1.4 },
    ring: [57, 158, 830, 226],
    ringLabel: "text without the stutters · nothing is saved",
    color: C.sky,
  },
  {
    src: "generated/screens/practice.png",
    title: "Practice",
    sub: "A mock interview about the role",
    icon: "message",
    view: { cx: 502, cy: 232, s: 1.3 },
    viewEnd: { cx: 454, cy: 246, s: 1.45 },
    ring: [49, 150, 765, 207],
    ringLabel: "it never comments on how you speak",
    color: C.sky,
  },
  {
    src: "generated/screens/journal.png",
    title: "Journal",
    sub: "Opt-in, only on your computer",
    icon: "lock",
    view: { cx: 500, cy: 250, s: 1.3 },
    viewEnd: { cx: 448, cy: 246, s: 1.5 },
    ring: [57, 152, 618, 284],
    ringLabel: "numbers and words, never audio",
    color: C.green,
  },
]

// coverflow geometry
const CF_Y = 410
const CF_W = 1200
const CF_H = 440

function coverPose(rel: number, view: View): Pose {
  const r = Math.max(-2.2, Math.min(2.2, rel))
  const ar = Math.abs(r)
  const near = Math.min(1, ar)
  const far = Math.max(0, ar - 1)
  const dx = Math.sign(r) * (near * 800 + far * 260)
  return {
    x: 960 + dx,
    y: CF_Y + near * 10,
    w: CF_W,
    h: CF_H,
    rx: 0,
    ry: -Math.sign(r) * (near * 34 + far * 8),
    sc: 1 - 0.22 * near - 0.08 * far,
    op: 1 - Math.min(1, far * 1.1),
    view,
  }
}

export default function Scene({ scene }: SceneProps) {
  const f = useCurrentFrame()
  const L0 = lineCue(scene, 0)
  const L1 = lineCue(scene, 1)
  const L2 = lineCue(scene, 2)

  // --- beats, from the narration ---
  const tRing = L0.at + 2 // the tab bar is ringed as "app" lands
  const tLift = L0.at + 20 // and lifts out as the scene's navigation
  const t1 = L1.at - 6 // the window makes room for the steps
  const tCalls = wordAt(L1, "calls", -2)
  const tGentle = wordAt(L1, "gentle", -2)
  const tStrong = wordAt(L1, "strong", -2)
  const tChoose = wordAt(L1, "and choose", -4)
  const tCable = wordAt(L1, "virtual cable", -2)
  const tMic = wordAt(L1, "as the mic", -4)
  const tZoom = wordAt(L1, "zoom", -12)
  const tStepsOut = L1.at + L1.frames - 6 // the steps leave as the line ends
  const tOut = L2.at - 6 // the coverflow comes in
  const tRec = L2.at - 4
  const tDict = wordAt(L2, "dictate", -6)
  const tPrac = wordAt(L2, "practice", -2)
  const tJournal = Math.min(scene.frames - 44, L2.at + L2.frames - 4)
  const tBand = L2.at + 18

  // --- the calls window's poses ---
  const viewAll: View = { cx: 840, cy: 380, s: 0.738 }
  const view1: View = { cx: 705, cy: 365, s: 1.4 }
  // view2 frames the whole Setup panel (image x ≈ 1109–1683, y ≈ 127–583) in the narrowed window,
  // so no sliced hero headline sits beside the output dropdown
  const view2: View = { cx: 1396, cy: 355, s: 1.36 }
  const pose0At = (fr: number): Pose => {
    const enter = springAt(fr, 0, { damping: 22, mass: 1, stiffness: 70 })
    return {
      x: 960,
      y: 452,
      w: 1240,
      h: 560,
      rx: lerp(26, 5, enter),
      ry: lerp(-18, -3, enter),
      sc: lerp(0.86, 1, enter),
      op: Math.min(1, enter * 2),
      view: viewAll,
    }
  }
  const pose0 = pose0At(f)
  const pan = ramp(f, tChoose, tChoose + 34, ease)
  const W1 = 1060
  const W2 = 780
  const H1 = 620
  const POSE1_X = 626
  const POSE1_Y = 518
  const pose1: Pose = { x: POSE1_X, y: POSE1_Y, w: lerp(W1, W2, pan), h: H1, rx: 0, ry: 3, sc: 1, op: 1, view: lerpView(view1, view2, pan) }
  const m1 = springAt(f, t1, SOFT)
  // which tab is in front: Calls → Recordings → Dictate → Practice → Journal
  const a =
    Math.min(1, springAt(f, tRec, SOFT)) +
    Math.min(1, springAt(f, tDict, SOFT)) +
    Math.min(1, springAt(f, tPrac, SOFT)) +
    Math.min(1, springAt(f, tJournal, SOFT))
  const m2 = springAt(f, tOut, SOFT)
  const callsCover = coverPose(0 - a, { cx: 840, cy: 420, s: 0.72 })
  const callsPose = lerpPose(lerpPose(pose0, pose1, m1), callsCover, m2)

  // rings on the calls screenshot
  const fadeOut = (at: number, d = 10) => 1 - ramp(f, at, at + d, ease)
  const pulseAt = (at: number) => interpolate(f, [at, at + 30], [0, 1], clamp)
  const callsRings: Ring[] = [
    // drawn, one echo, then gone by tLift + 4 as the bar itself lifts out
    { box: [34, 93, 470, 123], p: ramp(f, tRing, tRing + 10, easeOut), o: fadeOut(tLift, 4), pulse: pulseAt(tRing + 8) },
    { box: [347, 379, 535, 415], p: ramp(f, tGentle, tGentle + 14, easeOut), o: fadeOut(tChoose + 4), pulse: pulseAt(tStrong) },
    { box: [1148, 288, 1622, 348], p: ramp(f, tChoose + 22, tChoose + 38, easeOut), o: fadeOut(tStepsOut), color: C.teal, pulse: pulseAt(tCable) },
  ]
  // the output dropdown opened, picked, closed: the CABLE Input option drops in under the select
  // (covering the whole Detector label), folds away, and the select itself now reads CABLE Input
  const chipP = springAt(f, tCable, SNAPPY)
  const tPick = tCable + 14
  const chipC = ramp(f, tPick, tPick + 8, easeIn)
  const pickO = ramp(f, tPick + 2, tPick + 8, easeOut)
  const v2 = callsPose.view
  const cw = callsPose.w
  const ch = callsPose.h
  const chipA = toViewport(v2, cw, ch, 1144, 352)
  const chipB = toViewport(v2, cw, ch, 1626, 352)
  const callsPatches = (
    <>
      {/* the screenshot's own tab bar leaves the window as the rail lifts out of it */}
      <ImgBox view={v2} w={cw} h={ch} box={TAB_ROW} style={{ background: PAGE_BG, opacity: ramp(f, tLift, tLift + 2) }} />
      {/* the select's value after the pick */}
      {pickO > 0.002 && (
        <ImgBox
          view={v2}
          w={cw}
          h={ch}
          box={[1150, 318, 1594, 346]}
          style={{ background: "rgb(29,31,40)", opacity: pickO, display: "flex", alignItems: "center", paddingLeft: 9 * v2.s, boxSizing: "border-box" }}
        >
          <span
            style={{
              fontFamily: SANS,
              fontSize: 19 * v2.s,
              fontWeight: 700,
              color: C.text,
              whiteSpace: "nowrap",
              transform: `translateY(${(1 - pickO) * 6}px)`,
            }}
          >
            CABLE Input
          </span>
          <LightSweep at={tPick + 4} dur={26} color="rgba(94,234,212,0.32)" />
        </ImgBox>
      )}
    </>
  )
  const callsOverlay = (
    <>
      {/* the zoomed Calls view ends under the mode switch: no sliced Fluency Mirror strip */}
      {m1 * (1 - pan) > 0.01 && (
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            bottom: 0,
            height: 220,
            background: "linear-gradient(0deg, #05070d 50%, rgba(5,7,13,0))",
            opacity: Math.min(1, m1 * 1.6) * (1 - pan),
          }}
        />
      )}
      {chipP > 0.01 && chipC < 0.995 && (
        <div
          style={{
            position: "absolute",
            left: chipA.x,
            top: chipA.y,
            width: chipB.x - chipA.x,
            boxSizing: "border-box",
            padding: "12px 16px",
            borderRadius: 14,
            background: "rgba(6,10,18,0.97)",
            border: `1.5px solid ${C.teal}`,
            boxShadow: `0 24px 50px -20px rgba(0,0,0,0.9), 0 0 30px -8px ${C.teal}`,
            display: "flex",
            alignItems: "center",
            gap: 12,
            opacity: Math.min(1, chipP * 1.5) * (1 - chipC),
            transform: `translateY(${(1 - Math.min(1, chipP)) * -16}px) scaleY(${1 - chipC})`,
            transformOrigin: "50% 0",
          }}
        >
          <Icon name="check" size={24} color={C.teal} />
          <span style={{ fontFamily: SANS, fontSize: 26, fontWeight: 700, color: C.text }}>CABLE Input</span>
          <span style={{ marginLeft: "auto", fontFamily: MONO, fontSize: 18, color: C.dim }}>virtual cable</span>
        </div>
      )}
    </>
  )

  // stage point of the ringed select's right edge (once the pan has landed) and the mic card's field
  const vpLeft = POSE1_X - W2 / 2
  const vpTop = POSE1_Y - (H1 + CHROME) / 2 + CHROME
  const selRight = toViewport(view2, W2, H1, 1622, 318)
  const cableFrom = { x: vpLeft + selRight.x + 9, y: vpTop + selRight.y }
  const colX = 1232
  const cardTop = 556
  const cableTo = { x: colX + 18, y: cardTop + 122 }
  const cableD = `M ${cableFrom.x} ${cableFrom.y} C ${cableFrom.x + 60} ${cableFrom.y}, ${cableTo.x - 70} ${cableTo.y}, ${cableTo.x} ${cableTo.y}`
  const cableDraw = ramp(f, tMic + 8, tMic + 30, easeOut)
  const cableO = fadeOut(tStepsOut)

  // the tab rail: lifted out of the screenshot, then the scene's navigation
  // It starts exactly on the screenshot's tab bar (projected through the window's tilt at tLift), rises
  // clear of the window first while still small, then slides out and grows into place.
  // (it settles onto the bar over two frames, taking the bar's place, before it moves)
  const lift = springAt(f, tLift + 2, { damping: 20, mass: 1, stiffness: 90 })
  const liftY = springAt(f, tLift + 2, { damping: 20, mass: 1, stiffness: 190 })
  const liftPose = pose0At(tLift)
  const barA = projectToStage(liftPose, 32, 92)
  const barB = projectToStage(liftPose, 472, 124)
  const fromS = (barB.x - barA.x) / RAIL_W
  const railFrom = { x: barA.x, y: (barA.y + barB.y) / 2 - ((TAB_H + 16) * fromS) / 2, s: fromS }
  const railTo = { x: 96, y: 66, s: 1 }
  const railX = lerp(railFrom.x, railTo.x, lift)
  const railY = lerp(railFrom.y, railTo.y, Math.min(1, liftY))
  const railS = lerp(railFrom.s, railTo.s, Math.min(1, lift))
  const railO = Math.min(1, ramp(f, tLift, tLift + 2))

  // the GitHub band
  const bandP = springAt(f, tBand, SOFT)

  // scene exit for the last moments (the film's transition does the rest)
  const camShots = [
    { at: 0, zoom: 1, x: 960, y: 540 },
    { at: t1, zoom: 1.025, x: 964, y: 536, dur: Math.max(60, L1.frames) },
    { at: tOut, zoom: 1, x: 960, y: 540, dur: 30 },
    { at: tOut + 30, zoom: 1.035, x: 960, y: 532, dur: Math.max(60, scene.frames - tOut - 30) },
  ]

  return (
    <AbsoluteFill>
      <Particles count={22} color={C.sky} seed="app" opacity={0.4} rise={0.3} />
      <Camera shots={camShots} drift={5} seed="app">
        <GlowOrb x={lerp(960, 626, m1 * (1 - m2))} y={470} r={760} color="rgba(125,211,252,0.32)" opacity={0.5} />
        <GlowOrb x={1500} y={640} r={420} color="rgba(94,234,212,0.3)" opacity={0.45 * ramp(f, tMic, tMic + 20) * cableO} />

        {/* the coverflow of the other tabs (behind the calls window while it is in front) */}
        {CARDS.map((c, k) => {
          const i = k + 1
          const rel = i - a
          const front = Math.max(0, 1 - Math.abs(rel))
          const lv = ramp(f, [tRec, tDict, tPrac, tJournal][k] + 4, [tRec, tDict, tPrac, tJournal][k] + 40, ease)
          const pose = coverPose(rel, lerpView(c.view, c.viewEnd, lv))
          pose.op *= m2
          const cueAt = [tRec, tDict, tPrac, tJournal][k]
          const ring: Ring = {
            box: c.ring,
            p: ramp(f, cueAt + 10, cueAt + 26, easeOut),
            o: front,
            color: c.color,
            label: c.ringLabel,
            labelRight: c.labelRight,
            // no echo still spreading in the scene's last ~20 frames (the cross-transition)
            pulse: cueAt + 54 <= scene.frames - 20 ? pulseAt(cueAt + 24) : 0,
          }
          return (
            <AppWindow
              key={c.src}
              src={c.src}
              pose={pose}
              rings={[ring]}
              patches={<ImgBox view={pose.view} w={pose.w} h={pose.h} box={TAB_ROW} style={{ background: PAGE_BG }} />}
              sweepAt={cueAt + 4}
              dim={Math.min(1, Math.abs(rel))}
              z={100 - Math.round(Math.abs(rel) * 10)}
            />
          )
        })}

        <AppWindow
          src="generated/screens/calls.png"
          pose={callsPose}
          rings={callsRings}
          patches={callsPatches}
          overlay={callsOverlay}
          sweepAt={4}
          dim={m2 * Math.min(1, a)}
          z={m2 > 0.5 ? 100 - Math.round(Math.min(2.2, a) * 10) : 120}
        />

        {/* the virtual cable: from StutterClear's output to the meeting app's microphone */}
        <Diagram style={{ zIndex: 130 }}>
          <g opacity={cableO}>
            <FlowPath d={cableD} draw={cableDraw} flow={ramp(f, tZoom, tZoom + 10)} color={C.teal} width={4} packets={4} period={40} />
          </g>
        </Diagram>

        {/* the three steps for calls */}
        <div style={{ position: "absolute", left: colX, top: 196, width: 600, zIndex: 140, display: "flex", flexDirection: "column", gap: 0 }}>
          <div style={{ height: 120 }}>
            <StepRow n={1} f={f} at={tCalls} out={tStepsOut} title={<>Calls: Gentle or Strong</>}>
              <div style={{ display: "flex", gap: 10 }}>
                <Chip p={springAt(f, tGentle, SNAPPY)}>Gentle ≈0.5 s</Chip>
                <Chip p={springAt(f, tStrong, SNAPPY)}>Strong ≈0.85 s</Chip>
              </div>
            </StepRow>
          </div>
          <div style={{ height: 120, marginTop: 14 }}>
            <StepRow n={2} f={f} at={tChoose} out={tStepsOut} title={<>Send the cleaned voice to</>} color={C.teal}>
              <div>
                <Chip color={C.teal} p={springAt(f, tCable, SNAPPY)}>
                  CABLE Input
                </Chip>
              </div>
            </StepRow>
          </div>
          <div style={{ height: 60, marginTop: 14 }}>
            <StepRow n={3} f={f} at={tMic} out={tStepsOut} title={<>In Zoom, Meet or Teams</>} color={C.teal} />
          </div>
        </div>
        <div style={{ position: "absolute", left: colX + 18, top: cardTop, zIndex: 135, opacity: fadeOut(tStepsOut) }}>
          <MicCard f={f} at={tMic} valueAt={tZoom} live={ramp(f, tZoom + 6, tZoom + 20) * cableO} />
        </div>
        <div
          style={{
            position: "absolute",
            left: colX + 18,
            top: cardTop + 252,
            zIndex: 135,
            fontFamily: SANS,
            fontSize: 24,
            color: C.dim,
            opacity: ramp(f, tZoom + 14, tZoom + 26, easeOut) * fadeOut(tStepsOut),
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          <Icon name="layers" size={22} color={C.dim} />a virtual audio cable, e.g. VB-CABLE
        </div>

        {/* the front card's caption */}
        {CARDS.map((c, k) => {
          const rel = k + 1 - a
          const o = Math.max(0, 1 - Math.abs(rel) * 2.2) * m2
          if (o <= 0.01) return null
          return (
            <div
              key={c.title}
              style={{
                position: "absolute",
                left: 0,
                width: 1920,
                top: 668,
                zIndex: 150,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                opacity: o,
                transform: `translateX(${rel * -140}px)`,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 22 }}>
                <Icon name={c.icon} size={52} color={c.color} stroke={1.8} />
                <span style={{ fontFamily: SERIF, fontSize: 84, lineHeight: 1, letterSpacing: -1, color: C.text }}>{c.title}</span>
              </div>
              <div style={{ fontFamily: SANS, fontSize: 32, fontWeight: 500, color: C.dim, marginTop: 10 }}>{c.sub}</div>
            </div>
          )
        })}
      </Camera>

      <TabRail
        active={a}
        style={{
          left: railX,
          top: railY,
          opacity: railO,
          transform: `scale(${railS})`,
          transformOrigin: "0 0",
        }}
      />

      {/* how to get it */}
      <div
        style={{
          position: "absolute",
          left: 0,
          width: 1920,
          top: 830,
          display: "flex",
          justifyContent: "center",
          opacity: Math.min(1, bandP * 1.3),
          transform: `translateY(${(1 - Math.min(1, bandP)) * 16}px)`,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 18,
            padding: "8px 24px",
            borderRadius: 999,
            background: "rgba(3,6,12,0.6)",
            border: `1px solid ${C.border}`,
            fontFamily: MONO,
            fontSize: 22,
            color: C.dim,
          }}
        >
          <Icon name="globe" size={22} color={C.dim} />
          <span style={{ color: C.text }}>github.com/Pradyuman-aviator/stutterClear</span>
          <span style={{ color: C.faint }}>·</span>
          <span style={{ color: C.teal }}>python scripts/run_app.py</span>
        </div>
      </div>
    </AbsoluteFill>
  )
}
