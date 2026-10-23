import type { CSSProperties, ReactNode } from "react"
import { AbsoluteFill, Img, interpolate, interpolateColors, staticFile, useCurrentFrame } from "remotion"
import { evolvePath } from "@remotion/paths"

import type { SceneProps } from "../data"
import { lineCue } from "../data"
import {
  BOUNCY,
  Camera,
  clamp,
  ease,
  easeOut,
  GlowOrb,
  Icon,
  KineticText,
  LightSweep,
  Logo,
  Particles,
  ramp,
  Sfx,
  SNAPPY,
  springAt,
  Wave,
} from "../kit"
import type { IconName } from "../kit"
import { C, MONO, SANS, SERIF } from "../theme"

// His take, his choice. Two equal cards: send the take as it is, or clean it with StutterClear.
// Nothing is judged: both cards stay the same size and brightness; the choice is his.

const CW = 780
const CH = 470
const TOP = 300
const LEFT_X = 150
const RIGHT_X = 990
const PANEL = { x: 44, y: 130, w: 692, h: 236 } // the inner "screen" of each card, card coordinates

// The real Recordings screenshot (1680x1050), pieces of its "Clean my video" panel.
const SHOT = "generated/screens/recordings.png"
const SHOT_W = 1680
const SHOT_H = 1050

// Where "Thorough" sits on the stage (from the switch crop's placement below).
const SW = { sx: 1444, sy: 742, sw: 184, sh: 52, scale: 2, px: PANEL.w - 24 - 368, py: 10 }
// The switch, measured on the screenshot (px): its inside, and the thumb on Natural / on Thorough.
const SWI = { x0: 1453, x1: 1620, y0: 749, y1: 787 }
const THUMB = { y0: 752, y1: 784, nat: [1457, 1528] as const, tho: [1531, 1617] as const }
const thoroughStage = {
  x0: RIGHT_X + PANEL.x + SW.px + (THUMB.tho[0] - SW.sx) * SW.scale,
  x1: RIGHT_X + PANEL.x + SW.px + (THUMB.tho[1] - SW.sx) * SW.scale,
  y0: TOP + PANEL.y + SW.py + (THUMB.y0 - SW.sy) * SW.scale,
  y1: TOP + PANEL.y + SW.py + (THUMB.y1 - SW.sy) * SW.scale,
}

// The Natural | Thorough switch redrawn over its crop (same place, same colours as the screenshot),
// so the click visibly moves the thumb to Thorough. `show` fades it in over the crop, `t` slides it.
function SwitchFlip({ show, t }: { show: number; t: number }) {
  if (show <= 0) return null
  const k = SW.scale
  const X = (sx: number) => (sx - SW.sx) * k
  const Y = (sy: number) => (sy - SW.sy) * k
  const tl = THUMB.nat[0] + (THUMB.tho[0] - THUMB.nat[0]) * t
  const tr = THUMB.nat[1] + (THUMB.tho[1] - THUMB.nat[1]) * t
  const th = (THUMB.y1 - THUMB.y0) * k
  const sel = (on: number) => interpolateColors(on, [0, 1], ["#9a9b9c", "#fbfbfd"])
  const label = (text: string, x0: number, x1: number, on: number): ReactNode => (
    <div
      style={{
        position: "absolute",
        left: X(x0),
        width: (x1 - x0) * k,
        top: Y(THUMB.y0),
        height: th,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: SANS,
        fontWeight: 600,
        fontSize: 14 * k,
        letterSpacing: -0.2,
        color: sel(on),
      }}
    >
      {text}
    </div>
  )
  return (
    <div style={{ position: "absolute", left: SW.px, top: SW.py, width: SW.sw * k, height: SW.sh * k, opacity: show }}>
      {/* the switch's inside, empty */}
      <div
        style={{
          position: "absolute",
          left: X(SWI.x0),
          top: Y(SWI.y0),
          width: (SWI.x1 - SWI.x0) * k,
          height: (SWI.y1 - SWI.y0) * k,
          borderRadius: ((SWI.y1 - SWI.y0) * k) / 2,
          background: "linear-gradient(180deg, #21232d, #1f2129)",
        }}
      />
      {/* the thumb, sliding from Natural to Thorough */}
      <div
        style={{
          position: "absolute",
          left: X(tl),
          width: (tr - tl) * k,
          top: Y(THUMB.y0),
          height: th,
          borderRadius: th / 2,
          background: "#3f4148",
          border: "1px solid rgba(255,255,255,0.12)",
          boxSizing: "border-box",
          boxShadow: "inset 0 1px 0 rgba(255,255,255,0.08), 0 5px 12px -4px rgba(90,120,220,0.55)",
        }}
      />
      {label("Natural", THUMB.nat[0], THUMB.nat[1], 1 - t)}
      {label("Thorough", THUMB.tho[0], THUMB.tho[1], t)}
    </div>
  )
}

// A piece of the real screenshot, scaled, with feathered edges so it sits in the panel.
function Crop({ sx, sy, sw, sh, scale, feather = 6, style }: { sx: number; sy: number; sw: number; sh: number; scale: number; feather?: number; style?: CSSProperties }) {
  const m = `linear-gradient(to right, transparent, #000 ${feather}px, #000 calc(100% - ${feather}px), transparent), linear-gradient(to bottom, transparent, #000 ${feather}px, #000 calc(100% - ${feather}px), transparent)`
  return (
    <div
      style={{
        position: "absolute",
        width: sw * scale,
        height: sh * scale,
        overflow: "hidden",
        WebkitMaskImage: m,
        maskImage: m,
        WebkitMaskComposite: "source-in",
        maskComposite: "intersect",
        ...style,
      }}
    >
      <Img src={staticFile(SHOT)} style={{ position: "absolute", left: -sx * scale, top: -sy * scale, width: SHOT_W * scale, height: SHOT_H * scale, maxWidth: "none" }} />
    </div>
  )
}

// One of the two choices. Both are drawn by this one component so they stay exactly equal.
function ChoiceCard({
  x,
  enter,
  hover,
  icon,
  title,
  footIcon,
  footColor,
  foot,
  sweepAt,
  footPulse = 0,
  children,
}: {
  x: number
  enter: number
  hover: number
  icon: IconName
  title: string
  footIcon: IconName
  footColor: string
  foot: string
  sweepAt?: number
  footPulse?: number
  children: ReactNode
}) {
  const f = useCurrentFrame()
  const p = springAt(f, enter, { damping: 22, mass: 0.9, stiffness: 90 })
  const o = Math.min(1, p * 1.3)
  return (
    <div style={{ position: "absolute", left: x, top: TOP, width: CW, height: CH, perspective: 1600 }}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          opacity: o,
          transform: `translateY(${(1 - p) * 70 - hover * 6}px) rotateX(${(1 - p) * 14}deg) scale(${0.96 + 0.04 * p})`,
          transformOrigin: "50% 100%",
          borderRadius: 30,
          background: "linear-gradient(160deg, rgba(255,255,255,0.075), rgba(255,255,255,0.022))",
          border: `1px solid rgba(255,255,255,${0.1 + 0.12 * hover})`,
          boxShadow: `inset 0 1px 0 rgba(255,255,255,0.1), 0 40px 90px -36px rgba(0,0,0,0.85), 0 0 ${50 + 30 * hover}px -18px rgba(125,211,252,${0.18 + 0.2 * hover})`,
          overflow: "hidden",
        }}
      >
        {/* top edge light */}
        <div style={{ position: "absolute", left: 60, right: 60, top: 0, height: 1, background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent)" }} />
        {/* title */}
        <div style={{ position: "absolute", left: PANEL.x, top: 38, display: "flex", alignItems: "center", gap: 22 }}>
          <div
            style={{
              width: 62,
              height: 62,
              borderRadius: 31,
              border: `1px solid ${C.border2}`,
              background: "radial-gradient(circle at 35% 30%, rgba(125,211,252,0.22), rgba(255,255,255,0.03) 70%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Icon name={icon} size={30} color={C.text} stroke={1.8} />
          </div>
          <div style={{ fontFamily: SERIF, fontSize: 62, lineHeight: 1, letterSpacing: -1, color: C.text }}>{title}</div>
        </div>
        {/* inner screen */}
        <div
          style={{
            position: "absolute",
            left: PANEL.x,
            top: PANEL.y,
            width: PANEL.w,
            height: PANEL.h,
            borderRadius: 20,
            background: "linear-gradient(180deg, #181a23, #12141d)",
            border: `1px solid ${C.border}`,
            boxShadow: "inset 0 1px 0 rgba(255,255,255,0.05)",
            overflow: "hidden",
          }}
        >
          {children}
          {sweepAt !== undefined && <LightSweep at={sweepAt} dur={40} color="rgba(255,255,255,0.13)" />}
        </div>
        {/* footer */}
        <div style={{ position: "absolute", left: PANEL.x, top: PANEL.y + PANEL.h + 26, display: "flex", alignItems: "center", gap: 14 }}>
          <div
            style={{
              width: 38,
              height: 38,
              borderRadius: 19,
              background: `${footColor}22`,
              border: `1px solid ${footColor}66`,
              boxShadow: footPulse > 0 ? `0 0 ${24 * footPulse}px ${footColor}, 0 0 0 ${10 * footPulse}px ${footColor}22` : undefined,
              transform: `scale(${1 + 0.18 * footPulse})`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Icon name={footIcon} size={20} color={footColor} stroke={2.4} />
          </div>
          <span style={{ fontFamily: SANS, fontWeight: 500, fontSize: 27, color: C.dim }}>{foot}</span>
        </div>
      </div>
    </div>
  )
}

// A soft pointer.
function Cursor({ x, y, press, opacity }: { x: number; y: number; press: number; opacity: number }) {
  const s = 1.7 * (1 - 0.12 * press)
  return (
    <div style={{ position: "absolute", left: x, top: y, opacity, transform: `scale(${s})`, transformOrigin: "0 0", filter: "drop-shadow(0 6px 10px rgba(0,0,0,0.6)) drop-shadow(0 0 10px rgba(125,211,252,0.45))" }}>
      <svg width={24} height={32} viewBox="0 0 24 32" style={{ overflow: "visible" }}>
        <path d="M1 1 L1 25 L7.2 19.4 L11.4 29 L15.6 27.2 L11.5 17.8 L19.6 17.8 Z" fill={C.text} stroke="#0b1020" strokeWidth={1.6} strokeLinejoin="round" />
      </svg>
    </div>
  )
}

// A point along a gentle arc from a to b (lifted by `lift` px at the middle).
const arc = (a: [number, number], b: [number, number], t: number, lift: number): [number, number] => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t - Math.sin(Math.PI * t) * lift,
]

export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const c = d.clip
  const l0 = lineCue(scene, 0)
  const l1 = lineCue(scene, 1)

  // Beats, all tied to the narration.
  const headIn = l0.at // "Stuttering is a normal way of speaking."
  const restAt = l0.at + Math.round(0.56 * l0.frames) // "He could send this take as it is." (the second sentence starts ~56% in)
  const moveAt = l1.at - 4 // "Or he can choose StutterClear."
  const pickAt = moveAt + 22 // the cursor lands on Thorough
  const logoAt = l1.at + Math.round(0.56 * l1.frames) // "...StutterClear." (the word starts ~56% into the line)
  // the honesty pill lands just after the tagline, so the finished layout holds ~1 s before the fade
  const pillAt = Math.max(logoAt + 10, l1.at + Math.round(0.7 * l1.frames))
  const orAt = l1.at
  const readyAt = l0.at + Math.round(0.86 * l0.frames) // "...as it is."
  const readyPulse = Math.max(0, 1 - Math.abs(f - readyAt - 6) / 10)

  // The cursor: in from below on line 0, rests on "Send as it is", glides to "Thorough" on line 1.
  const start: [number, number] = [760, 860]
  const onSend: [number, number] = [LEFT_X + PANEL.x + 62 + 22 + 262, TOP + 64] // just past "is", off the title
  const onThorough: [number, number] = [thoroughStage.x1 - 40, thoroughStage.y1 - 14]
  const t1 = ramp(f, restAt - 22, restAt + 4, ease)
  const t2 = ramp(f, moveAt, pickAt, ease)
  const pos = t2 > 0 ? arc(onSend, onThorough, t2, 80) : arc(start, onSend, t1, 40)
  const cursorO = ramp(f, restAt - 22, restAt - 12, easeOut)
  const press = Math.max(0, 1 - Math.abs(f - pickAt - 2) / 5)

  const hoverRight = ramp(f, pickAt - 4, pickAt + 6, easeOut)

  // The ring that marks Thorough.
  const rx0 = thoroughStage.x0 - 10
  const ry0 = thoroughStage.y0 - 8
  const rw = thoroughStage.x1 - thoroughStage.x0 + 20
  const rh = thoroughStage.y1 - thoroughStage.y0 + 16
  const rr = rh / 2
  const ringPath = `M ${rx0 + rr} ${ry0} L ${rx0 + rw - rr} ${ry0} A ${rr} ${rr} 0 0 1 ${rx0 + rw - rr} ${ry0 + rh} L ${rx0 + rr} ${ry0 + rh} A ${rr} ${rr} 0 0 1 ${rx0 + rr} ${ry0} Z`
  const ring = ramp(f, pickAt + 1, pickAt + 17, easeOut)
  const ringPulse = Math.max(0, 1 - Math.abs(f - pickAt - 18) / 14)
  const ripple = ramp(f, pickAt, pickAt + 22, easeOut)
  const ev = evolvePath(ring, ringPath)
  // the switch follows the click: the thumb slides to Thorough
  const flipShow = ramp(f, pickAt, pickAt + 3, easeOut)
  const flipT = ramp(f, pickAt + 1, pickAt + 9, easeOut)

  // Heading on line 0 leaves as line 1 starts; the logo arrives on "StutterClear".
  const headOut = ramp(f, moveAt - 2, moveAt + 12, ease)
  const logoP = springAt(f, logoAt, SNAPPY)
  const tagP = springAt(f, logoAt + 7)
  const pillP = springAt(f, pillAt, BOUNCY)
  const orP = springAt(f, orAt, SNAPPY)
  const sheen = ramp(f, logoAt + 6, logoAt + 36, ease)

  const waveReveal = ramp(f, 8, 46, easeOut)

  return (
    <AbsoluteFill>
      <Particles count={22} seed="choice" color={C.sky} opacity={0.32} rise={0.3} y={120} h={820} />
      <Camera
        shots={[
          { at: 0, x: 960, y: 540, zoom: 1 },
          { at: 0, x: 960, y: 536, zoom: 1.02, dur: Math.max(30, restAt - 10) },
          { at: restAt - 10, x: 936, y: 534, zoom: 1.04, dur: Math.max(30, moveAt - restAt + 10) },
          { at: moveAt, x: 984, y: 530, zoom: 1.05, dur: scene.frames - moveAt },
        ]}
        drift={5}
        seed="his-choice"
      >
        <GlowOrb x={LEFT_X + CW / 2} y={TOP + CH / 2} r={520} color="rgba(125,211,252,0.5)" opacity={0.16} />
        <GlowOrb x={RIGHT_X + CW / 2} y={TOP + CH / 2} r={520} color="rgba(125,211,252,0.5)" opacity={0.16} />
        <GlowOrb x={onThorough[0]} y={onThorough[1]} r={260} color={C.sky} opacity={0.22 * hoverRight} />

        {/* Line 0: the statement */}
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: 116,
            textAlign: "center",
            opacity: 1 - headOut,
            transform: `translateY(${-headOut * 26}px)`,
            filter: headOut > 0 ? `blur(${headOut * 6}px)` : undefined,
          }}
        >
          <KineticText
            text="Stuttering is a normal way of speaking."
            at={headIn}
            per={3}
            effect="blur"
            highlight={{ normal: C.sky }}
            style={{ fontFamily: SERIF, fontSize: 80, lineHeight: 1.05, letterSpacing: -1, color: C.text }}
          />
        </div>

        {/* Line 1: the logo and its promise */}
        {f >= logoAt - 1 && (
          <div style={{ position: "absolute", left: 0, right: 0, top: 92, display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
            <GlowOrb x={960} y={60} r={300} color={C.sky} opacity={0.18 * Math.min(1, logoP)} />
            <div style={{ position: "relative", opacity: Math.min(1, logoP * 1.4), transform: `translateY(${(1 - logoP) * 30}px) scale(${0.86 + 0.14 * logoP})` }}>
              <Logo size={96} />
              {/* a sheen that crosses the letters once */}
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  fontFamily: SERIF,
                  fontSize: 96,
                  lineHeight: 1,
                  letterSpacing: -1,
                  color: "transparent",
                  backgroundImage: "linear-gradient(105deg, transparent 40%, rgba(255,255,255,0.95) 50%, transparent 60%)",
                  backgroundSize: "300% 100%",
                  backgroundPosition: `${100 - 100 * sheen}% 0`,
                  WebkitBackgroundClip: "text",
                  backgroundClip: "text",
                  opacity: sheen > 0 && sheen < 1 ? 1 : 0,
                }}
              >
                StutterClear
              </div>
            </div>
            <div style={{ fontFamily: SANS, fontSize: 31, fontWeight: 500, color: C.dim, opacity: Math.min(1, tagP), transform: `translateY(${(1 - tagP) * 14}px)`, marginTop: -6 }}>
              Your real voice, on your terms.
            </div>
          </div>
        )}

        {/* LEFT: send as it is */}
        {/* both cards enter together and stay exactly equal (no hover, no sweep) until line 1: nothing is judged */}
        <ChoiceCard x={LEFT_X} enter={2} hover={0} icon="arrow" title="Send as it is" footIcon="check" footColor={C.teal} foot="ready to send" footPulse={readyPulse}>
          <div style={{ position: "absolute", left: 24, top: 22, right: 24, display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{ width: 46, height: 46, borderRadius: 23, background: `${C.sky}22`, border: `1px solid ${C.sky}77`, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <svg width={18} height={20} viewBox="0 0 18 20">
                <path d="M3 2 L16 10 L3 18 Z" fill={C.sky} />
              </svg>
            </div>
            <span style={{ fontFamily: SANS, fontWeight: 600, fontSize: 28, color: C.text }}>Take 1</span>
            <span style={{ marginLeft: "auto", fontFamily: MONO, fontSize: 22, color: C.dim, fontVariantNumeric: "tabular-nums" }}>0:{Math.round(c.before_s).toString().padStart(2, "0")}</span>
          </div>
          <div style={{ position: "absolute", left: 24, top: 108 }}>
            <Wave env={c.env_before} seconds={c.before_s} width={PANEL.w - 48} height={104} color={C.sky} reveal={waveReveal} barSeconds={0.07} glow={3} />
          </div>
        </ChoiceCard>

        {/* RIGHT: clean my video (pieces of the real Recordings tab) */}
        <ChoiceCard x={RIGHT_X} enter={2} hover={hoverRight} icon="video" title="Clean my video" footIcon="layers" footColor={C.sky} foot="in StutterClear's Recordings tab" sweepAt={pickAt + 4}>
          <Crop
            sx={56}
            sy={839}
            sw={146}
            sh={34}
            scale={2}
            feather={3}
            style={{ left: 24, top: 28 }}
          />
          <Crop sx={SW.sx} sy={SW.sy} sw={SW.sw} sh={SW.sh} scale={SW.scale} feather={5} style={{ left: SW.px, top: SW.py }} />
          <SwitchFlip show={flipShow} t={flipT} />
          {/* the panel's own words, verbatim from the screenshot, set crisp */}
          <div style={{ position: "absolute", left: 34, top: 136, fontFamily: SANS, fontWeight: 500, fontSize: 27, color: C.dim, whiteSpace: "nowrap" }}>
            Natural keeps fillers; Thorough removes them too.
          </div>
        </ChoiceCard>

        {/* "or", between the two */}
        <div
          style={{
            position: "absolute",
            left: 960 - 34,
            top: TOP + CH / 2 - 34,
            width: 68,
            height: 68,
            borderRadius: 34,
            background: "rgba(8,12,22,0.92)",
            border: `1px solid ${C.border2}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: SERIF,
            fontStyle: "italic",
            fontSize: 40,
            color: C.text,
            opacity: Math.min(1, orP * 1.4),
            transform: `scale(${0.5 + 0.5 * orP})`,
            boxShadow: "0 10px 30px -8px rgba(0,0,0,0.8)",
          }}
        >
          <span style={{ marginTop: -6 }}>or</span>
        </div>

        {/* The ring around Thorough, and the click ripple */}
        <svg width={1920} height={1080} style={{ position: "absolute", left: 0, top: 0, overflow: "visible", pointerEvents: "none" }}>
          {ripple > 0 && ripple < 1 && (
            <circle cx={onThorough[0] - 4} cy={onThorough[1] - 4} r={14 + 70 * ripple} fill="none" stroke={C.sky} strokeWidth={2} opacity={0.6 * (1 - ripple)} />
          )}
          {ring > 0 && (
            <path
              d={ringPath}
              fill={`rgba(125,211,252,${0.13 * ring})`}
              stroke={C.sky}
              strokeWidth={3 + 1.5 * ringPulse}
              strokeDasharray={ev.strokeDasharray}
              strokeDashoffset={ev.strokeDashoffset}
              strokeLinecap="round"
              style={{ filter: `drop-shadow(0 0 ${8 + 14 * ringPulse}px ${C.sky})` }}
            />
          )}
        </svg>

        {/* Optional, always his call: under the take he could send as it is */}
        <div
          style={{
            position: "absolute",
            left: LEFT_X,
            width: CW,
            top: TOP + CH + 30,
            display: "flex",
            justifyContent: "center",
            opacity: Math.min(1, pillP * 1.5),
            transform: `translateY(${(1 - pillP) * 18}px) scale(${0.85 + 0.15 * pillP})`,
          }}
        >
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 14,
              padding: "12px 28px",
              borderRadius: 999,
              border: `1px solid ${C.teal}66`,
              background: "rgba(94,234,212,0.08)",
              boxShadow: `0 0 34px -12px ${C.teal}`,
              fontFamily: SANS,
              fontWeight: 600,
              fontSize: 28,
              color: C.text,
            }}
          >
            <Icon name="heart" size={24} color={C.teal} stroke={2.2} />
            Optional · always his call
          </div>
        </div>

        <Cursor x={pos[0]} y={pos[1]} press={press} opacity={cursorO * interpolate(f, [scene.frames - 36, scene.frames - 22], [1, 0], clamp)} />
      </Camera>
      <Sfx name="pop" at={pickAt} volume={0.3} />
    </AbsoluteFill>
  )
}
