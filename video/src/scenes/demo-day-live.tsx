import type { CSSProperties, ReactNode } from "react"
import { AbsoluteFill, Img, interpolateColors, staticFile, useCurrentFrame } from "remotion"
import { noise2D } from "@remotion/noise"

import type { SceneProps } from "../data"
import type { Shot } from "../kit"
import { lineCue } from "../data"
import {
  Avatar,
  Block,
  Camera,
  cameraAt,
  CallWindow,
  Counter,
  Diagram,
  easeIn,
  easeOut,
  envAt,
  Eyebrow,
  FlowPath,
  GlowOrb,
  Icon,
  LightSweep,
  Logo,
  Particles,
  ramp,
  Sfx,
  SNAPPY,
  springAt,
  Tag,
  Tile,
  Title,
} from "../kit"
import { C, CLASS_COLORS, MONO, SANS, SERIF } from "../theme"

// Demo day: half a second behind.
// Line 0: "Tonight" flips to "Demo day", the call draws in, a crop of the real Calls tab floats beside
// it and a cursor taps Gentle. Line 1: the camera pushes into Alex's tile, which opens into the live
// pipeline: microphone -> delay buffer (a ring: the play head trails the write head by ~0.5 s) -> a
// window of past audio + lookahead -> WavLM-Large on the laptop's RTX 4050 -> five schematic lanes
// leaving to the right, where the next scene picks them up.

// ---------- layout (stage px) ----------
// the call stage is laid out inside the safe area so the slow push (zoom CALL_ZOOM about the centre, plus
// drift) never takes the title, the window or the card past 96 px from the sides or 60 px from the top
const WIN = { x: 120, y: 200, w: 1100, h: 640 }
const ALEX = { x: WIN.x + 282.5, y: 401 } // centre of Alex's tile inside the call window
const CALL_ZOOM = 1.02
const TITLE_Y = 78
const CARD_X = 1110
const AV_A = 130 // avatar size in the call
const PUSH_ZOOM = 2.4

const MIC = { x: 215, y: 430 }
const RING = { x: 640, y: 430, r: 150 }
const W_ANGLE = -150 // the write head, upper left of the ring (deg, clockwise from 3 o'clock)
const BLK = { x: 1200, y: 305, w: 570, h: 196 }
const PATH_Y = 410

// The screenshot crop: the mode switch and its caption on the Calls tab (1680x1050 source).
const CROP = { x: 338, y: 370, w: 312, h: 80, s: 2.0 }
const GENTLE = { x: 429, y: 397 } // centre of the "Gentle" button in the source

const deg = (a: number) => (a * Math.PI) / 180
const polar = (cx: number, cy: number, r: number, a: number) => ({ x: cx + r * Math.cos(deg(a)), y: cy + r * Math.sin(deg(a)) })
function arcPath(cx: number, cy: number, r: number, a0: number, a1: number) {
  if (a1 - a0 < 0.5) return ""
  const p0 = polar(cx, cy, r, a0)
  const p1 = polar(cx, cy, r, a1)
  const large = a1 - a0 > 180 ? 1 : 0
  return `M ${p0.x.toFixed(1)} ${p0.y.toFixed(1)} A ${r} ${r} 0 ${large} 1 ${p1.x.toFixed(1)} ${p1.y.toFixed(1)}`
}

function At({ x, y, children, style }: { x: number; y: number; children: ReactNode; style?: CSSProperties }) {
  return <div style={{ position: "absolute", left: x, top: y, ...style }}>{children}</div>
}

// Where a stage point lands on screen under a Camera (same formula as the kit's Camera, without its
// tiny drift rotation), so the avatar can be drawn at its real size rather than scaled up by CSS
// (a scaled box-shadow leaves a faint rectangle at high zoom).
const DRIFT = 5
const SEED = "ddl"
function toScreen(f: number, shots: Shot[], x: number, y: number) {
  const c = cameraAt(f, shots)
  const dx = noise2D(`${SEED}x`, f / 110, 0) * DRIFT
  const dy = noise2D(`${SEED}y`, 0, f / 110) * DRIFT
  return { x: 960 + c.zoom * (x - c.x + dx), y: 540 + c.zoom * (y - c.y + dy), z: c.zoom }
}

function AvatarAt({ f, shots, x, y, size, level, badge = 0 }: { f: number; shots: Shot[]; x: number; y: number; size: number; level: number; badge?: number }) {
  const s = toScreen(f, shots, x, y)
  const S = size * s.z
  const b = 46 * s.z
  return (
    <div style={{ position: "absolute", left: s.x - S / 2, top: s.y - S / 2, width: S, height: S }}>
      <Avatar size={S} level={level} />
      {badge > 0 && (
        <div
          style={{
            position: "absolute",
            right: -8 * s.z,
            bottom: -6 * s.z,
            width: b,
            height: b,
            borderRadius: b / 2,
            background: "#0d1424",
            border: `${1.5 * Math.max(1, s.z * 0.8)}px solid ${C.sky}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: C.sky,
            opacity: badge,
            transform: `scale(${0.5 + 0.5 * badge})`,
          }}
        >
          <Icon name="mic" size={26 * s.z} />
        </div>
      )}
    </div>
  )
}

// Where a word starts in a narration line, 0..1: its character position, with each full stop and comma
// weighted as the pause a voice takes there, plus a little silence at the end (measured on the Magpie
// narration: a full stop rests about as long as 5 characters take to say, a comma 2, the tail 3). Beats
// stay on their words when the line is re-voiced and its length changes; `fallback` is used if the word
// is not in the line.
const STOP_W = 5
const COMMA_W = 2
const END_W = 3
function wordFrac(text: string | undefined, word: string, fallback: number) {
  if (!text) return fallback
  const i = text.indexOf(word)
  if (i < 0) return fallback
  const units = (s: string) => s.length + (s.match(/[.!?](?=\s)/g)?.length ?? 0) * STOP_W + (s.match(/[,;:](?=\s)/g)?.length ?? 0) * COMMA_W
  return units(text.slice(0, i)) / (units(text.replace(/[.!?]\s*$/, "")) + END_W)
}

// The pipeline's camera: it starts where the call's push-in lands (on Alex, same zoom), pulls back to
// the microphone and the buffer (the pair centred), opens wide for the budget and the GPU, then settles
// as the lanes leave to the right, into the next scene's pan.
const pipelineShots = (XF: number, BUDGET: number, LANES: number): Shot[] => [
  { at: 0, x: MIC.x, y: MIC.y, zoom: PUSH_ZOOM },
  { at: XF, x: MIC.x, y: MIC.y, zoom: PUSH_ZOOM },
  { at: XF + 4, x: 470, y: 440, zoom: 1.3, dur: 48 },
  // the wide stage spans the whole safe area (x 96-1824): it is framed a touch wide so drift never
  // takes the heading or the budget past the 96 px margin; only the lanes bleed off to the right
  { at: BUDGET - 8, x: 960, y: 540, zoom: 0.98, dur: 40 },
  { at: LANES, x: 962, y: 536, zoom: 0.99, dur: 70 },
]

// One button of the app's Off | Gentle | Strong switch, re-drawn over the screenshot crop (source px).
function SwitchPill({ x0, x1, y0, y1, bg, color, label, cx, selected = false }: { x0: number; x1: number; y0: number; y1: number; bg: string; color: string; label: string; cx: number; selected?: boolean }) {
  const s = CROP.s
  const w = 120 * s
  return (
    <>
      <div
        style={{
          position: "absolute",
          left: (x0 - CROP.x) * s,
          top: (y0 - CROP.y) * s,
          width: (x1 - x0) * s,
          height: (y1 - y0) * s,
          borderRadius: 999,
          background: bg,
          // selected: the app's lit pill; a cover: feathered so the screenshot's highlight edge does not show
          boxShadow: selected ? `inset 0 1px 0 rgba(255,255,255,0.12), 0 ${4 * s}px ${8 * s}px -${3 * s}px rgba(80,100,190,0.45)` : `0 0 ${3 * s}px ${1.5 * s}px ${bg}`,
        }}
      />
      <div
        style={{
          position: "absolute",
          left: (cx - CROP.x) * s - w / 2,
          top: (396.5 - CROP.y) * s - 20,
          width: w,
          height: 40,
          lineHeight: "40px",
          textAlign: "center",
          fontFamily: SANS,
          fontWeight: 600,
          fontSize: 15 * s,
          color,
        }}
      >
        {label}
      </div>
    </>
  )
}

// A mouse pointer.
function Pointer({ press }: { press: number }) {
  return (
    <svg width={44} height={52} viewBox="0 0 22 26" style={{ overflow: "visible", transform: `scale(${1 - 0.14 * press})`, transformOrigin: "4px 3px", filter: "drop-shadow(0 6px 10px rgba(0,0,0,0.6))" }}>
      <path d="M 2 1 L 2 20 L 7 15.5 L 10.5 23.5 L 14 22 L 10.6 14.2 L 17 14 Z" fill="#f5f7fb" stroke="#0a0f1c" strokeWidth={1.4} strokeLinejoin="round" />
    </svg>
  )
}

export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const L0 = lineCue(scene, 0)
  const L1 = lineCue(scene, 1)
  const at0 = (u: number) => L0.at + u * L0.frames
  const at1 = (u: number) => L1.at + u * L1.frames

  // ---------- beats (tied to the narration) ----------
  const w0 = (word: string, fallback: number) => at0(wordFrac(L0.text, word, fallback))
  const FLIP = L0.at // "Demo day."
  const CALL_IN = w0("On a live", 0.23) // "On a live video call" (after the pause that follows "Demo day.")
  const CARD_IN = w0("Alex picks", 0.6) // "Alex picks"
  const CURSOR_IN = CARD_IN + 3
  // "Gentle": measured onset 2.54 s of 3.58 s on the Magpie line (wordFrac lands ~8 frames late here);
  // the press lands just inside the word
  const TAP = at0(0.71) + 3
  const PUSH = Math.max(TAP + 14, L0.at + L0.frames - 6) // push into Alex's tile
  const XF = PUSH + 20 // the push-in has landed: the tile opens into the pipeline
  // line 1's fractions are measured word starts on the Magpie narration (faster-whisper word times):
  // its spelled-out "Wave L M Large" is slower than its characters suggest, so wordFrac would run late
  const VOICE = Math.max(XF + 6, at1(0.02)) // "His voice"
  const BUF = at1(0.15) // "buffer" (0.155)
  const HALF = at1(0.19) // "half a second behind him" (0.19)
  const BUDGET = at1(0.36) // "behind him"
  const WINDOW = at1(0.43) // "while" (0.42)
  const GPU = at1(0.45) // "WavLM-Large" (0.44)
  const NVGPU = at1(0.78) // "NVIDIA" (onset 5.21 s of 6.69 s)
  const RULER = at1(0.86) // "GPU" (onset 5.70 s)
  const LANES = at1(0.89)

  const xA = 1 - ramp(f, XF, XF + 14)
  const xB = ramp(f, XF, XF + 14)

  // Alex's voice level (decorative: his recording's loudness, looped)
  const tv = ((f / 30) * 0.9) % (d.clip.before_s - 0.2)
  const level = Math.min(1, Math.sqrt(envAt(d.clip.env_before, tv, tv + 0.15)))

  return (
    <AbsoluteFill>
      <Particles count={20} seed="ddl" color={C.sky} opacity={0.4} rise={0.3} />
      {xA > 0.001 && (
        <AbsoluteFill style={{ opacity: xA }}>
          <CallStage f={f} FLIP={FLIP} CALL_IN={CALL_IN} CARD_IN={CARD_IN} CURSOR_IN={CURSOR_IN} TAP={TAP} PUSH={PUSH} level={level} />
        </AbsoluteFill>
      )}
      {xB > 0.001 && (
        <AbsoluteFill style={{ opacity: xB }}>
          <PipelineStage
            f={f}
            XF={XF}
            VOICE={VOICE}
            BUF={BUF}
            HALF={HALF}
            BUDGET={BUDGET}
            WINDOW={WINDOW}
            GPU={GPU}
            NVGPU={NVGPU}
            RULER={RULER}
            LANES={LANES}
            level={level}
          />
        </AbsoluteFill>
      )}
      {/* Alex: one avatar through the cross-fade (drawn at full opacity, over both stages) */}
      {f >= XF && <AvatarAt f={f} shots={pipelineShots(XF, BUDGET, LANES)} x={MIC.x} y={MIC.y} size={AV_A} level={level} badge={springAt(f, VOICE, SNAPPY)} />}
      <Sfx name="pop" at={TAP} volume={0.35} />
    </AbsoluteFill>
  )
}

// ======================= line 0: the call =======================
function CallStage({ f, FLIP, CALL_IN, CARD_IN, CURSOR_IN, TAP, PUSH, level }: { f: number; FLIP: number; CALL_IN: number; CARD_IN: number; CURSOR_IN: number; TAP: number; PUSH: number; level: number }) {
  // the flip: "Tonight" turns away, "Demo day." turns in
  const out = ramp(f, FLIP, FLIP + 10, easeIn)
  const inn = springAt(f, FLIP + 7, SNAPPY)
  const win = springAt(f, CALL_IN, SNAPPY)
  const card = springAt(f, CARD_IN, SNAPPY)
  const tapped = f >= TAP
  const gentle = springAt(f, TAP, SNAPPY)
  const pre = 1 - ramp(f, TAP, TAP + 4) // the selection jumps from Off to Gentle on the press
  const badge = springAt(f, TAP + 6, SNAPPY)
  // cursor: glides from below right to "Gentle", presses at TAP, then drifts off a little
  const cx0 = { x: 470, y: 250 }
  const gx = (GENTLE.x - CROP.x) * CROP.s
  const gy = (GENTLE.y - CROP.y) * CROP.s
  const move = ramp(f, CURSOR_IN, TAP - 2, easeOut)
  const press = Math.max(0, 1 - Math.abs(f - TAP) / 4)
  const leave = ramp(f, TAP + 10, TAP + 30, easeOut)
  const cur = { x: cx0.x + (gx - cx0.x) * move + leave * 40, y: cx0.y + (gy - cx0.y) * move + leave * 50 }
  const curO = Math.min(ramp(f, CURSOR_IN, CURSOR_IN + 8), 1 - ramp(f, PUSH - 4, PUSH + 6))
  const ripple = ramp(f, TAP, TAP + 18, easeOut)

  const judges = [
    { i: "K", c: "#a5b4c8" },
    { i: "M", c: "#b8c1d6" },
    { i: "R", c: "#9fb0c2" },
  ]
  const tiles = [
    { key: "alex", name: "Alex" },
    { key: "j0", name: "Judge" },
    { key: "j1", name: "Judge" },
    { key: "j2", name: "Judge" },
  ]

  const shots: Shot[] = [
    { at: 0, x: 960, y: 540, zoom: 1 },
    { at: CALL_IN, x: 960, y: 540, zoom: CALL_ZOOM, dur: 110 },
    { at: PUSH, x: ALEX.x, y: ALEX.y, zoom: PUSH_ZOOM, dur: 20 },
  ]
  const lifted = f >= PUSH - 2 // from here the avatar is drawn over the camera, at its real size

  return (
    <>
    <Camera shots={shots} drift={DRIFT} seed={SEED}>
      <GlowOrb x={ALEX.x + 200} y={520} r={700} color="rgba(94,234,212,0.5)" opacity={0.12 * win} />

      {/* Tonight -> Demo day */}
      <div style={{ position: "absolute", left: WIN.x, top: TITLE_Y, height: 110, perspective: 900 }}>
        <div style={{ position: "absolute", left: 0, top: 0, transformOrigin: "50% 60%", transform: `rotateX(${out * 90}deg)`, opacity: 1 - out, whiteSpace: "nowrap" }}>
          <Title size={92} style={{ color: C.dim }}>Tonight</Title>
        </div>
        <div style={{ position: "absolute", left: 0, top: 0, transformOrigin: "50% 40%", transform: `rotateX(${(1 - inn) * -90}deg)`, opacity: Math.min(1, inn * 1.5), whiteSpace: "nowrap", display: "flex", alignItems: "baseline", gap: 26 }}>
          <Title size={92}>Demo day.</Title>
          <span style={{ fontFamily: MONO, fontSize: 22, letterSpacing: 4, color: C.teal, opacity: ramp(f, CALL_IN, CALL_IN + 12) }}>LIVE · VIDEO CALL</span>
        </div>
      </div>

      {/* the call */}
      <div style={{ position: "absolute", left: WIN.x, top: WIN.y, opacity: Math.min(1, win * 1.3), transform: `translateY(${(1 - win) * 50}px) scale(${0.94 + 0.06 * win})`, transformOrigin: "50% 60%" }}>
        <CallWindow title="Demo day · video call" width={WIN.w} height={WIN.h}>
          <div style={{ position: "absolute", inset: 0, display: "grid", gridTemplateColumns: "1fr 1fr", gridTemplateRows: "1fr 1fr", gap: 16 }}>
            {tiles.map((t, k) => {
              const p = springAt(f, CALL_IN + 6 + k * 4, SNAPPY)
              return (
                <div key={t.key} style={{ display: "flex", opacity: Math.min(1, p * 1.4), transform: `scale(${0.9 + 0.1 * p})` }}>
                  <Tile name={t.name} speaking={t.key === "alex" ? level * win : 0}>
                    {t.key === "alex" ? (
                      lifted ? <div style={{ width: AV_A, height: AV_A }} /> : <Avatar size={AV_A} level={level} />
                    ) : (
                      <div
                        style={{
                          width: 104,
                          height: 104,
                          borderRadius: "50%",
                          border: `1.5px solid ${C.border2}`,
                          background: "radial-gradient(circle at 35% 30%, rgba(255,255,255,0.14), rgba(10,14,26,0.9))",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontFamily: SANS,
                          fontWeight: 600,
                          fontSize: 44,
                          color: judges[k - 1].c,
                        }}
                      >
                        {judges[k - 1].i}
                      </div>
                    )}
                    {t.key === "alex" && (
                      <div
                        style={{
                          position: "absolute",
                          right: 16,
                          top: 14,
                          opacity: Math.min(1, badge * 1.5),
                          transform: `scale(${0.7 + 0.3 * badge})`,
                          transformOrigin: "100% 0",
                          fontFamily: MONO,
                          fontSize: 19,
                          fontWeight: 700,
                          color: "#05110f",
                          background: C.teal,
                          borderRadius: 999,
                          padding: "5px 12px",
                          boxShadow: `0 0 24px -4px ${C.teal}`,
                        }}
                      >
                        Gentle · ≈0.5 s
                      </div>
                    )}
                  </Tile>
                </div>
              )
            })}
          </div>
        </CallWindow>
      </div>

      {/* the real Calls tab, cropped: the mode switch */}
      <div
        style={{
          position: "absolute",
          left: CARD_X,
          top: 362,
          perspective: 1600,
          opacity: Math.min(1, card * 1.4),
          transform: `translateX(${(1 - card) * 140}px)`,
        }}
      >
        <div style={{ position: "relative", transform: `rotateY(${-9 + 4 * card}deg) rotateX(2deg)`, transformOrigin: "0 50%" }}>
          <div
            style={{
              width: CROP.w * CROP.s + 56,
              borderRadius: 26,
              padding: 28,
              boxSizing: "border-box",
              background: "linear-gradient(160deg, rgba(22,28,42,0.96), rgba(10,14,24,0.96))",
              border: `1px solid ${C.border2}`,
              boxShadow: `0 40px 90px -30px rgba(0,0,0,0.9), 0 0 ${60 * gentle}px -14px ${C.teal}`,
              position: "relative",
              overflow: "hidden",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 18 }}>
              <Logo size={34} />
              <Tag style={{ fontSize: 18 }}>Calls</Tag>
              <span style={{ marginLeft: "auto", fontFamily: MONO, fontSize: 18, color: C.faint }}>the app</span>
            </div>
            <div style={{ position: "relative", width: CROP.w * CROP.s, height: CROP.h * CROP.s, overflow: "hidden", borderRadius: 14 }}>
              <Img
                src={staticFile("generated/screens/calls.png")}
                style={{ position: "absolute", width: 1680 * CROP.s, left: -CROP.x * CROP.s, top: -CROP.y * CROP.s, maxWidth: "none" }}
              />
              {/* before the tap the switch is on Off: the screenshot's Gentle highlight is covered with the
                  track's own colour (sampled from calls.png) and a selected pill sits behind Off */}
              {pre > 0.001 && (
                <div style={{ position: "absolute", inset: 0, opacity: pre }}>
                  <SwitchPill x0={394} x1={466} y0={380} y1={415} bg="#1b1d28" color="#96979d" label="Gentle" cx={429.5} />
                  <SwitchPill x0={351} x1={393} y0={382} y1={414} bg="#3b3d45" color="#ffffff" label="Off" cx={371.5} selected />
                </div>
              )}
              {/* the tap: a ring around Gentle */}
              <div
                style={{
                  position: "absolute",
                  left: (398 - CROP.x) * CROP.s - 6,
                  top: (381 - CROP.y) * CROP.s - 6,
                  width: (462 - 398) * CROP.s + 12,
                  height: (413 - 381) * CROP.s + 12,
                  borderRadius: 999,
                  border: `2.5px solid ${C.teal}`,
                  opacity: tapped ? Math.min(1, gentle) : 0,
                  boxShadow: `0 0 26px ${C.teal}88`,
                }}
              />
              {tapped && (
                <div
                  style={{
                    position: "absolute",
                    left: gx - 70 * ripple,
                    top: gy - 70 * ripple,
                    width: 140 * ripple,
                    height: 140 * ripple,
                    borderRadius: "50%",
                    border: `2px solid ${C.teal}`,
                    opacity: 1 - ripple,
                  }}
                />
              )}
            </div>
            <LightSweep at={TAP + 2} dur={30} color="rgba(94,234,212,0.18)" />
          </div>
          <div style={{ position: "absolute", left: 28 + cur.x - 4, top: 28 + 52 + cur.y - 3, opacity: curO }}>
            <Pointer press={press} />
          </div>
        </div>
      </div>
    </Camera>
    {lifted && f < PUSH + 20 && <AvatarAt f={f} shots={shots} x={ALEX.x} y={ALEX.y} size={AV_A} level={level} />}
    </>
  )
}

// ======================= line 1: the live pipeline =======================
function PipelineStage({
  f,
  XF,
  VOICE,
  BUF,
  HALF,
  BUDGET,
  WINDOW,
  GPU,
  NVGPU,
  RULER,
  LANES,
  level,
}: {
  f: number
  XF: number
  VOICE: number
  BUF: number
  HALF: number
  BUDGET: number
  WINDOW: number
  GPU: number
  NVGPU: number
  RULER: number
  LANES: number
  level: number
}) {
  const shots = pipelineShots(XF, BUDGET, LANES)
  const mic = springAt(f, VOICE, SNAPPY)
  const ring = springAt(f, BUF, SNAPPY)
  const half = springAt(f, HALF, SNAPPY)
  const gpu = springAt(f, GPU, SNAPPY)
  const active = ramp(f, GPU + 6, NVGPU + 4)
  const nvLit = ramp(f, NVGPU, NVGPU + 14)

  // buffer: the heads stay put and the audio moves. His frames come in at the write head ("now",
  // upper left) and travel clockwise round the ring; the play head ("the call hears") slides back
  // ahead of them until it is ~0.5 s of audio away (the ring is 1 s round: 0.5 s = 180 deg; Strong's
  // 0.85 s = 306 deg). Frames leave at the play head, toward the call.
  const wA = W_ANGLE
  const gap = 180 * ramp(f, BUF + 4, HALF + 12, easeOut)
  const pA = wA + gap
  const SLOTS = 40
  const phase = Math.max(0, f - BUF) * 1.5 // deg the held audio has travelled
  // from "NVIDIA GPU" on, the buffer and the budget step back so the lit GPU and its lanes lead
  const back = 1 - 0.45 * ramp(f, NVGPU, NVGPU + 20)
  // text steps back less (its labels stay at least as bright as C.dim; the big serif stays above 0.75)
  const textBack = 1 - 0.25 * ramp(f, NVGPU, NVGPU + 20)
  const tagNow = ramp(f, BUF + 6, BUF + 16) * (1 - ramp(f, NVGPU, NVGPU + 14))
  const tagCall = ramp(f, BUF + 14, BUF + 24) * (1 - ramp(f, NVGPU, NVGPU + 14))

  return (
    <>
    <Camera shots={shots} drift={DRIFT} seed={SEED}>
      <GlowOrb x={RING.x} y={RING.y} r={330} color={C.teal} opacity={0.13 * ring * back} />
      <GlowOrb x={BLK.x + BLK.w / 2} y={BLK.y + BLK.h / 2} r={420} color={C.nv} opacity={0.05 * gpu + 0.14 * nvLit} />

      {/* heading */}
      <At x={96} y={92} style={{ opacity: mic * textBack }}>
        <Eyebrow color={C.teal}>Live engine · Gentle</Eyebrow>
      </At>
      <At x={96} y={128} style={{ opacity: Math.min(1, half * 1.3) * textBack, transform: `translateY(${(1 - half) * 24}px)`, filter: `blur(${(1 - Math.min(1, half)) * 8}px)` }}>
        <Title size={66}>Half a second behind him.</Title>
      </At>

      {/* microphone: Alex (drawn over the camera, below) */}
      <At x={MIC.x - 140} y={MIC.y + 102} style={{ width: 280, textAlign: "center", opacity: mic, transform: `translateY(${(1 - mic) * 14}px)` }}>
        <div style={{ fontFamily: MONO, fontSize: 20, color: C.text }}>Microphone · 16 kHz</div>
      </At>

      <Diagram>
        {/* mic -> buffer: his voice as 20 ms frames, arriving at the write head along the ring's direction */}
        {(() => {
          const e = polar(RING.x, RING.y, RING.r + 18, W_ANGLE)
          const d = `M ${MIC.x + 92} ${MIC.y} C ${MIC.x + 190} ${MIC.y}, ${e.x - 70} ${e.y + 70}, ${e.x.toFixed(1)} ${e.y.toFixed(1)}`
          return <FlowPath d={d} draw={ramp(f, VOICE + 2, VOICE + 18)} flow={ramp(f, VOICE + 6, VOICE + 16)} color={C.sky} kind="frame" packets={3} period={30} label="20 ms frames" labelAt={0.5} labelOffset={42} />
        })()}

        {/* the ring buffer */}
        <g opacity={Math.min(1, ring * 1.3) * back} transform={`translate(${RING.x} ${RING.y}) scale(${0.85 + 0.15 * ring}) translate(${-RING.x} ${-RING.y})`}>
          <circle cx={RING.x} cy={RING.y} r={RING.r} fill="rgba(10,16,30,0.55)" stroke={C.border2} strokeWidth={1.5} />
          <circle cx={RING.x} cy={RING.y} r={RING.r - 26} fill="none" stroke={C.border} strokeWidth={1} />
          {/* Strong: the deeper buffer, a ghost */}
          <path d={arcPath(RING.x, RING.y, RING.r + 26, wA, wA + 306 * ramp(f, HALF + 8, HALF + 30, easeOut))} fill="none" stroke={C.dim} strokeWidth={2} strokeDasharray="3 9" strokeLinecap="round" />
          {/* the audio held: frames between the write head and the play head, moving toward the play head */}
          {Array.from({ length: SLOTS }, (_, i) => {
            const a = (i * 360) / SLOTS + phase
            // how far this slot has travelled from the write head, 0..360
            const behind = (((a - wA) % 360) + 360) % 360
            if (behind > gap) return null
            const age = gap > 0 ? behind / Math.max(gap, 1) : 0
            const edge = Math.min(1, behind / 12, (gap - behind) / 12)
            const p = polar(RING.x, RING.y, RING.r - 13, a)
            const col = interpolateColors(age, [0, 1], [C.sky, C.teal])
            return <rect key={i} x={p.x - 7} y={p.y - 7} width={14} height={14} rx={3} fill={col} opacity={0.9 * Math.max(0, edge)} transform={`rotate(${a} ${p.x} ${p.y})`} />
          })}
          {/* Gentle's delay: the lit arc between the heads */}
          <path d={arcPath(RING.x, RING.y, RING.r, wA, pA)} fill="none" stroke={C.teal} strokeWidth={5} strokeLinecap="round" style={{ filter: `drop-shadow(0 0 8px ${C.teal})` }} />
          {/* heads */}
          {(() => {
            const w = polar(RING.x, RING.y, RING.r, wA)
            const p = polar(RING.x, RING.y, RING.r, pA)
            return (
              <>
                <circle cx={w.x} cy={w.y} r={11} fill={C.sky} style={{ filter: `drop-shadow(0 0 10px ${C.sky})` }} />
                <circle cx={p.x} cy={p.y} r={11} fill={C.teal} stroke="#05110f" strokeWidth={2} style={{ filter: `drop-shadow(0 0 10px ${C.teal})` }} />
              </>
            )
          })()}
        </g>

        {/* buffer -> GPU: the stream, and the window lifted off it */}
        <FlowPath d={`M ${RING.x + RING.r + 30} ${PATH_Y} L ${BLK.x - 12} ${PATH_Y}`} draw={ramp(f, WINDOW - 4, WINDOW + 14)} flow={ramp(f, WINDOW, WINDOW + 10)} color={C.sky} kind="frame" packets={6} period={36} />
        {(() => {
          const w = springAt(f, WINDOW, SNAPPY)
          const x0 = RING.x + RING.r + 40
          const x1 = BLK.x - 112
          const x2 = BLK.x - 26
          const y = PATH_Y - 34 - 8 * w
          return (
            <g opacity={Math.min(1, w * 1.4)}>
              <rect x={x0 - 8} y={y + 12} width={x2 - x0 + 16} height={44} rx={10} fill="rgba(125,211,252,0.07)" stroke={C.sky} strokeOpacity={0.35} strokeDasharray="4 6" />
              {/* 4 s is 20x the lookahead: the bracket breaks (//) so it is not read to scale */}
              {(() => {
                const xm = (x0 + x1 - 4) / 2
                return (
                  <>
                    <path d={`M ${x0} ${y + 8} L ${x0} ${y} L ${xm - 12} ${y} M ${xm + 12} ${y} L ${x1 - 4} ${y} L ${x1 - 4} ${y + 8}`} fill="none" stroke={C.sky} strokeWidth={2.5} />
                    <path d={`M ${xm - 13} ${y + 8} L ${xm - 3} ${y - 8} M ${xm + 3} ${y + 8} L ${xm + 13} ${y - 8}`} fill="none" stroke={C.sky} strokeWidth={2.5} strokeLinecap="round" />
                  </>
                )
              })()}
              <path d={`M ${x1 + 4} ${y + 8} L ${x1 + 4} ${y} L ${x2} ${y} L ${x2} ${y + 8}`} fill="none" stroke={C.teal} strokeWidth={2.5} />
            </g>
          )
        })()}

        {/* five schematic output lanes leave to the right */}
        {CLASS_COLORS.map((col, i) => {
          const y0 = BLK.y + 40 + i * 34
          const pts: string[] = []
          for (let k = 0; k <= 24; k++) {
            const x = BLK.x + BLK.w + 6 + k * 9
            const n = noise2D(`lane${i}`, k * 0.35 - f * 0.06, i) * 9 * ramp(f, LANES + 6, LANES + 24)
            pts.push(`${k ? "L" : "M"} ${x} ${(y0 + n).toFixed(1)}`)
          }
          return (
            <g key={i} opacity={ramp(f, LANES + i * 3, LANES + i * 3 + 10)}>
              <path d={pts.join(" ")} fill="none" stroke={col} strokeWidth={2.5} strokeLinecap="round" style={{ filter: `drop-shadow(0 0 5px ${col})` }} />
              <circle cx={BLK.x + BLK.w + 4} cy={y0} r={4} fill={col} />
            </g>
          )
        })}
      </Diagram>

      {/* ring centre */}
      <div style={{ position: "absolute", left: RING.x - 120, top: RING.y - 80, width: 240, height: 160, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", opacity: ring }}>
        <div style={{ fontFamily: MONO, fontSize: 19, letterSpacing: 3, color: C.dim, textTransform: "uppercase", transform: `translateY(${(1 - half) * 40}px)` }}>delay buffer</div>
        <div style={{ fontFamily: SERIF, fontSize: 104, lineHeight: 1, letterSpacing: -2, color: C.text, opacity: Math.min(1, half * 1.5) * textBack, transform: `scale(${0.7 + 0.3 * half})`, textShadow: `0 0 40px ${C.teal}66` }}>
          <Counter to={0.5} at={HALF} dur={22} decimals={1} prefix="≈" />
          <span style={{ fontSize: 60 }}> s</span>
        </div>
      </div>
      {/* the heads, named: what he says now, and what the call hears (outside the ring, along the radius) */}
      <HeadTag angle={wA} gap={40} w={40} h={24} color={C.sky} opacity={tagNow}>
        now
      </HeadTag>
      <HeadTag angle={pA} gap={40} w={170} h={24} color={C.teal} opacity={tagCall}>
        the call hears
      </HeadTag>

      {/* legend under the ring */}
      <At x={RING.x - 250} y={RING.y + RING.r + 40} style={{ width: 500, display: "flex", justifyContent: "center", gap: 34, fontFamily: MONO, fontSize: 20 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 10, color: C.teal, opacity: half }}>
          <span style={{ width: 28, height: 5, borderRadius: 3, background: C.teal, boxShadow: `0 0 8px ${C.teal}` }} />≈0.5 s · Gentle
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 10, color: C.dim, opacity: ramp(f, HALF + 10, HALF + 22) }}>
          <span style={{ width: 28, borderTop: `2px dashed ${C.dim}` }} />≈0.85 s · Strong
        </span>
      </At>

      {/* the window's label */}
      {(() => {
        const w = springAt(f, WINDOW, SNAPPY)
        const x0 = RING.x + RING.r + 40
        return (
          <At x={x0} y={PATH_Y - 140 - 8 * w} style={{ opacity: Math.min(1, w * 1.3), fontFamily: MONO, fontSize: 20, lineHeight: 1.32, whiteSpace: "nowrap" }}>
            <div style={{ color: C.sky }}>4 s of past audio</div>
            <div style={{ color: C.teal }}>+ 200 ms lookahead</div>
            <div style={{ color: C.dim }}>(Strong: 500 ms)</div>
          </At>
        )
      })()}

      {/* the GPU */}
      <Block
        x={BLK.x}
        y={BLK.y}
        w={BLK.w}
        h={BLK.h}
        title="WavLM-Large · 316M parameters"
        model="25 layers mixed → BiGRU → 5 classes"
        where="RTX 4050 laptop GPU · CUDA"
        icon="chip"
        accent={C.nv}
        nv
        appear={gpu}
        active={active}
      >
        <LightSweep at={NVGPU} dur={34} color="rgba(118,185,0,0.28)" />
      </Block>
      {/* over the lanes' column, its right edge inside the 96 px margin */}
      <At x={1824 - 140} y={BLK.y - 34} style={{ width: 140, textAlign: "right", fontFamily: MONO, fontSize: 20, color: C.dim, opacity: ramp(f, LANES + 8, LANES + 18) }}>
        schematic
      </At>

      {/* timing under the GPU: one window ~60 ms, re-run every 160 ms */}
      <Ruler f={f} at={RULER} />

      {/* the delay budget, to scale: it does not step back, so its numbers stay readable to the end */}
      <Budget f={f} at={BUDGET} />
    </Camera>
    </>
  )
}

// A label riding a head of the ring buffer: its box sits `gap` px outside the ring, along the radius.
function HeadTag({ angle, gap, w, h, color, opacity, children }: { angle: number; gap: number; w: number; h: number; color: string; opacity: number; children: ReactNode }) {
  if (opacity <= 0.001) return null
  const a = deg(angle)
  const r = RING.r + gap + (Math.abs(Math.cos(a)) * w) / 2 + (Math.abs(Math.sin(a)) * h) / 2
  const p = polar(RING.x, RING.y, r, angle)
  return (
    <div
      style={{
        position: "absolute",
        left: p.x - w / 2,
        top: p.y - h / 2,
        width: w,
        height: h,
        lineHeight: `${h}px`,
        textAlign: "center",
        whiteSpace: "nowrap",
        fontFamily: MONO,
        fontSize: 20,
        color,
        opacity,
        textShadow: "0 2px 10px rgba(0,0,0,0.8)",
      }}
    >
      {children}
    </div>
  )
}

// 0..480 ms: the GPU busy ~60 ms in every 160 ms hop (schematic, slowed down).
function Ruler({ f, at }: { f: number; at: number }) {
  const p = springAt(f, at, SNAPPY)
  const W = BLK.w
  const pxms = W / 480
  const period = 26 // frames per hop on screen (slowed: a real hop is 160 ms)
  const t = Math.max(0, f - at - 6)
  const hop = Math.floor(t / period)
  const inHop = (t % period) / period
  return (
    <div style={{ position: "absolute", left: BLK.x, top: BLK.y + BLK.h + 30, width: W, opacity: Math.min(1, p * 1.3), transform: `translateY(${(1 - p) * 20}px)` }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: MONO, fontSize: 19, marginBottom: 10 }}>
        <span style={{ color: C.nv }}>one window ≈ 60 ms</span>
        <span style={{ color: C.dim }}>re-run every 160 ms (slowed)</span>
      </div>
      <div style={{ position: "relative", height: 22, borderRadius: 6, background: "rgba(255,255,255,0.05)", border: `1px solid ${C.border}` }}>
        {[0, 1, 2].map((k) => {
          const lit = k < hop % 4 || (k === hop % 4 && inHop > 0) ? 1 : 0
          const fill = k === hop % 4 ? Math.min(1, inHop / (60 / 160)) : lit
          return (
            <div key={k} style={{ position: "absolute", left: k * 160 * pxms, top: 0, height: 20, width: 60 * pxms, borderRadius: 5, background: "rgba(118,185,0,0.16)", overflow: "hidden" }}>
              <div style={{ width: `${fill * 100}%`, height: "100%", background: C.nv, boxShadow: `0 0 14px ${C.nv}`, opacity: 0.9 }} />
            </div>
          )
        })}
        {[0, 1, 2, 3].map((k) => (
          <div key={k} style={{ position: "absolute", left: k * 160 * pxms - 1, top: -6, width: 2, height: 32, background: C.border2 }} />
        ))}
      </div>
      <div style={{ position: "relative", height: 26, marginTop: 6, fontFamily: MONO, fontSize: 18, color: C.dim }}>
        {[0, 160, 320, 480].map((ms, k) => (
          <span key={ms} style={{ position: "absolute", left: ms * pxms, transform: `translateX(${k === 0 ? 0 : k === 3 ? -100 : -50}%)`, whiteSpace: "nowrap" }}>
            {ms}
            {k === 3 ? " ms" : ""}
          </span>
        ))}
      </div>
    </div>
  )
}

// lookahead 200 + hop 160 + GPU and margin 100 + headroom 80 = 0.54 s (Gentle); 500 + ... = 0.84 s (Strong).
function Budget({ f, at }: { f: number; at: number }) {
  const pxms = 1.4
  const segs = [
    { name: "lookahead", g: 200, s: 500, c: C.teal },
    { name: "hop", g: 160, s: 160, c: C.sky },
    { name: "GPU + margin", g: 100, s: 100, c: C.nv },
    { name: "headroom", g: 80, s: 80, c: "#b8c2d4" },
  ]
  const head = springAt(f, at, SNAPPY)
  let xg = 0
  let xs = 0
  const total = springAt(f, at + 22, SNAPPY)
  const strong = springAt(f, at + 30, SNAPPY)
  return (
    <div style={{ position: "absolute", left: 96, top: 688 }}>
      <div style={{ fontFamily: MONO, fontSize: 20, letterSpacing: 3, color: C.dim, opacity: head, marginBottom: 12 }}>DELAY BUDGET</div>
      <div style={{ position: "relative", height: 64 }}>
        {segs.map((s, i) => {
          const p = springAt(f, at + 4 + i * 4, SNAPPY)
          const x = xg
          xg += s.g * pxms
          return (
            <div
              key={s.name}
              style={{
                position: "absolute",
                left: x,
                top: 0,
                width: s.g * pxms - 4,
                height: 64,
                borderRadius: 10,
                background: `linear-gradient(180deg, ${s.c}38, ${s.c}14)`,
                borderTop: `3px solid ${s.c}`,
                boxSizing: "border-box",
                padding: "6px 10px",
                opacity: Math.min(1, p * 1.4),
                transform: `scaleX(${0.4 + 0.6 * Math.min(1, p)})`,
                transformOrigin: "0 50%",
                overflow: "hidden",
                whiteSpace: "nowrap",
              }}
            >
              <div style={{ fontFamily: SANS, fontSize: 18, fontWeight: 600, color: C.dim, lineHeight: 1.1 }}>{s.name}</div>
              <div style={{ fontFamily: MONO, fontSize: 22, fontWeight: 700, color: C.text, lineHeight: 1.3 }}>{s.g} ms</div>
            </div>
          )
        })}
        <div
          style={{
            position: "absolute",
            left: xg + 22,
            top: 6,
            whiteSpace: "nowrap",
            fontFamily: SANS,
            fontWeight: 700,
            fontSize: 38,
            color: C.teal,
            opacity: Math.min(1, total * 1.3),
            transform: `translateX(${(1 - total) * -20}px)`,
            textShadow: `0 0 24px ${C.teal}55`,
          }}
        >
          = ≈0.54 s <span style={{ fontWeight: 500, fontSize: 26, color: C.dim }}>Gentle</span>
        </div>
      </div>
      <div style={{ position: "relative", height: 34, marginTop: 14, opacity: Math.min(1, strong * 1.3) }}>
        {segs.map((s) => {
          const x = xs
          xs += s.s * pxms
          return (
            <div
              key={s.name}
              style={{
                position: "absolute",
                left: x,
                top: 0,
                width: (s.s * pxms - 4) * Math.min(1, strong),
                height: 34,
                borderRadius: 8,
                background: `${s.c}1c`,
                borderTop: `2px solid ${s.c}88`,
                boxSizing: "border-box",
                padding: "3px 10px",
                fontFamily: MONO,
                fontSize: 18,
                color: C.dim,
                overflow: "hidden",
                whiteSpace: "nowrap",
              }}
            >
              {s.s}
            </div>
          )
        })}
        <div style={{ position: "absolute", left: xs + 22, top: 2, whiteSpace: "nowrap", fontFamily: MONO, fontSize: 22, color: C.dim }}>= ≈0.84 s · Strong</div>
      </div>
    </div>
  )
}

