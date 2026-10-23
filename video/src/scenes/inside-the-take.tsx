import type { CSSProperties, ReactNode } from "react"
import { AbsoluteFill, interpolate, interpolateColors, useCurrentFrame } from "remotion"

import type { Cut, SceneProps, Word } from "../data"
import { lineCue } from "../data"
import {
  Camera,
  clamp,
  Counter,
  easeIn,
  easeOut,
  FrameCells,
  GlowOrb,
  Icon,
  KineticText,
  LightSweep,
  NvTag,
  Particles,
  Pill,
  ProbLanes,
  ramp,
  Sfx,
  SNAPPY,
  springAt,
  Wave,
} from "../kit"
import { C, CLASS_COLORS, MONO, SANS, SERIF } from "../theme"

// Every 20 milliseconds, every word.
// The take laid out as a score: Alex's voice on top, the WavLM teacher's five lanes scored left to
// right (the real per-frame probabilities), then CrisperWhisper's words at their real times and the
// 18 real cuts snapping into the gaps between them.

// ---- the score's geometry (stage px) ----
// The stage is inset so that, at the camera's closest framing (zoom 1.03 plus drift), every label
// keeps 96 px from the frame's left and right edges.
const LEFT = 126
const X0 = 354 // time axis
const X1 = 1794
const AW = X1 - X0

const PILL_Y = 86
const TITLE_Y = 142
const RULER_Y = 236
const WAVE_Y = 262
const WAVE_H = 84
const TAG1_Y = 366
const SCAN_GAP_A = WAVE_Y + WAVE_H + 6 // the scan head skips the tag row
const SCAN_GAP_B = TAG1_Y + 34
const LANES_Y = 410
const LANE = [40, 22] // lane height: scoring, then compressed for the words
const GAP = [10, 4]
const TAG2_Y = 552
// word labels: four rows, top first. Each cluster reads as a descending staircase: the earliest
// word sits highest and every label hangs from a leader at its own left edge.
const TIER_Y = [592, 616, 640, 664]
const LABEL_LH = 22
const CHIP_Y = 698
const CHIP_H = 24
const BOX_PAD = 7
const CUT_TIER_Y = [742, 766]
const COUNT_Y = 802

const STUTTER_LABEL: Record<string, string> = {
  word_rep: "repeated word",
  sound_rep: "broken-off start",
  interjection: "filler",
}

// The magnifier: 40 frames (0.8 s) around the "um", where the filler lane lights.
const ZOOM_T0 = 7.9
const ZOOM_N = 40
const ZOOM_CELL = 21
const ZOOM_GAP = 4
const ZOOM_MARK = 15 // the cell the "20 ms" bracket points at

type Placed = { i: number; left: number; w: number; cx: number; tier: number }
type Flag = { left: number; w: number; lx: number; tier: number }

// Word labels as flags: left-aligned on a leader at the word's start, in the highest row where the
// label fits, no leader from a row above crosses it, and its own leader crosses no label below it.
function placeFlags(items: { lx: number; w: number }[], tiers: number, maxR: number, pad = 6): Flag[] {
  const right: number[] = Array(tiers).fill(-1e9)
  const placed: Flag[] = []
  for (const it of items) {
    const left = Math.min(maxR - it.w, it.lx - 3)
    const fits = (t: number) =>
      left >= right[t] + pad &&
      right.every((r, u) => u <= t || r + pad <= it.lx) &&
      placed.every((q) => q.tier >= t || q.lx < left - 2 || q.lx > left + it.w + 2)
    let tier = right.findIndex((_, t) => fits(t))
    if (tier < 0) tier = right.indexOf(Math.min(...right))
    right[tier] = left + it.w
    placed.push({ left, w: it.w, lx: it.lx, tier })
  }
  return placed
}

// Greedy label tiers: each label takes the lowest tier where it does not overlap its neighbour.
function placeLabels(items: { cx: number; w: number }[], tiers: number, minL: number, maxR: number, pad = 8): Placed[] {
  const right = Array(tiers).fill(-1e9)
  return items.map((it, i) => {
    const left = Math.max(minL, Math.min(maxR - it.w, it.cx - it.w / 2))
    let tier = right.findIndex((r) => left >= r + pad)
    if (tier < 0) tier = right.indexOf(Math.min(...right))
    right[tier] = left + it.w
    return { i, left, w: it.w, cx: it.cx, tier }
  })
}

function Abs({ x, y, children, style }: { x: number; y: number; children?: ReactNode; style?: CSSProperties }) {
  return <div style={{ position: "absolute", left: x, top: y, ...style }}>{children}</div>
}

export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const c = d.clip
  const secs = c.before_s
  const tx = (t: number) => X0 + (t / secs) * AW

  const L0 = lineCue(scene, 0)
  const L1 = lineCue(scene, 1)
  const at0 = (p: number) => L0.at + p * L0.frames
  const at1 = (p: number) => L1.at + p * L1.frames

  // ---- line 0: "On a recording, a WavLM model trained on NVIDIA GPUs scores every twenty milliseconds of his voice."
  const PILL_AT = Math.max(4, L0.at - 6)
  // beats from a word-level alignment of the narration (Magpie): line 0 "a WavLM" 0.15, "trained"
  // 0.37, "NVIDIA" 0.43, "scores" 0.64, "twenty" 0.74; line 1 "every word" 0.25-0.31,
  // "included" 0.47, "so cuts" 0.59-0.63, "between words" 0.81-0.87
  const LANES_AT = at0(0.15) // "a WavLM model"
  const NV_AT = at0(0.43) // "trained on NVIDIA GPUs"
  const SCAN_A = at0(0.2)
  const SCAN_B = at0(0.8)
  const ZOOM_AT = at0(0.64) // "scores every"
  const MS_AT = at0(0.745) // "twenty milliseconds"
  const ZOOM_OUT = Math.min(at0(0.97), L1.at - 14)
  // ---- line 1: "CrisperWhisper writes down every word, stutters included, so cuts land only between words."
  const COMPRESS_AT = L1.at - 4
  const TAG2_AT = L1.at + 2
  const WORDS_A = L1.at + 6
  const WORDS_B = at1(0.42)
  const EVERY_WORD_AT = at1(0.29)
  const INCLUDED_AT = at1(0.47) // "stutters included"
  const CUTS_A = at1(0.63) // "so cuts land only between words"
  // the cuts cascade quickly so the tally lands inside the line and holds ~1 s before the transition
  const CUT_STEP = Math.max(1.25, Math.min(2.2, (at1(0.8) - CUTS_A) / Math.max(1, c.cuts.length)))
  const cutAt = (k: number) => CUTS_A + k * CUT_STEP
  const COUNT_AT = Math.max(at1(0.84), cutAt(c.cuts.length - 1) + 6)
  // once the tally lands, everything but the cut row and the tally steps back
  const rest = ramp(f, COUNT_AT - 4, COUNT_AT + 18)

  // ---- scoring: the scan across the take ----
  const scan = interpolate(f, [SCAN_A, SCAN_B], [0, 1], clamp)
  const scanX = X0 + AW * scan
  const hitAt = (t: number) => SCAN_A + (t / secs) * (SCAN_B - SCAN_A)

  // lanes compress once the words arrive
  const comp = ramp(f, COMPRESS_AT, COMPRESS_AT + 22)
  const lh = LANE[0] + (LANE[1] - LANE[0]) * comp
  const lg = GAP[0] + (GAP[1] - GAP[0]) * comp
  const lanesH = 5 * lh + 4 * lg
  const laneTop = (k: number) => LANES_Y + k * (lh + lg)

  // threshold crossings: a bloom where a lane passes 0.5
  const blooms: { k: number; s: number; e: number; peak: number }[] = []
  for (let k = 0; k < c.classes.length; k++) {
    let s = -1
    let peak = 0
    c.probs.forEach((p, i) => {
      if (p[k] >= 0.5) {
        if (s < 0) {
          s = i
          peak = 0
        }
        peak = Math.max(peak, p[k])
      } else if (s >= 0) {
        blooms.push({ k, s: s * 0.02, e: i * 0.02, peak })
        s = -1
      }
    })
    if (s >= 0) blooms.push({ k, s: s * 0.02, e: c.probs.length * 0.02, peak })
  }

  // ---- words ----
  const words: Word[] = c.words
  const wordAt = (w: Word) => WORDS_A + (w.s / secs) * (WORDS_B - WORDS_A)
  const wordLabels = placeFlags(
    words.map((w) => ({ lx: tx(w.s) + 3, w: w.w.length * 10.4 + 4 })),
    TIER_Y.length,
    X1,
  )

  // ---- cuts ----
  const cuts: Cut[] = c.cuts
  const isStutter = (cut: Cut) => cut.kind !== "pause"
  const stutters = cuts.filter(isStutter)
  const pauses = cuts.filter((x) => !isStutter(x))
  const landed = (k: number) => f >= cutAt(k)
  const cutOf = (w: Word) => cuts.findIndex((x) => isStutter(x) && (w.s + w.e) / 2 >= x.start && (w.s + w.e) / 2 <= x.end)
  const stutterIdx = cuts.map((x, k) => (isStutter(x) ? k : -1)).filter((k) => k >= 0)
  const cutLabels = placeLabels(
    stutterIdx.map((k) => ({ cx: tx((cuts[k].start + cuts[k].end) / 2), w: (STUTTER_LABEL[cuts[k].kind] ?? "stutter").length * 10.9 + 4 })),
    2,
    X0 - 24,
    X1,
    14,
  )
  const clipped = Math.round((c.lost_pct / 100) * words.length)

  // wave: dim until scored, stutter cuts tint rose as their brackets land
  const waveCuts = cuts.map((x, k) => ({ x, k })).filter(({ x, k }) => isStutter(x) && landed(k)).map(({ x }) => x)

  // ---- camera: a slow push-in on the scoring, then a second move down onto the words, cuts and tally ----
  const shots = [
    { at: 0, x: 960, y: 532, zoom: 1 },
    { at: 4, x: 960, y: 522, zoom: 1.025, dur: Math.max(40, L1.at - 10) },
    { at: L1.at - 6, x: 960, y: 585, zoom: 1.03, dur: L1.frames },
  ]
  // the pill exits as the camera moves down past it
  const pillOut = ramp(f, L1.at + 4, L1.at + 18, easeIn)

  // ---- the magnifier ----
  const zIn = springAt(f, ZOOM_AT, SNAPPY)
  const zOut = ramp(f, ZOOM_OUT, ZOOM_OUT + 14, easeIn)
  const zoomO = Math.min(1, zIn * 1.4) * (1 - zOut)
  const lensL = tx(ZOOM_T0)
  const lensR = tx(ZOOM_T0 + ZOOM_N * 0.02)
  const cardW = ZOOM_N * (ZOOM_CELL + ZOOM_GAP) - ZOOM_GAP + 64
  const cardX = Math.max(X0, Math.min(X1 - cardW, (lensL + lensR) / 2 - cardW / 2))
  const cardY = 690
  const markX = 32 + ZOOM_MARK * (ZOOM_CELL + ZOOM_GAP)

  // ---- title ----
  const titleIn = MS_AT - 4

  return (
    <AbsoluteFill>
      <GlowOrb x={960} y={480} r={820} color="rgba(125,211,252,0.16)" opacity={1} />
      <GlowOrb x={tx(14)} y={760} r={520} color="rgba(251,113,133,0.10)" opacity={ramp(f, CUTS_A, CUTS_A + 40)} />
      <Particles count={18} color={C.sky} seed="itt" y={180} h={760} opacity={0.35} rise={0.25} />
      <Camera shots={shots} drift={4} seed="itt">
        {/* ---------- header ---------- */}
        <Abs x={LEFT} y={PILL_Y} style={{ opacity: Math.min(1, springAt(f, PILL_AT) * 1.3) * (1 - pillOut), transform: `translateY(${(1 - springAt(f, PILL_AT)) * 18 - pillOut * 14}px)` }}>
          <Pill color={C.sky} border="rgba(125,211,252,0.45)" style={{ fontSize: 24, padding: "8px 20px", fontWeight: 600 }}>
            <Icon name="wave" size={24} color={C.sky} />
            Thorough · recording
          </Pill>
        </Abs>
        <Abs x={LEFT} y={TITLE_Y} style={{ display: "flex", alignItems: "baseline", gap: 18, fontFamily: SERIF, whiteSpace: "nowrap" }}>
          <KineticText text="Every 20 milliseconds," at={titleIn} per={3} effect="blur" style={{ fontFamily: "inherit", fontSize: 70, letterSpacing: -1, lineHeight: 1, color: C.text }} highlight={{ "20": C.sky }} />
          <KineticText text="every word." at={EVERY_WORD_AT - 3} per={4} effect="blur" style={{ fontFamily: "inherit", fontStyle: "italic", fontSize: 70, letterSpacing: -1, lineHeight: 1, color: C.text }} highlight={{ word: C.teal }} />
        </Abs>

        {/* ---------- ruler: one tick a second ---------- */}
        <div style={{ position: "absolute", inset: 0, opacity: ramp(f, 2, 22, easeOut) }}>
          {Array.from({ length: Math.floor(secs) + 1 }, (_, s) => (
            <div key={s}>
              <div style={{ position: "absolute", left: tx(s) - 0.5, top: RULER_Y + 18, width: 1, height: s % 5 === 0 ? 10 : 5, background: "rgba(255,255,255,0.35)" }} />
              <div
                style={{
                  position: "absolute",
                  left: tx(s),
                  top: WAVE_Y - 4,
                  width: 1,
                  height: LANES_Y + lanesH + 12 - WAVE_Y,
                  borderLeft: `1px ${s % 5 === 0 ? "solid" : "dashed"} rgba(255,255,255,${s % 5 === 0 ? 0.07 : 0.035})`,
                }}
              />
              {s % 5 === 0 && (
                <div style={{ position: "absolute", left: tx(s) + 6, top: RULER_Y - 4, fontFamily: MONO, fontSize: 18, color: C.dim }}>
                  {`0:${String(s).padStart(2, "0")}`}
                </div>
              )}
            </div>
          ))}
          <div style={{ position: "absolute", left: X0, width: AW, top: RULER_Y + 18, height: 1, background: "rgba(255,255,255,0.18)" }} />
          <div style={{ position: "absolute", left: X1 - 120, width: 120, top: RULER_Y - 4, textAlign: "right", fontFamily: MONO, fontSize: 18, color: C.dim }}>
            {secs.toFixed(2)} s
          </div>
        </div>

        {/* ---------- row A: his voice ---------- */}
        <RowLabel y={WAVE_Y + WAVE_H / 2} at={2} color={C.sky} icon="mic">
          his voice
        </RowLabel>
        <Abs x={X0} y={WAVE_Y} style={{ opacity: 1 - 0.4 * rest }}>
          <Wave
            env={c.env_before}
            seconds={secs}
            width={AW}
            height={WAVE_H}
            color={C.sky}
            reveal={ramp(f, 0, 34, easeOut)}
            progress={scan}
            dimAfter
            cuts={waveCuts}
            cutTint={0.5}
            cutColor={C.rose}
            glow={4}
          />
        </Abs>

        {/* ---------- rows B-F: the WavLM teacher ---------- */}
        {c.classes.map((name, k) => (
          <div
            key={name}
            style={{
              position: "absolute",
              left: LEFT,
              top: laneTop(k),
              height: lh,
              display: "flex",
              alignItems: "center",
              gap: 12,
              fontFamily: MONO,
              fontSize: 20,
              whiteSpace: "nowrap",
              color: C.dim,
              opacity: Math.min(1, springAt(f, LANES_AT + 3) * 1.2),
            }}
          >
            <span style={{ width: 12, height: 12, borderRadius: 6, background: CLASS_COLORS[k], boxShadow: `0 0 10px ${CLASS_COLORS[k]}` }} />
            {name}
          </div>
        ))}
        <div style={{ position: "absolute", left: X0, top: LANES_Y, opacity: Math.min(1, springAt(f, LANES_AT + 3) * 1.2) * (1 - 0.25 * comp) * (1 - 0.47 * rest) }}>
          <ProbLanes
            probs={c.probs}
            classes={c.classes}
            width={AW}
            labels={false}
            laneHeight={lh}
            gap={lg}
            progress={scan}
            threshold={0.5}
          />
        </div>
        {/* blooms where a lane crosses the threshold */}
        {blooms.map((b, i) => {
          const h = hitAt(b.s)
          if (f < h) return null
          const flash = Math.max(0, 1 - (f - h) / 22)
          const o = (0.32 + 0.68 * flash) * (1 - 0.4 * comp) * (1 - 0.5 * rest)
          const col = CLASS_COLORS[b.k]
          const x0 = tx(b.s)
          const w = Math.max(6, tx(b.e) - x0)
          return (
            <div
              key={i}
              style={{
                position: "absolute",
                left: x0 - 4,
                top: laneTop(b.k) - 4,
                width: w + 8,
                height: lh + 8,
                borderRadius: 10,
                border: `1.5px solid ${col}`,
                background: `${col}1f`,
                boxShadow: `0 0 ${10 + 30 * flash}px ${col}, inset 0 0 ${12 * flash}px ${col}`,
                opacity: o,
                transform: `scale(${1 + 0.18 * flash})`,
              }}
            />
          )
        })}
        {/* the scan head */}
        {scan > 0 && scan < 1 && (
          <>
            {[
              [WAVE_Y - 10, SCAN_GAP_A],
              [SCAN_GAP_B, LANES_Y + lanesH + 10],
            ].map(([a, b], j) => (
              <div
                key={j}
                style={{
                  position: "absolute",
                  left: scanX - 140,
                  width: 140,
                  top: a,
                  height: b - a,
                  background: "linear-gradient(90deg, transparent, rgba(125,211,252,0.13))",
                }}
              />
            ))}
            {/* a readout riding the scan head: the teacher's frames counting by, until the magnifier takes over */}
            <div
              style={{
                position: "absolute",
                left: Math.min(X1 - 250, scanX - 125),
                width: 250,
                top: LANES_Y + lanesH + 18,
                textAlign: "center",
                opacity: Math.min(ramp(f, SCAN_A, SCAN_A + 8), 1 - ramp(f, ZOOM_AT - 10, ZOOM_AT)),
              }}
            >
              <span
                style={{
                  display: "inline-block",
                  padding: "6px 14px",
                  borderRadius: 10,
                  background: "rgba(10,14,26,0.85)",
                  border: `1px solid rgba(125,211,252,0.45)`,
                  fontFamily: MONO,
                  fontSize: 20,
                  color: C.dim,
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                frame <span style={{ color: C.text, fontWeight: 700 }}>{Math.min(c.probs.length, Math.floor(scan * c.probs.length) + 1)}</span> of {c.probs.length}
              </span>
            </div>
            {[
              [WAVE_Y - 16, SCAN_GAP_A],
              [SCAN_GAP_B, LANES_Y + lanesH + 16],
            ].map(([a, b], j) => (
              <div
                key={j}
                style={{
                  position: "absolute",
                  left: scanX - 1.5,
                  width: 3,
                  top: a,
                  height: b - a,
                  borderRadius: 2,
                  background: C.text,
                  boxShadow: `0 0 16px ${C.sky}, 0 0 40px ${C.sky}`,
                }}
              />
            ))}
          </>
        )}

        {/* the tag row sits above the scan head (which also skips it) */}
        <Abs
          x={X0}
          y={TAG1_Y}
          style={{ display: "flex", alignItems: "center", gap: 14, opacity: Math.min(1, springAt(f, LANES_AT) * 1.3), transform: `translateX(${(1 - springAt(f, LANES_AT)) * -24}px)` }}
        >
          <span style={{ fontFamily: MONO, fontSize: 21, color: C.text, fontWeight: 500 }}>WavLM teacher</span>
          <span style={{ fontFamily: MONO, fontSize: 21, color: C.dim }}>· 5 classes every 20 ms</span>
          <span
            style={{
              opacity: Math.min(1, springAt(f, NV_AT, SNAPPY) * 1.5),
              transform: `scale(${0.7 + 0.3 * springAt(f, NV_AT, SNAPPY)})`,
              display: "inline-block",
              boxShadow: `0 0 ${26 * Math.max(0, 1 - Math.abs(f - NV_AT - 8) / 24)}px ${C.nv}`,
              borderRadius: 6,
              marginLeft: 6,
            }}
          >
            <NvTag label="trained on NVIDIA GPUs" style={{ fontSize: 18 }} />
          </span>
        </Abs>

        {/* ---------- the magnifier: 40 frames of 20 ms ---------- */}
        {zoomO > 0.01 && (
          <div style={{ position: "absolute", inset: 0, opacity: zoomO }}>
            <svg width={1920} height={1080} style={{ position: "absolute", left: 0, top: 0, overflow: "visible" }}>
              <defs>
                <linearGradient id="itt-zoom" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor={C.text} stopOpacity={0.02} />
                  <stop offset="100%" stopColor={C.text} stopOpacity={0.1} />
                </linearGradient>
              </defs>
              <path
                d={`M ${lensL} ${LANES_Y + lanesH + 6} L ${lensR} ${LANES_Y + lanesH + 6} L ${cardX + cardW - 14} ${cardY} L ${cardX + 14} ${cardY} Z`}
                fill="url(#itt-zoom)"
                stroke="rgba(245,247,251,0.32)"
                strokeWidth={1.2}
              />
            </svg>
            <div
              style={{
                position: "absolute",
                left: lensL - 3,
                top: LANES_Y - 8,
                width: lensR - lensL + 6,
                height: lanesH + 14,
                borderRadius: 10,
                border: `2px solid ${C.text}`,
                boxShadow: `0 0 24px rgba(245,247,251,0.45), inset 0 0 20px rgba(245,247,251,0.12)`,
              }}
            />
            <div
              style={{
                position: "absolute",
                left: cardX,
                top: cardY,
                width: cardW,
                padding: "18px 32px 18px",
                boxSizing: "border-box",
                borderRadius: 22,
                background: "linear-gradient(160deg, rgba(20,28,46,0.94), rgba(10,14,26,0.94))",
                border: `1px solid ${C.border2}`,
                boxShadow: "0 30px 80px -30px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.08)",
                transform: `translateY(${(1 - zIn) * -40}px) scale(${(0.7 + 0.3 * Math.min(1, zIn)) * (1 - 0.25 * zOut)})`,
                transformOrigin: "50% 0%",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 14, fontFamily: MONO, fontSize: 20, color: C.dim, marginBottom: 14 }}>
                <span style={{ color: C.text }}>0.8 s of his voice</span>
                <span>· one cell per frame · {ZOOM_N} frames</span>
                <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 12, height: 12, borderRadius: 6, background: CLASS_COLORS[4], boxShadow: `0 0 10px ${CLASS_COLORS[4]}` }} />
                  {c.classes[4]}
                </span>
              </div>
              <FrameCells probs={c.probs} from={Math.round(ZOOM_T0 / 0.02)} count={ZOOM_N} cell={ZOOM_CELL} gap={ZOOM_GAP} progress={ramp(f, ZOOM_AT + 4, MS_AT + 4, easeOut)} />
              <div style={{ position: "relative", height: 44, opacity: ramp(f, MS_AT, MS_AT + 8) }}>
                <div
                  style={{
                    position: "absolute",
                    left: markX - 32 - 1,
                    top: 6,
                    width: ZOOM_CELL + 2,
                    height: 9,
                    borderLeft: `2px solid ${C.text}`,
                    borderRight: `2px solid ${C.text}`,
                    borderBottom: `2px solid ${C.text}`,
                    borderRadius: "0 0 4px 4px",
                    boxSizing: "border-box",
                  }}
                />
                <div
                  style={{
                    position: "absolute",
                    left: markX - 32 + ZOOM_CELL / 2 - 60,
                    width: 120,
                    top: 16,
                    textAlign: "center",
                    fontFamily: MONO,
                    fontWeight: 700,
                    fontSize: 24,
                    color: C.text,
                    transform: `translateY(${(1 - springAt(f, MS_AT, SNAPPY)) * 10}px)`,
                  }}
                >
                  20 ms
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ---------- row G: CrisperWhisper's words ---------- */}
        <Abs
          x={X0}
          y={TAG2_Y}
          style={{ display: "flex", alignItems: "center", gap: 14, opacity: Math.min(1, springAt(f, TAG2_AT) * 1.3), transform: `translateX(${(1 - springAt(f, TAG2_AT)) * -24}px)` }}
        >
          <span style={{ fontFamily: MONO, fontSize: 21, color: C.text, fontWeight: 500 }}>CrisperWhisper</span>
          <span style={{ fontFamily: MONO, fontSize: 21, color: C.dim }}>· every word with its timing</span>
        </Abs>
        <RowLabel y={CHIP_Y + CHIP_H / 2} at={TAG2_AT} color={C.teal} icon="text">
          words
        </RowLabel>
        {words.map((w, i) => {
          const at = wordAt(w)
          const p = springAt(f, at, SNAPPY)
          if (p <= 0.001) return null
          const pl = wordLabels[i]
          const x0 = tx(w.s)
          const cw = Math.max(5, tx(w.e) - x0 - 2)
          const ci = cutOf(w)
          const struck = ci >= 0 && landed(ci) ? ramp(f, cutAt(ci) + 2, cutAt(ci) + 12, easeOut) : 0
          const glint = w.removed ? Math.max(0, 1 - Math.abs(f - INCLUDED_AT - 8) / 14) : 0
          const ty = (1 - p) * -36
          const o = Math.min(1, p * 1.5)
          const labelY = TIER_Y[pl.tier]
          return (
            <div key={i} style={{ position: "absolute", inset: 0, opacity: o, transform: `translateY(${ty}px)` }}>
              <div
                style={{
                  position: "absolute",
                  left: pl.lx - 0.75,
                  top: labelY + LABEL_LH,
                  width: 1.5,
                  height: CHIP_Y - labelY - LABEL_LH - 1,
                  background: `rgba(245,247,251,${0.24 * (1 - 0.75 * rest)})`,
                }}
              />
              <div
                style={{
                  position: "absolute",
                  left: pl.left,
                  top: labelY,
                  whiteSpace: "nowrap",
                  fontFamily: SANS,
                  fontWeight: 500,
                  fontSize: 20,
                  lineHeight: `${LABEL_LH}px`,
                  color: struck > 0 ? interpolateColors(struck, [0, 1], [C.text, C.dim]) : C.text,
                  opacity: 1 - 0.15 * rest,
                  textShadow: glint > 0 ? `0 0 ${14 * glint}px ${C.sky}` : undefined,
                }}
              >
                <span style={{ position: "relative" }}>
                  {w.w}
                  {struck > 0 && (
                    <span style={{ position: "absolute", left: -1, right: -1, top: LABEL_LH, height: 2, borderRadius: 1, background: C.rose, transform: `scaleX(${struck})`, transformOrigin: "0 50%" }} />
                  )}
                </span>
              </div>
              <div
                style={{
                  position: "absolute",
                  left: x0 + 1,
                  top: CHIP_Y,
                  width: cw,
                  height: CHIP_H,
                  borderRadius: 6,
                  background: `linear-gradient(180deg, rgba(125,211,252,0.55), rgba(94,234,212,0.35))`,
                  border: `1px solid rgba(125,211,252,0.85)`,
                  boxSizing: "border-box",
                  boxShadow: `0 0 ${8 + 20 * glint}px rgba(125,211,252,${0.35 + 0.5 * glint})`,
                }}
              />
            </div>
          )
        })}

        {/* ---------- the cuts: brackets in time order ---------- */}
        {cuts.map((cut, k) => {
          const at = cutAt(k)
          const p = springAt(f, at, SNAPPY)
          if (p <= 0.001) return null
          const stutter = isStutter(cut)
          const col = stutter ? C.rose : "rgba(245,247,251,0.55)"
          const x0 = tx(cut.start)
          const x1 = tx(cut.end)
          const spread = (1 - Math.min(1, p)) * 16
          const guide = Math.max(0, 1 - (f - at) / 26) * (stutter ? 0.75 : 0.45)
          // a pause is shortened, not removed: a solid segment shows the ~0.45 s that stays
          const keepW = Math.min(tx(0.45) - tx(0), (x1 - x0) * 0.45)
          const showArrow = !stutter && x1 - x0 - keepW >= 26
          return (
            <div key={k} style={{ position: "absolute", inset: 0 }}>
              {guide > 0.01 &&
                [x0, x1].map((gx, j) => (
                  <div
                    key={j}
                    style={{
                      position: "absolute",
                      left: gx - 0.75,
                      top: WAVE_Y - 6,
                      width: 1.5,
                      height: CHIP_Y + CHIP_H - WAVE_Y + 12,
                      background: `linear-gradient(180deg, transparent, ${stutter ? C.rose : C.text})`,
                      opacity: guide,
                    }}
                  />
                ))}
              <div
                style={{
                  position: "absolute",
                  left: x0 - spread - 1,
                  top: CHIP_Y - BOX_PAD + (1 - p) * -30,
                  width: x1 - x0 + spread * 2 + 2,
                  height: CHIP_H + BOX_PAD * 2,
                  borderRadius: 8,
                  border: stutter ? `2px solid ${C.rose}` : `1.5px dashed ${col}`,
                  background: stutter ? "rgba(251,113,133,0.14)" : "rgba(245,247,251,0.04)",
                  boxShadow: stutter ? `0 0 ${18 * Math.max(0, 1 - (f - at) / 20) + 8}px rgba(251,113,133,0.55)` : undefined,
                  boxSizing: "border-box",
                  opacity: Math.min(1, p * 1.6),
                }}
              >
                {!stutter && (
                  <>
                    <div style={{ position: "absolute", left: 3, top: "50%", marginTop: -3, height: 6, width: Math.max(3, keepW - 4), borderRadius: 3, background: "rgba(245,247,251,0.55)" }} />
                    {showArrow && (
                      <div style={{ position: "absolute", left: keepW + 4, right: 8, top: "50%", height: 0, borderTop: "1.5px solid rgba(245,247,251,0.5)" }}>
                        <div style={{ position: "absolute", left: -1, top: -5.5, width: 0, height: 0, borderTop: "4.5px solid transparent", borderBottom: "4.5px solid transparent", borderRight: "7px solid rgba(245,247,251,0.6)" }} />
                      </div>
                    )}
                  </>
                )}
              </div>
              {stutter && <Sfx name="snap" at={at} volume={0.16} />}
            </div>
          )
        })}
        {cutLabels.map((pl, j) => {
          const k = stutterIdx[j]
          const at = cutAt(k) + 3
          const p = springAt(f, at, SNAPPY)
          if (p <= 0.001) return null
          const cut = cuts[k]
          const cx = tx((cut.start + cut.end) / 2)
          const y = CUT_TIER_Y[pl.tier]
          return (
            <div key={k} style={{ position: "absolute", inset: 0, opacity: Math.min(1, p * 1.5), transform: `translateY(${(1 - p) * 10}px)` }}>
              <div style={{ position: "absolute", left: cx - 0.75, top: CHIP_Y + CHIP_H + BOX_PAD, width: 1.5, height: y - (CHIP_Y + CHIP_H + BOX_PAD) + 2, background: "rgba(251,113,133,0.6)" }} />
              <div style={{ position: "absolute", left: pl.left, top: y, width: pl.w, textAlign: "center", fontFamily: MONO, fontSize: 18, lineHeight: "22px", color: C.rose, whiteSpace: "nowrap" }}>
                {STUTTER_LABEL[cut.kind] ?? "stutter"}
              </div>
            </div>
          )
        })}

        {/* ---------- the tally ---------- */}
        <GlowOrb x={X0 + 470} y={COUNT_Y + 32} r={380} color="rgba(110,231,183,0.20)" opacity={rest} />
        <Tally at={COUNT_AT} stutters={stutters.length} pauses={pauses.length} clipped={clipped} />
      </Camera>
      <Sfx name="pop" at={ZOOM_AT} volume={0.22} />
      <Sfx name="chime" at={COUNT_AT + 2} volume={0.32} />
    </AbsoluteFill>
  )
}

function RowLabel({ y, at, color, icon, children }: { y: number; at: number; color: string; icon: "mic" | "text"; children: ReactNode }) {
  const f = useCurrentFrame()
  const p = springAt(f, at)
  return (
    <div
      style={{
        position: "absolute",
        left: LEFT,
        top: y - 16,
        height: 32,
        display: "flex",
        alignItems: "center",
        gap: 10,
        fontFamily: SANS,
        fontWeight: 600,
        fontSize: 24,
        color,
        opacity: Math.min(1, p * 1.3),
        transform: `translateX(${(1 - p) * -20}px)`,
      }}
    >
      <Icon name={icon} size={24} color={color} />
      {children}
    </div>
  )
}

function Tally({ at, stutters, pauses, clipped }: { at: number; stutters: number; pauses: number; clipped: number }) {
  const f = useCurrentFrame()
  const p = springAt(f, at, SNAPPY)
  if (p <= 0.001) return null
  const item = (n: number, label: string, color: string, delay: number) => {
    const q = springAt(f, at + delay, SNAPPY)
    return (
      <span style={{ display: "inline-flex", alignItems: "baseline", gap: 10, opacity: Math.min(1, q * 1.5), transform: `translateY(${(1 - q) * 12}px)` }}>
        <Counter to={n} at={at + delay} dur={12} style={{ fontFamily: SANS, fontWeight: 800, fontSize: 38, color }} />
        <span style={{ fontFamily: SANS, fontWeight: 600, fontSize: 28, color: C.text }}>{label}</span>
      </span>
    )
  }
  const dot = <span style={{ fontFamily: SANS, fontSize: 28, color: C.faint }}>·</span>
  return (
    <div
      style={{
        position: "absolute",
        left: X0 - 26,
        top: COUNT_Y,
        display: "flex",
        alignItems: "center",
        gap: 22,
        padding: "8px 30px 10px",
        borderRadius: 999,
        background: "linear-gradient(160deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03))",
        border: `1px solid ${C.border2}`,
        boxShadow: "0 24px 60px -24px rgba(0,0,0,0.85), 0 0 50px -16px rgba(110,231,183,0.5)",
        opacity: Math.min(1, p * 1.4),
        transform: `translateY(${(1 - p) * 24}px) scale(${0.94 + 0.06 * p})`,
        overflow: "hidden",
      }}
    >
      {item(stutters, "stutters cut", C.rose, 0)}
      {dot}
      {item(pauses, "long pauses shortened", C.dim, 3)}
      {dot}
      {item(clipped, "words clipped", C.green, 6)}
      <Icon name="check" size={30} color={C.green} />
      <LightSweep at={at + 10} dur={34} color="rgba(255,255,255,0.18)" />
    </div>
  )
}
