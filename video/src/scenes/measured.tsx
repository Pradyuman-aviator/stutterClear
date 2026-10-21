import { AbsoluteFill, interpolate, interpolateColors, useCurrentFrame } from "remotion"

import type { Cue, SceneProps } from "../data"
import { lineCue } from "../data"
import {
  Camera,
  clamp,
  easeIn,
  easeOut,
  Eyebrow,
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

// Scene 11: Measured on unseen speakers. One chart on a shared 0-100% scale: each row's bars grow
// exactly when the narration names it (Gentle, Strong, Recordings); the 0% words lost gets a green
// pulse on "lose no words".

// The frame at which `word` is spoken inside a narration line, estimated by its position in the text
// (the narration is re-voiced later, so never hard-code seconds).
function wordAt(cue: Cue, word: string, lead = 3): number {
  const text = cue.text ?? ""
  const i = text.indexOf(word)
  if (i < 0 || !text.length) return cue.at
  return Math.round(cue.at + (i / text.length) * cue.frames) - lead
}

// layout (stage px)
// inset past the 96 px safe margin so the camera's push (max zoom 1.02) never breaks it
const L = 124
const RIGHT = 1920 - 124
const BAR_X = 640 // where 0% is
const BAR_W = 980 // the 0-100% scale
const ROW_TOP = 346
const ROW_H = 150
const ROW_GAP = 14
const GROW = 34 // frames a bar takes to grow

interface Row {
  key: string
  mode: string
  name: string
  note: string
  icon: IconName
  removed: number
  lost: number
  at: number // the frame the narration names it
}

export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const n = d.numbers
  const l0 = lineCue(scene, 0)
  const l1 = lineCue(scene, 1)

  const tGentle = wordAt(l0, "Gentle")
  const tStrong = wordAt(l0, "Strong", 6) // Magpie says "and Strong" a touch faster than the text estimate
  const tRec = wordAt(l1, "Recordings", 0)
  const tNoLoss = wordAt(l1, "lose no words")
  const tNever = wordAt(l0, "never")

  const rows: Row[] = [
    { key: "gentle", mode: "LIVE", name: "Gentle", note: "Live delay ≈0.5 s", icon: "headphones", removed: n.live_gentle_removed, lost: n.live_gentle_lost, at: tGentle },
    { key: "strong", mode: "LIVE", name: "Strong", note: "Live delay ≈0.85 s", icon: "headphones", removed: n.live_strong_removed, lost: n.live_strong_lost, at: tStrong },
    { key: "rec", mode: "RECORDINGS", name: "Thorough", note: "Cut only between words", icon: "wave", removed: n.rec_thorough_removed, lost: n.rec_thorough_lost, at: tRec },
  ]

  // the opening overlaps the previous scene's transition: the header is already arriving at frame 0 so
  // the title and the chart are settled and readable by about frame 20, before the first words land
  const headerIn = -4

  // the camera: a slow push-in that leans toward the row being named
  const rowY = (i: number) => ROW_TOP + i * (ROW_H + ROW_GAP) + ROW_H / 2
  const shots = [
    { at: 0, x: 960, y: 540, zoom: 0.985 },
    { at: 1, x: 960, y: 540, zoom: 1.0, dur: 70 },
    { at: tGentle - 6, x: 960, y: 536, zoom: 1.008, dur: 60 },
    { at: tStrong - 6, x: 960, y: 540, zoom: 1.012, dur: 50 },
    { at: tRec - 6, x: 960, y: 544, zoom: 1.016, dur: 60 },
    { at: tNoLoss + 16, x: 960, y: 548, zoom: 1.02, dur: 34 },
  ]

  const rowsIn = 2
  const ring = tNoLoss + 2
  const footIn = tRec + GROW + 8
  // the light settles well before the last ~20 frames, so the scene ends on a clean hold
  const settleAt = ring + 20
  // a moving spotlight on the row being named; it stays (softer) on Recordings so the green 0% is the
  // focal point of the final hold
  const spot = interpolate(f, [tGentle - 4, tStrong - 4, tRec - 4], [rowY(0), rowY(1), rowY(2)], { ...clamp, easing: easeOut })
  const spotOn = ramp(f, tGentle - 6, tGentle + 10) * (1 - 0.4 * ramp(f, settleAt, settleAt + 20))

  return (
    <AbsoluteFill>
      <Particles count={18} color={C.teal} seed="measured" x={BAR_X} y={ROW_TOP - 40} w={BAR_W} h={540} rise={0.25} opacity={0.35} size={[1.5, 3]} />
      <Camera shots={shots} drift={5} seed="measured">
        <GlowOrb x={430} y={210} r={520} color="rgba(125,211,252,0.16)" opacity={ramp(f, headerIn, headerIn + 20)} />
        <GlowOrb x={1080} y={spot} r={620} color="rgba(94,234,212,0.13)" opacity={spotOn} />

        <Header f={f} at={headerIn} speakers={n.speakers} mark={tNever} />
        <Scale f={f} at={rowsIn} />

        {rows.map((r, i) => (
          <RowCard key={r.key} f={f} row={r} i={i} enter={rowsIn + 2 + i * 3} next={rows[i + 1]?.at} settle={settleAt} ring={r.key === "rec" ? ring : undefined} />
        ))}

        <Footer f={f} at={footIn} f1={n.f1_sep28k} />
      </Camera>
      <Sfx name="chime" at={ring} volume={0.4} />
    </AbsoluteFill>
  )
}

function Header({ f, at, speakers, mark }: { f: number; at: number; speakers: number; mark: number }) {
  const words = ["Measured", "on", String(speakers), "speakers", "no", "model", "trained", "on"]
  const eb = springAt(f, at)
  // on "never": the unseen-speaker words brighten and a sky underline sweeps under them
  const hi = ramp(f, mark, mark + 14, easeOut)
  const renderWord = (w: string, i: number) => {
    const p = springAt(f, at + 2 + i * 2, SNAPPY)
    const isNum = i === 2
    const unseen = i >= 4
    return (
      <span
        key={i}
        style={{
          display: "inline-block",
          whiteSpace: "pre",
          opacity: Math.min(1, p * 1.4),
          transform: `translateY(${(1 - p) * 0.4}em)`,
          filter: `blur(${(1 - Math.min(1, p)) * 8}px)`,
          color: unseen ? interpolateColors(hi, [0, 1], [C.dim, C.text]) : C.text,
          fontStyle: unseen ? "italic" : "normal",
        }}
      >
        {isNum ? (
          <span
            style={{
              background: `linear-gradient(120deg, ${C.sky}, ${C.teal})`,
              WebkitBackgroundClip: "text",
              backgroundClip: "text",
              color: "transparent",
              fontSize: 104,
              lineHeight: 0.8,
            }}
          >
            {w}
          </span>
        ) : (
          w
        )}
        {i < words.length - 1 && i !== 3 ? " " : ""}
      </span>
    )
  }
  return (
    <div style={{ position: "absolute", left: L, top: 126, width: 1728 }}>
      <div style={{ opacity: eb, transform: `translateY(${(1 - eb) * 14}px)` }}>
        <Eyebrow color={C.sky} style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <span style={{ width: 34, height: 2, background: C.sky, display: "inline-block", transform: `scaleX(${eb})`, transformOrigin: "0 50%" }} />
          The numbers · UCLASS
        </Eyebrow>
      </div>
      <div style={{ fontFamily: SERIF, fontSize: 84, lineHeight: 1.04, letterSpacing: -1, color: C.text, marginTop: 14, whiteSpace: "nowrap" }}>
        {words.slice(0, 4).map((w, i) => renderWord(w, i))}
        <span style={{ display: "inline-block", whiteSpace: "pre" }}> </span>
        <span style={{ position: "relative", display: "inline-block", whiteSpace: "nowrap" }}>
          {words.slice(4).map((w, k) => renderWord(w, k + 4))}
          <span
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              bottom: 4,
              height: 3,
              borderRadius: 2,
              background: `linear-gradient(90deg, ${C.sky}, ${C.teal})`,
              boxShadow: `0 0 12px ${C.sky}88`,
              transform: `scaleX(${hi})`,
              transformOrigin: "0 50%",
            }}
          />
        </span>
      </div>
    </div>
  )
}

// The shared 0-100% scale: a legend, tick labels and gridlines through every row.
function Scale({ f, at }: { f: number; at: number }) {
  const ticks = [0, 25, 50, 75, 100]
  const grid = ramp(f, at, at + 18, easeOut)
  const top = ROW_TOP - 34
  const bottom = ROW_TOP + 3 * ROW_H + 2 * ROW_GAP + 10
  const lg = springAt(f, at + 2)
  return (
    <>
      <div style={{ position: "absolute", left: L, top: ROW_TOP - 52, display: "flex", gap: 30, alignItems: "center", fontFamily: SANS, fontSize: 24, fontWeight: 500, color: C.dim, opacity: lg, transform: `translateY(${(1 - lg) * 10}px)` }}>
        <span style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ width: 30, height: 14, borderRadius: 7, background: `linear-gradient(90deg, ${C.sky}, ${C.teal})`, boxShadow: `0 0 14px ${C.teal}66` }} />
          stutters removed
        </span>
        <span style={{ color: C.faint }}>·</span>
        <span style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ width: 30, height: 8, borderRadius: 4, background: C.rose }} />
          words lost
        </span>
      </div>
      {ticks.map((t, i) => {
        const x = BAR_X + (BAR_W * t) / 100
        const o = ramp(f, at + i * 2, at + 10 + i * 2, easeOut)
        return (
          <div key={t}>
            <div
              style={{
                position: "absolute",
                left: x,
                top,
                width: 1,
                height: (bottom - top) * grid,
                background: t === 0 ? "rgba(255,255,255,0.22)" : "linear-gradient(180deg, rgba(255,255,255,0.13), rgba(255,255,255,0.05))",
              }}
            />
            <div style={{ position: "absolute", left: x - 60, width: 120, top: top - 32, textAlign: "center", fontFamily: MONO, fontSize: 22, color: C.dim, opacity: o }}>
              {t}
              {t === 100 ? "%" : ""}
            </div>
          </div>
        )
      })}
    </>
  )
}

function RowCard({ f, row, i, enter, next, settle, ring }: { f: number; row: Row; i: number; enter: number; next?: number; settle: number; ring?: number }) {
  const top = ROW_TOP + i * (ROW_H + ROW_GAP)
  const inP = springAt(f, enter)
  // lit from the moment it is named; the "current" glow passes to the next row and fades at the end
  const lit = ramp(f, row.at - 4, row.at + 10, easeOut)
  // the row with the ring keeps half its light through the final hold, so the green 0% stays focal
  const settleDrop = ring === undefined ? 1 : 0.5
  const current = lit * (next === undefined ? 1 : 1 - ramp(f, next - 4, next + 12)) * (1 - settleDrop * ramp(f, settle, settle + 20))
  const removed = interpolate(f, [row.at, row.at + GROW], [0, row.removed], { ...clamp, easing: easeOut })
  const lost = interpolate(f, [row.at + 6, row.at + 6 + GROW], [0, row.lost], { ...clamp, easing: easeOut })
  const growing = f >= row.at && f <= row.at + GROW + 4
  const wR = (BAR_W * removed) / 100
  const wL = (BAR_W * lost) / 100
  const landed = ramp(f, row.at + GROW - 6, row.at + GROW + 8)
  const flash = Math.max(0, 1 - Math.abs(f - (row.at + GROW - 4)) / 14)
  const ringP = ring === undefined ? 0 : ramp(f, ring, ring + 10)
  const lostColor = ring === undefined ? C.rose : interpolateColors(ringP, [0, 1], [C.rose, C.green])
  const rpGlow = ring === undefined ? 0 : Math.min(1, springAt(f, ring, SNAPPY))
  // once the ring fires, the lit border turns from teal to the green of a good result
  const borderLit = interpolateColors(ringP, [0, 1], ["rgba(94,234,212,0.42)", "rgba(110,231,183,0.6)"])
  const haloLit = ringP > 0 ? `rgba(110,231,183,${0.55 * current})` : `rgba(94,234,212,${0.55 * current})`

  const rowOpacity = Math.min(1, inP) * (0.5 + 0.5 * lit)
  const accent = row.key === "rec" ? C.teal : C.sky
  const barY = top + 38
  const barH = 40
  const lostY = top + 100
  const lostH = 14

  return (
    <div style={{ opacity: rowOpacity, transform: `translateY(${(1 - inP) * 24}px)` }}>
      {/* the glass card */}
      <div
        style={{
          position: "absolute",
          left: L,
          top,
          width: RIGHT - L,
          height: ROW_H,
          borderRadius: 26,
          background: `linear-gradient(100deg, rgba(255,255,255,${0.05 + 0.03 * current}), rgba(255,255,255,0.018) 60%)`,
          border: `1px solid ${interpolateColors(current, [0, 1], ["rgba(255,255,255,0.09)", borderLit])}`,
          boxShadow: `inset 0 1px 0 rgba(255,255,255,0.08), 0 30px 70px -36px rgba(0,0,0,0.85), 0 0 ${50 * current}px -12px ${haloLit}`,
          overflow: "hidden",
        }}
      >
        <LightSweep at={row.at + GROW - 10} dur={36} color="rgba(255,255,255,0.10)" />
        {ring !== undefined && <LightSweep at={ring} dur={40} color="rgba(110,231,183,0.12)" />}
      </div>

      {/* label column */}
      <div style={{ position: "absolute", left: L + 30, top, height: ROW_H, display: "flex", gap: 24, alignItems: "center" }}>
        <div
          style={{
            width: 58,
            height: 58,
            borderRadius: 18,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: `rgba(125,211,252,${0.06 + 0.1 * lit})`,
            border: `1px solid ${interpolateColors(lit, [0, 1], ["rgba(255,255,255,0.1)", `${accent}88`])}`,
            color: interpolateColors(lit, [0, 1], [C.dim, accent]),
            boxShadow: `0 0 ${26 * lit}px -6px ${accent}`,
          }}
        >
          <Icon name={row.icon} size={30} stroke={1.8} />
        </div>
        <div>
          <div style={{ fontFamily: MONO, fontSize: 20, letterSpacing: 3, color: accent }}>{row.mode}</div>
          <div style={{ fontFamily: SERIF, fontSize: 56, lineHeight: 1, letterSpacing: -1, color: C.text, marginTop: 4 }}>{row.name}</div>
          <div style={{ fontFamily: SANS, fontSize: 22, fontWeight: 500, color: C.dim, marginTop: 6, whiteSpace: "nowrap", opacity: ramp(f, row.at + 8, row.at + 22) * 0.4 + 0.6 }}>
            {row.note}
          </div>
        </div>
      </div>

      {/* tracks */}
      <div style={{ position: "absolute", left: BAR_X, top: barY, width: BAR_W, height: barH, borderRadius: barH / 2, background: "rgba(255,255,255,0.035)" }} />
      <div style={{ position: "absolute", left: BAR_X, top: lostY, width: BAR_W, height: lostH, borderRadius: lostH / 2, background: "rgba(255,255,255,0.03)" }} />

      {/* stutters removed */}
      {wR > 0.5 && (
        <div
          style={{
            position: "absolute",
            left: BAR_X,
            top: barY,
            width: Math.max(barH, wR),
            height: barH,
            borderRadius: barH / 2,
            background: `linear-gradient(90deg, ${C.sky}, ${C.teal})`,
            boxShadow: `0 0 ${18 + 30 * flash}px rgba(94,234,212,${0.35 + 0.4 * flash}), inset 0 2px 0 rgba(255,255,255,0.45), inset 0 -6px 12px rgba(5,7,13,0.25)`,
            transformOrigin: "0 50%",
            overflow: "hidden",
          }}
        >
          <div style={{ position: "absolute", left: 0, right: 0, top: 4, height: 10, borderRadius: 6, background: "linear-gradient(180deg, rgba(255,255,255,0.35), rgba(255,255,255,0))" }} />
        </div>
      )}
      {/* the bright head while growing */}
      {growing && wR > 2 && (
        <div
          style={{
            position: "absolute",
            left: BAR_X + wR - 22,
            top: barY + barH / 2 - 22,
            width: 44,
            height: 44,
            borderRadius: 22,
            background: "radial-gradient(circle, rgba(255,255,255,0.95), rgba(94,234,212,0.5) 40%, transparent 70%)",
            opacity: 1 - landed,
          }}
        />
      )}
      {/* words lost */}
      {wL > 0.3 && (
        <div style={{ position: "absolute", left: BAR_X, top: lostY, width: Math.max(lostH, wL), height: lostH, borderRadius: lostH / 2, background: C.rose, boxShadow: `0 0 14px rgba(251,113,133,0.45)` }} />
      )}
      {/* a zero still hangs off the axis like the others: a thin tick, green once the ring fires */}
      {row.lost === 0 && (
        <div style={{ position: "absolute", left: BAR_X - 1, top: lostY, width: 3, height: lostH, borderRadius: 1.5, background: lostColor, opacity: ramp(f, row.at + 6, row.at + 14), boxShadow: `0 0 10px ${lostColor}` }} />
      )}
      {/* the punchline's glow, centred on the green 0% */}
      {ring !== undefined && <GlowOrb x={BAR_X + 56} y={lostY + lostH / 2} r={160} color={C.green} opacity={0.25 * rpGlow} />}

      {/* counters ride the tips */}
      <div
        style={{
          position: "absolute",
          left: BAR_X + Math.max(barH, wR) + 20,
          top: barY + barH / 2 - 34,
          height: 68,
          display: "flex",
          alignItems: "center",
          fontFamily: SANS,
          fontWeight: 700,
          fontSize: 56,
          letterSpacing: -1,
          color: C.text,
          fontVariantNumeric: "tabular-nums",
          opacity: ramp(f, row.at, row.at + 8),
          textShadow: `0 0 ${24 * flash}px rgba(94,234,212,0.8)`,
          whiteSpace: "nowrap",
        }}
      >
        {removed.toFixed(1)}%
      </div>
      <LostLabel f={f} x={row.lost === 0 ? BAR_X + 22 : BAR_X + Math.max(lostH, wL) + 16} y={lostY + lostH / 2} value={lost} decimals={row.lost % 1 === 0 ? 0 : String(row.lost).split(".")[1].length} at={row.at + 6} color={lostColor} ring={ring} />
    </div>
  )
}

function LostLabel({ f, x, y, value, decimals, at, color, ring }: { f: number; x: number; y: number; value: number; decimals: number; at: number; color: string; ring?: number }) {
  const o = ramp(f, at, at + 8)
  const isRing = ring !== undefined
  const rp = isRing ? springAt(f, ring, SNAPPY) : 0
  return (
    // as the value grows it settles a few px lower, so its pill stays clear of the bar above
    <div style={{ position: "absolute", left: x, top: y - 20 + 6 * Math.min(1, rp), height: 40, display: "flex", alignItems: "center", gap: 12, opacity: o, whiteSpace: "nowrap", zIndex: 3 }}>
      <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
        {isRing &&
          [0, 1].map((k) => {
            const t = ramp(f, ring + k * 8, ring + k * 8 + 34, easeOut)
            const fade = 1 - ramp(f, ring + k * 8 + 2, ring + k * 8 + 30, easeOut)
            return (
              <div
                key={k}
                style={{
                  position: "absolute",
                  left: "50%",
                  top: "50%",
                  width: 96,
                  height: 96,
                  marginLeft: -48,
                  marginTop: -48,
                  borderRadius: 48,
                  border: `3px solid ${C.green}`,
                  transform: `scale(${0.6 + 0.9 * t})`,
                  zIndex: 2,
                  opacity: f >= ring + k * 8 ? fade * 0.9 : 0,
                  boxShadow: `0 0 18px ${C.green}`,
                }}
              />
            )
          })}
        {isRing && (
          <div
            style={{
              position: "absolute",
              left: -14,
              right: -14,
              top: -6,
              bottom: -6,
              borderRadius: 22,
              border: `2px solid ${C.green}`,
              background: "rgba(110,231,183,0.08)",
              opacity: Math.min(1, rp),
              transform: `scale(${0.8 + 0.2 * rp})`,
              boxShadow: `0 0 26px -4px ${C.green}`,
            }}
          />
        )}
        <span style={{ position: "relative", fontFamily: SANS, fontWeight: 700, fontSize: 30 + 16 * rp, lineHeight: 1, color, fontVariantNumeric: "tabular-nums", textShadow: isRing ? `0 0 ${22 * Math.min(1, rp)}px rgba(110,231,183,0.7)` : undefined }}>{value.toFixed(decimals)}%</span>
      </div>
      <span style={{ fontFamily: SANS, fontWeight: 500, fontSize: 21, color: C.dim, marginLeft: isRing ? 20 * Math.min(1, rp) : 0 }}>words lost</span>
      {isRing && (
        <span style={{ display: "flex", color: C.green, opacity: Math.min(1, rp), transform: `scale(${0.5 + 0.5 * rp})` }}>
          <Icon name="check" size={28} stroke={2.6} />
        </span>
      )}
    </div>
  )
}

function Footer({ f, at, f1 }: { f: number; at: number; f1: number }) {
  const p = springAt(f, at)
  const top = ROW_TOP + 3 * ROW_H + 2 * ROW_GAP + 24
  return (
    <div
      style={{
        position: "absolute",
        left: L,
        top,
        width: RIGHT - L,
        display: "flex",
        justifyContent: "flex-start",
        alignItems: "center",
        fontFamily: SANS,
        fontSize: 22,
        fontWeight: 500,
        color: C.dim,
        opacity: p,
        transform: `translateY(${(1 - p) * 12}px)`,
      }}
    >
      <span style={{ fontFamily: MONO, fontSize: 20 }}>
        Teacher v2: macro F1 {f1.toFixed(3)} on SEP-28k-E
      </span>
    </div>
  )
}
