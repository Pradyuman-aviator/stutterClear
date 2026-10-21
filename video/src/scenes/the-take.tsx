import type { CSSProperties, ReactNode } from "react"
import { AbsoluteFill, interpolate, interpolateColors, useCurrentFrame } from "remotion"

import type { SceneProps, Word } from "../data"
import { clipCue, FPS, lineCue, sec } from "../data"
import {
  Avatar,
  BOUNCY,
  Camera,
  clamp,
  easeOut,
  envAt,
  GlowOrb,
  Icon,
  LightSweep,
  Particles,
  ramp,
  SNAPPY,
  springAt,
  Wave,
} from "../kit"
import { C, JP, MONO, SANS, SERIF } from "../theme"

// Tonight: the take. Alex records his pitch. Row 1 holds what he means, complete from the start
// and ghosted; Row 3 shows every word as he says it, repeats landing on top of each other; each
// meant word lights up the moment its spoken word ends. The meaning arrives as fast as speech allows.

// The stage spans x 136-1772 so the closing push-in (zoom 1.04 on 960,540) keeps it inside the
// 96 px margins, and the frame ends at y 882 so its bottom stays above the subtitles (y <= 900); it starts at y 200 so it
// clears the clip credit (y <= 175) during the clip.
const FRAME = { x: 136, y: 200, w: 1196, h: 682 }
const CARD = { x: 1362, y: 272, w: 410 }
const PADX = 48
const INNER = FRAME.w - PADX * 2 // 1100
const WAVE_W = 860
const DISC = 96
const HOP = INNER - WAVE_W - DISC // the gap the voice crosses to the listener
const GHOST = "rgba(245,247,251,0.26)"
const SAYS = 36 // Row 3 type size
// An older copy rises a full line above the newer one, a little smaller and fainter, and steps right,
// so 'they they they' and 'l- l- l- letters' can be counted at a glance. Steps are 40, 36, 32 px.
const liftOf = (k: number) => (k <= 1 ? 40 * k : k <= 2 ? 40 + 36 * (k - 1) : 76 + 32 * (k - 2))
const shrinkOf = (k: number) => Math.max(0.8, 1 - 0.1 * k)

const stem = (w: string) => w.toLowerCase().replace(/[^a-z]/g, "")

// Consecutive repeats ("they they they") and broken-off starts with their word ("writ- writ-
// writing") share one slot, so they can land on top of each other.
function groupWords(words: Word[]): Word[][] {
  const groups: Word[][] = []
  for (const w of words) {
    const g = groups[groups.length - 1]
    const last = g?.[g.length - 1]
    const same = last && (stem(last.w) === stem(w.w) || (last.w.endsWith("-") && stem(w.w).startsWith(stem(last.w))))
    if (g && same) g.push(w)
    else groups.push([w])
  }
  return groups
}

function Label({ children, dot, style }: { children: ReactNode; dot: string; style?: CSSProperties }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, fontFamily: MONO, fontSize: 20, letterSpacing: 4, textTransform: "uppercase", color: C.dim, ...style }}>
      <span style={{ width: 9, height: 9, borderRadius: 5, background: dot, boxShadow: `0 0 10px ${dot}` }} />
      {children}
    </div>
  )
}

export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const c = d.clip
  const l0 = lineCue(scene, 0)
  const l1 = lineCue(scene, 1)
  const clip = clipCue(scene, "before")
  const clipAt = clip?.at ?? l0.at + l0.frames + 12
  const clipFrames = clip?.frames ?? sec(c.before_s)
  const clipEnd = clipAt + clipFrames
  const wf = (s: number) => clipAt + s * FPS // a time in the clip, as a scene frame

  // elapsed seconds of the take
  const t = interpolate(f, [clipAt, clipEnd], [0, c.before_s], clamp)
  const playing = f >= clipAt && f < clipEnd
  const level = playing ? envAt(c.env_before, Math.max(0, t - 0.03), t + 0.05) : 0

  // beats on the narration
  const meansAt = l0.at + 2 // "He knows…": the meant sentence spells itself out with the line
  const listenAt = l0.at + 0.8 * l0.frames // "Listen."
  const thereAt = l1.at + 2 // "Every word was there from the start"
  const longerAt = l1.at + 0.62 * l1.frames // "It just takes longer": the glow peaks on "longer" (~0.65 of the line)
  const listenerAt = l1.at + 0.84 * l1.frames // "to reach the listener"

  // Row 1: the meant words (the kept ones, in order) against clip.clean
  const cleanWords = c.clean.split(" ")
  const kept = c.words.filter((w) => !w.removed)
  const litOf = (i: number) => {
    const w = kept[i]
    return w ? ramp(f, wf(w.e), wf(w.e) + 8, easeOut) : ramp(f, clipEnd - 8, clipEnd)
  }
  const litCount = cleanWords.reduce((n, _, i) => n + litOf(i), 0)
  const allLit = litCount / Math.max(1, cleanWords.length)
  const bloom = ramp(f, thereAt, thereAt + 12, easeOut) * (1 - ramp(f, thereAt + 34, thereAt + 70))
  const settle = ramp(f, thereAt, thereAt + 18, easeOut)
  // light travels through the sentence's glyphs, word by word, in reading order
  const sweepU = ramp(f, thereAt, thereAt + 30)
  const sweepOn = f >= thereAt && f <= thereAt + 30
  const sweepOf = (i: number) => {
    if (!sweepOn) return 0
    const w = 0.16
    const pos = i / Math.max(1, cleanWords.length - 1)
    return Math.max(0, 1 - Math.abs(sweepU * (1 + 2 * w) - w - pos) / w)
  }

  const groups = groupWords(c.words)

  // the sign on his slide answers when the words that name it land
  const writingEnd = c.words.find((w) => w.w.toLowerCase() === "writing")?.e ?? c.before_s * 0.4
  const lettersEnd = c.words.find((w) => w.w.toLowerCase() === "letters")?.e ?? c.before_s
  const kanjiLit = ramp(f, wf(writingEnd), wf(writingEnd) + 12, easeOut)
  const romajiLit = ramp(f, wf(lettersEnd), wf(lettersEnd) + 12, easeOut)

  // entrances
  const frameIn = springAt(f, 0)
  const slideIn = springAt(f, 8)
  const armed = ramp(f, listenAt, listenAt + 10, easeOut)
  const waveReveal = ramp(f, listenAt - 4, listenAt + 30, easeOut)
  const discIn = springAt(f, listenAt + 4, BOUNCY)

  // header: REC and the timer
  const secs = Math.min(Math.floor(t), Math.floor(c.before_s))
  const timer = `00:${String(secs).padStart(2, "0")}`
  const recPulse = playing ? 0.5 + 0.5 * Math.cos((f - clipAt) / 7) : 0
  const longer = ramp(f, longerAt, longerAt + 10, easeOut) * (1 - ramp(f, longerAt + 40, longerAt + 70))

  // the listener's ring
  const heard = t / c.before_s
  const ringR = DISC / 2 - 6
  const circ = 2 * Math.PI * ringR
  const pulse = f >= listenerAt ? interpolate(f, [listenerAt, listenerAt + 34], [0, 1], { ...clamp, easing: easeOut }) : 0
  const pulse2 = f >= listenerAt + 8 ? interpolate(f, [listenerAt + 8, listenerAt + 44], [0, 1], { ...clamp, easing: easeOut }) : 0
  const listenerGlow = Math.max(level * 0.8, ramp(f, listenerAt, listenerAt + 8) * (1 - ramp(f, listenerAt + 30, listenerAt + 60)))

  // Row 3 steps back a little while the meaning blooms
  // (the words only, never below C.dim; the row's label keeps its strength)
  const echoDim = 1 - 0.25 * ramp(f, thereAt, thereAt + 16)

  return (
    <AbsoluteFill>
      <Particles count={22} color={C.sky} seed="take" opacity={0.32} rise={0.25} size={[1.2, 3]} />
      <Camera
        shots={[
          { at: 0, x: 960, y: 540, zoom: 1 },
          // centred on the stage (x 136-1784) so every shot keeps it inside the 96 px margins
          { at: l0.at, x: 958, y: 538, zoom: 1.015, dur: l0.frames + 20 },
          { at: clipAt + 20, x: 960, y: 534, zoom: 1.03, dur: Math.max(40, clipFrames - 20) },
          { at: l1.at - 6, x: 960, y: 540, zoom: 1.04, dur: 80 },
        ]}
        drift={4}
        seed="take"
      >
        {/* light behind the frame; it warms as the meaning fills in */}
        <GlowOrb x={FRAME.x + FRAME.w * 0.45} y={FRAME.y + 270} r={620} color={C.sky} opacity={0.08 + 0.1 * allLit + 0.14 * bloom} />
        <GlowOrb x={CARD.x + CARD.w / 2} y={520} r={360} color={C.teal} opacity={0.09} />

        {/* the recording frame */}
        <div
          style={{
            position: "absolute",
            left: FRAME.x,
            top: FRAME.y,
            width: FRAME.w,
            height: FRAME.h,
            opacity: Math.min(1, frameIn * 1.3),
            transform: `translateY(${(1 - frameIn) * 26}px) scale(${0.975 + 0.025 * frameIn})`,
          }}
        >
          <div
            style={{
              position: "absolute",
              inset: 0,
              borderRadius: 30,
              background: "linear-gradient(165deg, rgba(255,255,255,0.075), rgba(255,255,255,0.02) 55%, rgba(125,211,252,0.035))",
              border: `1px solid ${C.border2}`,
              boxShadow: `inset 0 1px 0 rgba(255,255,255,0.1), 0 40px 110px -40px rgba(0,0,0,0.85), 0 0 ${60 + 50 * bloom}px -20px rgba(125,211,252,${0.18 + 0.3 * bloom})`,
              overflow: "hidden",
            }}
          >
            <LightSweep at={6} dur={50} color="rgba(255,255,255,0.12)" />
            {/* viewfinder corners */}
            {[
              { left: 18, top: 18, b: "borderLeft", c: "borderTop" },
              { right: 18, top: 18, b: "borderRight", c: "borderTop" },
              { left: 18, bottom: 18, b: "borderLeft", c: "borderBottom" },
              { right: 18, bottom: 18, b: "borderRight", c: "borderBottom" },
            ].map(({ b, c: cc, ...pos }, i) => (
              <div key={i} style={{ position: "absolute", width: 26, height: 26, [b]: `2px solid rgba(255,255,255,${0.18 + 0.2 * armed})`, [cc]: `2px solid rgba(255,255,255,${0.18 + 0.2 * armed})`, ...pos }} />
            ))}
          </div>

          <div style={{ position: "absolute", left: PADX, top: 20, width: INNER }}>
            {/* header */}
            <div style={{ display: "flex", alignItems: "center", height: 56 }}>
              <div style={{ opacity: Math.min(1, springAt(f, 4) * 1.4), transform: `scale(${0.7 + 0.3 * springAt(f, 4, SNAPPY)})`, margin: "0 26px 0 10px" }}>
                {/* damped a little so the outer ring, at the take's loudest, keeps clear of the clip credit above */}
                <Avatar size={56} level={level * 0.8} />
              </div>
              <div style={{ opacity: ramp(f, 7, 19, easeOut), transform: `translateY(${(1 - ramp(f, 7, 19, easeOut)) * 10}px)` }}>
                <div style={{ fontFamily: SANS, fontWeight: 650, fontSize: 30, color: C.text, lineHeight: 1.1 }}>Alex</div>
                <div style={{ fontFamily: SANS, fontWeight: 500, fontSize: 22, color: C.dim, marginTop: 3 }}>Demo-day pitch · Take 1</div>
              </div>
              <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 22, opacity: ramp(f, 10, 22, easeOut) }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "8px 16px",
                    borderRadius: 999,
                    border: `1px solid rgba(125,211,252,${0.12 + 0.4 * armed * (f < clipEnd ? 1 : 0.4)})`,
                    background: `rgba(125,211,252,${0.04 + 0.08 * armed * (f < clipEnd ? 1 : 0.3)})`,
                    fontFamily: MONO,
                    fontSize: 20,
                    letterSpacing: 3,
                    color: armed > 0.5 && f < clipEnd ? C.text : C.dim,
                  }}
                >
                  <span
                    style={{
                      width: 12,
                      height: 12,
                      borderRadius: 6,
                      background: f < clipEnd ? C.sky : C.dim,
                      opacity: (0.25 + 0.75 * armed) * (playing ? 0.45 + 0.55 * recPulse : 1),
                      boxShadow: playing ? `0 0 ${8 + 12 * recPulse}px ${C.sky}` : undefined,
                    }}
                  />
                  REC
                </div>
                <div
                  style={{
                    fontFamily: MONO,
                    fontWeight: 500,
                    fontSize: 34,
                    fontVariantNumeric: "tabular-nums",
                    letterSpacing: 1,
                    color: interpolateColors(Math.max(longer, playing ? 1 : 0), [0, 1], [C.dim, C.text]),
                    textShadow: longer > 0 ? `0 0 ${24 * longer}px rgba(125,211,252,${0.9 * longer})` : undefined,
                    transform: `scale(${1 + 0.1 * longer})`,
                    transformOrigin: "right center",
                  }}
                >
                  {timer}
                </div>
              </div>
            </div>

            {/* Row 1: what Alex means */}
            <div style={{ marginTop: 12 }}>
              <Label dot={C.sky} style={{ opacity: ramp(f, 14, 26, easeOut) }}>
                What Alex means
              </Label>
              <div style={{ position: "relative", marginTop: 6, padding: "2px 0 4px" }}>
                <div
                  style={{
                    fontFamily: SERIF,
                    fontSize: 56,
                    lineHeight: 1.04,
                    letterSpacing: -0.5,
                    transform: `scale(${1 + 0.02 * bloom})`,
                    transformOrigin: "left center",
                  }}
                >
                  {cleanWords.map((w, i) => {
                    const inP = springAt(f, meansAt + i * 2, SNAPPY)
                    const lit = litOf(i)
                    const fresh = lit > 0 ? 1 - ramp(f, wf(kept[i]?.e ?? 0) + 4, wf(kept[i]?.e ?? 0) + 30) : 0
                    const sw = sweepOf(i)
                    const glow = Math.max(fresh * lit, bloom, sw)
                    const base = interpolateColors(lit, [0, 1], [GHOST, interpolateColors(settle, [0, 1], [C.sky, "#bfe8fd"])])
                    return (
                      <span
                        key={i}
                        style={{
                          display: "inline-block",
                          whiteSpace: "pre",
                          opacity: Math.min(1, inP * 1.4),
                          transform: `translateY(${(1 - Math.min(1, inP)) * 18}px)`,
                          filter: `blur(${(1 - Math.min(1, inP)) * 8}px)`,
                          color: sw > 0.01 ? interpolateColors(sw, [0, 1], [base, "#ffffff"]) : base,
                          textShadow:
                            glow > 0.01
                              ? `0 0 ${26 * glow}px rgba(125,211,252,${0.85 * glow})${sw > 0.01 ? `, 0 0 ${10 * sw}px rgba(255,255,255,${0.7 * sw})` : ""}`
                              : undefined,
                        }}
                      >
                        {w}
                        {i < cleanWords.length - 1 ? " " : ""}
                      </span>
                    )
                  })}
                </div>
              </div>
            </div>

            {/* Row 2: his voice, and the listener it travels to */}
            <div style={{ marginTop: 6, display: "flex", alignItems: "center", gap: 0 }}>
              <div style={{ position: "relative", width: WAVE_W, height: 96 }}>
                <div style={{ position: "absolute", left: 0, right: 0, top: 47, height: 2, background: `rgba(255,255,255,${0.08 * (1 - waveReveal) + 0.04})`, borderRadius: 1, transform: `scaleX(${ramp(f, 16, 40, easeOut)})`, transformOrigin: "left" }} />
                <Wave env={c.env_before} seconds={c.before_s} width={WAVE_W} height={96} color={C.sky} progress={f < clipAt ? 0 : t / c.before_s} dimAfter reveal={waveReveal} glow={playing ? 3 + 5 * level : 2} />
              </div>
              {/* the hop from the voice to the listener */}
              <div style={{ position: "relative", width: HOP, height: 2, opacity: Math.min(1, discIn) }}>
                <div
                  style={{
                    position: "absolute",
                    left: 10,
                    right: 4,
                    top: 0,
                    height: 2,
                    borderRadius: 1,
                    background: `linear-gradient(90deg, rgba(125,211,252,${0.25 + 0.5 * level}), rgba(94,234,212,${0.45 + 0.55 * level}))`,
                    boxShadow: `0 0 ${6 + 10 * level}px rgba(94,234,212,${0.25 + 0.5 * level})`,
                    transform: `scaleX(${ramp(f, listenAt + 2, listenAt + 18, easeOut)})`,
                    transformOrigin: "left",
                  }}
                />
                {[0, 1, 2, 3].map((k) => {
                  const u = (((f - clipAt) / 26 + k / 4) % 1 + 1) % 1
                  return (
                    <span
                      key={k}
                      style={{
                        position: "absolute",
                        left: 10 + u * (HOP - 26),
                        top: -4,
                        width: 10,
                        height: 10,
                        borderRadius: 5,
                        background: interpolateColors(level, [0, 1], [C.teal, "#e6fffa"]),
                        opacity: playing ? (0.3 + 0.7 * level) * Math.sin(u * Math.PI) : 0,
                        boxShadow: `0 0 ${8 + 10 * level}px ${C.teal}`,
                      }}
                    />
                  )
                })}
              </div>
              <div style={{ position: "relative", width: DISC, height: DISC, opacity: Math.min(1, discIn * 1.4), transform: `scale(${0.6 + 0.4 * discIn})` }}>
                {[pulse, pulse2].map((p, k) =>
                  p > 0 && p < 1 ? (
                    <div key={k} style={{ position: "absolute", inset: 0, borderRadius: "50%", border: `2px solid ${C.teal}`, opacity: 0.8 * (1 - p), transform: `scale(${1 + 0.9 * p})` }} />
                  ) : null,
                )}
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    borderRadius: "50%",
                    background: "radial-gradient(circle at 40% 35%, rgba(94,234,212,0.16), rgba(10,14,26,0.85) 70%)",
                    border: `1px solid ${C.border2}`,
                    boxShadow: `0 0 ${30 + 50 * listenerGlow}px -8px rgba(94,234,212,${0.25 + 0.55 * listenerGlow})`,
                  }}
                />
                <svg width={DISC} height={DISC} style={{ position: "absolute", inset: 0, transform: "rotate(-90deg)" }}>
                  <circle cx={DISC / 2} cy={DISC / 2} r={ringR} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth={4} />
                  <circle
                    cx={DISC / 2}
                    cy={DISC / 2}
                    r={ringR}
                    fill="none"
                    stroke={C.teal}
                    strokeWidth={4}
                    strokeLinecap="round"
                    strokeDasharray={`${circ * heard} ${circ}`}
                    opacity={heard > 0.002 ? 1 : 0}
                    style={{ filter: `drop-shadow(0 0 6px ${C.teal})` }}
                  />
                </svg>
                <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <Icon name="headphones" size={40} color={interpolateColors(listenerGlow, [0, 1], [C.dim, C.teal])} stroke={1.8} />
                </div>
                <div style={{ position: "absolute", top: DISC + 10, left: -40, right: -40, textAlign: "center", fontFamily: MONO, fontSize: 20, letterSpacing: 3, color: C.dim }}>LISTENER</div>
              </div>
            </div>

            {/* Row 3: what he says, every word as it comes */}
            <div style={{ marginTop: 10 }}>
              <Label dot={C.text} style={{ opacity: ramp(f, 18, 30, easeOut) }}>
                What he says
              </Label>
              {/* room above each line for a full stack: two older copies on line 1, three on line 2 */}
              <div style={{ display: "flex", flexWrap: "wrap", columnGap: 14, rowGap: 106, paddingTop: 68, width: INNER, opacity: echoDim }}>
                {groups.map((g, gi) => (
                  <div key={gi} style={{ display: "inline-grid" }}>
                    {g.map((w, i) => {
                      const at = wf(w.s)
                      const p = springAt(f, at, SNAPPY)
                      // how many newer copies have landed on this one (springy, so it lifts as they land)
                      const depth = g.slice(i + 1).reduce((n, v) => n + Math.min(1, springAt(f, wf(v.s), SNAPPY)), 0)
                      // older copies settle at about half strength, never fainter
                      const fade = 1 - 0.4 * Math.min(1, depth) - 0.08 * Math.min(1, Math.max(0, depth - 1))
                      const enterY = i === 0 ? (1 - p) * 16 : -(1 - p) * 22
                      const arrive = f >= at ? 1 - ramp(f, at + 3, at + 26) : 0
                      return (
                        <span
                          key={i}
                          style={{
                            gridArea: "1 / 1",
                            fontFamily: SANS,
                            fontWeight: 600,
                            fontSize: SAYS,
                            lineHeight: 1.05,
                            fontStyle: w.label === "filler" ? "italic" : "normal",
                            whiteSpace: "pre",
                            color: C.text,
                            opacity: Math.min(1, p * 1.5) * fade,
                            transform: `translate(${depth * 12}px, ${-liftOf(depth) + enterY}px) scale(${(0.72 + 0.28 * p) * shrinkOf(depth)})`,
                            transformOrigin: "left bottom",
                            textShadow: arrive > 0.01 ? `0 0 ${20 * arrive}px rgba(125,211,252,${0.75 * arrive})` : "0 2px 12px rgba(0,0,0,0.5)",
                          }}
                        >
                          {w.w}
                        </span>
                      )
                    })}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* his slide: the question the pitch answers */}
        <div style={{ position: "absolute", left: CARD.x, top: CARD.y, width: CARD.w, perspective: 1600 }}>
          <div
            style={{
              opacity: Math.min(1, slideIn * 1.3),
              transform: `translateX(${(1 - slideIn) * 50}px) rotateY(${-9 + 3 * (1 - slideIn)}deg)`,
              transformOrigin: "left center",
              borderRadius: 26,
              padding: "30px 30px 28px",
              background: "linear-gradient(160deg, rgba(255,255,255,0.08), rgba(255,255,255,0.025))",
              border: `1px solid ${C.border}`,
              boxShadow: "inset 0 1px 0 rgba(255,255,255,0.08), 0 40px 90px -36px rgba(0,0,0,0.85)",
              position: "relative",
              overflow: "hidden",
            }}
          >
            <LightSweep at={14} dur={46} color="rgba(255,255,255,0.12)" />
            <div style={{ fontFamily: MONO, fontSize: 20, letterSpacing: 4, color: C.sky }}>HIS PITCH</div>
            <div style={{ fontFamily: SANS, fontWeight: 600, fontSize: 32, lineHeight: 1.22, color: C.text, marginTop: 12 }}>Why are train signs hard to read?</div>
            {/* the station sign */}
            <div
              style={{
                marginTop: 24,
                borderRadius: 14,
                padding: "16px 18px 18px",
                background: "linear-gradient(180deg, #13244a, #0b1631)",
                border: "2px solid rgba(255,255,255,0.85)",
                boxShadow: `0 18px 40px -18px rgba(0,0,0,0.9), 0 0 ${40 * Math.max(kanjiLit, romajiLit) * (1 - ramp(f, wf(lettersEnd) + 60, wf(lettersEnd) + 120) * 0.5)}px -12px rgba(125,211,252,0.6)`,
                textAlign: "center",
              }}
            >
              <div style={{ fontFamily: MONO, fontSize: 18, letterSpacing: 3, color: "rgba(255,255,255,0.7)", textAlign: "left" }}>STATION</div>
              <div
                style={{
                  fontFamily: JP,
                  fontWeight: 700,
                  fontSize: 92,
                  lineHeight: 1.1,
                  color: interpolateColors(kanjiLit, [0, 1], ["#ffffff", "#d6f0ff"]),
                  textShadow: kanjiLit > 0 ? `0 0 ${28 * kanjiLit}px rgba(125,211,252,${0.8 * kanjiLit})` : undefined,
                }}
              >
                東京
              </div>
              <div
                style={{
                  fontFamily: SANS,
                  fontWeight: 600,
                  fontSize: 40,
                  lineHeight: 1.15,
                  color: interpolateColors(romajiLit, [0, 1], ["#ffffff", "#d3fbf3"]),
                  textShadow: romajiLit > 0 ? `0 0 ${24 * romajiLit}px rgba(94,234,212,${0.8 * romajiLit})` : undefined,
                }}
              >
                Tōkyō
              </div>
            </div>
            {/* what his words name on the sign */}
            <div style={{ marginTop: 18, display: "flex", flexDirection: "column", gap: 10 }}>
              {[
                { text: "Chinese writing · kanji", color: C.sky, p: kanjiLit, at: wf(writingEnd) },
                { text: "Roman letters · romaji", color: C.teal, p: romajiLit, at: wf(lettersEnd) },
              ].map(({ text, color, p, at }) => {
                const s = springAt(f, at, SNAPPY)
                return (
                  <div
                    key={text}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      fontFamily: MONO,
                      fontSize: 20,
                      color: interpolateColors(p, [0, 1], [C.faint, color]),
                      opacity: 0.35 + 0.65 * Math.min(1, s),
                      transform: `translateX(${(1 - Math.min(1, s)) * 14}px)`,
                    }}
                  >
                    <span style={{ width: 22, height: 2, background: color, opacity: 0.3 + 0.7 * p, boxShadow: p > 0.5 ? `0 0 8px ${color}` : undefined }} />
                    {text}
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </Camera>
    </AbsoluteFill>
  )
}

