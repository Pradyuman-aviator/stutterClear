import type { CSSProperties, ReactNode } from "react"
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion"
import { evolvePath } from "@remotion/paths"

import type { Cue, SceneProps } from "../data"
import { lineCue } from "../data"
import {
  Avatar,
  Camera,
  ease,
  easeOut,
  GlowOrb,
  Icon,
  KineticText,
  LightSweep,
  Particles,
  ramp,
  Ribbon,
  Sfx,
  SNAPPY,
  SOFT,
  springAt,
} from "../kit"
import type { IconName } from "../kit"
import { C, JP, MONO, SANS, SERIF } from "../theme"

// Scene 1, "Alex has something to say": the title, then Alex (an abstract avatar, never a face), the
// app he built (a phone reading a Tokyo station sign: kanji above, romaji below) and his night before
// demo day. Every beat is tied to the narration's cues, so a new voice re-times the scene.

// Stage layout (1920x1080, inside the camera). The push-in ends centred on (830, 480) at zoom 1.06, left of
// centre so Alex drifts in, and everything stays within x 96..1824 and y 70..860 on screen.
const AV = { x: 470, y: 420, size: 240 } // Alex
const PH = { x: 1010, top: 118, w: 336, h: 664 } // the phone
const COL = 1262 // the right column: callouts, then the schedule

// The frame at which `word` is spoken in a narration line, estimated from where it sits in the text.
// A narrator pauses at punctuation inside a line, so each mark there counts as a few extra letters
// (checked against the Magpie take: "Tonight" and "demo-day" land within ~2 frames).
const PAUSE: Record<string, number> = { ".": 8, "!": 8, "?": 8, ";": 4, ":": 4, ",": 3 }
const spokenLength = (s: string) => s.length + s.split("").reduce((n, ch, i) => n + (s[i + 1] === " " ? PAUSE[ch] ?? 0 : 0), 0)
function wordAt(c: Cue, word: string, fallback = 0): number {
  const text = c.text ?? ""
  const i = text.indexOf(word)
  const u = i < 0 ? fallback : spokenLength(text.slice(0, i)) / Math.max(1, spokenLength(text))
  return Math.round(c.at + u * c.frames)
}

export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const c0 = lineCue(scene, 0)
  const c1 = lineCue(scene, 1)

  // Beats
  const tGlide = Math.max(24, c0.at - 2) // the title becomes the chapter mark as line 0 begins
  const tAlex = c0.at + 5 // "Alex" (the title has cleared his space by then)
  const tApp = wordAt(c0, "an app", 0.15) // "an app"
  const tRead = wordAt(c0, "read", 0.6) // "read Japanese train signs"
  const tSpeaks = c1.at // "He stutters; it's how he speaks."
  const tTonight = wordAt(c1, "Tonight", 0.46) // "Tonight, he records..."
  const tDemo = wordAt(c1, "demo-day", 0.76) // "...his demo-day pitch"

  const env = d.clip.env_before
  const secs = d.clip.before_s

  // The voice line: draws in before the first word, steps back when Alex arrives, warms on line 1.
  const ribbonO =
    0.7 * ramp(f, 0, 18, easeOut) -
    0.44 * ramp(f, tAlex - 4, tAlex + 20) +
    0.14 * ramp(f, tSpeaks, tSpeaks + 30) -
    0.12 * ramp(f, tDemo, tDemo + 20) // steps back again once the schedule lands, so the pills sit clear

  // "He stutters; it's how he speaks.": the spotlight moves to Alex; the phone and callouts step back
  // until "Tonight" brings in REC and the schedule.
  const spot = ramp(f, tSpeaks, tSpeaks + 14, easeOut) * (1 - ramp(f, tTonight - 8, tTonight + 6))

  // Alex
  const pA = springAt(f, tAlex, SNAPPY)
  const breathe = ramp(f, tSpeaks, tSpeaks + 24)
  const level = 0.12 + 0.05 * Math.sin(f / 11) + breathe * (0.14 + 0.12 * Math.sin((f - tSpeaks) / 9))

  // Title -> chapter mark
  const g = ramp(f, tGlide, tGlide + 20, ease) // a quicker glide, so the longer hold still clears Alex's space as he lands

  return (
    <AbsoluteFill>
      {/* far layer: the voice line and motes, moving less than the stage (parallax) */}
      <Camera shots={[{ at: 0, zoom: 1 }, { at: tAlex, x: 885, y: 515, zoom: 1.022, dur: scene.frames - tAlex }]} drift={3} seed="co-far">
        <div style={{ position: "absolute", left: -60, top: 772, opacity: Math.max(0, ribbonO) }}>
          <Ribbon env={env} seconds={secs} width={2040} height={150} calm={0.6} strands={4} progress={ramp(f, 0, 44, easeOut)} speed={0.8} />
        </div>
      </Camera>
      <Particles count={22} color={C.sky} seed="co" opacity={0.32} rise={0.28} size={[1.2, 3.2]} />

      {/* the stage */}
      <Camera shots={[{ at: 0, zoom: 1 }, { at: tAlex, x: 830, y: 480, zoom: 1.06, dur: scene.frames - tAlex }]} drift={5} seed="co">
        {/* glows behind the two focal objects */}
        <GlowOrb x={AV.x} y={AV.y} r={330} color={C.sky} opacity={(0.2 + 0.14 * spot) * Math.min(1, pA) * (1 + 0.35 * breathe * (0.5 + 0.5 * Math.sin((f - tSpeaks) / 9)))} />
        <GlowOrb x={PH.x} y={PH.top + PH.h * 0.42} r={420} color={C.violet} opacity={0.15 * ramp(f, tApp, tApp + 20)} />
        {/* a pool of light the phone stands in */}
        <div
          style={{
            position: "absolute",
            left: PH.x - 300,
            top: PH.top + PH.h - 26,
            width: 600,
            height: 80,
            borderRadius: "50%",
            background: "radial-gradient(closest-side, rgba(196,181,253,0.32), rgba(125,211,252,0.08) 60%, transparent)",
            opacity: ramp(f, tApp + 4, tApp + 26),
          }}
        />

        <AlexBlock f={f} tAlex={tAlex} tSpeaks={tSpeaks} tTonight={tTonight} pA={pA} level={level} spot={spot} />
        <Phone f={f} tApp={tApp} tRead={tRead} spot={spot} />
        <Callouts f={f} tRead={tRead} spot={spot} />
        <Schedule f={f} tTonight={tTonight} tDemo={tDemo} />
      </Camera>

      {/* the title, screen-fixed: it arrives centred, then glides up into the chapter mark */}
      <GlowOrb x={960} y={478} r={520} color={C.sky} opacity={0.14 * ramp(f, 0, 16) * (1 - g)} />
      <div
        style={{
          position: "absolute",
          left: interpolate(g, [0, 1], [960, 96]),
          top: interpolate(g, [0, 1], [478, 96]),
          transform: `translate(${-50 * (1 - g)}%, -50%) scale(${interpolate(g, [0, 1], [1, 0.4])})`,
          transformOrigin: "0 50%",
          whiteSpace: "nowrap",
          opacity: 1 - 0.18 * g,
        }}
      >
        <KineticText
          text="Alex has something to say."
          at={0}
          per={2}
          effect="blur"
          highlight={{ say: C.sky }}
          style={{ fontFamily: SERIF, fontSize: 124, lineHeight: 1.04, letterSpacing: -1, color: C.text }}
        />
      </div>

      <Sfx name="pop" at={tTonight} volume={0.22} />
    </AbsoluteFill>
  )
}

// ---------------------------------------------------------------------------------------------------

function AlexBlock({ f, tAlex, tSpeaks, tTonight, pA, level, spot }: { f: number; tAlex: number; tSpeaks: number; tTonight: number; pA: number; level: number; spot: number }) {
  const c = Math.min(1, pA)
  const name = springAt(f, tAlex + 5, SNAPPY)
  const note = springAt(f, tAlex + 10)
  const rec = springAt(f, tTonight, SNAPPY)
  return (
    <>
      {/* soft sonar rings while he is introduced as he speaks: warm, sky, never a warning colour */}
      {f >= tSpeaks &&
        [0, 1, 2, 3].map((k) => {
          const u = (f - tSpeaks - k * 24) / 72
          if (u < 0 || u > 1) return null
          const r = (AV.size / 2) * (1.15 + 0.85 * easeOut(u))
          return (
            <div
              key={k}
              style={{
                position: "absolute",
                left: AV.x - r,
                top: AV.y - r,
                width: r * 2,
                height: r * 2,
                borderRadius: "50%",
                border: `1.5px solid ${C.sky}`,
                opacity: (0.32 + 0.18 * spot) * (1 - u),
              }}
            />
          )
        })}
      <div
        style={{
          position: "absolute",
          left: AV.x - AV.size / 2,
          top: AV.y - AV.size / 2,
          opacity: Math.min(1, pA * 1.5),
          transform: `scale(${(0.72 + 0.28 * pA) * (1 + 0.05 * spot)})`,
          filter: c < 0.98 ? `blur(${(1 - c) * 12}px)` : undefined,
        }}
      >
        <Avatar size={AV.size} level={level} />
      </div>

      {/* REC, beside him, from "Tonight" */}
      <div
        style={{
          position: "absolute",
          left: AV.x + 168,
          top: AV.y - 150,
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "8px 16px",
          borderRadius: 999,
          background: "rgba(3,5,10,0.55)",
          border: `1px solid ${C.border2}`,
          fontFamily: MONO,
          fontSize: 20,
          letterSpacing: 3,
          color: C.text,
          opacity: Math.min(1, rec * 1.4),
          transform: `scale(${0.6 + 0.4 * rec})`,
          transformOrigin: "0 50%",
        }}
      >
        <span
          style={{
            width: 12,
            height: 12,
            borderRadius: 6,
            background: C.sky, // the same sky dot as the-take's REC chip: it carries straight over the cut
            boxShadow: `0 0 ${10 + 8 * Math.sin((f - tTonight) / 6)}px ${C.sky}`,
            opacity: 0.55 + 0.45 * (0.5 + 0.5 * Math.cos((f - tTonight) / 6)),
          }}
        />
        REC
      </div>

      {/* name and the honest note */}
      <div style={{ position: "absolute", left: AV.x - 400, width: 800, top: AV.y + 178, display: "flex", flexDirection: "column", alignItems: "center", gap: 18 }}>
        <div
          style={{
            opacity: Math.min(1, name * 1.4),
            transform: `translateY(${(1 - name) * 18}px) scale(${0.85 + 0.15 * name})`,
            padding: "8px 30px 10px",
            borderRadius: 999,
            border: `1px solid rgba(125,211,252,0.45)`,
            background: "rgba(125,211,252,0.08)",
            boxShadow: `0 0 30px -8px ${C.sky}`,
            fontFamily: SERIF,
            fontSize: 52,
            lineHeight: 1,
            letterSpacing: -0.5,
            color: C.text,
          }}
        >
          Alex
        </div>
        <div style={{ opacity: note, transform: `translateY(${(1 - note) * 14}px)`, fontFamily: SANS, fontSize: 24, fontWeight: 500, color: C.dim, whiteSpace: "nowrap" }}>
          Alex is fictional · his voice is a real recording
        </div>
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------------------------------

function Phone({ f, tApp, tRead, spot }: { f: number; tApp: number; tRead: number; spot: number }) {
  const p = springAt(f, tApp, SOFT)
  const c = Math.min(1, p)
  const sign = springAt(f, tApp + 10, SNAPPY)
  const frameIn = springAt(f, tApp + 16, SOFT)
  const focus = ramp(f, tRead - 4, tRead + 14, easeOut) // the sign is being read: brackets lock on
  const bracket = 1.14 - 0.14 * Math.min(1, frameIn) - 0.03 * focus
  const kanjiLit = ramp(f, tRead, tRead + 12, easeOut)
  const romajiLit = ramp(f, tRead + 6, tRead + 18, easeOut)
  return (
    <div
      style={{
        position: "absolute",
        left: PH.x - PH.w / 2,
        top: PH.top,
        width: PH.w,
        height: PH.h,
        perspective: 1800,
        opacity: Math.min(1, p * 1.3) * (1 - 0.45 * spot),
        transform: `translateY(${(1 - p) * 90}px)`,
        filter: spot > 0.01 ? `blur(${2 * spot}px)` : undefined,
      }}
    >
      <div
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          borderRadius: 50,
          transform: `rotateY(${-16 + 10 * c}deg) rotateX(${5 * (1 - c)}deg)`,
          background: "linear-gradient(150deg, rgba(255,255,255,0.16), rgba(255,255,255,0.03) 40%, rgba(255,255,255,0.08))",
          border: `1.5px solid rgba(255,255,255,0.22)`,
          boxShadow: "inset 0 1px 0 rgba(255,255,255,0.25), 0 60px 120px -40px rgba(0,0,0,0.95), 0 0 80px -30px rgba(196,181,253,0.6)",
          padding: 11,
          boxSizing: "border-box",
        }}
      >
        {/* the screen */}
        <div
          style={{
            position: "relative",
            width: "100%",
            height: "100%",
            borderRadius: 40,
            overflow: "hidden",
            background: "radial-gradient(120% 70% at 50% 18%, #16264a 0%, #0a1226 55%, #05080f 100%)",
            boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.05)",
          }}
        >
          {/* a station at night, out of focus: soft lights behind the sign */}
          {[
            [60, 120, 46, C.sky],
            [250, 90, 60, C.violet],
            [200, 560, 70, C.teal],
            [40, 470, 40, C.sky],
            [290, 620, 36, C.violet],
          ].map(([x, y, r, col], i) => (
            <div
              key={i}
              style={{
                position: "absolute",
                left: (x as number) - (r as number),
                top: (y as number) - (r as number) + Math.sin(f / 40 + i) * 4,
                width: (r as number) * 2,
                height: (r as number) * 2,
                borderRadius: "50%",
                background: `radial-gradient(circle, ${col} 0%, transparent 70%)`,
                opacity: 0.22,
              }}
            />
          ))}
          {/* the notch */}
          <div style={{ position: "absolute", left: "50%", top: 14, width: 96, height: 26, marginLeft: -48, borderRadius: 14, background: "#020308" }} />

          {/* the station sign */}
          <div
            style={{
              position: "absolute",
              left: 22,
              right: 22,
              top: 196,
              opacity: Math.min(1, sign * 1.4),
              transform: `translateY(${(1 - sign) * 26}px) scale(${0.9 + 0.1 * sign})`,
            }}
          >
            <div
              style={{
                position: "relative",
                borderRadius: 14,
                background: "linear-gradient(180deg, #0f2350, #0a1838)",
                border: "2px solid rgba(255,255,255,0.88)",
                boxShadow: "0 20px 50px -20px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.2)",
                padding: "22px 0 18px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                overflow: "hidden",
              }}
            >
              <div style={{ fontFamily: JP, fontWeight: 700, fontSize: 112, lineHeight: 1, color: "#fff", letterSpacing: 6, textShadow: `0 0 ${24 * kanjiLit}px rgba(125,211,252,${0.9 * kanjiLit})` }}>東京</div>
              <div style={{ fontFamily: SANS, fontWeight: 600, fontSize: 52, lineHeight: 1.1, color: "#fff", marginTop: 10, textShadow: `0 0 ${22 * romajiLit}px rgba(94,234,212,${0.9 * romajiLit})` }}>Tōkyō</div>
              <div style={{ width: "72%", height: 1, background: "rgba(255,255,255,0.3)", margin: "14px 0 10px" }} />
              <div style={{ fontFamily: SANS, fontWeight: 500, fontSize: 20, letterSpacing: 4, color: "rgba(255,255,255,0.72)", textTransform: "uppercase" }}>Station</div>
              <LightSweep at={tApp + 14} dur={34} color="rgba(255,255,255,0.28)" />
              <LightSweep at={tRead} dur={30} color="rgba(125,211,252,0.3)" />
            </div>
            {/* viewfinder brackets that lock on to the sign */}
            <Brackets scale={bracket} opacity={Math.min(1, frameIn) * (0.55 + 0.45 * focus)} color={focus > 0.5 ? C.sky : "rgba(255,255,255,0.7)"} />
          </div>

          {/* carousel dots */}
          <div style={{ position: "absolute", left: 0, right: 0, bottom: 34, display: "flex", justifyContent: "center", gap: 10, opacity: Math.min(1, frameIn) }}>
            {[0, 1, 2].map((i) => (
              <span key={i} style={{ width: i === 0 ? 26 : 9, height: 9, borderRadius: 5, background: i === 0 ? C.sky : "rgba(255,255,255,0.25)" }} />
            ))}
          </div>
          <LightSweep at={tApp + 4} dur={40} color="rgba(255,255,255,0.12)" angle={30} />
        </div>
      </div>
    </div>
  )
}

function Brackets({ scale, opacity, color }: { scale: number; opacity: number; color: string }) {
  const L = 34
  const corner = (pos: CSSProperties, rot: number) => (
    <div style={{ position: "absolute", width: L, height: L, ...pos, transform: `rotate(${rot}deg)` }}>
      <div style={{ position: "absolute", left: 0, top: 0, width: L, height: 3, borderRadius: 2, background: color }} />
      <div style={{ position: "absolute", left: 0, top: 0, width: 3, height: L, borderRadius: 2, background: color }} />
    </div>
  )
  return (
    <div style={{ position: "absolute", inset: -14, opacity, transform: `scale(${scale})`, filter: `drop-shadow(0 0 6px ${color})` }}>
      {corner({ left: 0, top: 0 }, 0)}
      {corner({ right: 0, top: 0 }, 90)}
      {corner({ right: 0, bottom: 0 }, 180)}
      {corner({ left: 0, bottom: 0 }, 270)}
    </div>
  )
}

// ---------------------------------------------------------------------------------------------------

// Leader lines from the sign to what each line of it is: Chinese writing (kanji), Roman letters (romaji).
const LEADS = [
  { d: `M 1121 414 L 1186 352 L ${COL - 8} 352`, y: 352, label: "Chinese writing", sub: "kanji", color: C.sky, dt: 0 },
  { d: `M 1104 506 L 1186 532 L ${COL - 8} 532`, y: 532, label: "Roman letters", sub: "rōmaji", color: C.teal, dt: 6 },
]

function Callouts({ f, tRead, spot }: { f: number; tRead: number; spot: number }) {
  return (
    <div style={{ position: "absolute", inset: 0, opacity: 1 - 0.45 * spot }}>
      <svg width={1920} height={1080} style={{ position: "absolute", left: 0, top: 0, overflow: "visible", pointerEvents: "none" }}>
        {LEADS.map((l, i) => {
          const p = ramp(f, tRead + l.dt, tRead + l.dt + 18, easeOut)
          if (p <= 0) return null
          const ev = evolvePath(p, l.d)
          const [x0, y0] = l.d.slice(2).split(" L ")[0].split(" ").map(Number)
          const dot = springAt(f, tRead + l.dt, SNAPPY)
          return (
            <g key={i} style={{ filter: `drop-shadow(0 0 6px ${l.color})` }}>
              <path d={l.d} fill="none" stroke={l.color} strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" strokeDasharray={ev.strokeDasharray} strokeDashoffset={ev.strokeDashoffset} />
              <circle cx={x0} cy={y0} r={6 * Math.min(1.2, dot)} fill={l.color} />
              <circle cx={x0} cy={y0} r={6 + 10 * Math.min(1, dot)} fill="none" stroke={l.color} strokeOpacity={0.5 * (1 - Math.min(1, dot) * 0.6)} />
            </g>
          )
        })}
      </svg>
      {LEADS.map((l, i) => {
        const p = springAt(f, tRead + l.dt + 10, SNAPPY)
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: COL + 6,
              top: l.y - 22,
              opacity: Math.min(1, p * 1.4),
              transform: `translateX(${(1 - p) * -24}px)`,
              filter: p < 0.95 ? `blur(${(1 - Math.min(1, p)) * 8}px)` : undefined,
            }}
          >
            <div style={{ fontFamily: MONO, fontSize: 30, fontWeight: 500, color: l.color, lineHeight: 1.45, whiteSpace: "nowrap" }}>{l.label}</div>
            <div style={{ fontFamily: MONO, fontSize: 20, color: C.dim, letterSpacing: 2, marginTop: 2 }}>{l.sub}</div>
          </div>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------------------------------

function Schedule({ f, tTonight, tDemo }: { f: number; tTonight: number; tDemo: number }) {
  const rows: { at: number; icon: IconName; text: ReactNode; color: string; y: number }[] = [
    { at: tTonight, icon: "video", text: "Pitch video: tonight", color: C.sky, y: 648 },
    { at: tDemo, icon: "clock", text: "Demo day: tomorrow", color: C.violet, y: 728 },
  ]
  const link = ramp(f, tDemo - 4, tDemo + 10, easeOut)
  return (
    <>
      {/* the thread between tonight and tomorrow */}
      <div
        style={{
          position: "absolute",
          left: COL + 27,
          top: rows[0].y + 30,
          width: 2,
          height: (rows[1].y - rows[0].y - 30) * link,
          background: `linear-gradient(${C.sky}, ${C.violet})`,
          opacity: 0.6,
        }}
      />
      {rows.map((r, i) => {
        const p = springAt(f, r.at, SNAPPY)
        const glow = Math.max(0, 1 - Math.abs(f - r.at - 8) / 22)
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: COL,
              top: r.y - 28,
              height: 56,
              display: "flex",
              alignItems: "center",
              gap: 14,
              padding: "0 26px 0 14px",
              borderRadius: 999,
              background: "rgba(10,15,28,0.72)",
              border: `1px solid ${r.color}66`,
              boxShadow: `0 0 ${18 + 30 * glow}px -6px ${r.color}, inset 0 1px 0 rgba(255,255,255,0.08)`,
              fontFamily: SANS,
              fontWeight: 600,
              fontSize: 28,
              color: C.text,
              whiteSpace: "nowrap",
              opacity: Math.min(1, p * 1.5),
              transform: `translateY(${(1 - p) * 22}px) scale(${0.8 + 0.2 * p})`,
              transformOrigin: "0 50%",
            }}
          >
            <span style={{ width: 30, height: 30, borderRadius: 15, display: "flex", alignItems: "center", justifyContent: "center", background: `${r.color}22` }}>
              <Icon name={r.icon} size={20} color={r.color} />
            </span>
            {r.text}
          </div>
        )
      })}
    </>
  )
}

