import type { CSSProperties, ReactNode } from "react"
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion"

import type { SceneProps } from "../data"
import { lineCue } from "../data"
import { Camera, clamp, ease, easeIn, easeOut, GlowOrb, Particles, ramp, Ribbon, Sfx, SNAPPY, SOFT, springAt } from "../kit"
import { C, MONO, SANS, SERIF } from "../theme"

// On his terms: the voice line breathes, the name lands, the promise follows, then the credits.

const TAG = "Your real voice, on your terms."

// When each word of the narrated line is said, as a share of the line (so it follows lineCue and
// scales with the line's length), a hair early so the word is on screen as it is heard. Measured on
// the NVIDIA Magpie narration (2.71 s): "Stutter" 0.00 s, "Clear" 0.50, ~0.15 s silence, "Your" 0.92,
// "real" 1.10, "voice," 1.42 (its "s" ends at 1.78), "on" 1.85, "your" 1.92, "terms." 2.13 (ends 2.60).
const LINE_S = 2.71
const WORD_S: [string, number][] = [
  ["StutterClear.", 0],
  ["Your", 0.92],
  ["real", 1.1],
  ["voice,", 1.42],
  ["on", 1.85],
  ["your", 1.92],
  ["terms.", 2.13],
]
function wordAt(cueAt: number, cueFrames: number, word: string, nth = 0) {
  const hits = WORD_S.filter(([w]) => w === word)
  const s = (hits[nth] ?? hits[0] ?? ["", 0])[1]
  return cueAt + (s / LINE_S) * cueFrames - 2
}

// The name, letter by letter; then waves of light run across it (each letter glows as a wave passes).
function Wordmark({ at, size, shines }: { at: number; size: number; shines: number[] }) {
  const f = useCurrentFrame()
  const letters = Array.from("StutterClear")
  const gradient = `linear-gradient(90deg, ${C.sky}, ${C.teal})`
  return (
    <div style={{ fontFamily: SERIF, fontSize: size, lineHeight: 1, letterSpacing: -2, whiteSpace: "nowrap" }}>
      {letters.map((ch, i) => {
        const p = springAt(f, at + i * 1.6, SNAPPY)
        const c = Math.min(1, p)
        let glow = 0
        for (const s of shines) {
          const pos = interpolate(f, [s, s + 30], [-3, letters.length + 3], { ...clamp, easing: ease })
          if (f >= s && f <= s + 30) glow = Math.max(glow, Math.exp(-((i - pos) ** 2) / 3))
        }
        const clear = i >= 7
        const style: CSSProperties = {
          display: "inline-block",
          opacity: Math.min(1, p * 1.5),
          transform: `translateY(${(1 - p) * 0.32 * size - glow * 4}px) scale(${0.9 + 0.1 * p})`,
          filter: `blur(${(1 - c) * 12}px) drop-shadow(0 0 ${6 + 22 * glow}px rgba(190,255,245,${0.75 * glow})) brightness(${1 + 0.35 * glow})`,
        }
        if (clear)
          Object.assign(style, {
            color: "transparent",
            backgroundImage: gradient,
            backgroundSize: "500% 100%",
            backgroundPosition: `${((i - 7) / 4) * 100}% 0`,
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
          })
        else style.color = C.text
        return (
          <span key={i} style={style}>
            {ch}
          </span>
        )
      })}
    </div>
  )
}

// The promise, each word arriving as it is said; "voice" and "terms" in Alex's colour.
function Tagline({ at, frames, size }: { at: number; frames: number; size: number }) {
  const f = useCurrentFrame()
  const words = TAG.split(" ")
  return (
    <div style={{ fontFamily: SERIF, fontSize: size, lineHeight: 1.05, letterSpacing: -1, color: C.text, whiteSpace: "nowrap", fontStyle: "italic" }}>
      {words.map((w, i) => {
        const wAt = wordAt(at, frames, w, words.slice(0, i).filter((x) => x === w).length)
        const p = springAt(f, wAt, SNAPPY)
        const key = w.replace(/[^a-z]/gi, "").toLowerCase()
        const lit = key === "voice" || key === "terms"
        return (
          <span
            key={i}
            style={{
              display: "inline-block",
              whiteSpace: "pre",
              opacity: Math.min(1, p * 1.4),
              transform: `translateY(${(1 - p) * 0.5}em)`,
              filter: `blur(${(1 - Math.min(1, p)) * 8}px)`,
              ...(lit
                ? { color: "transparent", backgroundImage: `linear-gradient(90deg, ${C.sky}, ${C.teal})`, WebkitBackgroundClip: "text", backgroundClip: "text" }
                : {}),
            }}
          >
            {w}
            {i < words.length - 1 ? " " : ""}
          </span>
        )
      })}
    </div>
  )
}

// the credit lines: brighter than C.dim so they read comfortably on the dark backdrop
const CREDIT = "rgba(245,247,251,0.8)"

const NV_STACK = ["Nemotron 3 Super on Nebius Token Factory", "NeMo FastConformer", "Magpie TTS", "GPUs with CUDA"]

function CreditRow({ at, out, children, style }: { at: number; out: number; children: ReactNode; style?: CSSProperties }) {
  const f = useCurrentFrame()
  const p = springAt(f, at, SOFT)
  const q = ramp(f, out, out + 24, easeIn)
  return (
    <div
      style={{
        opacity: Math.min(1, p) * (1 - q),
        transform: `translateY(${(1 - p) * 18 - q * 10}px)`,
        filter: `blur(${(1 - Math.min(1, p)) * 6}px)`,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        textAlign: "center",
        ...style,
      }}
    >
      {children}
    </div>
  )
}

export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const L = lineCue(scene, 0)
  const end = scene.frames
  const lineEnd = L.at + L.frames
  const nameAt = L.at - 2 // "StutterClear" is the first word said
  const tagAt = wordAt(L.at, L.frames, "Your")
  // the finished promise holds still for ~0.5 s after "terms." before the credits make room; the
  // credits are then fully readable for ~2.6 s before everything fades to the backdrop (last ~0.9 s)
  const credAt = lineEnd + 10
  const outAt = end - 27

  // the hero group (name, promise, voice line) rises a little when the promise arrives, and again
  // to make room for the credits
  const lift1 = springAt(f, tagAt - 4, SOFT)
  const lift2 = springAt(f, credAt - 2, SOFT)
  const heroY = -30 * lift1 - 92 * lift2
  const heroScale = 1 - 0.08 * lift2
  const heroOut = ramp(f, outAt, end - 4, easeIn)

  // the voice line: draws in from the left as the scene opens, then breathes
  const ribbonIn = ramp(f, 0, 52, easeOut)
  const breathe = 1 + 0.12 * Math.sin(f / 22)
  const ribbonH = 128 * breathe
  const ribbonGlow = 0.5 + 0.5 * springAt(f, nameAt + 6, SOFT)
  // a feathered leading edge while it draws (not a hard clip), with a soft light riding it
  const P = ribbonIn * 108
  const drawMask: CSSProperties =
    ribbonIn < 1
      ? {
          maskImage: `linear-gradient(90deg, #000 ${P - 8}%, transparent ${P}%)`,
          WebkitMaskImage: `linear-gradient(90deg, #000 ${P - 8}%, transparent ${P}%)`,
          maskSize: "100% 300%",
          WebkitMaskSize: "100% 300%",
          maskPosition: "center",
          WebkitMaskPosition: "center",
          maskRepeat: "no-repeat",
          WebkitMaskRepeat: "no-repeat",
        }
      : {}
  const headX = 160 + 1600 * Math.min(1, (P - 4) / 100)
  const headOpacity = 0.5 * Math.sin(Math.PI * ribbonIn)

  // the name's glow swells as it lands
  const nameGlow = springAt(f, nameAt + 4, SOFT)
  const tagGlow = springAt(f, tagAt + 10, SOFT)

  const env = d.clip.env_after
  const secs = d.clip.after_s

  return (
    <AbsoluteFill>
      <Sfx name="chime" at={nameAt + 10} volume={0.22} />

      {/* back layer: slow motes, behind everything (moves less than the hero: parallax) */}
      <div style={{ position: "absolute", inset: 0, opacity: (0.3 + 0.7 * ramp(f, 0, 40)) * (1 - heroOut) }}>
        <Particles count={22} color={C.teal} seed="oyt" x={140} y={160} w={1640} h={700} size={[1.4, 3.2]} rise={0.35} opacity={0.42} />
      </div>

      <Camera
        shots={[
          { at: 0, x: 960, y: 540, zoom: 1.0 },
          { at: nameAt, zoom: 1.045, dur: Math.max(60, lineEnd - nameAt) },
          { at: credAt, zoom: 1.0, dur: 60 },
          { at: outAt, zoom: 1.03, dur: end - outAt },
        ]}
        drift={5}
        seed="oyt"
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            transform: `translateY(${heroY}px) scale(${heroScale})`,
            transformOrigin: "960px 470px",
            opacity: 1 - heroOut,
          }}
        >
          {/* light behind the name */}
          <GlowOrb x={960} y={400} r={620} color="rgba(94,234,212,0.55)" opacity={0.1 + 0.22 * nameGlow} />
          <GlowOrb x={960} y={400} r={300} color="rgba(125,211,252,0.6)" opacity={0.12 * nameGlow + 0.08 * tagGlow} />

          {/* the voice line: Alex's cleaned voice, breathing across the lower middle */}
          <div style={{ position: "absolute", left: 160, top: 690 - ribbonH / 2, width: 1600, height: ribbonH, opacity: (0.55 + 0.45 * ribbonGlow) * (1 - 0.35 * lift2), ...drawMask }}>
            <Ribbon env={env} seconds={secs} width={1600} height={ribbonH} color={C.teal} color2={C.sky} strands={4} progress={1} calm={0.9} speed={0.55} />
          </div>
          {headOpacity > 0.01 && <GlowOrb x={headX} y={690} r={44} color="rgba(94,234,212,0.9)" opacity={headOpacity} />}
          {/* a faint mirror of it, for depth */}
          <div
            style={{
              position: "absolute",
              left: 160,
              top: 752,
              width: 1600,
              height: ribbonH * 0.6,
              opacity: 0.12 * ribbonIn * (1 - lift2),
              transform: "scaleY(-1)",
              filter: "blur(3px)",
              ...drawMask,
            }}
          >
            <Ribbon env={env} seconds={secs} width={1600} height={ribbonH * 0.6} color={C.teal} color2={C.sky} strands={2} progress={1} calm={0.9} speed={0.55} />
          </div>

          {/* the name */}
          <div style={{ position: "absolute", left: 0, right: 0, top: 296, display: "flex", justifyContent: "center" }}>
            <div style={{ filter: `drop-shadow(0 0 ${30 * nameGlow}px rgba(94,234,212,0.35))` }}>
              <Wordmark at={nameAt} size={168} shines={[nameAt + 24, credAt + 44]} />
            </div>
          </div>

          {/* the promise */}
          <div style={{ position: "absolute", left: 0, right: 0, top: 500, display: "flex", justifyContent: "center" }}>
            <Tagline at={L.at} frames={L.frames} size={76} />
          </div>
        </div>
      </Camera>

      {/* the credits: outside the camera, steady and readable */}
      <div style={{ position: "absolute", left: 96, right: 96, top: 662, display: "flex", flexDirection: "column", alignItems: "center", gap: 15 }}>
        <CreditRow at={credAt} out={outAt}>
          <div style={{ width: 320, height: 2, borderRadius: 1, background: `linear-gradient(90deg, transparent, ${C.teal}99, transparent)` }} />
        </CreditRow>
        <CreditRow at={credAt + 2} out={outAt} style={{ fontFamily: SANS, fontSize: 24, fontWeight: 500, color: C.text, gap: 14 }}>
          <span>Built for the Nebius × NVIDIA Global AI Hackathon</span>
          <span style={{ color: C.dim }}>·</span>
          <span style={{ fontFamily: MONO, fontSize: 22, color: C.teal }}>github.com/Pradyuman-aviator/stutterClear</span>
        </CreditRow>
        <CreditRow at={credAt + 4} out={outAt} style={{ gap: 10, flexWrap: "nowrap" }}>
          {NV_STACK.map((s, i) => (
            <span
              key={s}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 9,
                fontFamily: SANS,
                fontSize: 22,
                fontWeight: 500,
                color: C.text,
                padding: "6px 14px 6px 6px",
                borderRadius: 999,
                border: `1px solid rgba(118,185,0,0.35)`,
                background: "rgba(118,185,0,0.07)",
                opacity: Math.min(1, springAt(f, credAt + 4 + i * 2, SOFT)),
              }}
            >
              <span style={{ fontFamily: MONO, fontSize: 22, fontWeight: 700, letterSpacing: 0.5, color: "#0b0f02", background: C.nv, borderRadius: 999, padding: "2px 10px" }}>
                NVIDIA
              </span>
              {s}
            </span>
          ))}
        </CreditRow>
        <CreditRow at={credAt + 6} out={outAt} style={{ maxWidth: 1500, fontFamily: SANS, fontSize: 23, lineHeight: 1.45, color: CREDIT }}>
          <span>
            <span style={{ color: C.text }}>Alex is fictional.</span> His voice: a real recording from UCLASS, University College London (Howell, Davis &amp;
            Bartrip, 2009);
            <br />
            data collection supported by the Wellcome Trust.
          </span>
        </CreditRow>
        <CreditRow at={credAt + 8} out={outAt} style={{ fontFamily: MONO, fontSize: 22, letterSpacing: 0.5, color: CREDIT }}>
          {d.voice_label}
        </CreditRow>
      </div>
    </AbsoluteFill>
  )
}
