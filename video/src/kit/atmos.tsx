import type { CSSProperties } from "react"
import { AbsoluteFill, Img, random, staticFile, useCurrentFrame } from "remotion"
import { noise2D } from "@remotion/noise"

import { C } from "../theme"
import { clamp, easeOut } from "./motion"
import { interpolate } from "remotion"

// The film's floor: slow aurora glows that drift on noise, a vignette and film grain, so the frame
// never sits still. `tint` shifts the glows (e.g. violet for the cloud, nv green for the GPU).
export function Backdrop({ tint, intensity = 1 }: { tint?: string; intensity?: number }) {
  const f = useCurrentFrame()
  const blobs = [
    { c: `rgba(125,211,252,${0.15 * intensity})`, x: 430, y: 270, r: 950, s: "a" },
    { c: `rgba(196,181,253,${0.12 * intensity})`, x: 1520, y: 830, r: 1050, s: "b" },
    { c: `rgba(94,234,212,${0.07 * intensity})`, x: 1180, y: 150, r: 760, s: "c" },
  ]
  const layers = blobs.map((b) => {
    const x = b.x + noise2D(`${b.s}x`, f / 280, 0) * 260
    const y = b.y + noise2D(`${b.s}y`, 0, f / 280) * 170
    return `radial-gradient(${b.r}px ${Math.round(b.r * 0.78)}px at ${x.toFixed(1)}px ${y.toFixed(1)}px, ${b.c}, transparent 70%)`
  })
  if (tint) layers.unshift(`radial-gradient(1200px 800px at 960px 600px, ${tint}, transparent 72%)`)
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{ background: `${layers.join(", ")}, ${C.bg}` }} />
      <AbsoluteFill style={{ background: "radial-gradient(ellipse 75% 70% at 50% 48%, transparent 55%, rgba(0,0,0,0.6) 100%)" }} />
      <Grain />
    </AbsoluteFill>
  )
}

// Film grain: a noise tile that jumps every frame.
export function Grain({ opacity = 0.055 }: { opacity?: number }) {
  const f = useCurrentFrame()
  const x = Math.floor(random(`gx${f}`) * 512)
  const y = Math.floor(random(`gy${f}`) * 512)
  return (
    <AbsoluteFill style={{ pointerEvents: "none", mixBlendMode: "overlay", opacity }}>
      <Img src={staticFile("grain.png")} style={{ display: "none" }} />
      <AbsoluteFill style={{ backgroundImage: `url(${staticFile("grain.png")})`, backgroundPosition: `${x}px ${y}px` }} />
    </AbsoluteFill>
  )
}

// Glowing motes drifting upward on noise; deterministic for a seed.
export function Particles({
  count = 36,
  color = C.sky,
  seed = "p",
  x = 0,
  y = 0,
  w = 1920,
  h = 1080,
  size = [1.5, 4],
  rise = 0.4,
  opacity = 0.55,
  style,
}: {
  count?: number
  color?: string
  seed?: string
  x?: number
  y?: number
  w?: number
  h?: number
  size?: [number, number]
  rise?: number // pixels per frame, on average
  opacity?: number
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  return (
    <div style={{ position: "absolute", left: x, top: y, width: w, height: h, pointerEvents: "none", ...style }}>
      {Array.from({ length: count }, (_, i) => {
        const bx = random(`${seed}x${i}`) * w
        const by = random(`${seed}y${i}`) * h
        const sz = size[0] + random(`${seed}s${i}`) * (size[1] - size[0])
        const v = rise * (0.4 + random(`${seed}v${i}`) * 1.2)
        const px = bx + noise2D(`${seed}nx`, i * 7.3, f / 260) * 70
        const py = (((by - f * v) % h) + h) % h + noise2D(`${seed}ny`, i * 3.1, f / 260) * 30
        const o = opacity * (0.45 + 0.55 * (0.5 + 0.5 * noise2D(`${seed}o`, i * 1.7, f / 45)))
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: px,
              top: py,
              width: sz,
              height: sz,
              borderRadius: "50%",
              background: color,
              opacity: o,
              boxShadow: `0 0 ${sz * 4}px ${sz}px ${color}`,
            }}
          />
        )
      })}
    </div>
  )
}

// A soft band of light that sweeps across its parent once, from `at` over `dur` frames. Put it
// inside a parent with position: relative and overflow: hidden.
export function LightSweep({ at = 0, dur = 36, color = "rgba(255,255,255,0.22)", angle = 18 }: {
  at?: number
  dur?: number
  color?: string
  angle?: number
}) {
  const f = useCurrentFrame()
  const p = interpolate(f, [at, at + dur], [-0.4, 1.4], { ...clamp, easing: easeOut })
  if (f < at || f > at + dur) return null
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        mixBlendMode: "screen",
        background: `linear-gradient(${90 + angle}deg, transparent ${p * 100 - 18}%, ${color} ${p * 100}%, transparent ${p * 100 + 18}%)`,
      }}
    />
  )
}

// A big soft glow, e.g. behind a title or a model block that is working.
export function GlowOrb({ x, y, r = 300, color = C.sky, opacity = 0.35 }: { x: number; y: number; r?: number; color?: string; opacity?: number }) {
  return (
    <div
      style={{
        position: "absolute",
        left: x - r,
        top: y - r,
        width: r * 2,
        height: r * 2,
        borderRadius: "50%",
        background: `radial-gradient(circle, ${color} 0%, transparent 68%)`,
        opacity,
        pointerEvents: "none",
      }}
    />
  )
}
