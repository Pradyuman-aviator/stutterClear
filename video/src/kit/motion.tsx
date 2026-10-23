import type { CSSProperties, ReactNode } from "react"
import { Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion"
import type { SpringConfig } from "remotion"
import { noise2D } from "@remotion/noise"

export const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const
export const ease = Easing.bezier(0.65, 0, 0.35, 1) // in-out: camera moves, slides
export const easeOut = Easing.bezier(0.16, 1, 0.3, 1) // expo-out: entrances
export const easeIn = Easing.bezier(0.7, 0, 0.84, 0) // exits

export const SOFT: Partial<SpringConfig> = { damping: 200 } // no overshoot
export const SNAPPY: Partial<SpringConfig> = { damping: 18, mass: 0.7, stiffness: 140 } // a little overshoot
export const BOUNCY: Partial<SpringConfig> = { damping: 11, mass: 0.6, stiffness: 170 }

// Spring progress 0..1 for something that starts `at` frames into the scene (pure: use it in loops).
export const springAt = (f: number, at: number, config: Partial<SpringConfig> = SOFT, fps = 30) =>
  spring({ frame: f - at, fps, config })

// Linear 0..1 between two frames, eased.
export const ramp = (f: number, a: number, b: number, easing: (t: number) => number = ease) =>
  interpolate(f, [a, b], [0, 1], { ...clamp, easing })

export function useSpring(at: number, config: Partial<SpringConfig> = SOFT) {
  const f = useCurrentFrame()
  const { fps } = useVideoConfig()
  return spring({ frame: f - at, fps, config })
}

// Rises in with a spring at `at`; leaves (fades and lifts) at `out` if given.
export function Rise({
  at = 0,
  out,
  y = 30,
  blur = 0,
  children,
  style,
}: {
  at?: number
  out?: number
  y?: number
  blur?: number
  children: ReactNode
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  const p = springAt(f, at)
  const q = out === undefined ? 0 : ramp(f, out, out + 14, easeIn)
  const o = Math.min(1, p) * (1 - q)
  return (
    <div
      style={{
        opacity: o,
        transform: `translateY(${(1 - p) * y - q * y * 0.6}px)`,
        filter: blur ? `blur(${(1 - Math.min(1, p)) * blur + q * blur}px)` : undefined,
        ...style,
      }}
    >
      {children}
    </div>
  )
}

// Scales in with a little overshoot.
export function Pop({
  at = 0,
  out,
  from = 0.6,
  children,
  style,
}: {
  at?: number
  out?: number
  from?: number
  children: ReactNode
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  const p = springAt(f, at, BOUNCY)
  const q = out === undefined ? 0 : ramp(f, out, out + 12, easeIn)
  return (
    <div style={{ opacity: Math.min(1, p * 1.6) * (1 - q), transform: `scale(${(from + (1 - from) * p) * (1 - 0.15 * q)})`, ...style }}>
      {children}
    </div>
  )
}

// Text that arrives word by word (or letter by letter): each piece rises, unblurs and settles.
export function KineticText({
  text,
  at = 0,
  per = 3,
  by = "word",
  effect = "rise",
  highlight = {},
  style,
}: {
  text: string
  at?: number
  per?: number // frames between pieces
  by?: "word" | "char"
  effect?: "rise" | "blur" | "drop" | "scale"
  highlight?: Record<string, string> // word (lower case, no punctuation) -> colour
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  const pieces = by === "word" ? text.split(" ") : Array.from(text)
  return (
    <div style={style}>
      {pieces.map((w, i) => {
        const p = springAt(f, at + i * per, SNAPPY)
        const c = Math.min(1, p)
        const key = w.toLowerCase().replace(/[^a-z0-9'%.-]/g, "").replace(/\.$/, "")
        const t =
          effect === "drop"
            ? `translateY(${(p - 1) * 0.6}em)`
            : effect === "scale"
              ? `scale(${0.4 + 0.6 * p})`
              : `translateY(${(1 - p) * 0.45}em)`
        return (
          <span
            key={i}
            style={{
              display: "inline-block",
              whiteSpace: "pre",
              opacity: Math.min(1, p * 1.4),
              transform: t,
              filter: effect === "blur" || effect === "rise" ? `blur(${(1 - c) * (effect === "blur" ? 14 : 6)}px)` : undefined,
              color: highlight[key],
            }}
          >
            {w}
            {by === "word" && i < pieces.length - 1 ? " " : ""}
          </span>
        )
      })}
    </div>
  )
}

// Typed out at `cps` characters a second, with a blinking caret.
export function Typewriter({
  text,
  at = 0,
  cps = 28,
  caret = "#7dd3fc",
  hideCaretAfter,
  style,
}: {
  text: string
  at?: number
  cps?: number
  caret?: string
  hideCaretAfter?: number // frames after the text is complete
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  const { fps } = useVideoConfig()
  const n = Math.max(0, Math.min(text.length, Math.floor(((f - at) / fps) * cps)))
  const doneAt = at + (text.length / cps) * fps
  const showCaret = f >= at && (hideCaretAfter === undefined || f < doneAt + hideCaretAfter)
  return (
    <span style={style}>
      {text.slice(0, n)}
      {showCaret && <span style={{ color: caret, opacity: Math.floor(f / 15) % 2 === 0 ? 1 : 0 }}>▍</span>}
    </span>
  )
}

// A number that counts up.
export function Counter({
  to,
  from = 0,
  at = 0,
  dur = 45,
  decimals = 0,
  prefix = "",
  suffix = "",
  style,
}: {
  to: number
  from?: number
  at?: number
  dur?: number
  decimals?: number
  prefix?: string
  suffix?: string
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  const v = interpolate(f, [at, at + dur], [from, to], { ...clamp, easing: easeOut })
  return (
    <span style={{ fontVariantNumeric: "tabular-nums", ...style }}>
      {prefix}
      {v.toFixed(decimals)}
      {suffix}
    </span>
  )
}

// A camera over a 1920x1080 stage: each shot eases the view to centre (x, y) at `zoom`, starting at
// frame `at` and taking `dur` frames. The first shot is where the camera starts. `drift` adds a slow
// handheld float, in pixels.
export interface Shot {
  at: number
  x?: number
  y?: number
  zoom?: number
  rotate?: number
  dur?: number
}
type Cam = { x: number; y: number; zoom: number; rotate: number }

export function cameraAt(f: number, shots: Shot[]): Cam {
  const s = [...shots].sort((a, b) => a.at - b.at)
  if (!s.length) return { x: 960, y: 540, zoom: 1, rotate: 0 }
  const targets: Cam[] = []
  s.forEach((sh, i) => {
    const prev = i ? targets[i - 1] : { x: 960, y: 540, zoom: 1, rotate: 0 }
    targets.push({ x: sh.x ?? prev.x, y: sh.y ?? prev.y, zoom: sh.zoom ?? prev.zoom, rotate: sh.rotate ?? prev.rotate })
  })
  const lerp = (a: Cam, b: Cam, p: number): Cam => ({
    x: a.x + (b.x - a.x) * p,
    y: a.y + (b.y - a.y) * p,
    // zoom moves in log space so a push-in feels even
    zoom: Math.exp(Math.log(a.zoom) + (Math.log(b.zoom) - Math.log(a.zoom)) * p),
    rotate: a.rotate + (b.rotate - a.rotate) * p,
  })
  const starts: Cam[] = [targets[0]]
  const evalAt = (t: number, upto: number) => {
    let cur = targets[0]
    for (let i = 1; i <= upto && t > s[i].at; i++) cur = lerp(starts[i], targets[i], ramp(t, s[i].at, s[i].at + (s[i].dur ?? 40)))
    return cur
  }
  for (let i = 1; i < s.length; i++) starts[i] = evalAt(s[i].at, i - 1)
  return evalAt(f, s.length - 1)
}

export function Camera({
  shots,
  drift = 0,
  seed = "cam",
  children,
  style,
}: {
  shots: Shot[]
  drift?: number
  seed?: string
  children: ReactNode
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  const c = cameraAt(f, shots)
  const dx = drift ? noise2D(`${seed}x`, f / 110, 0) * drift : 0
  const dy = drift ? noise2D(`${seed}y`, 0, f / 110) * drift : 0
  const dr = drift ? noise2D(`${seed}r`, f / 160, 3) * drift * 0.012 : 0
  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        top: 0,
        width: 1920,
        height: 1080,
        transformOrigin: "0 0",
        transform: `translate(960px, 540px) rotate(${c.rotate + dr}deg) scale(${c.zoom}) translate(${-c.x + dx}px, ${-c.y + dy}px)`,
        ...style,
      }}
    >
      {children}
    </div>
  )
}

// Fades a whole group in at `at` and out at `out` (frames), for building a scene in beats.
export function Beat({
  at = 0,
  out,
  fade = 12,
  children,
  style,
}: {
  at?: number
  out?: number
  fade?: number
  children: ReactNode
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  const a = ramp(f, at, at + fade, easeOut)
  const b = out === undefined ? 0 : ramp(f, out, out + fade, easeIn)
  const o = a * (1 - b)
  if (o <= 0.001) return null
  return <div style={{ position: "absolute", inset: 0, opacity: o, ...style }}>{children}</div>
}
