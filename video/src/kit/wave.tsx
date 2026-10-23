import type { CSSProperties } from "react"
import { useId } from "react"
import { interpolateColors, useCurrentFrame } from "remotion"

import { C, CLASS_COLORS, MONO } from "../theme"

// Peak of a 10 ms envelope between two times, 0..1 of the clip's loudest moment.
export function envAt(env: number[], t0: number, t1 = t0 + 0.05): number {
  const peak = Math.max(1e-4, ...env)
  const a = Math.max(0, Math.floor(t0 * 100))
  const b = Math.min(env.length, Math.max(a + 1, Math.ceil(t1 * 100)))
  let v = 0
  for (let i = a; i < b; i++) v = Math.max(v, env[i])
  return v / peak
}

// Alex's voice as bars. Bars inside a cut turn `cutColor` with `cutTint`, and `collapse` (one
// 0..1 per cut) shrinks each cut out of the line, the rest closing up behind it: the signature
// "snap" of a stutter leaving the recording. `progress` (0..1 of the clip) draws a playhead;
// `reveal` draws the bars in from the left.
export function Wave({
  env,
  seconds,
  width,
  height,
  color = C.sky,
  cuts = [],
  cutTint = 0,
  cutColor = C.rose,
  collapse = [],
  progress,
  dimAfter = false,
  reveal = 1,
  barSeconds = 0.04,
  align = "left",
  glow = 0,
  style,
}: {
  env: number[]
  seconds: number
  width: number
  height: number
  color?: string
  cuts?: { start: number; end: number }[]
  cutTint?: number
  cutColor?: string
  collapse?: number[]
  progress?: number
  dimAfter?: boolean
  reveal?: number
  barSeconds?: number
  align?: "left" | "center"
  glow?: number
  style?: CSSProperties
}) {
  const n = Math.ceil(seconds / barSeconds)
  const slot = width / n
  const bars: { x: number; w: number; h: number; c: string; o: number; t: number }[] = []
  let x = 0
  for (let i = 0; i < n; i++) {
    const t0 = i * barSeconds
    const t1 = Math.min(seconds, t0 + barSeconds)
    const tc = (t0 + t1) / 2
    const ci = cuts.findIndex((c) => tc >= c.start && tc < c.end)
    const k = ci >= 0 ? 1 - Math.min(1, Math.max(0, collapse[ci] ?? 0)) : 1
    const v = envAt(env, t0, t1)
    const shown = Math.min(1, Math.max(0, (reveal * n - i) / 8))
    const played = progress === undefined || !dimAfter || tc <= progress * seconds ? 1 : 0.32
    bars.push({
      x,
      w: slot * k,
      h: Math.max(3, Math.sqrt(v) * height) * (0.25 + 0.75 * k),
      c: ci >= 0 ? interpolateColors(cutTint, [0, 1], [color, cutColor]) : color,
      o: shown * played * (ci >= 0 ? 0.25 + 0.75 * k : 1),
      t: tc,
    })
    x += slot * k
  }
  const off = align === "center" ? (width - x) / 2 : 0
  let head: number | undefined
  if (progress !== undefined) {
    const t = progress * seconds
    const b = bars.find((q) => q.t + barSeconds / 2 >= t) ?? bars[bars.length - 1]
    head = off + b.x + b.w * Math.min(1, Math.max(0, (t - (b.t - barSeconds / 2)) / barSeconds))
  }
  return (
    <div style={{ position: "relative", width, height, ...style }}>
      <svg width={width} height={height} style={{ overflow: "visible", filter: glow ? `drop-shadow(0 0 ${glow}px ${color})` : undefined }}>
        {bars.map((b, i) =>
          b.w < 0.05 || b.o <= 0.01 ? null : (
            <rect
              key={i}
              x={off + b.x + b.w * 0.19}
              y={(height - b.h) / 2}
              width={Math.max(0.5, b.w * 0.62)}
              height={b.h}
              rx={Math.min(b.w * 0.31, 3)}
              fill={b.c}
              opacity={b.o}
            />
          ),
        )}
      </svg>
      {head !== undefined && progress! > 0 && progress! < 1 && (
        <div
          style={{
            position: "absolute",
            left: head - 1.5,
            top: -14,
            bottom: -14,
            width: 3,
            borderRadius: 2,
            background: C.text,
            boxShadow: `0 0 18px ${C.sky}`,
          }}
        />
      )}
    </div>
  )
}

// The x position (in px, 0..width) of time t in a Wave of `seconds` drawn `width` wide with some
// cuts collapsed: for pinning labels above a cut or a word.
export function waveX(t: number, seconds: number, width: number, cuts: { start: number; end: number }[] = [], collapse: number[] = [], align: "left" | "center" = "left") {
  let removed = 0
  let removedBefore = 0
  cuts.forEach((c, i) => {
    const k = Math.min(1, Math.max(0, collapse[i] ?? 0))
    const len = (c.end - c.start) * k
    removed += len
    if (t >= c.end) removedBefore += len
    else if (t > c.start) removedBefore += (t - c.start) * k
  })
  const scale = width / seconds
  const off = align === "center" ? (removed * scale) / 2 : 0
  return off + (t - removedBefore) * scale
}

// A flowing neon line for a voice: several silk strands whose height follows the envelope.
// `phase` moves it (pass frame / 20 or so); `progress` draws it from the left.
export function Ribbon({
  env,
  seconds,
  width,
  height,
  color = C.sky,
  color2 = C.violet,
  strands = 4,
  progress = 1,
  calm = 0,
  speed = 1,
  style,
}: {
  env: number[]
  seconds: number
  width: number
  height: number
  color?: string
  color2?: string
  strands?: number
  progress?: number
  calm?: number // 0..1: flattens the jagged parts (the voice after cleaning)
  speed?: number
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  const N = 140
  const mid = height / 2
  const id = `rib${useId().replace(/[^a-zA-Z0-9]/g, "")}` // unique: two ribbons on screen must not share defs
  const paths = Array.from({ length: strands }, (_, s) => {
    const pts: [number, number][] = []
    for (let i = 0; i <= N; i++) {
      const u = i / N
      const t = u * seconds
      const a = envAt(env, Math.max(0, t - 0.08), t + 0.08)
      const jag = Math.sin(u * 90 + s * 1.3 + f * 0.21 * speed) * 0.25 * (1 - calm)
      const wave = Math.sin(u * Math.PI * (6 + s * 1.5) + (f / 18) * speed * (1 + s * 0.15) + s)
      const amp = Math.sqrt(a) * (mid - 4) * (1 - 0.18 * s)
      pts.push([u * width, mid + (wave + jag) * amp * (0.55 + 0.45 * Math.sin(u * Math.PI))])
    }
    let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`
    for (let i = 1; i < pts.length - 1; i++) {
      const xc = (pts[i][0] + pts[i + 1][0]) / 2
      const yc = (pts[i][1] + pts[i + 1][1]) / 2
      d += ` Q ${pts[i][0].toFixed(1)} ${pts[i][1].toFixed(1)} ${xc.toFixed(1)} ${yc.toFixed(1)}`
    }
    return d
  })
  return (
    <svg width={width} height={height} style={{ overflow: "visible", ...style }}>
      <defs>
        <linearGradient id={id} x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stopColor={color} stopOpacity={0} />
          <stop offset="12%" stopColor={color} />
          <stop offset="88%" stopColor={color2} />
          <stop offset="100%" stopColor={color2} stopOpacity={0} />
        </linearGradient>
        <clipPath id={`${id}c`}>
          <rect x={0} y={-height} width={width * progress} height={height * 3} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${id}c)`} style={{ filter: `drop-shadow(0 0 10px ${color})` }}>
        {paths.map((d, s) => (
          <path key={s} d={d} fill="none" stroke={`url(#${id})`} strokeWidth={s === 0 ? 3.2 : 1.6} strokeOpacity={s === 0 ? 1 : 0.5 - s * 0.08} strokeLinecap="round" />
        ))}
      </g>
    </svg>
  )
}

// The teacher's output: one lane per class, the probability of that class every 20 ms as a lit
// area. `progress` reveals it left to right (in step with the clip, for instance), `threshold`
// draws the line a cut needs to cross.
export function ProbLanes({
  probs,
  classes,
  width,
  laneHeight = 56,
  gap = 16,
  from = 0,
  to,
  progress = 1,
  threshold,
  emphasize,
  labels = true,
  labelWidth = 290,
  playhead = false,
  style,
}: {
  probs: number[][]
  classes: string[]
  width: number
  laneHeight?: number
  gap?: number
  from?: number // first frame (20 ms) shown
  to?: number
  progress?: number
  threshold?: number
  emphasize?: number[] // class indexes drawn brighter; the others dim
  labels?: boolean
  labelWidth?: number
  playhead?: boolean
  style?: CSSProperties
}) {
  const end = to ?? probs.length
  const n = Math.max(1, end - from)
  const plotW = width - (labels ? labelWidth : 0)
  const x0 = labels ? labelWidth : 0
  const height = classes.length * laneHeight + (classes.length - 1) * gap
  const id = `pl${useId().replace(/[^a-zA-Z0-9]/g, "")}`
  return (
    <div style={{ position: "relative", width, height, ...style }}>
      <svg width={width} height={height} style={{ overflow: "visible" }}>
        <defs>
          {CLASS_COLORS.map((c, k) => (
            <linearGradient key={k} id={`${id}g${k}`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={c} stopOpacity={0.9} />
              <stop offset="100%" stopColor={c} stopOpacity={0.05} />
            </linearGradient>
          ))}
          <clipPath id={`${id}c`}>
            <rect x={0} y={-10} width={x0 + plotW * progress} height={height + 20} />
          </clipPath>
        </defs>
        {classes.map((_, k) => {
          const top = k * (laneHeight + gap)
          const dim = emphasize && !emphasize.includes(k) ? 0.28 : 1
          let d = `M ${x0} ${top + laneHeight}`
          for (let i = 0; i < n; i++) {
            const p = probs[from + i]?.[k] ?? 0
            d += ` L ${(x0 + (i / (n - 1 || 1)) * plotW).toFixed(1)} ${(top + laneHeight - p * laneHeight).toFixed(1)}`
          }
          d += ` L ${x0 + plotW} ${top + laneHeight} Z`
          return (
            <g key={k} opacity={dim}>
              <rect x={x0} y={top} width={plotW} height={laneHeight} rx={8} fill="rgba(255,255,255,0.035)" />
              <g clipPath={`url(#${id}c)`}>
                <path d={d} fill={`url(#${id}g${k})`} stroke={CLASS_COLORS[k]} strokeWidth={1.6} />
              </g>
              {threshold !== undefined && (
                <line
                  x1={x0}
                  x2={x0 + plotW}
                  y1={top + laneHeight * (1 - threshold)}
                  y2={top + laneHeight * (1 - threshold)}
                  stroke="rgba(255,255,255,0.45)"
                  strokeDasharray="6 7"
                  strokeWidth={1.4}
                />
              )}
            </g>
          )
        })}
        {playhead && progress > 0 && progress < 1 && (
          <line x1={x0 + plotW * progress} x2={x0 + plotW * progress} y1={-12} y2={height + 12} stroke={C.text} strokeWidth={2.5} style={{ filter: `drop-shadow(0 0 8px ${C.sky})` }} />
        )}
      </svg>
      {labels &&
        classes.map((c, k) => (
          <div
            key={c}
            style={{
              position: "absolute",
              left: 0,
              top: k * (laneHeight + gap),
              height: laneHeight,
              width: labelWidth - 24,
              display: "flex",
              alignItems: "center",
              gap: 12,
              fontFamily: MONO,
              fontSize: 20,
              whiteSpace: "nowrap",
              color: C.dim,
              opacity: emphasize && !emphasize.includes(k) ? 0.4 : 1,
            }}
          >
            <span style={{ width: 12, height: 12, borderRadius: 6, background: CLASS_COLORS[k], boxShadow: `0 0 10px ${CLASS_COLORS[k]}` }} />
            {c}
          </div>
        ))}
    </div>
  )
}

// The model's view of time: one cell per 20 ms frame, lit by how sure it is that the frame is a
// stutter (the brightest class decides the colour). `progress` lights the cells left to right.
export function FrameCells({
  probs,
  from = 0,
  count = 100,
  cell = 12,
  gap = 3,
  perRow,
  progress = 1,
  threshold = 0.5,
  style,
}: {
  probs: number[][]
  from?: number
  count?: number
  cell?: number
  gap?: number
  perRow?: number
  progress?: number
  threshold?: number
  style?: CSSProperties
}) {
  const row = perRow ?? count
  const rows = Math.ceil(count / row)
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${row}, ${cell}px)`, gap, width: row * (cell + gap) - gap, height: rows * (cell + gap) - gap, ...style }}>
      {Array.from({ length: count }, (_, i) => {
        const p = probs[from + i] ?? [0, 0, 0, 0, 0]
        let k = 0
        for (let j = 1; j < p.length; j++) if (p[j] > p[k]) k = j
        const v = p[k] ?? 0
        const on = i / count <= progress
        return (
          <div
            key={i}
            style={{
              width: cell,
              height: cell,
              borderRadius: Math.max(2, cell / 4),
              background: on ? CLASS_COLORS[k] : "rgba(255,255,255,0.08)",
              opacity: on ? 0.14 + 0.86 * v : 1,
              boxShadow: on && v >= threshold ? `0 0 ${cell}px ${CLASS_COLORS[k]}` : undefined,
            }}
          />
        )
      })}
    </div>
  )
}
