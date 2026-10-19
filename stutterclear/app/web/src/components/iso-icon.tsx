import type { ReactNode } from "react"

// Small isometric marks. Each one is measured and fitted to the view box so it
// sits in the middle of the round badge, not low in the corner.

const COS = Math.cos(Math.PI / 6)
const SIN = 0.5

function project(x: number, y: number, z: number): [number, number] {
  return [(x - y) * COS, (x + y) * SIN - z]
}

function path(points: [number, number, number][]) {
  return (
    points
      .map(([x, y, z], i) => {
        const [px, py] = project(x, y, z)
        return `${i === 0 ? "M" : "L"}${px.toFixed(2)} ${py.toFixed(2)}`
      })
      .join(" ") + "Z"
  )
}

type Tone = "lo" | "mid" | "hi" | "lit"

const TOP: Record<Tone, string> = {
  lo: "#1e293b",
  mid: "#334155",
  hi: "#7dd3fc",
  lit: "#5eead4",
}
const LEFT: Record<Tone, string> = {
  lo: "#0f172a",
  mid: "#1e293b",
  hi: "#38bdf8",
  lit: "#2dd4bf",
}
const RIGHT: Record<Tone, string> = {
  lo: "#020617",
  mid: "#0f172a",
  hi: "#0369a1",
  lit: "#0f766e",
}

export type IsoPart = {
  x: number
  y: number
  z: number
  w: number
  d: number
  h: number
  tone?: Tone
}

function corners(p: IsoPart): [number, number, number][] {
  const { x, y, z, w, d, h } = p
  return [
    [x, y, z],
    [x + w, y, z],
    [x, y + d, z],
    [x + w, y + d, z],
    [x, y, z + h],
    [x + w, y, z + h],
    [x, y + d, z + h],
    [x + w, y + d, z + h],
  ]
}

export function IsoIcon({ label, parts }: { label: string; parts: IsoPart[] }) {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const part of parts) {
    for (const c of corners(part)) {
      const [px, py] = project(...c)
      minX = Math.min(minX, px)
      minY = Math.min(minY, py)
      maxX = Math.max(maxX, px)
      maxY = Math.max(maxY, py)
    }
  }
  const pad = 1.2
  const width = maxX - minX + pad * 2
  const height = maxY - minY + pad * 2
  const side = Math.max(width, height)
  const ox = minX - pad - (side - width) / 2
  const oy = minY - pad - (side - height) / 2

  return (
    <svg viewBox={`${ox} ${oy} ${side} ${side}`} className="block size-5" role="img" aria-label={label}>
      {parts.map((p, i) => {
        const tone = p.tone ?? "mid"
        const { x, y, z, w, d, h } = p
        return (
          <g key={i}>
            <path d={path([[x, y, z], [x, y + d, z], [x, y + d, z + h], [x, y, z + h]])} fill={LEFT[tone]} />
            <path d={path([[x, y + d, z], [x + w, y + d, z], [x + w, y + d, z + h], [x, y + d, z + h]])} fill={RIGHT[tone]} />
            <path d={path([[x, y, z + h], [x + w, y, z + h], [x + w, y + d, z + h], [x, y + d, z + h]])} fill={TOP[tone]} />
          </g>
        )
      })}
    </svg>
  )
}

export function IsoLogo() {
  return (
    <IsoIcon
      label="StutterClear"
      parts={[
        { x: 1, y: 2, z: 0, w: 14, d: 10, h: 1.4, tone: "lo" },
        { x: 3, y: 4, z: 1.4, w: 2.2, d: 6, h: 4, tone: "mid" },
        { x: 6.2, y: 3.4, z: 1.4, w: 2.2, d: 7.2, h: 8, tone: "hi" },
        { x: 9.4, y: 4.2, z: 1.4, w: 2.2, d: 5.4, h: 5.5, tone: "mid" },
        { x: 6.6, y: 5.2, z: 9.4, w: 1.4, d: 3.6, h: 0.8, tone: "lit" },
      ]}
    />
  )
}

export function IsoSetup() {
  return (
    <IsoIcon
      label="Setup"
      parts={[
        { x: 2, y: 2, z: 0, w: 14, d: 10, h: 1.2, tone: "lo" },
        { x: 3, y: 3, z: 1.2, w: 5, d: 8, h: 6, tone: "mid" },
        { x: 9, y: 4, z: 1.2, w: 5, d: 6, h: 3, tone: "hi" },
        { x: 10.2, y: 5.2, z: 4.2, w: 2.6, d: 1.2, h: 0.6, tone: "lit" },
      ]}
    />
  )
}

export function IsoMirror() {
  return (
    <IsoIcon
      label="Fluency mirror"
      parts={[
        { x: 1, y: 4, z: 0, w: 16, d: 8, h: 1, tone: "lo" },
        { x: 3, y: 5, z: 1, w: 1.2, d: 6, h: 8, tone: "mid" },
        { x: 4.2, y: 5.4, z: 3, w: 10, d: 0.8, h: 7, tone: "hi" },
        { x: 6, y: 5.6, z: 5, w: 6, d: 0.5, h: 0.6, tone: "lit" },
      ]}
    />
  )
}

export function IsoContext() {
  return (
    <IsoIcon
      label="Context card"
      parts={[
        { x: 1, y: 2, z: 0, w: 15, d: 11, h: 0.8, tone: "lo" },
        { x: 3, y: 3, z: 0.8, w: 12, d: 8, h: 0.5, tone: "hi" },
        { x: 4.2, y: 4.2, z: 1.3, w: 7, d: 0.4, h: 0.35, tone: "lit" },
        { x: 4.2, y: 5.4, z: 1.3, w: 5, d: 0.35, h: 0.3, tone: "mid" },
      ]}
    />
  )
}

export function IsoNote() {
  return (
    <IsoIcon
      label="Voice note"
      parts={[
        { x: 4, y: 3, z: 0, w: 8, d: 8, h: 1, tone: "lo" },
        { x: 6.6, y: 5.2, z: 1, w: 2.8, d: 2.8, h: 7, tone: "hi" },
        { x: 6.2, y: 4.8, z: 8, w: 3.6, d: 3.6, h: 1.4, tone: "lit" },
        { x: 7.4, y: 6, z: 9.4, w: 1.2, d: 1.2, h: 2.2, tone: "mid" },
      ]}
    />
  )
}

export function IsoHear() {
  return (
    <IsoIcon
      label="Hears the stutter"
      parts={[
        { x: 2, y: 6, z: 0, w: 12, d: 4, h: 1, tone: "lo" },
        { x: 4, y: 6.6, z: 1, w: 8, d: 2.8, h: 3, tone: "mid" },
        { x: 5, y: 7, z: 4, w: 6, d: 2, h: 2.2, tone: "hi" },
        { x: 6.4, y: 7.4, z: 6.2, w: 3.2, d: 1.2, h: 1.2, tone: "lit" },
      ]}
    />
  )
}

export function IsoCut() {
  return (
    <IsoIcon
      label="Cuts between words"
      parts={[
        { x: 1, y: 4, z: 0, w: 14, d: 3, h: 0.6, tone: "lo" },
        { x: 2, y: 4.4, z: 0.6, w: 5, d: 2.2, h: 1.2, tone: "hi" },
        { x: 9, y: 4.4, z: 0.6, w: 5, d: 2.2, h: 1.2, tone: "hi" },
        { x: 7.2, y: 3.2, z: 0.4, w: 1.6, d: 4.6, h: 0.5, tone: "lit" },
      ]}
    />
  )
}

export function IsoGuard() {
  return (
    <IsoIcon
      label="Meaning guard"
      parts={[
        { x: 4, y: 3, z: 0, w: 8, d: 8, h: 1, tone: "lo" },
        { x: 5, y: 4, z: 1, w: 6, d: 6, h: 6, tone: "mid" },
        { x: 6.2, y: 5.2, z: 7, w: 3.6, d: 3.6, h: 1.4, tone: "lit" },
      ]}
    />
  )
}

export function IsoListen() {
  return (
    <IsoIcon
      label="They hear you"
      parts={[
        { x: 2, y: 5, z: 0, w: 12, d: 6, h: 1, tone: "lo" },
        { x: 3, y: 6, z: 1, w: 4, d: 4, h: 5, tone: "hi" },
        { x: 8, y: 6.4, z: 1, w: 4.5, d: 3.2, h: 3.2, tone: "mid" },
        { x: 4, y: 6.8, z: 6, w: 2, d: 2, h: 0.8, tone: "lit" },
      ]}
    />
  )
}

export function IsoStage() {
  return (
    <IsoIcon
      label="On stage"
      parts={[
        { x: 1, y: 2, z: 0, w: 14, d: 10, h: 1, tone: "lo" },
        { x: 6, y: 5, z: 1, w: 3, d: 3, h: 7, tone: "hi" },
        { x: 5.2, y: 4.2, z: 8, w: 4.6, d: 4.6, h: 1.2, tone: "lit" },
      ]}
    />
  )
}

export function IsoPractice() {
  return (
    <IsoIcon
      label="Practice"
      parts={[
        { x: 2, y: 3, z: 0, w: 6, d: 6, h: 8, tone: "mid" },
        { x: 9, y: 4, z: 0, w: 5, d: 5, h: 6, tone: "hi" },
        { x: 3.2, y: 4.2, z: 8, w: 3.6, d: 0.4, h: 0.4, tone: "lit" },
      ]}
    />
  )
}

export function IsoBadge({ children }: { children: ReactNode }) {
  return (
    <span className="grid size-8 shrink-0 place-items-center rounded-full bg-white/[0.06] text-sky-200">
      {children}
    </span>
  )
}
