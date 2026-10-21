import type { CSSProperties, ReactNode } from "react"
import { useMemo } from "react"
import { useCurrentFrame } from "remotion"
import { evolvePath, getLength, getPointAtLength } from "@remotion/paths"

import { C, MONO, SANS } from "../theme"
import type { IconName } from "./icons"
import { Icon } from "./icons"

// An SVG layer over the whole 1920x1080 stage, for FlowPaths. Put Blocks (HTML) next to it.
export function Diagram({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <svg width={1920} height={1080} viewBox="0 0 1920 1080" style={{ position: "absolute", left: 0, top: 0, overflow: "visible", ...style }}>
      {children}
    </svg>
  )
}

// A connection that draws itself (`draw` 0..1) and carries packets along it (`flow` 0..1 is their
// opacity). Packets are glowing dots, small audio-frame tiles, or words (pass `words`) so the
// viewer sees WHAT travels: sound frames to the GPU model, text (never audio) to Nemotron.
export function FlowPath({
  d,
  draw = 1,
  flow = 0,
  color = C.sky,
  width = 3,
  packets = 5,
  period = 70,
  kind = "dot",
  words = [],
  dashed = false,
  label,
  labelAt = 0.5,
  labelOffset = -22,
}: {
  d: string
  draw?: number
  flow?: number
  color?: string
  width?: number
  packets?: number
  period?: number // frames for a packet to cross the path
  kind?: "dot" | "frame" | "word"
  words?: string[]
  dashed?: boolean
  label?: string
  labelAt?: number // 0..1 along the path
  labelOffset?: number // px above the path
}) {
  const f = useCurrentFrame()
  const len = useMemo(() => getLength(d), [d])
  const ev = evolvePath(Math.min(1, Math.max(0, draw)), d)
  const mid = useMemo(() => getPointAtLength(d, len * labelAt), [d, len, labelAt])
  const count = kind === "word" ? Math.max(1, words.length) : packets
  return (
    <g>
      <path d={d} fill="none" stroke={color} strokeOpacity={0.12} strokeWidth={width * 4} strokeDasharray={ev.strokeDasharray} strokeDashoffset={ev.strokeDashoffset} strokeLinecap="round" />
      <path
        d={d}
        fill="none"
        stroke={color}
        strokeOpacity={0.75}
        strokeWidth={width}
        strokeLinecap="round"
        strokeDasharray={dashed ? "2 12" : ev.strokeDasharray}
        strokeDashoffset={dashed ? -f * 0.6 : ev.strokeDashoffset}
        opacity={dashed ? Math.min(1, draw * 1.5) : 1}
      />
      {flow > 0.01 &&
        Array.from({ length: count }, (_, i) => {
          const u = (f / period + i / count) % 1
          if (u > draw) return null
          const p = getPointAtLength(d, u * len)
          if (!p) return null
          const edge = Math.min(1, u * 10, (1 - u) * 10)
          const o = flow * edge
          if (kind === "frame") {
            return (
              <rect key={i} x={p.x - 7} y={p.y - 7} width={14} height={14} rx={3} fill={color} opacity={o} style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
            )
          }
          if (kind === "word") {
            const w = words[i % words.length] ?? ""
            const wpx = 14 + w.length * 11
            return (
              <g key={i} opacity={o} transform={`translate(${p.x} ${p.y})`}>
                <rect x={-wpx / 2} y={-17} width={wpx} height={34} rx={17} fill="rgba(10,14,26,0.92)" stroke={color} strokeWidth={1.5} />
                <text x={0} y={7} textAnchor="middle" fontFamily={MONO} fontSize={18} fill={C.text}>
                  {w}
                </text>
              </g>
            )
          }
          return <circle key={i} cx={p.x} cy={p.y} r={width * 1.7} fill={color} opacity={o} style={{ filter: `drop-shadow(0 0 7px ${color})` }} />
        })}
      {label && mid && draw > labelAt && (
        <text x={mid.x} y={mid.y + labelOffset} textAnchor="middle" fontFamily={MONO} fontSize={19} fill={C.dim} opacity={Math.min(1, (draw - labelAt) * 6)}>
          {label}
        </text>
      )}
    </g>
  )
}

// A part of the system: what it is, which model, where it runs. `nv` marks an NVIDIA model,
// `active` (0..1) lights it while it works, `appear` (0..1) brings it in.
export function Block({
  x,
  y,
  w = 360,
  h = 150,
  title,
  model,
  where,
  icon,
  accent = C.sky,
  nv = false,
  appear = 1,
  active = 0,
  children,
  style,
}: {
  x: number
  y: number
  w?: number
  h?: number
  title: string
  model?: string
  where?: string
  icon?: IconName
  accent?: string
  nv?: boolean
  appear?: number
  active?: number
  children?: ReactNode
  style?: CSSProperties
}) {
  const f = useCurrentFrame()
  const pulse = active > 0 ? 0.75 + 0.25 * Math.sin(f / 6) : 0
  const glow = active * pulse
  const a = Math.min(1, Math.max(0, appear))
  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        width: w,
        height: h,
        opacity: a,
        transform: `translateY(${(1 - a) * 24}px) scale(${0.94 + 0.06 * a})`,
        borderRadius: 22,
        padding: "18px 22px",
        boxSizing: "border-box",
        background: `linear-gradient(160deg, rgba(255,255,255,0.075), rgba(255,255,255,0.025))`,
        border: `1.5px solid ${glow > 0.05 ? accent : C.border2}`,
        boxShadow: `0 0 ${40 * glow}px ${accent}66, inset 0 1px 0 rgba(255,255,255,0.08), 0 24px 60px -24px rgba(0,0,0,0.8)`,
        display: "flex",
        flexDirection: "column",
        gap: 8,
        fontFamily: SANS,
        color: C.text,
        ...style,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        {icon && (
          <div style={{ width: 46, height: 46, borderRadius: 12, background: `${accent}22`, display: "flex", alignItems: "center", justifyContent: "center", color: accent, flexShrink: 0 }}>
            <Icon name={icon} size={28} />
          </div>
        )}
        <div style={{ fontSize: 28, fontWeight: 600, lineHeight: 1.15 }}>{title}</div>
      </div>
      {model && <div style={{ fontFamily: MONO, fontSize: 19, color: nv ? C.nv : C.dim, lineHeight: 1.3 }}>{model}</div>}
      {(where || nv) && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: "auto" }}>
          {nv && <NvTag />}
          {where && <Tag>{where}</Tag>}
        </div>
      )}
      {children}
    </div>
  )
}

// "NVIDIA" set in NVIDIA green (text only: no logo).
export function NvTag({ label = "NVIDIA", style }: { label?: string; style?: CSSProperties }) {
  return (
    <span
      style={{
        fontFamily: MONO,
        fontSize: 16,
        fontWeight: 700,
        letterSpacing: 1.5,
        color: "#0b0f02",
        background: C.nv,
        borderRadius: 6,
        padding: "3px 9px",
        ...style,
      }}
    >
      {label}
    </span>
  )
}

export function Tag({ children, color = C.dim, style }: { children: ReactNode; color?: string; style?: CSSProperties }) {
  return (
    <span
      style={{
        fontFamily: MONO,
        fontSize: 16,
        color,
        border: `1px solid ${C.border2}`,
        borderRadius: 6,
        padding: "3px 9px",
        background: "rgba(255,255,255,0.04)",
        ...style,
      }}
    >
      {children}
    </span>
  )
}
