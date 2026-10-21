import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion"

import type { Data, Scene } from "../data"
import { sec } from "../data"
import { C } from "../theme"
import { Rise } from "../ui"

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const

// The meaning guard on a typed sentence, with Nemotron's real verdicts and reasons.
export function Guard({ d }: { d: Data; scene: Scene }) {
  const f = useCurrentFrame()
  const { fps } = useVideoConfig()
  const flag = interpolate(f, [sec(2.0), sec(2.6)], [0, 1], clamp)
  const scan = interpolate(f, [sec(4.2), sec(6.4)], [0, 1], clamp)
  const verdict = spring({ frame: f - sec(6.6), fps, config: { damping: 200 } })
  const final = interpolate(f, [sec(11.5), sec(12.3)], [0, 1], clamp)
  const words = d.guard.words
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 70, paddingBottom: 170 }}>
      <Rise delay={2} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
        <div style={{ fontSize: 30, color: C.dim, letterSpacing: 3, fontWeight: 600 }}>THE MEANING GUARD</div>
        <div style={{ fontSize: 24, color: C.faint }}>NVIDIA Nemotron on Nebius Token Factory · {d.model}</div>
      </Rise>
      <div style={{ position: "relative", display: "flex", gap: 12, alignItems: "flex-start" }}>
        {scan > 0 && scan < 1 && (
          <div style={{ position: "absolute", top: -14, bottom: -14, left: `${scan * 100}%`, width: 6, borderRadius: 3,
            background: C.violet, boxShadow: `0 0 30px ${C.violet}` }} />
        )}
        {words.map((w, i) => {
          const removed = w.candidate && !w.kept
          const kept = w.candidate && w.kept
          const color = !w.candidate ? C.text : flag < 1 ? C.text : verdict > 0.5 ? (kept ? C.green : C.rose) : C.amber
          return (
            <Rise key={i} delay={sec(0.3) + i * 3}>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
                <div style={{
                  fontSize: 54,
                  fontWeight: 650,
                  padding: "8px 16px",
                  borderRadius: 18,
                  border: `2px solid ${w.candidate ? `${color}${flag > 0 ? "aa" : "00"}` : "transparent"}`,
                  background: w.candidate ? `${color}${Math.round(flag * 24).toString(16).padStart(2, "0")}` : "transparent",
                  color,
                  textDecoration: removed && verdict > 0.5 ? "line-through" : "none",
                  textDecorationThickness: 5,
                  opacity: removed ? 1 - 0.55 * verdict : 1,
                  transform: `scale(${removed ? 1 - 0.12 * verdict : kept ? 1 + 0.06 * verdict : 1})`,
                }}>
                  {w.w}
                </div>
                {kept && (
                  <div style={{ opacity: verdict, fontSize: 24, color: C.green, maxWidth: 260, textAlign: "center", lineHeight: 1.3 }}>
                    kept: {w.reason}
                  </div>
                )}
              </div>
            </Rise>
          )
        })}
      </div>
      <div style={{ opacity: final, fontSize: 46, color: C.teal, transform: `translateY(${(1 - final) * 20}px)` }}>
        “{words.filter((w) => !(w.candidate && !w.kept)).map((w) => w.w).join(" ")}”
      </div>
    </AbsoluteFill>
  )
}
