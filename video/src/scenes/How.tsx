import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion"

import type { Data, Scene } from "../data"
import { sec } from "../data"
import { C } from "../theme"
import { Card, Rise } from "../ui"

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const

const NODES = [
  ["Your microphone", "your real voice"],
  ["Delay buffer", "about 0.5 s behind you"],
  ["WavLM detector", "trained on real stuttered speech, on the GPU"],
  ["Splicer", "cuts between words, joins with your room tone"],
  ["The listener", "hears you, without the repeats"],
]

// The live pipeline, then a strip of speech losing its repeats and closing the gaps.
export function How(_: { d: Data; scene: Scene }) {
  const f = useCurrentFrame()
  const pulse = (f % sec(2.2)) / sec(2.2)
  const shrink = interpolate(f, [sec(6.5), sec(8.5)], [1, 0], clamp)
  const segments: [string, boolean][] = [["we came upon", false], ["a", true], ["a", true], ["a village", false],
    ["village", true], ["which", true], ["which", true], ["which started", false]]
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 90, paddingBottom: 150 }}>
      <div style={{ position: "relative", display: "flex", gap: 36 }}>
        <div style={{ position: "absolute", left: 140, right: 140, top: 92, height: 4, borderRadius: 2,
          background: `linear-gradient(90deg, ${C.sky}55, ${C.teal}55)` }} />
        <div style={{ position: "absolute", left: 140 + pulse * (5 * 300 + 4 * 36 - 280), top: 84, width: 20, height: 20,
          borderRadius: 10, background: C.sky, boxShadow: `0 0 26px ${C.sky}` }} />
        {NODES.map(([t, s], i) => (
          <Rise key={t} delay={sec(0.4 + i * 0.55)}>
            <Card style={{ width: 300, height: 190, padding: 24, display: "flex", flexDirection: "column", justifyContent: "center", gap: 10,
              background: "#0f141e", borderColor: i === 2 ? `${C.sky}66` : C.border }}>
              <div style={{ fontSize: 31, fontWeight: 650 }}>{t}</div>
              <div style={{ fontSize: 22, color: C.dim, lineHeight: 1.35 }}>{s}</div>
            </Card>
          </Rise>
        ))}
      </div>
      <Rise delay={sec(4.5)}>
        <div style={{ display: "flex", alignItems: "center", gap: 0, fontSize: 34 }}>
          {segments.map(([t, cut], i) => (
            <div key={i} style={{
              maxWidth: cut ? 260 * shrink : 400,
              overflow: "hidden",
              whiteSpace: "nowrap",
              padding: cut ? `14px ${18 * shrink}px` : "14px 18px",
              margin: cut ? `0 ${4 * shrink}px` : "0 4px",
              borderRadius: 14,
              background: cut ? "rgba(251,113,133,0.18)" : "rgba(94,234,212,0.12)",
              border: cut ? `1px solid rgba(251,113,133,${0.33 * shrink})` : `1px solid ${C.teal}55`,
              color: cut ? C.rose : C.text,
              opacity: cut ? 0.25 + 0.75 * shrink : 1,
            }}>
              {t}
            </div>
          ))}
        </div>
      </Rise>
    </AbsoluteFill>
  )
}
