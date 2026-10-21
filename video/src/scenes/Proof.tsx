import { AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame } from "remotion"

import type { Data, Scene } from "../data"
import { sec } from "../data"
import { C } from "../theme"
import { Card, Rise } from "../ui"

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const

const SHOTS: [string, string][] = [
  ["recordings", "Recordings & video"],
  ["calls", "Calls: live cleaning and captions"],
  ["dictate", "Dictate: clean text"],
]

// The real app: three of its tabs.
export function App(_: { d: Data; scene: Scene }) {
  const f = useCurrentFrame()
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", paddingBottom: 170 }}>
      <div style={{ position: "relative", width: 1800, height: 700 }}>
        {SHOTS.map(([name, label], i) => {
          const side = i - 1
          const enter = interpolate(f, [sec(0.2 + i * 0.5), sec(1.2 + i * 0.5)], [0, 1], clamp)
          const w = side === 0 ? 920 : 660
          return (
            <div key={name} style={{
              position: "absolute",
              left: 900 - w / 2 + side * 560,
              top: side === 0 ? 0 : 40,
              width: w,
              zIndex: side === 0 ? 2 : 1,
              opacity: enter,
              transform: `translateY(${(1 - enter) * 40}px) perspective(1600px) rotateY(${-side * 14}deg)`,
            }}>
              {side !== 0 && (
                <div style={{ marginBottom: 14, fontSize: 26, color: C.dim, textAlign: side < 0 ? "left" : "right", padding: "0 30px" }}>
                  {label}
                </div>
              )}
              <Card style={{ padding: 10, overflow: "hidden" }}>
                <Img src={staticFile(`generated/screens/${name}.png`)} style={{ width: "100%", borderRadius: 20, display: "block" }} />
              </Card>
              {side === 0 && <div style={{ marginTop: 18, fontSize: 26, color: C.text, textAlign: "center" }}>{label}</div>}
            </div>
          )
        })}
      </div>
      <Rise delay={sec(3)}>
        <div style={{ fontSize: 28, color: C.dim, marginTop: 10 }}>
          Plus: Practice interviews · Fluency journal · Review your cuts · Esc pauses cleaning
        </div>
      </Rise>
    </AbsoluteFill>
  )
}

function Stat({ value, unit, label, sub, delay, decimals = 0 }: { value: number; unit: string; label: string; sub: string; delay: number; decimals?: number }) {
  const f = useCurrentFrame()
  const v = interpolate(f, [delay, delay + sec(1.4)], [0, value], { ...clamp, easing: (t) => 1 - (1 - t) ** 3 })
  return (
    <Rise delay={delay}>
      <Card style={{ width: 500, padding: "40px 40px", display: "flex", flexDirection: "column", gap: 14 }}>
        <div style={{ fontSize: 120, fontWeight: 700, letterSpacing: -3, color: C.sky, lineHeight: 1 }}>
          {v.toFixed(decimals)}
          <span style={{ fontSize: 64 }}>{unit}</span>
        </div>
        <div style={{ fontSize: 34, fontWeight: 600 }}>{label}</div>
        <div style={{ fontSize: 24, color: C.dim, lineHeight: 1.4 }}>{sub}</div>
      </Card>
    </Rise>
  )
}

// Measured on speakers the models never trained on.
export function Numbers({ d }: { d: Data; scene: Scene }) {
  const n = d.numbers
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 50, paddingBottom: 170 }}>
      <Rise delay={2}>
        <div style={{ fontSize: 30, color: C.dim, letterSpacing: 3, fontWeight: 600 }}>MEASURED ON {n.speakers} SPEAKERS IT NEVER TRAINED ON</div>
      </Rise>
      <div style={{ display: "flex", gap: 40 }}>
        <Stat value={n.live_strong_removed} unit="%" label="of stutters removed, live" delay={sec(1)}
          sub={`Strong mode on a laptop GPU · ${n.live_strong_lost.toFixed(1)}% of words lost`} />
        <Stat value={n.rec_thorough_removed} unit="%" label="removed in recordings" delay={sec(2.6)}
          sub={`cuts only between words · ${n.rec_thorough_lost}% of words lost`} />
        <Stat value={0.5} decimals={1} unit=" s" label="behind you, live" delay={sec(4.2)} sub="the delay that lets it hear a repeat before the listener does" />
      </div>
    </AbsoluteFill>
  )
}
