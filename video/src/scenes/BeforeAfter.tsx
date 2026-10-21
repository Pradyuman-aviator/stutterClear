import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion"

import type { Data, Scene } from "../data"
import { C } from "../theme"
import { Card, Rise, Wave } from "../ui"

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const

// The real clip, as the app cleans it: what was said with its cuts in red, then what the listener
// hears, each with a playhead that follows its audio.
export function BeforeAfter({ d, scene }: { d: Data; scene: Scene }) {
  const f = useCurrentFrame()
  const cue = (id: string) => scene.cues.find((c) => c.id === id)!
  const before = cue("before")
  const afterLine = scene.cues.filter((c) => c.id === "after")[0]
  const after = scene.cues.filter((c) => c.id === "after")[1]
  const pb = interpolate(f, [before.at, before.at + before.frames], [0, 1], clamp)
  const pa = interpolate(f, [after.at, after.at + after.frames], [0, 1], clamp)
  const t = pb * d.clip.before_s
  const afterIn = interpolate(f, [afterLine.at - 8, afterLine.at + 10], [0, 1], clamp)
  const W = 1580

  return (
    <AbsoluteFill style={{ alignItems: "center", paddingTop: 70, gap: 26 }}>
      <Rise delay={2}>
        <Card style={{ width: 1720, padding: "28px 40px", display: "flex", flexDirection: "column", gap: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 26, color: C.dim }}>
            <span style={{ letterSpacing: 3, fontWeight: 600 }}>BEFORE · WHAT WAS SAID</span>
            <span>{d.clip.before_s.toFixed(1)} s</span>
          </div>
          <Wave env={d.clip.env_before} seconds={d.clip.before_s} width={W} height={140} color={C.sky} cuts={d.clip.cuts}
            progress={pb > 0 ? pb : 1} />
          <div style={{ fontSize: 30, lineHeight: 1.55, color: C.dim, minHeight: 96 }}>
            {d.clip.words.map((w, i) => {
              const said = w.s <= t || pb >= 1
              return (
                <span key={i} style={{
                  color: w.removed ? C.rose : said ? C.text : C.faint,
                  textDecoration: w.removed && (said || f > after.at - 30) ? "line-through" : "none",
                  textDecorationThickness: 3,
                  opacity: w.removed && pb >= 1 ? 0.75 : 1,
                }}>
                  {w.w}{" "}
                </span>
              )
            })}
          </div>
        </Card>
      </Rise>
      <div style={{ opacity: afterIn, transform: `translateY(${(1 - afterIn) * 24}px)` }}>
        <Card style={{ width: 1720, padding: "28px 40px", display: "flex", flexDirection: "column", gap: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 26, color: C.dim }}>
            <span style={{ letterSpacing: 3, fontWeight: 600, color: C.teal }}>AFTER · WHAT THE LISTENER HEARS</span>
            <span>{d.clip.after_s.toFixed(1)} s · {d.clip.cuts.length} cuts, all between words</span>
          </div>
          <Wave env={d.clip.env_after} seconds={d.clip.after_s} width={(W * d.clip.after_s) / d.clip.before_s} height={140}
            color={C.teal} progress={pa > 0 ? pa : 1} />
          <div style={{ fontSize: 30, lineHeight: 1.55 }}>{d.clip.clean}</div>
        </Card>
      </div>
      <div style={{ position: "absolute", left: 100, bottom: 130, fontSize: 22, color: C.faint }}>{d.clip.credit}</div>
    </AbsoluteFill>
  )
}
