import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion"

import type { Data, Scene } from "../data"
import { sec } from "../data"
import { C } from "../theme"
import { Logo, Pill, Rise, Wave } from "../ui"

// 1 in 100 people stutter; a waveform with its repeats glowing red.
export function Hook({ d }: { d: Data; scene: Scene }) {
  const f = useCurrentFrame()
  const show = 6 // seconds of the clip drawn
  const n = Math.floor((show / d.clip.before_s) * d.clip.env_before.length)
  const glow = 0.55 + 0.45 * Math.sin(f / 7)
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 46, paddingBottom: 140 }}>
      <Rise delay={4}>
        <div style={{ fontSize: 132, fontWeight: 700, letterSpacing: -3, textAlign: "center", lineHeight: 1.05 }}>
          About <span style={{ color: C.sky }}>1 in 100</span> people
          <br />
          stutter.
        </div>
      </Rise>
      <Rise delay={sec(3.4)}>
        <div style={{ fontSize: 40, color: C.dim, textAlign: "center", maxWidth: 1300 }}>
          On a call, every repeat is something the listener has to wait through.
        </div>
      </Rise>
      <Rise delay={sec(1.4)} style={{ filter: `drop-shadow(0 0 ${18 * glow}px rgba(251,113,133,0.45))` }}>
        <Wave env={d.clip.env_before.slice(0, n)} seconds={show} width={1500} height={130} color={C.sky}
          cuts={d.clip.cuts.filter((c) => c.start < show)} />
      </Rise>
    </AbsoluteFill>
  )
}

export function Intro() {
  const f = useCurrentFrame()
  const shine = interpolate(f, [sec(0.5), sec(2.5)], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 40, paddingBottom: 140 }}>
      <Rise delay={2} style={{ transform: `scale(${0.96 + 0.04 * shine})` }}>
        <Logo size={190} />
      </Rise>
      <Rise delay={sec(1.6)}>
        <div style={{ fontSize: 56, fontWeight: 600, textAlign: "center" }}>
          Your real voice. <span style={{ color: C.dim }}>Without the repeats.</span>
        </div>
      </Rise>
      <Rise delay={sec(3.2)} style={{ display: "flex", gap: 20 }}>
        <Pill>Live in calls</Pill>
        <Pill>Recordings &amp; video</Pill>
        <Pill>On stage</Pill>
        <Pill>Practice</Pill>
      </Rise>
    </AbsoluteFill>
  )
}

export function End({ d }: { d: Data; scene: Scene }) {
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 34, paddingBottom: 150 }}>
      <Rise delay={2}>
        <Logo size={170} />
      </Rise>
      <Rise delay={sec(0.8)}>
        <div style={{ fontSize: 54, fontWeight: 600 }}>Your real voice, on your terms.</div>
      </Rise>
      <Rise delay={sec(1.8)}>
        <div style={{ fontSize: 34, color: C.sky }}>github.com/Pradyuman-aviator/stutterClear</div>
      </Rise>
      <Rise delay={sec(2.4)}>
        <div style={{ fontSize: 26, color: C.faint, textAlign: "center", lineHeight: 1.6, maxWidth: 1500 }}>
          Built for the Nebius × NVIDIA Global AI Hackathon · NVIDIA Nemotron on Nebius Token Factory ·
          NVIDIA FastConformer captions · WavLM · {d.voice_label} · {d.clip.credit}
        </div>
      </Rise>
    </AbsoluteFill>
  )
}
