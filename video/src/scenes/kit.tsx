import { AbsoluteFill, useCurrentFrame } from "remotion"

import type { SceneProps } from "../data"
import {
  Avatar,
  Block,
  Camera,
  Counter,
  Diagram,
  Eyebrow,
  FlowPath,
  FrameCells,
  Icon,
  ICONS,
  KineticText,
  LightSweep,
  Particles,
  ProbLanes,
  ramp,
  Ribbon,
  Typewriter,
  Wave,
} from "../kit"
import { C, MONO } from "../theme"

// A gallery of the kit, for checking it renders.
export default function Scene({ d, scene }: SceneProps) {
  const f = useCurrentFrame()
  const c = d.clip
  const collapse = c.cuts.map((_, i) => ramp(f, 60 + i * 5, 72 + i * 5))
  return (
    <AbsoluteFill>
      <Particles count={30} seed="kit" />
      <Camera shots={[{ at: 0, zoom: 1 }, { at: 150, zoom: 1.08, x: 900, y: 500, dur: 60 }]} drift={6}>
        <div style={{ position: "absolute", left: 80, top: 60 }}>
          <Eyebrow>Kit · {scene.frames} frames</Eyebrow>
          <KineticText text="Alex has something to say." at={4} style={{ fontSize: 64, marginTop: 10 }} highlight={{ say: C.teal }} />
        </div>
        <div style={{ position: "absolute", left: 80, top: 210 }}>
          <Wave env={c.env_before} seconds={c.before_s} width={900} height={120} cuts={c.cuts} cutTint={ramp(f, 20, 50)} collapse={collapse} progress={0.4} glow={6} />
        </div>
        <div style={{ position: "absolute", left: 80, top: 360 }}>
          <ProbLanes probs={c.probs} classes={c.classes} width={900} laneHeight={34} gap={8} progress={ramp(f, 0, 120)} threshold={0.5} playhead labelWidth={230} />
        </div>
        <div style={{ position: "absolute", left: 80, top: 600 }}>
          <FrameCells probs={c.probs} from={830} count={140} cell={10} gap={3} perRow={70} progress={ramp(f, 10, 100)} />
        </div>
        <div style={{ position: "absolute", left: 80, top: 680 }}>
          <Ribbon env={c.env_before} seconds={c.before_s} width={900} height={140} progress={ramp(f, 0, 90)} />
        </div>
        <Diagram>
          <FlowPath d="M 1100 260 C 1300 260, 1300 520, 1500 520" draw={ramp(f, 10, 60)} flow={1} color={C.nv} kind="frame" label="audio frames" />
          <FlowPath d="M 1100 700 C 1300 700, 1300 560, 1480 560" draw={ramp(f, 20, 70)} flow={1} color={C.violet} kind="word" words={["they", "they", "they", "have"]} period={120} />
        </Diagram>
        <Block x={1500} y={440} w={360} h={170} title="Stutter detector" model="WavLM-Large · 5 classes" where="GPU · CUDA" icon="chip" accent={C.nv} nv active={ramp(f, 40, 60)} appear={ramp(f, 0, 30)} />
        <div style={{ position: "absolute", left: 1520, top: 680 }}>
          <Avatar size={150} level={0.5 + 0.5 * Math.sin(f / 7)} />
        </div>
        <div style={{ position: "absolute", left: 1100, top: 820, fontSize: 60, fontWeight: 700, color: C.green }}>
          <Counter to={c.removed_pct} decimals={0} suffix="%" at={20} />
        </div>
        <div style={{ position: "absolute", left: 1100, top: 920, fontFamily: MONO, fontSize: 26, width: 700 }}>
          <Typewriter text={c.clean} at={10} cps={30} />
        </div>
        <div style={{ position: "absolute", left: 1100, top: 120, display: "flex", gap: 14, color: C.dim, flexWrap: "wrap", width: 760 }}>
          {ICONS.map((n) => (
            <Icon key={n} name={n} size={34} />
          ))}
        </div>
        <div style={{ position: "absolute", left: 1100, top: 30, width: 400, height: 60, borderRadius: 14, overflow: "hidden", background: "rgba(255,255,255,0.06)" }}>
          <LightSweep at={20} dur={60} />
        </div>
      </Camera>
    </AbsoluteFill>
  )
}
