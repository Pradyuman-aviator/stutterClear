import type { ReactNode } from "react"
import { AbsoluteFill, Html5Audio, Sequence, staticFile } from "remotion"

import type { Data, Scene } from "./data"
import { timeline } from "./data"
import { BeforeAfter } from "./scenes/BeforeAfter"
import { Guard } from "./scenes/Guard"
import { How } from "./scenes/How"
import { Calls, Rescue } from "./scenes/Live"
import { End, Hook, Intro } from "./scenes/Opening"
import { App, Numbers } from "./scenes/Proof"
import { C, SANS } from "./theme"
import { Background, SceneFade, Subtitle } from "./ui"

const SCENES: Record<string, (p: { d: Data; scene: Scene }) => ReactNode> = {
  hook: Hook,
  intro: Intro,
  beforeafter: BeforeAfter,
  how: How,
  guard: Guard,
  calls: Calls,
  rescue: Rescue,
  app: App,
  numbers: Numbers,
  end: End,
}

export function Video({ data }: { data: Data | null }) {
  if (!data) {
    return (
      <AbsoluteFill style={{ background: C.bg, color: C.text, fontFamily: SANS, alignItems: "center", justifyContent: "center", fontSize: 40 }}>
        Run scripts/export_video_data.py first.
      </AbsoluteFill>
    )
  }
  const scenes = timeline(data)
  return (
    <AbsoluteFill>
      <Background />
      {scenes.map((s) => {
        const View = SCENES[s.id]
        return (
          <Sequence key={s.id} from={s.from} durationInFrames={s.frames} name={s.id}>
            <SceneFade frames={s.frames}>
              <View d={data} scene={s} />
            </SceneFade>
          </Sequence>
        )
      })}
      {scenes.flatMap((s) =>
        s.cues.map((c) => (
          <Sequence key={`${s.id}-${c.id}-${c.kind}`} from={s.from + c.at} durationInFrames={c.frames} name={`${c.kind}:${c.id}`}>
            <Html5Audio src={staticFile(c.file)} />
            {c.kind === "line" && c.text && <Subtitle text={c.text} frames={c.frames} />}
          </Sequence>
        ))
      )}
    </AbsoluteFill>
  )
}
