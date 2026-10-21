import { Composition, staticFile } from "remotion"

import type { Data } from "./data"
import { FPS, timeline } from "./data"
import { Video } from "./Video"

export function Root() {
  return (
    <Composition
      id="StutterClear"
      component={Video}
      width={1920}
      height={1080}
      fps={FPS}
      durationInFrames={FPS * 10}
      defaultProps={{ data: null as Data | null }}
      calculateMetadata={async () => {
        const data: Data = await fetch(staticFile("generated/data.json")).then((r) => r.json())
        const scenes = timeline(data)
        const last = scenes[scenes.length - 1]
        return { durationInFrames: last.from + last.frames, props: { data } }
      }}
    />
  )
}
