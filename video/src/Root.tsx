import { Composition } from "remotion"

import type { Data } from "./data"
import { filmFrames, FPS, timeline } from "./data"
import { sceneCompositions } from "./preview"
import { VIEWS } from "./scenes/registry"
import { Film, loadData } from "./stage"

function FilmView({ data }: { data: Data | null }) {
  return <Film data={data} views={VIEWS} />
}

export function Root() {
  return (
    <>
      <Composition
        id="StutterClear"
        component={FilmView}
        width={1920}
        height={1080}
        fps={FPS}
        durationInFrames={FPS * 10}
        defaultProps={{ data: null as Data | null }}
        calculateMetadata={async () => {
          const data = await loadData()
          return { durationInFrames: filmFrames(timeline(data)), props: { data } }
        }}
      />
      {Object.entries(VIEWS).flatMap(([id, View]) => sceneCompositions(id, View))}
    </>
  )
}
