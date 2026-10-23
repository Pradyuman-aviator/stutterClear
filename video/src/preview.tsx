import { Composition } from "remotion"

import type { Data } from "./data"
import { boardOf, FPS, timeline } from "./data"
import type { SceneComponent } from "./stage"
import { loadData, ScenePreview, SceneSheet } from "./stage"

const made = new Map<string, { Preview: (p: { data: Data | null }) => React.ReactNode; Sheet: (p: { data: Data | null; frames: number[] }) => React.ReactNode }>()

function views(id: string, View: SceneComponent) {
  let v = made.get(id)
  if (!v) {
    v = {
      Preview: ({ data }) => <ScenePreview id={id} View={View} data={data} />,
      Sheet: ({ data, frames }) => <SceneSheet id={id} View={View} data={data} frames={frames} />,
    }
    made.set(id, v)
  }
  return v
}

// Scene-<id> plays one scene as it looks in the film; Sheet-<id> is a still of nine of its moments
// (or the frames passed as inputProps.frames).
export function sceneCompositions(id: string, View: SceneComponent) {
  const { Preview, Sheet } = views(id, View)
  const meta = async () => {
    const data = await loadData()
    const s = timeline(data, boardOf(id)).find((x) => x.id === id)
    return { data, frames: s ? s.frames : FPS * 5 }
  }
  return [
    <Composition
      key={`scene-${id}`}
      id={`Scene-${id}`}
      component={Preview}
      width={1920}
      height={1080}
      fps={FPS}
      durationInFrames={FPS * 5}
      defaultProps={{ data: null as Data | null }}
      calculateMetadata={async ({ props }) => {
        const m = await meta()
        return { durationInFrames: m.frames, props: { ...props, data: m.data } }
      }}
    />,
    <Composition
      key={`sheet-${id}`}
      id={`Sheet-${id}`}
      component={Sheet}
      width={1920}
      height={1080}
      fps={FPS}
      durationInFrames={1}
      defaultProps={{ data: null as Data | null, frames: [] as number[] }}
      calculateMetadata={async ({ props }) => {
        const m = await meta()
        // as long as the scene: Remotion clamps a frozen frame to the composition's length
        return { durationInFrames: m.frames, props: { ...props, data: m.data } }
      }}
    />,
  ]
}

// A root with only one scene, so a broken scene file never stops the others from previewing.
export const previewRoot = (id: string, View: SceneComponent) =>
  function PreviewRoot() {
    return <>{sceneCompositions(id, View)}</>
  }
