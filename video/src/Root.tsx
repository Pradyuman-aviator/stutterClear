import { AbsoluteFill, Composition, Freeze } from "remotion"

import type { Data } from "./data"
import { BOARD_DEMO, filmFrames, FPS, timeline } from "./data"
import { sceneCompositions } from "./preview"
import { VIEWS } from "./scenes/registry"
import { Film, loadData } from "./stage"
import { MONO } from "./theme"

function FilmView({ data }: { data: Data | null }) {
  return <Film data={data} views={VIEWS} />
}

// The film with a 30-second screen recording of the real app (storyboard-demo.json).
function FilmDemoView({ data }: { data: Data | null }) {
  return <Film data={data} views={VIEWS} board={BOARD_DEMO} />
}

// Sixteen moments of the whole film on one image (or the frames passed as inputProps.frames), for
// checking continuity from scene to scene.
function FilmSheetView({ data, frames }: { data: Data | null; frames: number[] }) {
  if (!data) return <FilmView data={data} />
  const total = filmFrames(timeline(data))
  const picks = frames.length ? frames : Array.from({ length: 16 }, (_, i) => Math.round(((i + 0.5) / 16) * (total - 1)))
  const cols = 4
  const w = 1920 / cols
  const h = (w * 9) / 16
  return (
    <AbsoluteFill style={{ background: "#000" }}>
      {picks.slice(0, 16).map((fr, i) => (
        <div key={i} style={{ position: "absolute", left: (i % cols) * w, top: Math.floor(i / cols) * h, width: w, height: h, overflow: "hidden" }}>
          <div style={{ width: 1920, height: 1080, transform: `scale(${w / 1920})`, transformOrigin: "0 0", position: "relative" }}>
            <Freeze frame={fr}>
              <FilmView data={data} />
            </Freeze>
          </div>
          <div style={{ position: "absolute", left: 6, top: 4, fontFamily: MONO, fontSize: 15, color: "#fff", background: "rgba(0,0,0,0.65)", padding: "1px 6px", borderRadius: 5 }}>
            {Math.floor(fr / FPS / 60)}:{String(Math.floor((fr / FPS) % 60)).padStart(2, "0")} · f{fr}
          </div>
          <div style={{ position: "absolute", inset: 0, border: "1px solid rgba(255,255,255,0.25)" }} />
        </div>
      ))}
    </AbsoluteFill>
  )
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
      <Composition
        id="StutterClearDemo"
        component={FilmDemoView}
        width={1920}
        height={1080}
        fps={FPS}
        durationInFrames={FPS * 10}
        defaultProps={{ data: null as Data | null }}
        calculateMetadata={async () => {
          const data = await loadData()
          return { durationInFrames: filmFrames(timeline(data, BOARD_DEMO)), props: { data } }
        }}
      />
      <Composition
        id="FilmSheet"
        component={FilmSheetView}
        width={1920}
        height={1080}
        fps={FPS}
        durationInFrames={FPS * 10}
        defaultProps={{ data: null as Data | null, frames: [] as number[] }}
        calculateMetadata={async ({ props }) => {
          const data = await loadData()
          // as long as the film: a frozen frame is clamped to the composition's length
          return { durationInFrames: filmFrames(timeline(data)), props: { ...props, data } }
        }}
      />
      {Object.entries(VIEWS).flatMap(([id, View]) => sceneCompositions(id, View))}
    </>
  )
}
