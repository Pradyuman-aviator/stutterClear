import type { ComponentType } from "react"
import { linearTiming, TransitionSeries } from "@remotion/transitions"
import { AbsoluteFill, Freeze, Html5Audio, interpolate, Sequence, staticFile, useCurrentFrame } from "remotion"

import type { Data, Scene, SceneProps } from "./data"
import { filmFrames, FPS, musicVolume, speechSpans, timeline, TRANSITION } from "./data"
import { Backdrop, clamp, Credit, ease, Note, NvTracker, Sfx, Subtitle } from "./kit"
import { C, MONO, SANS } from "./theme"
import { presentation } from "./transitions"

export type SceneComponent = ComponentType<SceneProps>

export const loadData = async (): Promise<Data> => fetch(staticFile("generated/data.json")).then((r) => r.json())

function Missing() {
  return (
    <AbsoluteFill style={{ background: C.bg, color: C.text, fontFamily: SANS, alignItems: "center", justifyContent: "center", fontSize: 40 }}>
      Run scripts/export_video_data.py first.
    </AbsoluteFill>
  )
}

// Narration subtitles, the clip credit while the real recording plays, and (in the first scene
// with narration) the note that the narrator's voice is synthetic. Frames are the scene's own.
export function SceneOverlay({ d, scene, first }: { d: Data; scene: Scene; first: boolean }) {
  const firstLine = scene.cues.find((c) => c.kind === "line")
  return (
    <>
      {scene.cues.map((c) =>
        c.kind === "line" ? (
          <Sequence key={c.id} from={c.at} durationInFrames={c.frames} name={`sub:${c.id}`}>
            <Subtitle text={c.text ?? ""} frames={c.frames} />
          </Sequence>
        ) : c.id === "before" || c.id === "after" ? (
          <Sequence key={c.id} from={c.at - 8} durationInFrames={c.frames + 16} name={`credit:${c.id}`}>
            <Credit text={d.clip.credit} frames={c.frames + 16} />
          </Sequence>
        ) : null,
      )}
      {first && firstLine && (
        <Sequence from={firstLine.at} durationInFrames={Math.max(firstLine.frames, FPS * 5)} name="voice-note">
          <Note text={d.voice_label} frames={Math.max(firstLine.frames, FPS * 5)} top={scene.nvidia ? 96 : 52} />
        </Sequence>
      )}
    </>
  )
}

const firstNarrated = (scenes: Scene[]) => scenes.find((s) => s.cues.some((c) => c.kind === "line"))?.id

// One scene on its own, as it looks in the film (for previews and stills).
export function ScenePreview({ id, View, data }: { id: string; View: SceneComponent; data: Data | null }) {
  if (!data) return <Missing />
  const scenes = timeline(data)
  const scene = scenes.find((s) => s.id === id)
  if (!scene) return <Missing />
  return (
    <AbsoluteFill style={{ background: C.bg, fontFamily: SANS, color: C.text }}>
      <Backdrop />
      <AbsoluteFill>
        <View d={data} scene={scene} />
      </AbsoluteFill>
      <SceneOverlay d={data} scene={scene} first={firstNarrated(scenes) === id} />
      {scene.nvidia && <NvTracker lit={scene.nvidia} fresh={freshNv(scenes, scene.index)} at={NV_AT} />}
    </AbsoluteFill>
  )
}

const NV_AT = 18 // frames into a scene when its new NVIDIA pills light up

// The pills a scene lights that the scene before it did not.
const freshNv = (scenes: Scene[], i: number) => (scenes[i].nvidia ?? []).filter((k) => !(scenes[i - 1]?.nvidia ?? []).includes(k))

// The tracker over the whole film: it follows the scene on screen, fading in and out where scenes
// start or stop showing it.
function FilmNvTracker({ scenes }: { scenes: Scene[] }) {
  const f = useCurrentFrame()
  let i = 0
  while (i + 1 < scenes.length && scenes[i + 1].from <= f) i++
  const s = scenes[i]
  const prev = scenes[i - 1]
  if (!s.nvidia) {
    if (!prev?.nvidia) return null
    const o = interpolate(f - s.from, [0, TRANSITION], [1, 0], clamp)
    return o > 0 ? <NvTracker lit={prev.nvidia} opacity={o} /> : null
  }
  const o = prev?.nvidia ? 1 : interpolate(f - s.from, [0, TRANSITION], [0, 1], clamp)
  return (
    <Sequence from={s.from} durationInFrames={s.frames} layout="none">
      <NvTracker lit={s.nvidia} fresh={freshNv(scenes, i)} at={NV_AT} opacity={o} />
    </Sequence>
  )
}

// Nine moments of a scene on one image, for checking a scene at a glance.
export function SceneSheet({ id, View, data, frames }: { id: string; View: SceneComponent; data: Data | null; frames: number[] }) {
  if (!data) return <Missing />
  const scene = timeline(data).find((s) => s.id === id)
  if (!scene) return <Missing />
  const picks = frames.length ? frames : [0.04, 0.15, 0.27, 0.38, 0.5, 0.62, 0.73, 0.85, 0.96].map((u) => Math.round(u * (scene.frames - 1)))
  const cols = 3
  const w = 1920 / cols
  const h = (w * 9) / 16
  return (
    <AbsoluteFill style={{ background: "#000" }}>
      {picks.slice(0, 9).map((fr, i) => (
        <div key={i} style={{ position: "absolute", left: (i % cols) * w, top: Math.floor(i / cols) * h, width: w, height: h, overflow: "hidden" }}>
          <div style={{ width: 1920, height: 1080, transform: `scale(${w / 1920})`, transformOrigin: "0 0", position: "relative" }}>
            <Freeze frame={fr}>
              <ScenePreview id={id} View={View} data={data} />
            </Freeze>
          </div>
          <div style={{ position: "absolute", left: 8, top: 6, fontFamily: MONO, fontSize: 18, color: "#fff", background: "rgba(0,0,0,0.65)", padding: "2px 8px", borderRadius: 6 }}>
            f{fr} · {(fr / FPS).toFixed(1)}s
          </div>
          <div style={{ position: "absolute", inset: 0, border: "1px solid rgba(255,255,255,0.25)" }} />
        </div>
      ))}
    </AbsoluteFill>
  )
}

// The whole film: one backdrop, the scenes with their transitions, narration and the real clip,
// subtitles, music that ducks under speech, and a whoosh on every transition.
export function Film({ data, views }: { data: Data | null; views: Record<string, SceneComponent> }) {
  if (!data) return <Missing />
  const scenes = timeline(data)
  const total = filmFrames(scenes)
  const spans = speechSpans(scenes)
  const first = firstNarrated(scenes)
  const series = scenes.flatMap((s, i) => {
    const View = views[s.id]
    const seq = (
      <TransitionSeries.Sequence key={s.id} durationInFrames={s.frames} name={s.id}>
        <AbsoluteFill style={{ fontFamily: SANS, color: C.text }}>{View ? <View d={data} scene={s} /> : null}</AbsoluteFill>
      </TransitionSeries.Sequence>
    )
    if (i === 0 || s.transition === "none") return [seq]
    return [
      <TransitionSeries.Transition
        key={`t-${s.id}`}
        timing={linearTiming({ durationInFrames: TRANSITION, easing: ease })}
        presentation={presentation(s.transition)}
      />,
      seq,
    ]
  })
  return (
    <AbsoluteFill style={{ background: C.bg }}>
      <Backdrop />
      <TransitionSeries>{series}</TransitionSeries>
      {scenes.map((s) => (
        <Sequence key={`o-${s.id}`} from={s.from} durationInFrames={s.frames} name={`overlay:${s.id}`}>
          <SceneOverlay d={data} scene={s} first={s.id === first} />
        </Sequence>
      ))}
      {scenes.flatMap((s) =>
        s.cues.map((c) => (
          <Sequence key={`a-${s.id}-${c.id}`} from={s.from + c.at} durationInFrames={c.frames} name={`${c.kind}:${c.id}`}>
            <Html5Audio src={staticFile(c.file)} />
          </Sequence>
        )),
      )}
      <FilmNvTracker scenes={scenes} />
      <Html5Audio src={staticFile("generated/sound/music.wav")} volume={(f) => musicVolume(f, spans, total)} />
      {scenes.slice(1).map((s) => (s.transition === "none" ? null : <Sfx key={`w-${s.id}`} name="whoosh" at={s.from} volume={0.28} />))}
    </AbsoluteFill>
  )
}
