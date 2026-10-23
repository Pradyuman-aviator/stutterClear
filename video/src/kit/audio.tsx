import { Html5Audio, Sequence, staticFile } from "remotion"

export type SfxName = "whoosh" | "snap" | "chime" | "pop"

// A sound effect at frame `at` of the scene (whoosh: a move; snap: a cut leaves; chime: the meaning
// guard keeps a word, or a result lands; pop: something appears).
export function Sfx({ name, at, volume = 0.5 }: { name: SfxName; at: number; volume?: number }) {
  return (
    <Sequence from={Math.round(at)} durationInFrames={60} layout="none" name={`sfx:${name}`}>
      <Html5Audio src={staticFile(`generated/sound/${name}.wav`)} volume={volume} />
    </Sequence>
  )
}
