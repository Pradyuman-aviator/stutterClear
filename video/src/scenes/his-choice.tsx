import { AbsoluteFill } from "remotion"

import type { SceneProps } from "../data"
import { Eyebrow, Title } from "../kit"

// His take, his choice
export default function Scene({ scene }: SceneProps) {
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", gap: 24 }}>
      <Eyebrow>{scene.id}</Eyebrow>
      <Title>{scene.title}</Title>
    </AbsoluteFill>
  )
}
