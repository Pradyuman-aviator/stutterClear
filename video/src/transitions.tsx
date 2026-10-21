import type { TransitionPresentation, TransitionPresentationComponentProps } from "@remotion/transitions"
import { clockWipe } from "@remotion/transitions/clock-wipe"
import { fade } from "@remotion/transitions/fade"
import { flip } from "@remotion/transitions/flip"
import { iris } from "@remotion/transitions/iris"
import { pushCut } from "@remotion/transitions/push-cut"
import { slide } from "@remotion/transitions/slide"
import { wipe } from "@remotion/transitions/wipe"
import { AbsoluteFill } from "remotion"

import type { TransitionKind } from "./data"

type None = Record<string, never>

// The outgoing scene rushes past the camera while the next one comes into focus.
function ZoomThrough({ children, presentationDirection, presentationProgress }: TransitionPresentationComponentProps<None>) {
  const p = presentationProgress
  const entering = presentationDirection === "entering"
  const scale = entering ? 0.88 + 0.12 * p : 1 + 0.28 * p
  const blur = entering ? (1 - p) * 16 : p * 16
  return <AbsoluteFill style={{ transform: `scale(${scale})`, filter: `blur(${blur.toFixed(2)}px)`, opacity: entering ? p : 1 - p }}>{children}</AbsoluteFill>
}

const zoomThrough = (): TransitionPresentation<None> => ({ component: ZoomThrough, props: {} })

// CSS-only presentations (the shader ones need an experimental browser feature).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function presentation(kind: TransitionKind): TransitionPresentation<any> {
  switch (kind) {
    case "slide":
      return slide({ direction: "from-right" })
    case "slide-up":
      return slide({ direction: "from-bottom" })
    case "wipe":
      return wipe({ direction: "from-left" })
    case "flip":
      return flip({ direction: "from-right", perspective: 2400 })
    case "clock":
      return clockWipe({ width: 1920, height: 1080 })
    case "iris":
      return iris({ width: 1920, height: 1080 })
    case "push":
      return pushCut({ flashColor: "#ffffff", flashOpacity: 0.35 })
    case "zoom":
      return zoomThrough()
    default:
      return fade()
  }
}
