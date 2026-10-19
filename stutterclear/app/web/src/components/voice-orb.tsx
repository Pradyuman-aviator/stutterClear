import { lazy, Suspense, useEffect, useRef, type RefObject } from "react"
import type { Signals } from "@/hooks/use-signals"
import { cn } from "@/lib/utils"

// three.js is most of the bundle: load it after the page is up, the orb fades in when ready.
const Orb = lazy(() => import("@/components/ui/orb").then((m) => ({ default: m.Orb })))

const IDLE: [string, string] = ["#CADCFC", "#A0B9D1"]
const LIVE: [string, string] = ["#A5C8FF", "#7FE3D6"]
const CUT: [string, string] = ["#FF9DB0", "#FFC4A8"]
const BLOCK: [string, string] = ["#D9C2FF", "#A98BFF"]

// The ElevenLabs orb, driven by the voice: it swells with the mic, flashes rose when a stutter
// is removed and turns violet when a pause is kept. It leans toward the cursor and is the
// start/stop button.
export function VoiceOrb({
  signals,
  running,
  onToggle,
  className,
}: {
  signals: RefObject<Signals>
  running: boolean
  onToggle: () => void
  className?: string
}) {
  const home = useRef<HTMLDivElement>(null)
  const glow = useRef<HTMLDivElement>(null)
  const inRef = useRef(0)
  const outRef = useRef(0.3)
  const colorsRef = useRef<[string, string]>(IDLE)
  const runningRef = useRef(running)

  useEffect(() => {
    runningRef.current = running
  }, [running])

  useEffect(() => {
    const apply = () => {
      const s = signals.current
      if (runningRef.current) {
        inRef.current = 0.12 + 0.88 * s.level
        outRef.current = 0.3 + 0.7 * s.outLevel
      } else {
        inRef.current = 0.12
        outRef.current = 0.32
      }
      colorsRef.current = s.cut > 0.35 ? CUT : s.block > 0.35 ? BLOCK : runningRef.current ? LIVE : IDLE
      if (glow.current) {
        glow.current.style.opacity = runningRef.current ? String(0.35 + 0.5 * s.level) : "0.35"
        glow.current.style.transform = "none"
        glow.current.style.background = "radial-gradient(circle, rgb(125 170 255 / 0.45), transparent 65%)"
      }
    }
    apply()
    if (!running) return
    let raf = 0
    const loop = () => {
      apply()
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [signals, running])

  return (
    <div ref={home} className={cn("relative aspect-square", className)}>
      <div
        ref={glow}
        aria-hidden
        className="pointer-events-none absolute -inset-[18%] rounded-full blur-2xl transition-[background] duration-500"
      />
      <button
        type="button"
        aria-label={running ? "Stop StutterClear" : "Start StutterClear"}
        onClick={onToggle}
        className="relative block size-full cursor-pointer rounded-full outline-none focus-visible:ring-2 focus-visible:ring-sky-300/70"
      >
        <span className="absolute inset-0 rounded-full bg-[radial-gradient(circle_at_30%_25%,rgb(255_255_255/0.08),transparent_60%)] shadow-[inset_0_0_0_1px_rgb(255_255_255/0.08),0_30px_80px_-20px_rgb(0_0_0/0.8)]" />
        <Suspense
          fallback={<span className="absolute inset-[6%] rounded-full bg-[radial-gradient(circle_at_40%_35%,rgb(202_220_252/0.35),rgb(160_185_209/0.08)_70%)]" />}
        >
          <Orb
            className="pointer-events-none size-full"
            luminous
            colorsRef={colorsRef}
            volumeMode="manual"
            inputVolumeRef={inRef}
            outputVolumeRef={outRef}
          />
        </Suspense>
      </button>
    </div>
  )
}
