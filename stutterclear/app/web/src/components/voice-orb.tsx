import { useEffect, useRef, type RefObject } from "react"
import { motion, useReducedMotion, useSpring } from "motion/react"
import { Orb } from "@/components/ui/orb"
import type { Signals } from "@/hooks/use-signals"
import { cn } from "@/lib/utils"

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
  const reduced = useReducedMotion()
  const home = useRef<HTMLDivElement>(null)
  const glow = useRef<HTMLDivElement>(null)
  const inRef = useRef(0)
  const outRef = useRef(0.3)
  const colorsRef = useRef<[string, string]>(IDLE)
  const hover = useRef(0)
  const hoverTarget = useRef(0)
  const runningRef = useRef(running)
  const spring = { stiffness: 120, damping: 14, mass: 0.8 }
  const x = useSpring(0, spring)
  const y = useSpring(0, spring)
  const rot = useSpring(0, spring)

  useEffect(() => {
    runningRef.current = running
  }, [running])

  useEffect(() => {
    let raf = 0
    let last = performance.now()
    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const s = signals.current
      hover.current += (hoverTarget.current - hover.current) * (1 - Math.exp(-dt * 6))
      const breath = 0.5 + 0.5 * Math.sin(now / 900)
      const hv = hover.current
      if (runningRef.current) {
        inRef.current = 0.12 + 0.88 * s.level + 0.1 * hv
        outRef.current = 0.3 + 0.7 * s.outLevel + 0.1 * hv
      } else {
        inRef.current = 0.06 + 0.08 * breath + 0.3 * hv
        outRef.current = 0.28 + 0.06 * breath + 0.35 * hv
      }
      colorsRef.current = s.cut > 0.35 ? CUT : s.block > 0.35 ? BLOCK : runningRef.current ? LIVE : IDLE
      if (glow.current) {
        const g = 0.35 + 0.5 * Math.max(s.level, hv * 0.6)
        glow.current.style.opacity = String(g)
        glow.current.style.transform = `scale(${1 + 0.18 * s.level + 0.06 * hv})`
        glow.current.style.background =
          s.cut > 0.35
            ? "radial-gradient(circle, rgb(251 113 133 / 0.55), transparent 65%)"
            : s.block > 0.35
              ? "radial-gradient(circle, rgb(192 132 252 / 0.5), transparent 65%)"
              : "radial-gradient(circle, rgb(125 170 255 / 0.45), transparent 65%)"
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [signals])

  useEffect(() => {
    if (reduced) return
    const onMove = (e: PointerEvent) => {
      const r = home.current?.getBoundingClientRect()
      if (!r) return
      const dx = (e.clientX - (r.left + r.width / 2)) / innerWidth
      const dy = (e.clientY - (r.top + r.height / 2)) / innerHeight
      x.set(dx * 36)
      y.set(dy * 36)
      rot.set(dx * 8)
    }
    addEventListener("pointermove", onMove)
    return () => removeEventListener("pointermove", onMove)
  }, [reduced, x, y, rot])

  return (
    <div ref={home} className={cn("relative aspect-square", className)}>
      <div
        ref={glow}
        aria-hidden
        className="pointer-events-none absolute -inset-[18%] rounded-full blur-2xl transition-[background] duration-500"
      />
      <motion.button
        type="button"
        aria-label={running ? "Stop StutterClear" : "Start StutterClear"}
        onClick={onToggle}
        onPointerEnter={() => (hoverTarget.current = 1)}
        onPointerLeave={() => (hoverTarget.current = 0)}
        style={{ x, y, rotate: rot }}
        whileHover={{ scale: 1.04 }}
        whileTap={{ scale: 0.95 }}
        transition={{ type: "spring", stiffness: 300, damping: 14 }}
        className="relative block size-full cursor-pointer rounded-full outline-none focus-visible:ring-2 focus-visible:ring-sky-300/70"
      >
        <span className="absolute inset-0 rounded-full bg-[radial-gradient(circle_at_30%_25%,rgb(255_255_255/0.08),transparent_60%)] shadow-[inset_0_0_0_1px_rgb(255_255_255/0.08),0_30px_80px_-20px_rgb(0_0_0/0.8)]" />
        <Orb
          className="pointer-events-none size-full"
          luminous
          colorsRef={colorsRef}
          volumeMode="manual"
          inputVolumeRef={inRef}
          outputVolumeRef={outRef}
        />
      </motion.button>
    </div>
  )
}
