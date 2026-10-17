import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Pause, Play } from "lucide-react"
import type { PlaybackInfo } from "@/lib/api"
import { SPRING } from "@/lib/motion"
import { cn } from "@/lib/utils"

type Which = "clean" | "original"

const TRACKS: { which: Which; label: string; color: string; ring: string }[] = [
  { which: "clean", label: "Clean voice", color: "#5eead4", ring: "rgb(94 234 212 / 0.18)" },
  { which: "original", label: "Original", color: "#93c5fd", ring: "rgb(147 197 253 / 0.16)" },
]

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`

// Play the last session back: what the listener heard (clean) and what you said (original).
// The audio comes from the app server's memory; one track plays at a time.
export function Playback({ info, version }: { info: PlaybackInfo; version: number }) {
  const audio = useRef<HTMLAudioElement | null>(null)
  const [playing, setPlaying] = useState<Which | null>(null)
  const progress = useRef<Record<Which, SVGCircleElement | null>>({ clean: null, original: null })
  const [elapsed, setElapsed] = useState<Record<Which, number>>({ clean: 0, original: 0 })

  // A new recording: forget positions and stop whatever was playing.
  useEffect(() => {
    setElapsed({ clean: 0, original: 0 })
    setPlaying(null)
    for (const ring of Object.values(progress.current)) if (ring) ring.style.strokeDashoffset = "1"
    return () => {
      audio.current?.pause()
      audio.current = null
    }
  }, [version])

  // Progress ring at display rate while something plays.
  useEffect(() => {
    if (!playing) return
    let raf = 0
    let lastShown = -1
    const tick = () => {
      const a = audio.current
      const ring = progress.current[playing]
      if (a && ring && a.duration > 0) {
        ring.style.strokeDashoffset = String(1 - a.currentTime / a.duration)
        const s = Math.floor(a.currentTime)
        if (s !== lastShown) {
          lastShown = s
          setElapsed((e) => ({ ...e, [playing]: a.currentTime }))
        }
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing])

  const toggle = (which: Which) => {
    const cur = audio.current
    if (playing === which && cur) {
      cur.pause()
      setPlaying(null)
      return
    }
    cur?.pause()
    const a = new Audio(`/api/playback/${which}.wav?v=${version}`)
    const resumeAt = elapsed[which]
    a.addEventListener("loadedmetadata", () => {
      if (resumeAt > 0 && resumeAt < a.duration - 0.2) a.currentTime = resumeAt
    })
    a.addEventListener("ended", () => {
      setPlaying(null)
      setElapsed((e) => ({ ...e, [which]: 0 }))
      const ring = progress.current[which]
      if (ring) ring.style.strokeDashoffset = "1"
    })
    a.play().catch(() => setPlaying(null))
    audio.current = a
    setPlaying(which)
  }

  return (
    <div className="flex flex-wrap items-center justify-center gap-2.5 sm:justify-start">
      {TRACKS.map(({ which, label, color, ring }) => {
        const total = which === "clean" ? info.clean_s : info.original_s
        const on = playing === which
        return (
          <motion.button
            key={which}
            type="button"
            onClick={() => toggle(which)}
            aria-label={`${on ? "Pause" : "Play"} ${label.toLowerCase()}`}
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.96 }}
            transition={SPRING.press}
            className={cn(
              "group/play flex cursor-pointer items-center gap-3 rounded-full border py-1.5 pr-4 pl-1.5 text-left",
              "transition-[background-color,border-color] duration-500 hover:duration-150",
              "outline-none focus-visible:ring-2 focus-visible:ring-sky-300/60",
              on ? "border-white/20 bg-white/10" : "border-white/10 bg-white/[0.04] hover:bg-white/[0.08]"
            )}
          >
            <span className="relative grid size-10 place-items-center rounded-full" style={{ background: ring }}>
              <svg viewBox="0 0 40 40" className="absolute inset-0 -rotate-90" aria-hidden>
                <circle cx="20" cy="20" r="18" fill="none" stroke="rgb(255 255 255 / 0.08)" strokeWidth="2.5" />
                <circle
                  ref={(el) => {
                    // The ring is driven only from the animation loop, never by React renders,
                    // so the once-a-second time text cannot pull it backwards.
                    if (el && !el.style.strokeDashoffset) el.style.strokeDashoffset = "1"
                    progress.current[which] = el
                  }}
                  cx="20"
                  cy="20"
                  r="18"
                  fill="none"
                  stroke={color}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  pathLength={1}
                  strokeDasharray="1"
                />
              </svg>
              <AnimatePresence mode="popLayout" initial={false}>
                <motion.span
                  key={on ? "pause" : "play"}
                  initial={{ scale: 0.4, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.4, opacity: 0 }}
                  transition={SPRING.press}
                  style={{ color }}
                >
                  {on ? <Pause className="size-4 fill-current" /> : <Play className="size-4 translate-x-px fill-current" />}
                </motion.span>
              </AnimatePresence>
            </span>
            <span className="flex flex-col leading-tight">
              <span className="text-sm font-medium text-white/90">{label}</span>
              <span className="text-[11px] text-white/45 tabular-nums">
                {on || elapsed[which] > 0 ? `${clock(elapsed[which])} / ` : ""}
                {clock(total)}
              </span>
            </span>
          </motion.button>
        )
      })}
    </div>
  )
}
