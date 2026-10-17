import { type RefObject } from "react"
import { AnimatePresence, motion } from "motion/react"
import { AudioLines, Download, LoaderCircle, Pause, Play, X } from "lucide-react"
import { Segmented } from "@/components/segmented"
import type { DemoState, Hear } from "@/hooks/use-demo"
import type { DemoEngine, DemoSample } from "@/lib/api"
import { SPRING } from "@/lib/motion"
import { cn } from "@/lib/utils"

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`

const HEAR = [
  { value: "original" as Hear, label: "Original" },
  { value: "clean" as Hear, label: "Clean voice" },
]
const ENGINES: { value: DemoEngine; label: string }[] = [
  { value: "natural", label: "Natural" },
  { value: "thorough", label: "Thorough" },
  { value: "live", label: "Live" },
]

// Before a sample is chosen: real recordings anyone can try, no mic needed.
export function DemoPicker({ samples, onPick }: { samples: DemoSample[]; onPick: (id: string) => void }) {
  if (!samples.length) return null
  return (
    <div className="flex w-full flex-col items-center gap-2 sm:items-start">
      <div className="text-xs font-medium text-white/50">No mic handy? Hear it on a real recording</div>
      <div className="flex flex-wrap justify-center gap-2 sm:justify-start">
        {samples.map((s) => (
          <motion.button
            key={s.id}
            type="button"
            onClick={() => onPick(s.id)}
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.96 }}
            transition={SPRING.press}
            className="flex cursor-pointer items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] py-1.5 pr-3.5 pl-1.5 text-left transition-[background-color,border-color] duration-500 outline-none hover:border-white/20 hover:bg-white/[0.08] hover:duration-150 focus-visible:ring-2 focus-visible:ring-sky-300/60"
          >
            <span className="grid size-7 place-items-center rounded-full bg-teal-300/15 text-teal-200">
              <Play className="size-3.5 translate-x-px fill-current" />
            </span>
            <span className="text-sm text-white/85">
              {s.title} <span className="text-white/45">· {s.who}</span>
            </span>
            <span className="text-[11px] text-white/35 tabular-nums">{clock(s.duration_s)}</span>
          </motion.button>
        ))}
      </div>
    </div>
  )
}

// While a sample is loaded: play/pause with a progress ring, A/B between the original and the
// cleaned voice, and which engine cleaned it.
export function DemoBar({
  state,
  sample,
  credit,
  progress,
  onPlay,
  onPause,
  onHear,
  onEngine,
  onClose,
  downloadUrl,
}: {
  state: DemoState
  sample: DemoSample | undefined
  credit: string
  progress: RefObject<SVGCircleElement | null>
  onPlay: () => void
  onPause: () => void
  onHear: (h: Hear) => void
  onEngine: (e: DemoEngine) => void
  onClose: () => void
  downloadUrl?: string
}) {
  const tl = state.timeline
  const total = !tl ? sample?.duration_s ?? 0 : state.hear === "clean" ? tl.clean_s : tl.original_s
  return (
    <div className="flex w-full flex-col items-center gap-3 rounded-2xl border border-white/8 bg-white/[0.03] p-3 sm:items-start">
      <div className="flex w-full items-center gap-3">
        <motion.button
          type="button"
          onClick={state.playing ? onPause : onPlay}
          disabled={!tl}
          aria-label={state.playing ? "Pause the sample" : "Play the sample"}
          whileHover={{ scale: 1.06 }}
          whileTap={{ scale: 0.92 }}
          transition={SPRING.press}
          className="relative grid size-11 shrink-0 cursor-pointer place-items-center rounded-full bg-teal-300/15 text-teal-200 outline-none focus-visible:ring-2 focus-visible:ring-sky-300/60 disabled:cursor-wait"
        >
          <svg viewBox="0 0 44 44" className="absolute inset-0 -rotate-90" aria-hidden>
            <circle cx="22" cy="22" r="20" fill="none" stroke="rgb(255 255 255 / 0.08)" strokeWidth="2.5" />
            <circle
              ref={(el) => {
                if (el && !el.style.strokeDashoffset) el.style.strokeDashoffset = "1"
                progress.current = el
              }}
              cx="22"
              cy="22"
              r="20"
              fill="none"
              stroke="#5eead4"
              strokeWidth="2.5"
              strokeLinecap="round"
              pathLength={1}
              strokeDasharray="1"
            />
          </svg>
          <AnimatePresence mode="popLayout" initial={false}>
            <motion.span
              key={!tl ? "load" : state.playing ? "pause" : "play"}
              initial={{ scale: 0.4, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.4, opacity: 0 }}
              transition={SPRING.press}
            >
              {!tl ? (
                <LoaderCircle className="size-4 animate-spin" />
              ) : state.playing ? (
                <Pause className="size-4 fill-current" />
              ) : (
                <Play className="size-4 translate-x-px fill-current" />
              )}
            </motion.span>
          </AnimatePresence>
        </motion.button>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium text-white/90">
            {sample ? `${sample.title} · ${sample.who}` : "Sample"}
          </div>
          <div className="flex items-center gap-1.5 text-[11px] text-white/45">
            <AudioLines className="size-3" />
            <span className="truncate">{tl ? tl.label : "Cleaning…"}</span>
            <span className="tabular-nums">
              · {clock(state.t)} / {clock(total)}
            </span>
          </div>
        </div>
        <motion.button
          type="button"
          onClick={onClose}
          aria-label="Close the sample"
          whileHover={{ scale: 1.12, rotate: 90 }}
          whileTap={{ scale: 0.88 }}
          transition={SPRING.press}
          className="cursor-pointer rounded-full p-1.5 text-white/45 outline-none hover:bg-white/10 hover:text-white focus-visible:ring-2 focus-visible:ring-sky-300/60"
        >
          <X className="size-4" />
        </motion.button>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2 sm:justify-start">
        <Segmented
          label="Hear"
          value={state.hear}
          options={HEAR}
          onChange={onHear}
          className="text-xs"
          pillClassName="bg-teal-300/20 shadow-[inset_0_1px_0_rgb(255_255_255/0.15),0_6px_20px_-8px_rgb(94_234_212/0.6)]"
        />
        {sample && sample.engines.length > 1 && (
          <Segmented label="Engine" value={state.engine} options={ENGINES.filter((e) => sample.engines.includes(e.value))} onChange={onEngine} />
        )}
        {downloadUrl && tl && (
          <motion.a
            href={downloadUrl}
            download="voice-note-clean.wav"
            whileHover={{ y: -2 }}
            whileTap={{ scale: 0.95 }}
            transition={SPRING.press}
            className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs text-white/75 transition-colors hover:bg-white/[0.08] hover:text-white"
          >
            <Download className="size-3.5" /> Download clean
          </motion.a>
        )}
      </div>
      {tl && (
        <div className={cn("text-[11px] leading-relaxed text-white/40")}>
          {tl.cuts.length} stutter{tl.cuts.length === 1 ? "" : "s"} removed, {tl.removed_s.toFixed(1)} s shorter.{" "}
          {credit}
        </div>
      )}
      {state.error && <div className="text-xs text-rose-300">{state.error}</div>}
    </div>
  )
}
