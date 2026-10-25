import { type RefObject } from "react"
import { AnimatePresence, motion } from "motion/react"
import { AudioLines, CircleAlert, Download, LoaderCircle, Pause, Play, ShieldCheck, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import type { DemoState, Hear } from "@/hooks/use-demo"
import type { DemoEngine, DemoSample } from "@/lib/api"
import { SPRING } from "@/lib/motion"
import { cn } from "@/lib/utils"

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`

const HEAR = [
  { value: "original" as Hear, label: "Original" },
  { value: "clean" as Hear, label: "Clean voice" },
]
// Pause cuts only shorten a silence: they are counted apart from the stutters.
function summary(cuts: { kind: string }[]) {
  const stutters = cuts.filter((c) => c.kind !== "pause").length
  const pauses = cuts.length - stutters
  return `${stutters} stutter${stutters === 1 ? "" : "s"} removed` + (pauses ? `, ${pauses} long pause${pauses === 1 ? "" : "s"} shortened` : "")
}

const ENGINES: { value: DemoEngine; label: string; hint: string }[] = [
  { value: "natural", label: "Natural", hint: "Repeated and broken-off words go, cut only between words; um and uh stay." },
  { value: "thorough", label: "Thorough", hint: "Natural, plus um, uh and repeated phrases; long pauses shrink to under half a second." },
  { value: "live", label: "Live", hint: "Exactly what a call hears, decided in real time." },
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
            whileTap={{ scale: 0.98 }}
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
  extra,
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
  extra?: React.ReactNode // e.g. Nemotron insights for this recording
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
          whileTap={{ scale: 0.98 }}
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
              key={!tl ? (state.error ? "failed" : "load") : state.playing ? "pause" : "play"}
              initial={{ scale: 0.4, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.4, opacity: 0 }}
              transition={SPRING.press}
            >
              {!tl && state.error ? (
                <CircleAlert className="size-4 text-rose-300" />
              ) : !tl ? (
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
            <span className="truncate">{tl ? tl.label : state.error ? "Could not load" : "Cleaning…"}</span>
            <span className="shrink-0 whitespace-nowrap tabular-nums">
              · {clock(state.t)} / {clock(total)}
            </span>
          </div>
        </div>
        <motion.button
          type="button"
          onClick={onClose}
          aria-label="Close the sample"
          whileTap={{ scale: 0.98 }}
          transition={SPRING.press}
          className="cursor-pointer rounded-full p-1.5 text-white/45 outline-none hover:bg-white/10 hover:text-white focus-visible:ring-2 focus-visible:ring-sky-300/60"
        >
          <X className="size-4" />
        </motion.button>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2 sm:justify-start">
        <ToggleGroup aria-label="Hear" value={[state.hear]} onValueChange={(v) => v[0] && onHear(v[0] as Hear)} variant="outline" spacing={1}>
          {HEAR.map((o) => (
            <ToggleGroupItem key={o.value} value={o.value}>{o.label}</ToggleGroupItem>
          ))}
        </ToggleGroup>
        {sample && sample.engines.length > 1 && (
          <ToggleGroup aria-label="Engine" value={[state.engine]} onValueChange={(v) => v[0] && onEngine(v[0] as DemoEngine)} variant="outline" spacing={1}>
            {ENGINES.filter((e) => sample.engines.includes(e.value)).map((o) => (
              <ToggleGroupItem key={o.value} value={o.value}>{o.label}</ToggleGroupItem>
            ))}
          </ToggleGroup>
        )}
        {downloadUrl && tl && (
          <Button variant="outline" size="sm" render={<a href={downloadUrl} download="voice-note-clean.wav" />}>
            <Download data-icon="inline-start" /> Download clean
          </Button>
        )}
        {tl && (
          <>
            <Button variant="outline" size="sm" render={<a href={`/api/demo/${state.id}/cuts.csv?engine=${state.engine}`} />}>
              Timestamps
            </Button>
            <Button variant="outline" size="sm" render={<a href={`/api/demo/${state.id}/cuts.srt?engine=${state.engine}`} />}>
              Subtitles
            </Button>
          </>
        )}
      </div>
      {tl?.guard && (
        <div className="flex w-full items-start gap-2 rounded-xl border border-violet-300/15 bg-violet-300/[0.06] px-3 py-2 text-[11px] leading-relaxed text-violet-100/80">
          <ShieldCheck className="mt-px size-3.5 shrink-0 text-violet-300" />
          <span>
            <span className="font-medium text-violet-100">Meaning guard · Nemotron</span> checked {tl.guard.checked} word
            {tl.guard.checked === 1 ? "" : "s"} marked for cutting
            {tl.guard.error
              ? ` but did not answer, so the cuts stand as detected.`
              : tl.guard.kept.length
                ? `; kept ${tl.guard.kept.map((k) => `"${k.word}"${k.reason ? ` (${k.reason})` : ""}`).join(", ")} because it was meant.`
                : "; every one was a real stutter."}
          </span>
        </div>
      )}
      {tl && (
        <div className={cn("text-[11px] leading-relaxed text-white/40")}>
          <span className="text-white/60">{ENGINES.find((e) => e.value === state.engine)?.hint}</span>{" "}
          {summary(tl.cuts)}, {tl.removed_s.toFixed(1)} s shorter.{" "}
          {credit}
        </div>
      )}
      {tl && extra}
      {state.error && <div className="text-xs text-rose-300">{state.error}</div>}
    </div>
  )
}
