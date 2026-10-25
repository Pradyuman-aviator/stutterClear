import { motion } from "motion/react"
import type { GpuStatus } from "@/lib/api"
import { typical, type Gpu } from "@/hooks/use-gpu"
import { behind, clock, useNow } from "@/hooks/use-now"
import { SPRING } from "@/lib/motion"
import { cn } from "@/lib/utils"

type Feature = "voice-note" | "dictate" | "video" | "live"

const WAKES_IT: Record<Feature, string> = {
  "voice-note": "Recording or choosing a file wakes it.",
  dictate: "Speaking wakes it.",
  video: "Choosing a video wakes it.",
  live: "",
}

// What each step of a wake is doing, in the visitor's words; the Kaggle steps are the slow ones.
const STEPS: Record<string, string> = {
  laptop: "calling the team's computer",
  asking: "asking Kaggle for one",
  queued: "waiting in Kaggle's queue",
  starting: "starting the machine",
  loading: "loading the models",
}
const KAGGLE_STEPS = ["asking", "queued", "starting"]

const DOT: Record<GpuStatus["state"], string> = {
  off: "bg-white/45",
  waking: "bg-amber-300",
  connecting: "bg-amber-300",
  ready: "bg-emerald-300",
  busy: "bg-sky-300",
  resting: "bg-violet-300",
  unavailable: "bg-rose-400",
}

const waking = (s: GpuStatus | null) => !!s && (s.state === "waking" || s.state === "connecting")

// Seconds since the wake began: the server's count, carried on by the page's own clock.
function useElapsed(gpu: Gpu) {
  const now = useNow(waking(gpu.status), gpu.at)
  const e = gpu.status?.elapsed_s
  return e == null ? null : e + (now - gpu.at) / 1000
}

function Dot({ state, className }: { state: GpuStatus["state"]; className?: string }) {
  const ping = state === "waking" || state === "connecting" || state === "busy"
  return (
    <span className={cn("relative flex size-2 shrink-0", className)}>
      {ping && <span className={cn("absolute inline-flex size-full animate-ping rounded-full opacity-60", DOT[state])} />}
      <span className={cn("relative inline-flex size-2 rounded-full", DOT[state])} />
    </span>
  )
}

// In the header, next to the engine: the GPU worker at a glance. Hidden without one.
export function GpuPill({ gpu }: { gpu: Gpu }) {
  const elapsed = useElapsed(gpu)
  const s = gpu.status
  if (!s) return null
  const t = elapsed === null || s.step === "reconnecting" ? "" : ` · ${clock(elapsed)}`
  const label = {
    off: "GPU asleep",
    waking: "GPU waking",
    connecting: "GPU waking",
    ready: "GPU ready",
    busy: "GPU busy",
    resting: "GPU resting",
    unavailable: "GPU offline",
  }[s.state]
  return (
    <motion.div
      layout
      transition={SPRING.glide}
      className="glass relative flex shrink-0 items-center gap-2 overflow-hidden rounded-full px-3.5 py-1.5 text-xs whitespace-nowrap text-white/75"
    >
      <Dot state={s.state} />
      <span>
        {label}
        {waking(s) && <span className="tabular-nums" aria-hidden>{t}</span>}
      </span>
    </motion.div>
  )
}

function Action({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="ml-1 inline-flex cursor-pointer items-center rounded-full border border-sky-300/25 bg-sky-300/10 px-2.5 py-0.5 align-baseline text-[11px] font-medium text-sky-100 transition-colors hover:bg-sky-300/20"
    >
      {children}
    </button>
  )
}

function Link({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="cursor-pointer text-sky-300/90 underline-offset-4 transition-colors hover:text-sky-200 hover:underline"
    >
      {children}
    </button>
  )
}

// One line inside a card: what the GPU is doing for this feature, honestly (the server's own steps
// and clock, never a made-up percentage), and what the visitor can do meanwhile.
export function GpuNote({
  gpu,
  feature,
  job = false,
  mode = "gentle",
  onWake,
  onGoto,
  className,
}: {
  gpu: Gpu
  feature: Feature
  job?: boolean // this card has a recording waiting for the GPU
  mode?: "gentle" | "strong" // live: the delay to show once Best (GPU) is ready
  onWake?: () => void // wake it on purpose (a button); omitted: no button
  onGoto?: (tab: "calls" | "recordings") => void // links to what works without the GPU, while it wakes
  className?: string
}) {
  const elapsed = useElapsed(gpu)
  const s = gpu.status
  if (!s) return null
  const where = s.where ? ` on ${s.where}` : ""
  const usually = typical(s.typical_s)
  let text: React.ReactNode
  let late = ""

  if (s.state === "off") {
    if (gpu.error) {
      text = <>{gpu.error.message} {gpu.error.retry && onWake && <Action onClick={onWake}>Try again</Action>}</>
    } else if (feature === "live") {
      text = s.standby ? (
        <>Best (GPU) runs on our GPU, which is standing by and wakes in seconds. {onWake && <Action onClick={onWake}>Wake the GPU</Action>}</>
      ) : (
        <>
          Best (GPU) runs on our free GPU, which sleeps when nobody uses it.{" "}
          {onWake && <Action onClick={onWake}>Wake the GPU</Action>} (about {usually})
        </>
      )
    } else {
      text = (
        <>
          {s.standby
            ? "Our GPU is standing by and wakes in seconds."
            : `Our free GPU sleeps when nobody uses it; waking it takes about ${typical(s.typical_s, "minutes")} (seconds when the team's computer is on).`}{" "}
          {WAKES_IT[feature]} {onWake && <Action onClick={onWake}>Wake it now</Action>}
        </>
      )
    }
  } else if (waking(s)) {
    const step = s.step ?? "asking"
    const t = elapsed === null ? "" : clock(elapsed)
    text =
      step === "reconnecting" ? (
        "Reconnecting to the GPU…"
      ) : (
        <>
          Waking the GPU: {STEPS[step] ?? "getting it ready"}
          <span aria-hidden className="tabular-nums">{t && ` · ${t}`}</span>
          {KAGGLE_STEPS.includes(step) && ` · usually ${usually}`}
        </>
      )
    if (step !== "reconnecting" && elapsed !== null && elapsed > s.typical_s[1] + 60) {
      late = KAGGLE_STEPS.includes(step) ? "Taking longer than usual: Kaggle may be short of free GPUs." : "Taking longer than usual."
      if (job) late += " Your recording keeps its place."
    }
  } else if (s.state === "ready" || s.state === "busy") {
    if (feature === "live") {
      const ms = s.live.delay_ms?.[mode]
      text = s.live.available
        ? `Best (GPU) is ready.${ms ? ` About ${behind(ms)} behind you.` : ""}`
        : s.live.reason ?? "Best (GPU) is not available right now."
    } else if (s.state === "busy") {
      text = `GPU busy${where}.`
    } else {
      const m = s.sleeps_in_s === null ? null : Math.max(1, Math.ceil(s.sleeps_in_s / 60))
      text = `GPU awake${where}.${m === null ? "" : ` It sleeps again in ${m} min unless someone uses it.`}`
    }
  } else if (s.state === "resting") {
    const until = s.until ? new Date(s.until) : null
    const when =
      until && !Number.isNaN(until.getTime())
        ? `until ${until.toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "numeric", minute: "2-digit" })}`
        : "until they come back"
    const spent = s.step === "day" ? "Today's share of the free GPU hours is used up" : "This week's free GPU hours are used up"
    text = `${spent}, so the GPU rests ${when}. Live cleaning, captions, the sample clips and Nemotron still work.`
  } else {
    const m = s.retry_s && s.retry_s > 0 ? Math.max(1, Math.ceil(s.retry_s / 60)) : null
    text = (
      <>
        {s.reason ?? "The GPU is offline right now."}{" "}
        {m !== null ? `Try again in ${m} min. ` : s.can_wake && onWake ? <><Action onClick={onWake}>Try again</Action>{" "}</> : null}
        Live cleaning and the sample clips still work.
      </>
    )
  }

  return (
    <div
      role="status"
      className={cn(
        "flex items-start gap-2.5 rounded-xl border border-white/6 bg-white/[0.03] px-3 py-2 text-xs leading-relaxed text-white/60",
        className
      )}
    >
      <Dot state={s.state} className="mt-[5px]" />
      <div className="min-w-0">
        <span>{text}</span>
        {late && <span className="text-amber-200/80"> {late}</span>}
        {waking(s) && onGoto && feature !== "live" && (
          <div className="mt-0.5 text-white/45">
            Meanwhile: <Link onClick={() => onGoto("calls")}>Try live cleaning</Link> ·{" "}
            <Link onClick={() => onGoto("recordings")}>Hear a sample</Link>
          </div>
        )}
      </div>
    </div>
  )
}
