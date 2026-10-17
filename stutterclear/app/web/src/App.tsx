import { useCallback, useEffect, useRef, useState, type ReactNode, type RefObject } from "react"
import { AnimatePresence, MotionConfig, motion, type Variants } from "motion/react"
import { AudioLines, Captions, Check, Cpu, Headphones, Lock, Mic, Square, WandSparkles } from "lucide-react"

import { DelayMatrix } from "@/components/delay-matrix"
import { GlassCard } from "@/components/glass-card"
import { LiquidBackground } from "@/components/liquid-background"
import { LiquidSwitch } from "@/components/liquid-switch"
import { MagneticButton } from "@/components/magnetic-button"
import { COLORS, MirrorWaveform, type Layer } from "@/components/mirror-waveform"
import { RollingNumber } from "@/components/rolling-number"
import { Segmented, type SegmentOption } from "@/components/segmented"
import { SessionToast, type ToastData } from "@/components/session-toast"
import { ShimmeringText } from "@/components/ui/shimmering-text"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { VoiceOrb } from "@/components/voice-orb"
import { useLive, type LiveData, type Stats } from "@/hooks/use-live"
import { useSignals, type Signals } from "@/hooks/use-signals"
import {
  getDevices, getInfo, KIND_LABEL, startSession, stopSession,
  type DetectorKind, type Devices, type Mode, type StartBody,
} from "@/lib/api"
import { EASE_OUT, SPRING } from "@/lib/motion"
import { useStore, type Store } from "@/lib/store"
import { cn } from "@/lib/utils"

const MODES: SegmentOption<Mode>[] = [
  { value: "off", label: "Off" },
  { value: "gentle", label: "Gentle" },
  { value: "strong", label: "Strong" },
]
const MODE_HINT: Record<Mode, string> = {
  off: "Mirror only: nothing is cut, so you can watch your speech as it is.",
  gentle: "Removes clear repeats and fillers. About 250 ms behind you.",
  strong: "Removes the most stutters; may clip a word now and then. About 600 ms behind you.",
}
const DETECTORS: SegmentOption<DetectorKind>[] = [
  { value: "student", label: "AI model" },
  { value: "heuristic", label: "Rule-based" },
]
const NO_OUTPUT = "none"

// Entrances animate opacity and position only: blurring big panels while they move costs frames.
const page: Variants = { hidden: {}, show: { transition: { staggerChildren: 0.07, delayChildren: 0.05 } } }
const rise: Variants = {
  hidden: { opacity: 0, y: 22, scale: 0.985 },
  show: { opacity: 1, y: 0, scale: 1, transition: SPRING.enter },
}

function stored(key: string, fallback: string) {
  try {
    return localStorage.getItem(key) ?? fallback
  } catch {
    return fallback
  }
}
function store(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
  } catch {
    // private window: settings just are not remembered
  }
}

export default function App() {
  const { data, stats, connected, serverRunning, reset } = useLive()
  const [running, setRunning] = useState(false)
  const busy = useRef(false)
  const [devices, setDevices] = useState<Devices | null>(null)
  // Devices are remembered by name: Windows renumbers them whenever one connects or drops.
  const [inDev, setInDev] = useState("")
  const [outDev, setOutDev] = useState(() => stored("sc.out.name", NO_OUTPUT))
  const [mode, setMode] = useState<Mode>(() => stored("sc.mode", "gentle") as Mode)
  const [detector, setDetector] = useState<DetectorKind>(() => stored("sc.det", "student") as DetectorKind)
  const [save, setSave] = useState(false) // never remembered: recording is opt-in every time
  const [detName, setDetName] = useState("")
  const [hasStudent, setHasStudent] = useState(true)
  const [toast, setToast] = useState<ToastData | null>(null)
  const highlight = useRef<Layer | null>(null)
  const [hl, setHl] = useState<Layer | null>(null)
  const signals = useSignals(data, running)

  const loadDevices = useCallback(() => {
    getDevices()
      .then((d) => {
        setDevices(d)
        const fallback = d.inputs.find((x) => x.index === d.default_input)?.name ?? d.inputs[0]?.name ?? ""
        setInDev((cur) => {
          const want = cur || stored("sc.in.name", "")
          return d.inputs.some((x) => x.name === want) ? want : fallback
        })
        setOutDev((o) => (o === NO_OUTPUT || d.outputs.some((x) => x.name === o) ? o : NO_OUTPUT))
      })
      .catch(() => setDevices(null))
  }, [])

  useEffect(() => {
    if (!connected) return
    getInfo().then((i) => setHasStudent(i.student)).catch(() => {})
    loadDevices()
  }, [connected, loadDevices])

  // A session can outlive the page (reload while talking): follow what the server says.
  useEffect(() => {
    if (!busy.current) setRunning(serverRunning)
  }, [serverRunning])

  // Start and stop are optimistic: the UI flips on the click, the server catches up behind it.
  const start = useCallback(
    async (over: Partial<StartBody> = {}) => {
      busy.current = true
      setRunning(true)
      reset()
      try {
        const r = await startSession({
          mode, detector, save,
          in_device: null,
          out_device: null,
          in_name: inDev || null,
          out_name: outDev === NO_OUTPUT ? null : outDev,
          ...over,
        })
        setDetName(r.detector)
      } catch (e) {
        setRunning(false)
        loadDevices() // the usual cause is a device that came or went
        setToast({ kind: "error", text: (e as Error).message, id: Date.now() })
      } finally {
        busy.current = false
      }
    },
    [mode, detector, save, inDev, outDev, reset, loadDevices]
  )

  const stop = useCallback(async () => {
    busy.current = true
    setRunning(false)
    try {
      const r = await stopSession()
      if (r.summary) setToast({ kind: "summary", summary: r.summary, id: Date.now() })
    } catch (e) {
      setToast({ kind: "error", text: `Could not stop: ${(e as Error).message}`, id: Date.now() })
    } finally {
      busy.current = false
    }
  }, [])

  const toggle = useCallback(() => {
    if (busy.current) return
    if (running) stop()
    else start()
  }, [running, start, stop])

  // Space starts and stops, unless you are typing or on another control.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat) return
      const el = document.activeElement
      if (el && el !== document.body && el.matches("input, textarea, select, button, [role=switch], [role=radio], [contenteditable]")) return
      e.preventDefault()
      toggle()
    }
    addEventListener("keydown", onKey)
    return () => removeEventListener("keydown", onKey)
  }, [toggle])

  // Settings changed mid-session restart it, so you can flip Off / Gentle / Strong while talking.
  const change = (over: Partial<StartBody>) => {
    if (over.mode) {
      setMode(over.mode)
      store("sc.mode", over.mode)
    }
    if (over.detector) {
      setDetector(over.detector)
      store("sc.det", over.detector)
    }
    if (running && !busy.current) start(over)
  }

  const setHighlight = (layer: Layer | null) => {
    highlight.current = layer
    setHl(layer)
  }

  const inItems = (devices?.inputs ?? []).map((d) => ({ value: d.name, label: d.name }))
  const outItems = [
    { value: NO_OUTPUT, label: "Nowhere (preview only)" },
    ...(devices?.outputs ?? []).map((d) => ({
      value: d.name,
      label: /cable/i.test(d.name) ? `${d.name} (Zoom / Meet mic)` : d.name,
    })),
  ]

  return (
    <MotionConfig reducedMotion="user" transition={SPRING.glide}>
      <div className="relative min-h-dvh overflow-x-clip text-white">
        <LiquidBackground signals={signals} />
        <div className="grain" aria-hidden />

        <motion.div
          variants={page}
          initial="hidden"
          animate="show"
          className="mx-auto max-w-6xl px-4 py-5 sm:px-6 sm:py-6 lg:py-8 2xl:max-w-7xl"
        >
          <motion.header variants={rise} className="flex items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <motion.div
                whileHover={{ rotate: -8, scale: 1.06 }}
                whileTap={{ scale: 0.92 }}
                transition={SPRING.press}
                className="grid size-10 shrink-0 place-items-center rounded-2xl bg-gradient-to-br from-sky-300 via-indigo-300 to-teal-300 text-slate-950 shadow-[0_8px_30px_-8px_rgb(125_170_255/0.8)]"
              >
                <AudioLines className="size-5" strokeWidth={2.4} />
              </motion.div>
              <div className="min-w-0">
                <div className="text-lg leading-tight font-semibold tracking-tight">
                  Stutter<span className="text-gradient">Clear</span>
                </div>
                <div className="hidden text-xs text-white/45 min-[420px]:block">Your real voice, on your terms</div>
              </div>
            </div>
            <EnginePill connected={connected} running={running} detector={detName} stats={stats} />
          </motion.header>

          {/* Desktop: 3 columns. Tablet: 2 columns, packed densely so no cell is left empty.
              Phone: one column. */}
          <div className="mt-5 grid grid-flow-row-dense gap-4 sm:mt-6 sm:gap-5 md:grid-cols-2 lg:grid-cols-3">
            {/* Hero: orb, status, start */}
            <motion.div variants={rise} className="md:col-span-2">
              <GlassCard tilt={1.2} className="h-full p-5 sm:p-8">
                <div className="relative flex flex-col items-center gap-6 sm:flex-row sm:gap-10">
                  <VoiceOrb
                    signals={signals}
                    running={running}
                    onToggle={toggle}
                    className="w-44 shrink-0 min-[400px]:w-52 sm:w-56 lg:w-60"
                  />
                  <div className="flex min-w-0 flex-1 flex-col items-center gap-4 text-center sm:items-start sm:gap-5 sm:text-left">
                    <StatusLine signals={signals} data={data} running={running} connected={connected} mode={mode} />
                    <h1 className="text-[28px] leading-[1.1] font-semibold tracking-tight text-balance sm:text-4xl">
                      Say it your way.
                      <span className="block text-white/45">The listener hears you, without the repeats.</span>
                    </h1>
                    <div className="flex flex-wrap items-center justify-center gap-4 sm:justify-start">
                      <MagneticButton
                        onClick={toggle}
                        disabled={!connected}
                        aria-label={running ? "Stop" : "Start listening"}
                        fillClassName={running ? "bg-white/15" : "bg-white/40"}
                        className={cn(
                          "h-12 w-48 text-[15px] font-semibold transition-[background-color,box-shadow,color] duration-500",
                          running
                            ? "bg-rose-500/85 text-white shadow-[0_12px_40px_-12px_rgb(244_63_94/0.9)]"
                            : "bg-sky-300 text-slate-950 shadow-[0_12px_40px_-12px_rgb(125_211_252/0.9)]"
                        )}
                      >
                        <AnimatePresence mode="popLayout" initial={false}>
                          <motion.span
                            key={running ? "stop" : "start"}
                            initial={{ y: 18, opacity: 0 }}
                            animate={{ y: 0, opacity: 1 }}
                            exit={{ y: -18, opacity: 0 }}
                            transition={SPRING.glide}
                            className="inline-flex items-center gap-2"
                          >
                            {running ? <Square className="size-4 fill-current" /> : <Mic className="size-4" />}
                            {running ? "Stop" : "Start listening"}
                          </motion.span>
                        </AnimatePresence>
                      </MagneticButton>
                      <span className="hidden items-center gap-1.5 text-xs text-white/35 [@media(hover:hover)]:sm:inline-flex">
                        or press
                        <kbd className="rounded-md border border-white/15 bg-white/5 px-1.5 py-0.5 font-sans text-[11px] text-white/60">
                          Space
                        </kbd>
                      </span>
                    </div>
                    <div className="flex w-full flex-col items-center gap-2 sm:items-start">
                      <Segmented label="Correction" value={mode} options={MODES} onChange={(m) => change({ mode: m })} />
                      <div className="grid">
                        <AnimatePresence initial={false}>
                          <motion.p
                            key={mode}
                            initial={{ opacity: 0, y: 6 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -6 }}
                            transition={{ duration: 0.35, ease: EASE_OUT }}
                            className="max-w-md text-[13px] text-white/50 [grid-area:1/1]"
                          >
                            {MODE_HINT[mode]}
                          </motion.p>
                        </AnimatePresence>
                      </div>
                    </div>
                  </div>
                </div>
              </GlassCard>
            </motion.div>

            {/* Setup */}
            <motion.div variants={rise}>
              <GlassCard tilt={2} className="flex h-full flex-col gap-5 p-5 sm:p-6">
                <CardTitle icon={<Headphones className="size-4" />}>Setup</CardTitle>
                <Field label="Microphone">
                  <DeviceSelect
                    items={inItems}
                    value={inDev}
                    placeholder={devices ? "Choose a microphone" : "Loading devices…"}
                    onOpen={loadDevices}
                    onChange={(v) => {
                      setInDev(v)
                      store("sc.in.name", v)
                      if (running && !busy.current) start({ in_name: v })
                    }}
                  />
                </Field>
                <Field label="Send the cleaned voice to">
                  <DeviceSelect
                    items={outItems}
                    value={outDev}
                    onOpen={loadDevices}
                    onChange={(v) => {
                      setOutDev(v)
                      store("sc.out.name", v)
                      if (running && !busy.current) start({ out_name: v === NO_OUTPUT ? null : v })
                    }}
                  />
                </Field>
                <Field label="Detector">
                  <Segmented
                    label="Detector"
                    className="w-full"
                    value={detector}
                    options={DETECTORS}
                    onChange={(v) => change({ detector: v })}
                  />
                  <AnimatePresence initial={false}>
                    {detector === "student" && !hasStudent && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="overflow-hidden text-xs text-amber-200/80"
                      >
                        The AI model is still training, so the rule-based detector is used for now.
                      </motion.div>
                    )}
                  </AnimatePresence>
                </Field>
                <div className="mt-auto flex items-center justify-between gap-4 rounded-2xl border border-white/6 bg-white/[0.03] p-3.5">
                  <div className="text-sm">
                    <div className="font-medium">Save this session</div>
                    <div className="text-xs text-white/45">Off by default. Stays on this laptop.</div>
                  </div>
                  <LiquidSwitch label="Save this session's audio" checked={save} onChange={setSave} />
                </div>
              </GlassCard>
            </motion.div>

            {/* Fluency Mirror */}
            <motion.div variants={rise} className="md:col-span-2">
              <GlassCard tilt={0.6} className="flex h-full flex-col p-5 sm:p-6">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <CardTitle icon={<AudioLines className="size-4" />}>Fluency Mirror</CardTitle>
                  <div className="flex flex-wrap gap-1.5" onPointerLeave={() => setHighlight(null)}>
                    {(
                      [
                        ["you", "You said"],
                        ["cut", "Stutter removed"],
                        ["block", "Pause kept"],
                        ["heard", "Listener hears"],
                      ] as const
                    ).map(([layer, label]) => (
                      <motion.button
                        key={layer}
                        type="button"
                        onPointerEnter={() => setHighlight(layer)}
                        onFocus={() => setHighlight(layer)}
                        onBlur={() => setHighlight(null)}
                        onClick={() => setHighlight(hl === layer ? null : layer)} // tap to toggle on touch
                        whileHover={{ y: -2 }}
                        whileTap={{ scale: 0.95 }}
                        transition={SPRING.press}
                        className={cn(
                          "inline-flex cursor-default items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors duration-500 hover:duration-100",
                          hl === layer ? "border-white/20 bg-white/10 text-white" : "border-white/8 bg-white/[0.03] text-white/60"
                        )}
                      >
                        <span className="size-2 rounded-full" style={{ background: COLORS[layer], boxShadow: `0 0 10px ${COLORS[layer]}` }} />
                        {label}
                      </motion.button>
                    ))}
                  </div>
                </div>
                <MirrorWaveform
                  data={data}
                  signals={signals}
                  highlight={highlight}
                  className="mt-4 min-h-48 w-full flex-1 touch-pan-y sm:min-h-56"
                />
                <StatsRow stats={stats} running={running} />
              </GlassCard>
            </motion.div>

            {/* Context + what comes next */}
            <motion.div variants={rise} className="flex flex-col gap-4 sm:gap-5">
              <ContextCard />
              <GlassCard tilt={2} className="flex flex-1 flex-col gap-4 p-5 sm:p-6">
                <div className="flex items-center justify-between gap-3">
                  <CardTitle icon={<WandSparkles className="size-4" />}>Coming next</CardTitle>
                  <span className="shrink-0 rounded-full border border-sky-300/25 bg-sky-300/10 px-2 py-0.5 text-[11px] text-sky-200">
                    In progress
                  </span>
                </div>
                <Upcoming icon={<WandSparkles className="size-4" />} title="Blocked-word rescue">
                  When you get stuck on a word, a few likely words appear here, from your context card. You choose; nothing is said for you.
                </Upcoming>
                <Upcoming icon={<Captions className="size-4" />} title="Live captions">
                  Captions of what the listener hears, as you speak.
                </Upcoming>
              </GlassCard>
            </motion.div>
          </div>

          <motion.footer
            variants={rise}
            className="mt-8 flex items-start justify-center gap-2 px-2 text-center text-xs text-white/35 sm:items-center"
          >
            <Lock className="mt-0.5 size-3.5 shrink-0 sm:mt-0" />
            Audio is processed on this laptop. Nothing is recorded unless you switch saving on.
          </motion.footer>
        </motion.div>

        <SessionToast toast={toast} onClose={() => setToast(null)} />
      </div>
    </MotionConfig>
  )
}

type Tone = "idle" | "live" | "cut" | "block" | "warn"
const TONE: Record<Tone, { color: string; shimmer: string }> = {
  idle: { color: "rgb(255 255 255 / 0.55)", shimmer: "rgb(255 255 255 / 0.95)" },
  live: { color: "rgb(147 197 253 / 0.7)", shimmer: "rgb(224 242 254)" },
  cut: { color: "rgb(251 113 133 / 0.75)", shimmer: "rgb(255 228 230)" },
  block: { color: "rgb(192 132 252 / 0.75)", shimmer: "rgb(243 232 255)" },
  warn: { color: "rgb(252 211 77 / 0.7)", shimmer: "rgb(254 243 199)" },
}

// Its own component: it re-checks the live signals ten times a second without touching the page.
function StatusLine({
  signals, data, running, connected, mode,
}: {
  signals: RefObject<Signals>
  data: RefObject<LiveData>
  running: boolean
  connected: boolean
  mode: Mode
}) {
  const [status, setStatus] = useState<{ text: string; tone: Tone }>({ text: "Connecting…", tone: "warn" })
  const everConnected = useRef(false)
  useEffect(() => {
    if (connected) everConnected.current = true
    const id = window.setInterval(() => {
      const s = signals.current
      let next: { text: string; tone: Tone }
      if (!connected) next = { text: everConnected.current ? "Reconnecting to the engine…" : "Connecting…", tone: "warn" }
      else if (!running) next = { text: "Tap the orb to start", tone: "idle" }
      else if (s.cut > 0.2) {
        const kind = (KIND_LABEL[data.current.lastCutKind] ?? "stutter").toLowerCase()
        next = { text: `Removed a ${kind}`, tone: "cut" }
      } else if (s.block > 0.25) next = { text: "Pause kept. Take your time.", tone: "block" }
      else if (mode === "off") next = { text: "Mirror only. Nothing is cut.", tone: "live" }
      else next = { text: "Listening…", tone: "live" }
      setStatus((cur) => (cur.text === next.text ? cur : next))
    }, 100)
    return () => window.clearInterval(id)
  }, [signals, data, running, connected, mode])

  return (
    <div className="grid h-6 items-center">
      <AnimatePresence initial={false}>
        <motion.div
          key={status.text}
          className="[grid-area:1/1]"
          initial={{ opacity: 0, y: 8, filter: "blur(4px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          exit={{ opacity: 0, y: -8, filter: "blur(4px)" }}
          transition={{ duration: 0.4, ease: EASE_OUT }}
        >
          <ShimmeringText
            text={status.text}
            className="text-sm font-medium"
            duration={1.8}
            startOnView={false}
            color={TONE[status.tone].color}
            shimmerColor={TONE[status.tone].shimmer}
          />
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

// Live numbers, subscribed on their own so only these tiles re-render with each update.
function StatsRow({ stats, running }: { stats: Store<Stats | null>; running: boolean }) {
  const s = useStore(stats)
  return (
    <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Stat label="Delay (ms)">
        <div className="flex h-[30px] items-center">
          <SteadyDelay ms={running && s ? s.lag_ms : null} />
        </div>
      </Stat>
      <Stat label="Stutters removed" pulseKey={s?.n_cuts ?? 0} pulseColor={COLORS.cut}>
        <RollingNumber value={s?.n_cuts ?? 0} />
      </Stat>
      <Stat label="Seconds removed">
        <RollingNumber value={s?.removed_s ?? 0} decimals={1} />
      </Stat>
      <Stat label="Detector time (ms)" icon={<Cpu className="size-3.5" />}>
        <RollingNumber value={s?.detect_ms ?? 0} decimals={2} />
      </Stat>
    </div>
  )
}

// The splicer's lag wobbles by a frame (250, 260, 250...). Show 10 ms steps and only move when it
// really changes, so the dot matrix does not flicker.
function SteadyDelay({ ms }: { ms: number | null }) {
  const [shown, setShown] = useState<number | null>(null)
  useEffect(() => {
    if (ms === null) setShown(null)
    else setShown((cur) => (cur === null || Math.abs(ms - cur) >= 15 ? Math.round(ms / 10) * 10 : cur))
  }, [ms])
  return <DelayMatrix ms={shown} />
}

function CardTitle({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-white/85">
      <span className="grid size-7 place-items-center rounded-xl bg-white/[0.06] text-sky-200">{icon}</span>
      {children}
    </h2>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="text-xs font-medium text-white/50">{label}</div>
      {children}
    </div>
  )
}

function DeviceSelect({
  items,
  value,
  onChange,
  onOpen,
  placeholder,
}: {
  items: { value: string; label: string }[]
  value: string
  onChange: (v: string) => void
  onOpen?: () => void // refresh the list: earbuds may have connected since the page loaded
  placeholder?: string
}) {
  return (
    <Select
      items={items}
      value={value || null}
      onValueChange={(v) => v != null && onChange(String(v))}
      onOpenChange={(open) => open && onOpen?.()}
    >
      <SelectTrigger className="w-full min-w-0 cursor-pointer rounded-xl border-white/10 bg-white/[0.04] px-3 transition-[background-color,border-color,transform] duration-300 hover:border-white/20 hover:bg-white/[0.08] hover:duration-100 active:scale-[0.985] data-[size=default]:h-11 dark:bg-white/[0.04] dark:hover:bg-white/[0.08]">
        <SelectValue placeholder={placeholder} className="min-w-0 truncate" />
      </SelectTrigger>
      <SelectContent className="glass-blur rounded-xl p-1">
        {items.map((it) => (
          <SelectItem key={it.value} value={it.value} className="cursor-pointer rounded-lg py-2 transition-colors">
            {it.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function Stat({
  label,
  children,
  icon,
  pulseKey,
  pulseColor,
}: {
  label: string
  children: ReactNode
  icon?: ReactNode
  pulseKey?: number
  pulseColor?: string
}) {
  return (
    <motion.div
      whileHover={{ y: -3 }}
      whileTap={{ scale: 0.98 }}
      transition={SPRING.press}
      className="tile relative overflow-hidden rounded-2xl p-3.5 sm:p-4"
    >
      {pulseKey !== undefined && pulseKey > 0 && (
        <motion.span
          key={pulseKey}
          aria-hidden
          className="pointer-events-none absolute inset-0 rounded-2xl"
          initial={{ opacity: 0.9 }}
          animate={{ opacity: 0 }}
          transition={{ duration: 1.4, ease: EASE_OUT }}
          style={{ boxShadow: `inset 0 0 0 1px ${pulseColor}, inset 0 0 30px -6px ${pulseColor}` }}
        />
      )}
      <div className="text-2xl font-semibold tracking-tight sm:text-[28px]">{children}</div>
      <div className="mt-2 flex items-center gap-1.5 text-xs text-white/50">
        {icon}
        {label}
      </div>
    </motion.div>
  )
}

function EnginePill({
  connected, running, detector, stats,
}: {
  connected: boolean
  running: boolean
  detector: string
  stats: Store<Stats | null>
}) {
  const live = useStore(stats, (s) => s?.detector ?? "")
  const det = detector || live
  const label = !connected ? "Engine offline" : running ? `Live · ${det === "student" ? "AI model" : "Rule-based"}` : "Engine ready"
  const dot = !connected ? "bg-amber-300" : running ? "bg-rose-400" : "bg-emerald-300"
  return (
    <motion.div
      layout
      transition={SPRING.glide}
      className="glass relative flex shrink-0 items-center gap-2 overflow-hidden rounded-full px-3.5 py-1.5 text-xs whitespace-nowrap text-white/75"
    >
      <span className="relative flex size-2">
        <span className={cn("absolute inline-flex size-full animate-ping rounded-full opacity-60", dot)} />
        <span className={cn("relative inline-flex size-2 rounded-full", dot)} />
      </span>
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.span
          key={label}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={SPRING.glide}
        >
          {label}
        </motion.span>
      </AnimatePresence>
    </motion.div>
  )
}

function ContextCard() {
  const [text, setText] = useState(() => stored("ctx", ""))
  const [saved, setSaved] = useState(false)
  const timer = useRef<number | undefined>(undefined)
  return (
    <GlassCard tilt={0} className="flex flex-col gap-3 p-5 sm:p-6">
      <div className="flex items-center justify-between">
        <CardTitle icon={<Check className="size-4" />}>Context card</CardTitle>
        <AnimatePresence>
          {saved && (
            <motion.span
              initial={{ opacity: 0, scale: 0.7 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.85 }}
              transition={SPRING.press}
              className="inline-flex items-center gap-1 text-[11px] text-emerald-300"
            >
              <Check className="size-3" /> Saved
            </motion.span>
          )}
        </AnimatePresence>
      </div>
      <textarea
        value={text}
        onChange={(e) => {
          setText(e.target.value)
          store("ctx", e.target.value)
          setSaved(true)
          window.clearTimeout(timer.current)
          timer.current = window.setTimeout(() => setSaved(false), 1200)
        }}
        rows={4}
        placeholder="My name is Sam. Talking to Dr. Mehta about my project, StutterClear."
        className="w-full resize-none rounded-2xl border border-white/8 bg-white/[0.03] p-3.5 text-base leading-relaxed text-white/90 transition-[border-color,background-color,box-shadow] duration-500 outline-none placeholder:text-white/30 hover:border-white/15 hover:duration-150 focus:border-sky-300/40 focus:bg-white/[0.05] focus:shadow-[0_0_0_4px_rgb(125_211_252/0.12)] focus:duration-200 sm:text-sm"
      />
      <div className="text-xs text-white/40">Names and topics you might say. Stays on this laptop.</div>
    </GlassCard>
  )
}

function Upcoming({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <motion.div whileHover={{ x: 3 }} transition={SPRING.press} className="tile flex gap-3 rounded-2xl p-3.5">
      <span className="mt-0.5 text-white/50">{icon}</span>
      <div className="text-sm">
        <div className="font-medium text-white/85">{title}</div>
        <div className="mt-0.5 text-xs leading-relaxed text-white/45">{children}</div>
      </div>
    </motion.div>
  )
}
