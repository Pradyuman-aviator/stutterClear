import { useCallback, useEffect, useRef, useState, type ReactNode, type RefObject } from "react"
import { AnimatePresence, motion, type Variants } from "motion/react"
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
import { useLive, type LiveData } from "@/hooks/use-live"
import { useSignals, type Signals } from "@/hooks/use-signals"
import {
  getDevices, getInfo, KIND_LABEL, startSession, stopSession,
  type DetectorKind, type Devices, type Mode, type StartBody,
} from "@/lib/api"
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

const page: Variants = { hidden: {}, show: { transition: { staggerChildren: 0.08, delayChildren: 0.05 } } }
const rise: Variants = {
  hidden: { opacity: 0, y: 18, filter: "blur(10px)" },
  show: { opacity: 1, y: 0, filter: "blur(0px)", transition: { type: "spring", stiffness: 140, damping: 20 } },
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

type Tone = "idle" | "live" | "cut" | "block" | "warn"
const TONE: Record<Tone, { color: string; shimmer: string }> = {
  idle: { color: "rgb(255 255 255 / 0.55)", shimmer: "rgb(255 255 255 / 0.95)" },
  live: { color: "rgb(147 197 253 / 0.7)", shimmer: "rgb(224 242 254)" },
  cut: { color: "rgb(251 113 133 / 0.75)", shimmer: "rgb(255 228 230)" },
  block: { color: "rgb(192 132 252 / 0.75)", shimmer: "rgb(243 232 255)" },
  warn: { color: "rgb(252 211 77 / 0.7)", shimmer: "rgb(254 243 199)" },
}

function useStatus(
  signals: RefObject<Signals>, data: RefObject<LiveData>, running: boolean, connected: boolean, mode: Mode
) {
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
  return status
}

export default function App() {
  const { data, stats, connected, serverRunning, reset } = useLive()
  const [running, setRunning] = useState(false)
  const [busy, setBusy] = useState(false)
  const [devices, setDevices] = useState<Devices | null>(null)
  const [inDev, setInDev] = useState("")
  const [outDev, setOutDev] = useState(() => stored("sc.out", NO_OUTPUT))
  const [mode, setMode] = useState<Mode>(() => (stored("sc.mode", "gentle") as Mode))
  const [detector, setDetector] = useState<DetectorKind>(() => (stored("sc.det", "student") as DetectorKind))
  const [save, setSave] = useState(false) // never remembered: recording is opt-in every time
  const [detName, setDetName] = useState("")
  const [hasStudent, setHasStudent] = useState(true)
  const [toast, setToast] = useState<ToastData | null>(null)
  const highlight = useRef<Layer | null>(null)
  const [hl, setHl] = useState<Layer | null>(null)
  const signals = useSignals(data, running)
  const status = useStatus(signals, data, running, connected, mode)

  useEffect(() => {
    if (!connected) return
    getInfo().then((i) => setHasStudent(i.student)).catch(() => {})
    getDevices()
      .then((d) => {
        setDevices(d)
        const keep = stored("sc.in", "")
        setInDev(d.inputs.some((x) => String(x.index) === keep) ? keep : String(d.default_input))
        setOutDev((o) => (o === NO_OUTPUT || d.outputs.some((x) => String(x.index) === o) ? o : NO_OUTPUT))
      })
      .catch(() => setDevices(null))
  }, [connected])

  // A session can outlive the page (reload while talking): follow what the server says.
  useEffect(() => {
    if (!busy) setRunning(serverRunning)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serverRunning])

  const start = useCallback(
    async (over: Partial<StartBody> = {}) => {
      setBusy(true)
      try {
        reset()
        const body: StartBody = {
          mode, detector, save,
          in_device: inDev === "" ? null : +inDev,
          out_device: outDev === NO_OUTPUT ? null : +outDev,
          ...over,
        }
        const r = await startSession(body)
        setDetName(r.detector)
        setRunning(true)
      } catch (e) {
        setToast({ kind: "error", text: `Could not start: ${(e as Error).message}`, id: Date.now() })
      } finally {
        setBusy(false)
      }
    },
    [mode, detector, save, inDev, outDev, reset]
  )

  const stop = useCallback(async () => {
    setBusy(true)
    try {
      const r = await stopSession()
      setRunning(false)
      if (r.summary) setToast({ kind: "summary", summary: r.summary, id: Date.now() })
    } catch (e) {
      setToast({ kind: "error", text: `Could not stop: ${(e as Error).message}`, id: Date.now() })
    } finally {
      setBusy(false)
    }
  }, [])

  const toggle = useCallback(() => {
    if (busy) return
    if (running) stop()
    else start()
  }, [busy, running, start, stop])

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
    if (running && !busy) start(over)
  }

  const inItems = (devices?.inputs ?? []).map((d) => ({ value: String(d.index), label: d.name }))
  const outItems = [
    { value: NO_OUTPUT, label: "Nowhere (preview only)" },
    ...(devices?.outputs ?? []).map((d) => ({
      value: String(d.index),
      label: /cable/i.test(d.name) ? `${d.name} (Zoom / Meet mic)` : d.name,
    })),
  ]

  return (
    <div className="relative min-h-dvh text-white">
      <LiquidBackground signals={signals} />
      <div className="grain" aria-hidden />

      <motion.div variants={page} initial="hidden" animate="show" className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:py-8">
        <motion.header variants={rise} className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <motion.div
              whileHover={{ rotate: -8, scale: 1.08 }}
              whileTap={{ scale: 0.9 }}
              className="grid size-10 place-items-center rounded-2xl bg-gradient-to-br from-sky-300 via-indigo-300 to-teal-300 text-slate-950 shadow-[0_8px_30px_-8px_rgb(125_170_255/0.8)]"
            >
              <AudioLines className="size-5" strokeWidth={2.4} />
            </motion.div>
            <div>
              <div className="text-lg leading-tight font-semibold tracking-tight">
                Stutter<span className="text-gradient">Clear</span>
              </div>
              <div className="text-xs text-white/45">Your real voice, on your terms</div>
            </div>
          </div>
          <EnginePill connected={connected} running={running} detector={detName || stats?.detector || ""} />
        </motion.header>

        <div className="mt-6 grid gap-5 lg:grid-cols-3">
          {/* Hero: orb, status, start */}
          <motion.div variants={rise} className="lg:col-span-2">
            <GlassCard tilt={1.2} className="h-full p-6 sm:p-8">
              <div className="relative flex flex-col items-center gap-8 sm:flex-row sm:gap-10">
                <VoiceOrb signals={signals} running={running} onToggle={toggle} className="w-52 shrink-0 sm:w-60" />
                <div className="flex min-w-0 flex-1 flex-col items-center gap-5 text-center sm:items-start sm:text-left">
                  <div className="grid h-6 items-center">
                    <AnimatePresence initial={false}>
                      <motion.div
                        key={status.text}
                        className="[grid-area:1/1]"
                        initial={{ opacity: 0, y: 8, filter: "blur(6px)" }}
                        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                        exit={{ opacity: 0, y: -8, filter: "blur(6px)" }}
                        transition={{ duration: 0.3 }}
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
                  <h1 className="text-3xl leading-[1.1] font-semibold tracking-tight text-balance sm:text-4xl">
                    Say it your way.
                    <span className="block text-white/45">The listener hears you, without the repeats.</span>
                  </h1>
                  <div className="flex flex-wrap items-center justify-center gap-4 sm:justify-start">
                    <MagneticButton
                      onClick={toggle}
                      disabled={busy || !connected}
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
                          initial={{ y: 16, opacity: 0, filter: "blur(4px)" }}
                          animate={{ y: 0, opacity: 1, filter: "blur(0px)" }}
                          exit={{ y: -16, opacity: 0, filter: "blur(4px)" }}
                          transition={{ type: "spring", stiffness: 400, damping: 26 }}
                          className="inline-flex items-center gap-2"
                        >
                          {running ? <Square className="size-4 fill-current" /> : <Mic className="size-4" />}
                          {running ? "Stop" : "Start listening"}
                        </motion.span>
                      </AnimatePresence>
                    </MagneticButton>
                    <span className="hidden items-center gap-1.5 text-xs text-white/35 sm:inline-flex">
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
                          initial={{ opacity: 0, x: -8 }}
                          animate={{ opacity: 1, x: 0 }}
                          exit={{ opacity: 0, x: 8 }}
                          transition={{ duration: 0.25 }}
                          className="text-[13px] text-white/50 [grid-area:1/1]"
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
            <GlassCard tilt={2} className="flex h-full flex-col gap-5 p-6">
              <CardTitle icon={<Headphones className="size-4" />}>Setup</CardTitle>
              <Field label="Microphone">
                <DeviceSelect
                  items={inItems}
                  value={inDev}
                  placeholder={devices ? "Choose a microphone" : "Loading devices…"}
                  onChange={(v) => {
                    setInDev(v)
                    store("sc.in", v)
                    if (running && !busy) start({ in_device: +v })
                  }}
                />
              </Field>
              <Field label="Send the cleaned voice to">
                <DeviceSelect
                  items={outItems}
                  value={outDev}
                  onChange={(v) => {
                    setOutDev(v)
                    store("sc.out", v)
                    if (running && !busy) start({ out_device: v === NO_OUTPUT ? null : +v })
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
          <motion.div variants={rise} className="lg:col-span-2">
            <GlassCard tilt={0.6} className="flex h-full flex-col p-6">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <CardTitle icon={<AudioLines className="size-4" />}>Fluency Mirror</CardTitle>
                <div className="flex flex-wrap gap-1.5" onPointerLeave={() => { highlight.current = null; setHl(null) }}>
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
                      onPointerEnter={() => { highlight.current = layer; setHl(layer) }}
                      onFocus={() => { highlight.current = layer; setHl(layer) }}
                      onBlur={() => { highlight.current = null; setHl(null) }}
                      whileHover={{ y: -2 }}
                      whileTap={{ scale: 0.94 }}
                      className={cn(
                        "inline-flex cursor-default items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors duration-300",
                        hl === layer ? "border-white/20 bg-white/10 text-white" : "border-white/8 bg-white/[0.03] text-white/60"
                      )}
                    >
                      <span className="size-2 rounded-full" style={{ background: COLORS[layer], boxShadow: `0 0 10px ${COLORS[layer]}` }} />
                      {label}
                    </motion.button>
                  ))}
                </div>
              </div>
              <MirrorWaveform data={data} signals={signals} highlight={highlight} className="mt-4 min-h-56 w-full flex-1" />
              <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Delay (ms)">
                  <div className="flex h-[30px] items-center">
                    <DelayMatrix ms={running && stats ? stats.lag_ms : null} />
                  </div>
                </Stat>
                <Stat label="Stutters removed" pulseKey={stats?.n_cuts ?? 0} pulseColor={COLORS.cut}>
                  <RollingNumber value={stats?.n_cuts ?? 0} />
                </Stat>
                <Stat label="Seconds removed">
                  <RollingNumber value={stats?.removed_s ?? 0} decimals={1} />
                </Stat>
                <Stat label="Detector time (ms)" icon={<Cpu className="size-3.5" />}>
                  <RollingNumber value={stats?.detect_ms ?? 0} decimals={2} />
                </Stat>
              </div>
            </GlassCard>
          </motion.div>

          {/* Context + what comes next */}
          <motion.div variants={rise} className="flex flex-col gap-5">
            <ContextCard />
            <GlassCard tilt={2} className="flex flex-1 flex-col gap-4 p-6">
              <div className="flex items-center justify-between">
                <CardTitle icon={<WandSparkles className="size-4" />}>Coming next</CardTitle>
                <span className="rounded-full border border-sky-300/25 bg-sky-300/10 px-2 py-0.5 text-[11px] text-sky-200">
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

        <motion.footer variants={rise} className="mt-8 flex items-center justify-center gap-2 text-xs text-white/35">
          <Lock className="size-3.5" />
          Audio is processed on this laptop. Nothing is recorded unless you switch saving on.
        </motion.footer>
      </motion.div>

      <SessionToast toast={toast} onClose={() => setToast(null)} />
    </div>
  )
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
  placeholder,
}: {
  items: { value: string; label: string }[]
  value: string
  onChange: (v: string) => void
  placeholder?: string
}) {
  return (
    <Select items={items} value={value || null} onValueChange={(v) => v != null && onChange(String(v))}>
      <SelectTrigger className="w-full cursor-pointer rounded-xl border-white/10 bg-white/[0.04] px-3 transition-[background-color,border-color,transform] duration-200 hover:border-white/20 hover:bg-white/[0.08] active:scale-[0.98] data-[size=default]:h-10 dark:bg-white/[0.04] dark:hover:bg-white/[0.08]">
        <SelectValue placeholder={placeholder} className="truncate" />
      </SelectTrigger>
      <SelectContent className="rounded-xl p-1 backdrop-blur-xl">
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
      whileTap={{ scale: 0.97 }}
      transition={{ type: "spring", stiffness: 400, damping: 20 }}
      className="tile relative overflow-hidden rounded-2xl p-4"
    >
      {pulseKey !== undefined && pulseKey > 0 && (
        <motion.span
          key={pulseKey}
          aria-hidden
          className="pointer-events-none absolute inset-0 rounded-2xl"
          initial={{ opacity: 0.9 }}
          animate={{ opacity: 0 }}
          transition={{ duration: 1.2, ease: "easeOut" }}
          style={{ boxShadow: `inset 0 0 0 1px ${pulseColor}, inset 0 0 30px -6px ${pulseColor}` }}
        />
      )}
      <div className="text-[28px] font-semibold tracking-tight">{children}</div>
      <div className="mt-2 flex items-center gap-1.5 text-xs text-white/50">
        {icon}
        {label}
      </div>
    </motion.div>
  )
}

function EnginePill({ connected, running, detector }: { connected: boolean; running: boolean; detector: string }) {
  const label = !connected ? "Engine offline" : running ? `Live · ${detector === "student" ? "AI model" : "Rule-based"}` : "Engine ready"
  const dot = !connected ? "bg-amber-300" : running ? "bg-rose-400" : "bg-emerald-300"
  return (
    <motion.div layout transition={{ type: "spring", stiffness: 400, damping: 30 }} className="glass flex items-center gap-2 rounded-full px-3.5 py-1.5 text-xs text-white/75">
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
          transition={{ duration: 0.2 }}
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
    <GlassCard tilt={0} className="flex flex-col gap-3 p-6">
      <div className="flex items-center justify-between">
        <CardTitle icon={<Check className="size-4" />}>Context card</CardTitle>
        <AnimatePresence>
          {saved && (
            <motion.span
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={{ type: "spring", stiffness: 500, damping: 22 }}
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
        className="w-full resize-none rounded-2xl border border-white/8 bg-white/[0.03] p-3.5 text-sm leading-relaxed text-white/90 transition-[border-color,background-color,box-shadow] duration-300 outline-none placeholder:text-white/30 hover:border-white/15 focus:border-sky-300/40 focus:bg-white/[0.05] focus:shadow-[0_0_0_4px_rgb(125_211_252/0.12)]"
      />
      <div className="text-xs text-white/40">Names and topics you might say. Stays on this laptop.</div>
    </GlassCard>
  )
}

function Upcoming({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <motion.div
      whileHover={{ x: 3 }}
      transition={{ type: "spring", stiffness: 400, damping: 25 }}
      className="tile flex gap-3 rounded-2xl p-3.5"
    >
      <span className="mt-0.5 text-white/50">{icon}</span>
      <div className="text-sm">
        <div className="font-medium text-white/85">{title}</div>
        <div className="mt-0.5 text-xs leading-relaxed text-white/45">{children}</div>
      </div>
    </motion.div>
  )
}
