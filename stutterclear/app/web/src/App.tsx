import { useCallback, useEffect, useEffectEvent, useRef, useState, type ReactNode, type RefObject } from "react"
import { AnimatePresence, MotionConfig, motion, type Variants } from "motion/react"
import { Check, Cpu, Lock, Mic, Play, Square } from "lucide-react"
import { IsoBadge, IsoContext, IsoLogo, IsoMirror, IsoSetup } from "@/components/iso-icon"

import { CaptionStrip } from "@/components/caption-strip"
import { DelayMatrix } from "@/components/delay-matrix"
import { DemoBar, DemoPicker } from "@/components/demo-player"
import { HowItWorks } from "@/components/how-it-works"
import { InsightsPanel } from "@/components/insights-panel"
import { GlassCard } from "@/components/glass-card"
import { Button } from "@/components/ui/button"
import { COLORS, MirrorWaveform, type Layer } from "@/components/mirror-waveform"
import { Playback } from "@/components/playback"
import { RollingNumber } from "@/components/rolling-number"
import { CutReview } from "@/components/cut-review"
import { DictatePanel } from "@/components/dictate-panel"
import { JournalPanel } from "@/components/journal-panel"
import { VideoCard } from "@/components/video-card"
import { TranscriptView } from "@/components/transcript-view"
import { PracticePanel } from "@/components/practice-panel"
import { StagePanel } from "@/components/stage-panel"
import { Field as FormField, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Toaster } from "@/components/ui/sonner"
import { toast as notify } from "sonner"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { SessionToast, type ToastData } from "@/components/session-toast"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { VoiceNoteCard } from "@/components/voice-note"
import { VoiceOrb } from "@/components/voice-orb"
import { useDemo } from "@/hooks/use-demo"
import { useLive, type LiveData, type Stats } from "@/hooks/use-live"
import { useSignals, type Signals } from "@/hooks/use-signals"
import {
  askRescue, playSaid, sayInCall, getDemos, getDevices, getInfo, getPlayback, KIND_LABEL, prepareContext, startSession, stopSession,
  type DemoTimeline, type InsightsBody,
  type DemoSample, type DetectorInfo, type DetectorKind, type Devices, type Mode, type PlaybackInfo, type StartBody,
} from "@/lib/api"
import { EASE_OUT, SPRING } from "@/lib/motion"
import { useStore, type Store } from "@/lib/store"
import { BrowserMic } from "@/lib/browser-mic"
import { localAdjust } from "@/lib/feedback"
import { cn } from "@/lib/utils"

const MODES: { value: Mode; label: string }[] = [
  { value: "off", label: "Off" },
  { value: "gentle", label: "Gentle" },
  { value: "strong", label: "Strong" },
]
// How far behind you the listener hears you depends on the detector: the big model needs to hear
// a little after each moment before deciding (see stutterclear/live.py delay_overrides).
function modeHint(mode: Mode, detector: DetectorKind) {
  if (mode === "off") return "Mirror only: nothing is cut, so you can watch your speech as it is."
  const behind = detector === "teacher" ? (mode === "gentle" ? "0.5 s" : "0.85 s") : mode === "gentle" ? "250 ms" : "600 ms"
  return mode === "gentle"
    ? `Removes clear repeats. About ${behind} behind you.`
    : `Removes the most stutters, fillers too; may clip a word now and then. About ${behind} behind you.`
}
const DETECTOR_NOTE: Record<DetectorKind, string> = {
  teacher: "The big model, run live on your GPU.",
  student: "A small model that runs on any computer.",
  heuristic: "Light rules: safe, but removes little.",
}
const NO_OUTPUT = "none"
const BROWSER_MIC = "__browser__" // live from this browser's mic, streamed to the server

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
  const { data, stats, caption, connected, serverRunning, reset } = useLive()
  const [running, setRunning] = useState(false)
  const [starting, setStarting] = useState(false)
  const busy = useRef(false)
  const [devices, setDevices] = useState<Devices | null>(null)
  // Devices are remembered by name: Windows renumbers them whenever one connects or drops.
  const [inDev, setInDev] = useState("")
  const [outDev, setOutDev] = useState(() => stored("sc.out.name", NO_OUTPUT))
  const [mode, setMode] = useState<Mode>(() => stored("sc.mode", "gentle") as Mode)
  const [detector, setDetector] = useState<DetectorKind>(() => (stored("sc.det", "") || "heuristic") as DetectorKind)
  const [save, setSave] = useState(false) // never remembered: recording is opt-in every time
  const [detName, setDetName] = useState("")
  const [detectorList, setDetectorList] = useState<DetectorInfo[]>([])
  const [recommendedDet, setRecommendedDet] = useState<DetectorKind>("heuristic")
  const [notesOn, setNotesOn] = useState(false)
  const [infoReady, setInfoReady] = useState(false)
  const [videoOn, setVideoOn] = useState(false)
  const [captionsOn, setCaptionsOn] = useState(false)
  const [rescueAvailable, setRescueAvailable] = useState(false)
  const [searchOn, setSearchOn] = useState(false)
  const [hosted, setHosted] = useState(false)
  const [ttsOn, setTtsOn] = useState(false)
  const [interviewOn, setInterviewOn] = useState(false)
  const [feedbackBrowser, setFeedbackBrowser] = useState(false)
  const [aidOn, setAidOn] = useState(() => stored("sc.aid", "") === "on")
  const [aidMs, setAidMs] = useState(() => Number(stored("sc.aidms", "80")) || 80)
  const [tab, setTab] = useState("calls")
  const [rescueOn, setRescueOn] = useState(() => stored("sc.rescue", "") === "on")
  const [rescue, setRescue] = useState<{ loading: boolean; words: string[]; error?: string }>({ loading: false, words: [] })
  const [noteSamples, setNoteSamples] = useState<DemoSample[]>([])
  const [toast, setToast] = useState<ToastData | null>(null)
  const [playback, setPlayback] = useState<PlaybackInfo | null>(null)
  const [playbackVersion, setPlaybackVersion] = useState(0)
  const highlight = useRef<Layer | null>(null)
  const [hl, setHl] = useState<Layer | null>(null)
  const demo = useDemo(data, stats, reset, caption)
  const [demos, setDemos] = useState<{ samples: DemoSample[]; credit: string }>({ samples: [], credit: "" })
  const demoOn = demo.state !== null
  const demoPlaying = !!demo.state?.playing
  const active = running || demoPlaying // the orb, mirror and stats come alive for either
  const signals = useSignals(data, active)

  const loadDevices = useCallback(() => {
    getDevices()
      .then((d) => {
        setDevices(d)
        // No mic on the server (a hosted demo): live audio comes from the browser.
        const fallback = d.inputs.find((x) => x.index === d.default_input)?.name ?? d.inputs[0]?.name ?? BROWSER_MIC
        setInDev((cur) => {
          const want = cur || stored("sc.in.name", "")
          return want === BROWSER_MIC || d.inputs.some((x) => x.name === want) ? want : fallback
        })
        setOutDev((o) => (o === NO_OUTPUT || d.outputs.some((x) => x.name === o) ? o : NO_OUTPUT))
      })
      .catch(() => setDevices(null))
  }, [])

  useEffect(() => {
    if (!connected) return
    getInfo()
      .then((i) => {
        setInfoReady(true)
        setNotesOn(i.notes)
        setVideoOn(!!i.video)
        setCaptionsOn(i.captions)
        setRescueAvailable(i.nemotron)
        setSearchOn(i.search)
        setHosted(i.hosted)
        setTtsOn(!!i.tts)
        setInterviewOn(!!i.interview)
        setFeedbackBrowser(i.feedback === "browser")
        setDetectorList(i.detectors)
        setRecommendedDet(i.recommended)
        // Use the best measured detector unless one was picked before (and is still available here).
        const kept = stored("sc.det", "") as DetectorKind
        const ok = i.detectors.some((d) => d.kind === kept && d.available)
        setDetector(ok ? kept : i.recommended)
      })
      .catch(() => {})
    getPlayback().then((p) => setPlayback(p.available ? p : null)).catch(() => {})
    getDemos().then(setDemos).catch(() => {})
    loadDevices()
  }, [connected, loadDevices])

  const goTab = (next: string) => {
    setTab(next)
    history.replaceState(null, "", `#${next}`)
  }

  // The tabs this server can show. A link to one (#dictate, #stage...) waits for the server to say
  // which features it has.
  const tabs = ["calls", "recordings", ...(notesOn ? ["dictate"] : []), ...(ttsOn ? ["stage"] : []),
    ...(interviewOn ? ["practice"] : []), ...(!hosted ? ["journal"] : [])].join(",")
  useEffect(() => {
    if (!infoReady) return
    const fromHash = location.hash.replace("#", "")
    const next = tabs.split(",").includes(fromHash) ? fromHash : hosted ? "recordings" : "calls"
    setTab(next)
    if (location.hash !== `#${next}`) history.replaceState(null, "", `#${next}`)
  }, [infoReady, hosted, tabs])

  // A session can outlive the page (reload while talking): follow what the server says.
  useEffect(() => {
    if (!busy.current) setRunning(serverRunning)
  }, [serverRunning])

  // Start and stop are optimistic: the UI flips on the click, the server catches up behind it.
  const browserMic = useRef<BrowserMic | null>(null)

  const start = useCallback(
    async (over: Partial<StartBody> = {}) => {
      busy.current = true
      setRunning(true)
      setStarting(true)
      setPlayback(null) // the server drops the previous recording when a session starts
      reset()
      try {
        if (browserMic.current) {
          await browserMic.current.stop() // restarting with new settings
          browserMic.current = null
        }
        if ((over.in_name ?? inDev) === BROWSER_MIC) {
          // The server lets one live session run at a time; give the previous one a moment to end.
          for (let attempt = 0; ; attempt++) {
            const mic = new BrowserMic()
            try {
              // On a public server this browser's own reviews tune the detector (the server keeps none).
              const adjust = feedbackBrowser ? await localAdjust() : undefined
              const r = await mic.start({ mode: over.mode ?? mode, detector: over.detector ?? detector, aidMs: aidOn ? aidMs : 0, adjust }, (end) => {
                if (browserMic.current !== mic) return
                browserMic.current = null
                mic.stop()
                setRunning(false)
                if (!end) {
                  setToast({ kind: "error", text: "The live connection dropped. Press Start to go again.", id: Date.now() })
                  return
                }
                if (end.summary) setToast({ kind: "summary", summary: end.summary, note: end.ended ?? undefined, id: Date.now() })
                if (end.playback?.available) {
                  setPlayback(end.playback)
                  setPlaybackVersion((v) => v + 1)
                }
              })
              browserMic.current = mic
              setDetName(r.detector)
              break
            } catch (e) {
              await mic.stop()
              if (attempt >= 4 || !/already running|someone else is live/i.test((e as Error).message)) throw e
              await new Promise((r) => setTimeout(r, 400))
            }
          }
          return
        }
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
        setStarting(false)
      }
    },
    [mode, detector, save, inDev, outDev, reset, loadDevices, aidOn, aidMs, feedbackBrowser]
  )

  const stop = useCallback(async () => {
    busy.current = true
    setRunning(false)
    try {
      const mic = browserMic.current
      browserMic.current = null
      const r = mic ? ((await mic.stop()) ?? { summary: null }) : await stopSession()
      if (r.summary) setToast({ kind: "summary", summary: r.summary, id: Date.now() })
      if (r.playback?.available) {
        setPlayback(r.playback)
        setPlaybackVersion((v) => v + 1)
      }
    } catch (e) {
      setToast({ kind: "error", text: `Could not stop: ${(e as Error).message}`, id: Date.now() })
    } finally {
      busy.current = false
    }
  }, [])

  const toggle = useCallback(() => {
    if (busy.current) return
    if (running) stop()
    else {
      if (demo.state) demo.close()
      start()
    }
  }, [running, start, stop, demo])

  // Space starts and stops, unless you are typing or on another control.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat) return
      if (tab === "stage" && !e.ctrlKey) return
      const el = document.activeElement
      if (el && el !== document.body && el.matches("input, textarea, select, button, [role=switch], [role=radio], [contenteditable]")) return
      e.preventDefault()
      toggle()
    }
    addEventListener("keydown", onKey)
    return () => removeEventListener("keydown", onKey)
  }, [toggle, tab])

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

  const lastMode = useRef<Mode>(mode === "off" ? "gentle" : mode)
  useEffect(() => {
    if (mode !== "off") lastMode.current = mode
  }, [mode])
  const pauseCleaning = useEffectEvent(() => {
    const next = mode === "off" ? lastMode.current : "off"
    change({ mode: next })
    notify.message(next === "off" ? "Cleaning paused: the listener hears you as you are. Esc resumes." : "Cleaning resumed.")
  })

  // Esc pauses the cleaning mid-session (Off) and brings the last mode back: if something sounds
  // wrong in a call, one key gives the listener your voice exactly as it is.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Escape" || e.repeat || !running) return
      e.preventDefault()
      pauseCleaning()
    }
    addEventListener("keydown", onKey)
    return () => removeEventListener("keydown", onKey)
  }, [running])

  // Say it: the synthetic voice speaks a word you picked. Into the call when the cleaned voice goes to
  // an output (a virtual cable), otherwise here in the browser.
  const sayWord = useCallback(
    async (word: string) => {
      try {
        if (running && inDev !== BROWSER_MIC && outDev !== NO_OUTPUT) {
          const r = await sayInCall(word)
          notify.success(`Said in the call: "${word}" (${r.label})`)
        } else {
          const label = await playSaid(word)
          notify.message(`"${word}" (${label})`)
        }
      } catch (e) {
        notify.error((e as Error).message)
      }
    },
    [running, inDev, outDev]
  )

  // Word rescue: ask Nemotron for the word you are stuck on, from the captions so far and the
  // context card. Opt-in; asked automatically when a block is detected, or with the button.
  const askForWord = useCallback(async () => {
    const c = caption.get()
    const said = c ? [...c.lines, c.partial].join(" ") : ""
    setRescue({ loading: true, words: [] })
    try {
      const r = await askRescue(said, stored("ctx", ""))
      setRescue({ loading: false, words: r.suggestions })
    } catch (e) {
      setRescue({ loading: false, words: [], error: (e as Error).message })
    }
  }, [caption])

  useEffect(() => {
    if (!rescueOn || !rescueAvailable || !active) return
    let handled = data.current.lastBlockAt
    const id = window.setInterval(() => {
      const at = data.current.lastBlockAt
      if (at > handled) {
        handled = at
        askForWord()
      }
    }, 200)
    return () => window.clearInterval(id)
  }, [rescueOn, rescueAvailable, active, data, askForWord])

  // Suggestions fade away after a while: they are a prompt, not a transcript.
  useEffect(() => {
    if (!rescue.words.length && !rescue.error) return
    const id = window.setTimeout(() => setRescue({ loading: false, words: [] }), 12000)
    return () => window.clearTimeout(id)
  }, [rescue])

  const setHighlight = (layer: Layer | null) => {
    highlight.current = layer
    setHl(layer)
  }

  const inItems = [
    ...(devices?.inputs ?? []).map((d) => ({ value: d.name, label: d.name })),
    { value: BROWSER_MIC, label: "This browser's mic" },
  ]
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
        <div className="w-full px-4 py-5 sm:px-6 sm:py-6 lg:px-8 lg:py-8">
          <header className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <IsoBadge>
                <IsoLogo />
              </IsoBadge>
              <div className="min-w-0">
                <div className="font-logo text-2xl leading-none tracking-tight">
                  Stutter<span className="text-gradient">Clear</span>
                </div>
                <div className="hidden text-xs text-white/45 min-[420px]:block">Your real voice, on your terms</div>
              </div>
            </div>
            <EnginePill connected={connected} running={running} demo={demoOn} detector={detName} stats={stats} />
          </header>

          <Tabs
            key={tabs} // new tabs register afresh, so the highlight follows a tab that appears with a link to it
            className="mt-5 min-w-0"
            value={tab}
            onValueChange={(v) => goTab(String(v))}
          >
            <TabsList className="sticky top-2 z-20 h-auto max-w-full flex-wrap sm:static">
              <TabsTrigger value="calls">Calls</TabsTrigger>
              <TabsTrigger value="recordings">Recordings</TabsTrigger>
              {notesOn && <TabsTrigger value="dictate">Dictate</TabsTrigger>}
              {ttsOn && <TabsTrigger value="stage">On stage</TabsTrigger>}
              {interviewOn && <TabsTrigger value="practice">Practice</TabsTrigger>}
              {!hosted && <TabsTrigger value="journal">Journal</TabsTrigger>}
            </TabsList>
            <TabsContent value="calls">
              <div className="grid grid-flow-row-dense grid-cols-1 gap-4 sm:gap-5 md:grid-cols-2 lg:grid-cols-3">
            {/* Hero: orb, status, start */}
            <div id="hero" className="scroll-mt-4 md:col-span-2">
              <GlassCard tilt={0} className="h-full p-5 sm:p-8">
                <div className="relative flex flex-col items-center gap-6 sm:h-full sm:flex-row sm:gap-10">
                  <VoiceOrb
                    signals={signals}
                    running={active}
                    onToggle={toggle}
                    className="w-44 shrink-0 min-[400px]:w-52 sm:w-56 lg:w-60"
                  />
                  <div className="flex min-w-0 flex-1 flex-col items-center gap-4 text-center sm:items-start sm:gap-5 sm:text-left">
                    <StatusLine
                      signals={signals}
                      data={data}
                      running={active}
                      starting={starting}
                      demo={demoPlaying}
                      connected={connected}
                      mode={running ? mode : "gentle"}
                    />
                    <h1 className="text-[28px] leading-[1.1] font-semibold tracking-tight text-balance sm:text-4xl">
                      Say it your way.
                      <span className="block text-white/45">The listener hears you, without the repeats.</span>
                    </h1>
                    <div className="flex flex-wrap items-center justify-center gap-4 sm:justify-start">
                      <Button
                        onClick={toggle}
                        disabled={!connected}
                        aria-label={running ? "Stop" : "Start listening"}
                        className={cn(
                          "h-8 rounded-full px-3 text-sm [&_svg]:size-3.5",
                          running ? "bg-rose-500 text-white hover:bg-rose-500/90" : "bg-sky-300 text-slate-950 hover:bg-sky-200"
                        )}
                      >
                        {running ? <Square data-icon="inline-start" className="fill-current" /> : <Mic data-icon="inline-start" />}
                        {running ? "Stop" : "Start listening"}
                      </Button>
                      <span className="hidden items-center gap-1.5 text-xs text-white/35 [@media(hover:hover)]:sm:inline-flex">
                        or press
                        <kbd className="rounded-md border border-white/15 bg-white/5 px-1.5 py-0.5 font-sans text-[11px] text-white/60">
                          Space
                        </kbd>
                        {running && (
                          <>
                            · pause cleaning
                            <kbd className="rounded-md border border-white/15 bg-white/5 px-1.5 py-0.5 font-sans text-[11px] text-white/60">
                              Esc
                            </kbd>
                          </>
                        )}
                      </span>
                    </div>
                    <div className="flex w-full flex-col items-center gap-2 sm:items-start">
                      <ToggleGroup aria-label="Correction" value={[mode]} onValueChange={(v) => { const n = v[0] as Mode | undefined; if (n) change({ mode: n }) }} variant="outline" spacing={1}>
                        {MODES.map((m) => (
                          <ToggleGroupItem key={m.value} value={m.value}>{m.label}</ToggleGroupItem>
                        ))}
                      </ToggleGroup>
                      <p className="max-w-md text-[13px] leading-relaxed text-white/50">{modeHint(mode, detector)}</p>
                      {!running && demos.samples.length > 0 && (
                        <button
                          type="button"
                          onClick={() => goTab("recordings")}
                          className="mt-1 inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs text-white/70 transition-colors hover:border-sky-300/30 hover:text-sky-100"
                        >
                          <Play className="size-3 fill-current text-sky-300" />
                          No mic handy? Hear it on real recordings
                        </button>
                      )}
                    </div>
                    <AnimatePresence initial={false}>
                      {!running && !demoOn && playback && (
                        <motion.div
                          key="playback"
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          transition={SPRING.glide}
                          className="-my-1 w-full overflow-hidden py-1"
                        >
                          <div className="mb-2 text-xs font-medium text-white/50">
                            Listen back
                          </div>
                          <Playback info={playback} version={playbackVersion} />
                          <div className="mt-2">
                            <InsightsPanel
                              key={playbackVersion}
                              enabled={rescueAvailable}
                              build={() => {
                                const kinds: Record<string, number> = {}
                                for (const c of data.current.cuts) kinds[c.kind] = (kinds[c.kind] ?? 0) + 1
                                const cap = caption.get()
                                return {
                                  kinds,
                                  duration_s: playback.original_s,
                                  removed_s: stats.get()?.removed_s ?? 0,
                                  words: [],
                                  transcript: cap ? [...cap.lines, cap.partial].join(" ") : "",
                                }
                              }}
                            />
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              </GlassCard>
            </div>

            {/* Setup */}
            <motion.div variants={rise}>
              <GlassCard tilt={0} className="flex h-full flex-col gap-5 p-5 sm:p-6">
                <CardTitle icon={<IsoSetup />}>Setup</CardTitle>
                <FieldGroup>
                  <FormField>
                  <FieldLabel>Microphone</FieldLabel>
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
                  </FormField>
                  {inDev === BROWSER_MIC ? null : (
                  <FormField>
                    <FieldLabel>Send the cleaned voice to</FieldLabel>
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
                  </FormField>
                  )}
                  <FormField>
                  <FieldLabel>Detector</FieldLabel>
                  <ToggleGroup aria-label="Detector" className="flex-wrap" value={[detector]} onValueChange={(v) => { const n = v[0] as DetectorKind | undefined; if (n) change({ detector: n }) }} variant="outline" spacing={1}>
                    {detectorOptions(detectorList).map((d) => (
                      <ToggleGroupItem key={d.value} value={d.value}>{d.label}</ToggleGroupItem>
                    ))}
                  </ToggleGroup>
                  <DetectorNote info={detectorList.find((d) => d.kind === detector)} recommended={recommendedDet} />
                  </FormField>
                  {inDev === BROWSER_MIC && (
                  <FormField>
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <FieldLabel>Fluency aid</FieldLabel>
                        <p className="text-xs text-white/45">Use headphones. Many people find a short delay helps.</p>
                      </div>
                      <Switch
                        checked={aidOn}
                        onCheckedChange={(on) => {
                          setAidOn(on)
                          store("sc.aid", on ? "on" : "")
                        }}
                        aria-label="Hear your voice a little late"
                      />
                    </div>
                    {aidOn && (
                      <Slider
                        min={50}
                        max={200}
                        step={10}
                        value={[aidMs]}
                        onValueChange={(v) => {
                          const n = Array.isArray(v) ? v[0] : aidMs
                          setAidMs(n)
                          store("sc.aidms", String(n))
                        }}
                        aria-label="Fluency aid delay"
                      />
                    )}
                  </FormField>
                  )}
                </FieldGroup>
                <div
                  className={cn(
                    "mt-auto flex items-center justify-between gap-4 rounded-2xl border border-white/6 bg-white/[0.03] p-3.5",
                    hosted && "hidden" // a public server never stores recordings
                  )}
                >
                  <div className="text-sm">
                    <div className="font-medium">Save this session</div>
                  </div>
                  <Switch checked={save} onCheckedChange={setSave} aria-label="Save this session's audio" />
                </div>
              </GlassCard>
            </motion.div>

            {/* Fluency Mirror */}
            <motion.div variants={rise} className="md:col-span-2">
              <GlassCard tilt={0} className="flex h-full flex-col p-5 sm:p-6">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <CardTitle icon={<IsoMirror />}>Fluency Mirror</CardTitle>
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
                        whileTap={{ scale: 0.98 }}
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
                {captionsOn && (
                  <CaptionStrip
                    caption={caption}
                    rescue={{ enabled: rescueOn && rescueAvailable, loading: rescue.loading, words: rescue.words, error: rescue.error, onAsk: askForWord, onSay: ttsOn ? sayWord : undefined }}
                    label={demoOn ? `Captions · ${demo.state?.hear === "original" ? "original" : "clean voice"}` : "Live captions · what the listener hears"}
                  />
                )}
                <StatsRow stats={stats} running={active} />
              </GlassCard>
            </motion.div>

            {/* Context + voice notes */}
            <motion.div variants={rise} className="flex flex-col gap-4 sm:gap-5">
              <ContextCard
                searchOn={searchOn}
                rescueAvailable={rescueAvailable}
                rescueOn={rescueOn}
                onRescue={(on) => {
                  setRescueOn(on)
                  store("sc.rescue", on ? "on" : "")
                }}
              />
            </motion.div>
              </div>
            </TabsContent>

            <TabsContent value="recordings" className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <GlassCard tilt={0} className="flex flex-col gap-4 p-5 sm:p-6 md:col-span-2">
              <VoiceNoteCard
                enabled={notesOn}
                busy={running}
                onReady={(id, seconds) => {
                  setNoteSamples((ns) => [
                    ...ns,
                    { id, title: "Your voice note", who: "you", duration_s: seconds, engines: ["natural", "thorough", "live"] },
                  ])
                  demo.load(id, "natural")
                }}
              />
              {demo.state ? (
                <DemoBar
                  state={demo.state}
                  sample={[...demos.samples, ...noteSamples].find((s) => s.id === demo.state?.id)}
                  credit={
                    demo.state.id.startsWith("note-")
                      ? "Your recording stays on this computer and is forgotten within an hour."
                      : demos.credit
                  }
                  downloadUrl={
                    demo.state.id.startsWith("note-")
                      ? `/api/demo/${demo.state.id}/clean.wav?engine=${demo.state.engine}`
                      : `/api/demo/${demo.state.id}/clean.wav?engine=${demo.state.engine}`
                  }
                  progress={demo.progress}
                  onPlay={demo.play}
                  onPause={demo.pause}
                  onHear={demo.setHear}
                  onEngine={demo.setEngine}
                  onClose={demo.close}
                  extra={
                    demo.state.timeline && (
                      <>
                        {demo.state.timeline.text && (
                          <div className="flex flex-col gap-1.5">
                            <span className="text-xs text-white/45">Transcript: struck-through words were removed</span>
                            <TranscriptView text={demo.state.timeline.text} />
                          </div>
                        )}
                        <InsightsPanel
                          key={`${demo.state.id}-${demo.state.engine}`}
                          enabled={rescueAvailable}
                          build={() => insightsFromTimeline(demo.state!.timeline!)}
                        />
                        <CutReview
                          id={demo.state.id}
                          cuts={demo.state.timeline.cuts}
                          at={demo.state.t}
                          hear={demo.state.hear}
                          lagS={demo.state.engine === "live" ? demo.state.timeline.lag_ms / 1000 : 0}
                          browserStore={feedbackBrowser}
                        />
                      </>
                    )
                  }
                />
              ) : (
                <DemoPicker
                  samples={demos.samples}
                  onPick={(id) => demo.load(id, demos.samples.find((s) => s.id === id)?.engines[0] ?? "live")}
                />
              )}
            </GlassCard>
            {videoOn && <VideoCard busy={running} />}
            </TabsContent>
            {notesOn && (
              <TabsContent value="dictate">
                <DictatePanel enabled={notesOn} busy={running} />
              </TabsContent>
            )}
            {ttsOn && (
              <TabsContent value="stage">
                <StagePanel running={running} caption={caption} data={data} onToggle={toggle} />
              </TabsContent>
            )}
            {interviewOn && (
              <TabsContent value="practice">
                <PracticePanel
                  running={running}
                  caption={caption}
                  playback={playback}
                  playbackVersion={playbackVersion}
                  insights={rescueAvailable}
                  onStart={() => start()}
                  onStop={() => stop()}
                />
              </TabsContent>
            )}
            {!hosted && (
              <TabsContent value="journal">
                <JournalPanel nemotron={rescueAvailable} visible={tab === "journal"} />
              </TabsContent>
            )}
          </Tabs>

          <motion.div variants={rise} className="mt-4 sm:mt-5">
            <HowItWorks nemotron={rescueAvailable} />
          </motion.div>

          <motion.footer
            variants={rise}
            className="mt-8 flex items-start justify-center gap-2 px-2 text-center text-xs text-white/35 sm:items-center"
          >
            <Lock className="mt-0.5 size-3.5 shrink-0 sm:mt-0" />
            {hosted
              ? "Audio is processed on StutterClear's server and is not stored. Recordings you clean are kept in memory for an hour, then forgotten. Live turns last up to 5 minutes, so everyone gets one."
              : "Audio is processed on this laptop. Nothing is recorded unless you switch saving on."}
          </motion.footer>
        </div>

        <SessionToast toast={toast} onClose={() => setToast(null)} />
        <Toaster />
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
  signals, data, running, starting, demo, connected, mode,
}: {
  signals: RefObject<Signals>
  data: RefObject<LiveData>
  running: boolean
  starting: boolean
  demo: boolean
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
      else if (!running) next = { text: "", tone: "idle" }
      else if (starting) next = { text: "Opening the microphone…", tone: "idle" }
      else if (!data.current.heardSound && performance.now() - data.current.lastSoundAt > 4000)
        next = { text: "Waiting for your voice… (nothing from the mic yet)", tone: "warn" }
      else if (s.cut > 0.2) {
        const kind = (KIND_LABEL[data.current.lastCutKind] ?? "stutter").toLowerCase()
        next = { text: `Removed a ${kind}`, tone: "cut" }
      } else if (s.block > 0.25) next = { text: "Pause kept. Take your time.", tone: "block" }
      else if (demo) next = { text: "Playing a real recording…", tone: "live" }
      else if (mode === "off") next = { text: "Mirror only. Nothing is cut.", tone: "live" }
      else next = { text: "Listening…", tone: "live" }
      setStatus((cur) => (cur.text === next.text ? cur : next))
    }, 100)
    return () => window.clearInterval(id)
  }, [signals, data, running, starting, demo, connected, mode])

  if (!status.text) return null
  return (
    <div className="h-6 text-sm font-medium" style={{ color: TONE[status.tone].color }}>
      {status.text}
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
      <IsoBadge>{icon}</IsoBadge>
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
      <SelectTrigger className="h-8 w-full min-w-0 cursor-pointer rounded-lg border-white/10 bg-white/[0.04] px-2.5 text-sm transition-colors hover:border-white/20 hover:bg-white/[0.08] data-[size=default]:h-8 dark:bg-white/[0.04] dark:hover:bg-white/[0.08]">
        <SelectValue placeholder={placeholder} className="min-w-0 truncate" />
      </SelectTrigger>
      <SelectContent
        alignItemWithTrigger={false}
        align="start"
        sideOffset={6}
        className="glass-blur rounded-lg border border-white/10 bg-[#10131c] p-1 shadow-lg"
      >
        {items.map((it) => (
          <SelectItem key={it.value} value={it.value} className="cursor-pointer rounded-md py-1.5 pr-7 text-sm whitespace-normal">
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
  connected, running, demo, detector, stats,
}: {
  connected: boolean
  running: boolean
  demo: boolean
  detector: string
  stats: Store<Stats | null>
}) {
  const live = useStore(stats, (s) => s?.detector ?? "")
  const det = detector || live
  const label = !connected
    ? "Engine offline"
    : running
      ? `Live · ${det === "teacher" ? "Best (GPU)" : det === "student" ? "AI model" : "Rule-based"}`
      : demo
        ? "Demo · real recording"
        : "Engine ready"
  const dot = !connected ? "bg-amber-300" : running ? "bg-rose-400" : demo ? "bg-teal-300" : "bg-emerald-300"
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

function insightsFromTimeline(tl: DemoTimeline): InsightsBody {
  const kinds: Record<string, number> = {}
  for (const c of tl.cuts) kinds[c.kind] = (kinds[c.kind] ?? 0) + 1
  return {
    kinds,
    duration_s: tl.original_s,
    removed_s: tl.removed_s,
    words: tl.cuts.map((c) => c.word ?? "").filter(Boolean),
    transcript: (tl.captions?.original ?? []).map((w) => w.w).join(" "),
  }
}

function ContextCard({
  searchOn,
  rescueAvailable,
  rescueOn,
  onRescue,
}: {
  searchOn: boolean
  rescueAvailable: boolean
  rescueOn: boolean
  onRescue: (on: boolean) => void
}) {
  const [text, setText] = useState(() => stored("ctx", ""))
  const [saved, setSaved] = useState(false)
  const timer = useRef<number | undefined>(undefined)
  const [situation, setSituation] = useState("")
  const [agent, setAgent] = useState<{ loading: boolean; sources: string[]; error?: string }>({ loading: false, sources: [] })
  const update = (v: string) => {
    setText(v)
    store("ctx", v)
    setSaved(true)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setSaved(false), 1200)
  }
  // Context agent: Nemotron (and Tavily search, if set up) prepares the card for a situation.
  const prepare = async () => {
    if (!situation.trim()) return
    setAgent({ loading: true, sources: [] })
    try {
      const r = await prepareContext(situation, text)
      update(text.trim() ? `${r.card}\n\n${text.trim()}` : r.card)
      setAgent({ loading: false, sources: r.sources })
    } catch (e) {
      setAgent({ loading: false, sources: [], error: (e as Error).message })
    }
  }
  return (
    <GlassCard tilt={0} className="flex flex-col gap-3 p-5 sm:p-6">
      <div className="flex items-center justify-between">
        <CardTitle icon={<IsoContext />}>Context card</CardTitle>
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
      {rescueAvailable && (
        <div className="flex flex-col gap-1.5">
          <div className="flex gap-2">
            <input
              value={situation}
              onChange={(e) => setSituation(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && prepare()}
              placeholder="Where will you speak? e.g. Interview at NVIDIA on Friday"
              className="min-w-0 flex-1 rounded-xl border border-white/8 bg-white/[0.03] px-3 py-2 text-base text-white/90 transition-[border-color,background-color] duration-300 outline-none placeholder:text-white/30 hover:border-white/15 focus:border-violet-300/40 sm:text-sm"
            />
            <motion.button
              type="button"
              onClick={prepare}
              disabled={agent.loading || !situation.trim()}
              whileTap={{ scale: 0.98 }}
              transition={SPRING.press}
              className="inline-flex shrink-0 cursor-pointer items-center gap-1.5 rounded-xl border border-violet-300/25 bg-violet-300/10 px-3 text-xs text-violet-100 hover:bg-violet-300/20 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {agent.loading ? "Preparing…" : "Prepare"}
            </motion.button>
          </div>
          <div className="text-[11px] text-white/35">
            {agent.error
              ? <span className="text-rose-300/80">{agent.error}</span>
              : agent.sources.length
                ? `Nemotron, with Tavily search: ${agent.sources.map((u) => new URL(u).hostname.replace(/^www\./, "")).join(", ")}`
                : `Nemotron fills the card with names and terms you may need${searchOn ? ", searching the web with Tavily" : ""}.`}
          </div>
        </div>
      )}
      <textarea
        value={text}
        onChange={(e) => update(e.target.value)}
        rows={4}
        placeholder="My name is Sam. Talking to Dr. Mehta about my project, StutterClear."
        className="w-full resize-none rounded-2xl border border-white/8 bg-white/[0.03] p-3.5 text-base leading-relaxed text-white/90 transition-[border-color,background-color,box-shadow] duration-500 outline-none placeholder:text-white/30 hover:border-white/15 hover:duration-150 focus:border-sky-300/40 focus:bg-white/[0.05] focus:shadow-[0_0_0_4px_rgb(125_211_252/0.12)] focus:duration-200 sm:text-sm"
      />
      <div className="text-xs text-white/40">Names and topics you might say. Stays in this browser.</div>
      <div className="flex items-center justify-between gap-4 rounded-2xl border border-white/6 bg-white/[0.03] p-3.5">
        <div className="text-sm">
          <div className="font-medium">Word rescue · Nemotron</div>
          <div className="text-xs leading-relaxed text-white/45">
            {rescueAvailable
              ? "When you get stuck, suggests the word you mean. Sends your last sentences and this card (never audio) to NVIDIA Nemotron on Nebius."
              : "Needs NEBIUS_API_KEY set where the app runs."}
          </div>
        </div>
        <Switch checked={rescueOn && rescueAvailable} onCheckedChange={(v) => rescueAvailable && onRescue(v)} aria-label="Word rescue" disabled={!rescueAvailable} />
      </div>
    </GlassCard>
  )
}


// What each detector is and how it measured on held-out speakers: the page shows its evidence.
function DetectorNote({ info, recommended }: { info: DetectorInfo | undefined; recommended: DetectorKind }) {
  if (!info) return null
  const q = info.quality
  return (
    <div className="grid">
      <AnimatePresence initial={false}>
        <motion.div
          key={info.kind}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.25, ease: EASE_OUT }}
          className="text-xs leading-relaxed text-white/50 [grid-area:1/1]"
        >
          {DETECTOR_NOTE[info.kind]}{" "}
          {q ? (
            <>
              Unseen speakers: removes <span className="text-white/80">{q.removed_pct}%</span> of stutters, loses{" "}
              <span className={q.words_lost_pct > 3 ? "text-amber-200/90" : "text-white/80"}>{q.words_lost_pct}%</span> of
              words.
            </>
          ) : (
            "Not measured yet."
          )}
          {info.kind === recommended && <span className="ml-1 text-emerald-300/80">Recommended.</span>}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

function detectorOptions(list: DetectorInfo[]): { value: DetectorKind; label: string }[] {
  const avail = list.filter((d) => d.available)
  return (avail.length ? avail : [{ kind: "heuristic", label: "Rule-based" } as DetectorInfo]).map((d) => ({
    value: d.kind,
    label: d.label,
  }))
}
