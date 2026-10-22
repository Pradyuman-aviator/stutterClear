import { useEffect, useEffectEvent, useState, useCallback, useRef } from "react"
import { motion, AnimatePresence } from "motion/react"
import {
  Sparkles,
  ChevronRight,
  ChevronLeft,
  X,
  Trophy,
  Zap,
  CheckCircle2,
  Gamepad2,
  Volume2,
  VolumeX,
  Target,
  ArrowRight,
  RotateCcw,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export interface TutorialStep {
  id: string
  targetSelector: string
  title: string
  tag: string
  actionPrompt: string
  description: string
  tab?: string
  position?: "top" | "bottom" | "left" | "right" | "auto"
}

export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    id: "voice-orb",
    targetSelector: "#tutorial-voice-orb",
    title: "Neural Voice Core",
    tag: "AUDIO SENSOR",
    actionPrompt: "Tap or click the orb to trigger live capture",
    description:
      "This 3D sphere is your visual fluency hub. Powered by WebGL harmonics, it breathes and reacts in real-time as you speak, lighting up with electric pulses whenever stutter cutting engages.",
    tab: "calls",
  },
  {
    id: "start-btn",
    targetSelector: "#tutorial-start-btn",
    title: "Live Speech Engine",
    tag: "MAIN TRIGGER",
    actionPrompt: "Press here (or tap Spacebar) to go live",
    description:
      "Toggle real-time speech processing on and off. When live, StutterClear listens to your microphone, cuts stutters a fraction of a second behind you, and sends on your cleaned voice.",
    tab: "calls",
  },
  {
    id: "modes",
    targetSelector: "#tutorial-modes",
    title: "Correction Intensity",
    tag: "PROFILE MATRIX",
    actionPrompt: "Select Off, Gentle, or Strong",
    description:
      "• Gentle: Removes clear repeats, about 250 ms behind you (0.5 s with the Teacher model).\n• Strong: Removes the most stutters, fillers too; may clip a word now and then.\n• Off: Mirror only. Nothing is cut, so you can watch your speech as it is.",
    tab: "calls",
  },
  {
    id: "demo-btn",
    targetSelector: "#tutorial-demo-btn",
    title: "Audio Simulation Deck",
    tag: "INSTANT DEMO",
    actionPrompt: "Hear sample clips without setting up a mic",
    description:
      "No microphone plugged in? Open the sample clips in Recordings and hear each one before and after cleaning, before you try your own voice.",
    tab: "calls",
  },
  {
    id: "mic-select",
    targetSelector: "#tutorial-mic-select",
    title: "Microphone Input",
    tag: "INPUT ROUTING",
    actionPrompt: "Choose your mic hardware or browser stream",
    description:
      "Select your physical desktop microphone, USB headset, or choose 'This browser's mic' to stream live audio directly via WebSocket with zero drivers required.",
    tab: "calls",
  },
  {
    id: "out-select",
    targetSelector: "#tutorial-out-select",
    title: "Call Output Routing",
    tag: "VIRTUAL CABLE",
    actionPrompt: "Route cleaned voice into Zoom, Teams, Meet or Discord",
    description:
      "Send your cleaned speech to a virtual microphone (like VB-Audio Cable or BlackHole). In Zoom or Google Meet, choose that cable as your mic—your audience only hears your fluent voice!",
    tab: "calls",
  },
  {
    id: "detector-select",
    targetSelector: "#tutorial-detector-select",
    title: "Detection Neural Model",
    tag: "AI ARCHITECTURE",
    actionPrompt: "Pick the model this computer can run",
    description:
      "• Teacher: the big model, run live on a GPU.\n• Student: a small model that runs on any computer.\n• Rule-based: light rules, safe but removes little.\nBelow the choice: how each one scored on speakers it never heard.",
    tab: "calls",
  },
  {
    id: "fluency-mirror",
    targetSelector: "#tutorial-fluency-mirror",
    title: "4-Layer Fluency Mirror",
    tag: "ACOUSTIC RADAR",
    actionPrompt: "Hover over the color pills to isolate layers",
    description:
      "Your voice live, in four layers: Blue is what you said, Pink is a stutter removed, Purple is a pause kept, and Teal is what the listener hears.",
    tab: "calls",
  },
  {
    id: "tabs-list",
    targetSelector: "#tutorial-tabs-list",
    title: "Navigation Deck",
    tag: "COMMAND CENTER",
    actionPrompt: "Switch between Calls, Recordings, Dictate, On stage & Practice",
    description:
      "Explore StutterClear's full arsenal: 'Recordings' cleans voice notes and sample clips, 'Dictate' turns speech into text without the stutters, 'On stage' follows your script as you speak, and 'Practice' runs a mock interview with an AI interviewer. The tabs shown depend on what this server can run.",
    tab: "calls",
  },
]

// Synthesizes crisp retro-futuristic game sound effects via Web Audio API
class GameAudio {
  private ctx: AudioContext | null = null
  public enabled = true

  private getContext() {
    if (typeof window === "undefined") return null
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      if (AudioCtx) this.ctx = new AudioCtx()
    }
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume().catch(() => {})
    }
    return this.ctx
  }

  // Futuristic blip when switching steps
  stepBlip() {
    if (!this.enabled) return
    try {
      const ctx = this.getContext()
      if (!ctx) return
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.type = "sine"
      const now = ctx.currentTime
      osc.frequency.setValueAtTime(600, now)
      osc.frequency.exponentialRampToValueAtTime(1100, now + 0.08)
      gain.gain.setValueAtTime(0.04, now)
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.09)
      osc.connect(gain)
      gain.connect(ctx.destination)
      osc.start(now)
      osc.stop(now + 0.09)
    } catch {
      // Audio autoplay policy or unavailable
    }
  }

  // Level Up / Victory Arpeggio
  victoryFanfare() {
    if (!this.enabled) return
    try {
      const ctx = this.getContext()
      if (!ctx) return
      const notes = [523.25, 659.25, 783.99, 1046.5] // C5, E5, G5, C6
      const now = ctx.currentTime
      notes.forEach((freq, idx) => {
        const osc = ctx.createOscillator()
        const gain = ctx.createGain()
        osc.type = "triangle"
        const start = now + idx * 0.08
        osc.frequency.setValueAtTime(freq, start)
        gain.gain.setValueAtTime(0.05, start)
        gain.gain.exponentialRampToValueAtTime(0.001, start + 0.25)
        osc.connect(gain)
        gain.connect(ctx.destination)
        osc.start(start)
        osc.stop(start + 0.26)
      })
    } catch {
      // Audio autoplay policy
    }
  }

  // Skip / Exit descending tone
  dismiss() {
    if (!this.enabled) return
    try {
      const ctx = this.getContext()
      if (!ctx) return
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.type = "sine"
      const now = ctx.currentTime
      osc.frequency.setValueAtTime(750, now)
      osc.frequency.exponentialRampToValueAtTime(320, now + 0.12)
      gain.gain.setValueAtTime(0.03, now)
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.13)
      osc.connect(gain)
      gain.connect(ctx.destination)
      osc.start(now)
      osc.stop(now + 0.13)
    } catch {
      // Audio autoplay policy
    }
  }
}

const sfx = new GameAudio()

interface GameTutorialProps {
  isOpen: boolean
  onClose: () => void
  onEnsureTab?: (tabName: string) => void
  userName?: string
}

export function GameTutorial({ isOpen, onClose, onEnsureTab, userName }: GameTutorialProps) {
  const [currentStepIdx, setCurrentStepIdx] = useState(0)
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null)
  const [soundOn, setSoundOn] = useState(true)
  const [isCompleted, setIsCompleted] = useState(false)
  // Only the steps whose target is on the page: the demo button hides while live, the output
  // picker when the browser's mic is the input (always, on the hosted demo).
  const [steps, setSteps] = useState(TUTORIAL_STEPS)
  const step = steps[Math.min(currentStepIdx, steps.length - 1)]
  const xp = (currentStepIdx + 1) * 50
  const prevStepIdx = useRef(currentStepIdx)

  sfx.enabled = soundOn

  // App passes a new function on every render; reading it through an effect event keeps the
  // effects below from re-running (and re-scrolling) each time the app re-renders.
  const ensureTab = useEffectEvent((tab: string) => onEnsureTab?.(tab))

  // Every opening starts a fresh run from the first step.
  useEffect(() => {
    if (!isOpen) return
    setCurrentStepIdx(0)
    prevStepIdx.current = 0
    setIsCompleted(false)
    ensureTab("calls")
    const timer = setTimeout(() => {
      const present = TUTORIAL_STEPS.filter((s) => document.querySelector(s.targetSelector))
      setSteps(present.length ? present : TUTORIAL_STEPS)
    }, 100)
    return () => clearTimeout(timer)
  }, [isOpen])

  // Synchronize target bounding rect
  const updateRect = useCallback(() => {
    if (!step) return
    const el = document.querySelector(step.targetSelector)
    if (el) {
      const rect = el.getBoundingClientRect()
      setTargetRect(rect)
    } else {
      setTargetRect(null)
    }
  }, [step])

  // Scroll target element into view smoothly if offscreen
  useEffect(() => {
    if (!isOpen || isCompleted || !step) return

    if (step.tab) {
      ensureTab(step.tab)
    }

    const timer = setTimeout(() => {
      const el = document.querySelector(step.targetSelector)
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" })
        const rect = el.getBoundingClientRect()
        setTargetRect(rect)
      } else {
        setTargetRect(null)
      }
    }, 150)

    return () => clearTimeout(timer)
  }, [isOpen, currentStepIdx, step, isCompleted])

  // Update rect on scroll / resize
  useEffect(() => {
    if (!isOpen || isCompleted) return
    window.addEventListener("resize", updateRect)
    window.addEventListener("scroll", updateRect, { passive: true })
    const interval = setInterval(updateRect, 300)
    return () => {
      window.removeEventListener("resize", updateRect)
      window.removeEventListener("scroll", updateRect)
      clearInterval(interval)
    }
  }, [isOpen, isCompleted, updateRect])

  // Play blip sound on step change
  useEffect(() => {
    if (isOpen && !isCompleted && prevStepIdx.current !== currentStepIdx) {
      sfx.stepBlip()
      prevStepIdx.current = currentStepIdx
    }
  }, [currentStepIdx, isOpen, isCompleted])

  const handleNext = () => {
    if (currentStepIdx < steps.length - 1) {
      setCurrentStepIdx((i) => i + 1)
    } else {
      // Completed!
      sfx.victoryFanfare()
      setIsCompleted(true)
      try {
        localStorage.setItem("sc.tutorial_completed", "true")
      } catch {
        // storage disabled
      }
    }
  }

  const handlePrev = () => {
    if (currentStepIdx > 0) {
      setCurrentStepIdx((i) => i - 1)
    }
  }

  const handleSkip = () => {
    sfx.dismiss()
    try {
      localStorage.setItem("sc.tutorial_completed", "true")
    } catch {
      // storage disabled
    }
    onClose()
  }

  const handleFinish = () => {
    sfx.dismiss()
    onClose()
  }

  const handleRestart = () => {
    setIsCompleted(false)
    setCurrentStepIdx(0)
    sfx.stepBlip()
  }

  // Keyboard navigation. Captured before the app's own keys and kept from them: Esc would also
  // pause the cleaning mid-session. Enter on a focused button presses that button.
  useEffect(() => {
    if (!isOpen) return
    const handleKeyDown = (e: KeyboardEvent) => {
      const onControl = (e.target as Element | null)?.closest?.("button, a, input, textarea, select")
      let handled = true
      if (e.key === "Escape") {
        if (isCompleted) handleFinish()
        else handleSkip()
      } else if (e.key === "Enter" && !onControl) {
        if (isCompleted) handleFinish()
        else handleNext()
      } else if (e.key === "ArrowRight" && !isCompleted) {
        handleNext()
      } else if (e.key === "ArrowLeft" && !isCompleted) {
        handlePrev()
      } else {
        handled = false
      }
      if (handled) {
        e.preventDefault()
        e.stopPropagation()
      }
    }
    window.addEventListener("keydown", handleKeyDown, true)
    return () => window.removeEventListener("keydown", handleKeyDown, true)
  })

  if (!isOpen) return null

  // Calculate HUD positioning based on target element position
  const getHudStyle = () => {
    if (!targetRect || typeof window === "undefined") {
      return { top: "50%", left: "50%", transform: "translate(-50%, -50%)" }
    }
    const vw = window.innerWidth
    const vh = window.innerHeight

    // On mobile screens, dock HUD near bottom with safe margins
    if (vw < 768) {
      return {
        bottom: "16px",
        left: "16px",
        right: "16px",
        maxWidth: "calc(100vw - 32px)",
      }
    }

    const spaceBelow = vh - targetRect.bottom
    const spaceAbove = targetRect.top

    // Target is in top portion: place HUD below
    if (spaceBelow >= 320 || spaceBelow >= spaceAbove) {
      const topPos = Math.min(targetRect.bottom + 20, vh - 340)
      const leftPos = Math.max(24, Math.min(targetRect.left, vw - 480))
      return {
        top: `${topPos}px`,
        left: `${leftPos}px`,
        maxWidth: "460px",
      }
    }

    // Target is in bottom portion: place HUD above
    const topPos = Math.max(20, targetRect.top - 330)
    const leftPos = Math.max(24, Math.min(targetRect.left, vw - 480))
    return {
      top: `${topPos}px`,
      left: `${leftPos}px`,
      maxWidth: "460px",
    }
  }

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 overflow-hidden pointer-events-auto">
        {/* Darkened Game Backdrop. With a target, the spotlight's shadow does the darkening:
            a dark, blurred layer here would cover the cutout and hide the very element shown. */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          className={cn("absolute inset-0", targetRect && !isCompleted ? "bg-transparent" : "bg-slate-950/80 backdrop-blur-sm")}
          onClick={handleNext}
        />

        {/* Animated Target Spotlight & Reticle */}
        {targetRect && !isCompleted && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{
              opacity: 1,
              scale: 1,
              top: targetRect.top - 8,
              left: targetRect.left - 8,
              width: targetRect.width + 16,
              height: targetRect.height + 16,
            }}
            transition={{ type: "spring", stiffness: 350, damping: 30 }}
            className="absolute pointer-events-none z-50 rounded-2xl"
            style={{
              boxShadow: "0 0 0 9999px rgba(3, 7, 18, 0.78), 0 0 35px rgba(56, 189, 248, 0.6), inset 0 0 20px rgba(56, 189, 248, 0.35)",
              border: "2px solid rgba(56, 189, 248, 0.8)",
            }}
          >
            {/* Cyberpunk corner reticles */}
            <span className="absolute -top-1.5 -left-1.5 size-3 border-t-2 border-l-2 border-sky-300" />
            <span className="absolute -top-1.5 -right-1.5 size-3 border-t-2 border-r-2 border-sky-300" />
            <span className="absolute -bottom-1.5 -left-1.5 size-3 border-b-2 border-l-2 border-sky-300" />
            <span className="absolute -bottom-1.5 -right-1.5 size-3 border-b-2 border-r-2 border-sky-300" />

            {/* Glowing radar ping */}
            <span className="absolute -inset-1 rounded-2xl animate-ping border border-sky-400/40 pointer-events-none" />

            {/* Floating Target Pin Callout */}
            <div className="absolute -top-7 left-1/2 -translate-x-1/2 flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-sky-500/90 text-slate-950 font-mono text-[10px] font-bold tracking-wider uppercase whitespace-nowrap shadow-lg shadow-sky-500/50">
              <Target className="size-3 animate-spin" style={{ animationDuration: "6s" }} />
              TARGET ACTIVE
            </div>
          </motion.div>
        )}

        {/* Quest HUD Dialogue Box */}
        {!isCompleted ? (
          <motion.div
            key={currentStepIdx}
            initial={{ opacity: 0, y: 15, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -15, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 380, damping: 28 }}
            className="fixed z-50"
            style={getHudStyle()}
          >
            <div className="relative rounded-2xl border border-sky-500/30 bg-slate-950/90 p-5 shadow-2xl shadow-sky-950/80 backdrop-blur-xl sm:p-6 text-white overflow-hidden ring-1 ring-white/10">
              {/* Sci-fi top glowing bar */}
              <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-sky-400 via-indigo-500 to-amber-400" />

              {/* Header: Mission, XP, and Mute/Skip */}
              <div className="flex items-center justify-between gap-3 pb-3 border-b border-white/10">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 rounded-md bg-sky-500/15 border border-sky-400/30 px-2 py-0.5 font-mono text-xs font-semibold text-sky-300 whitespace-nowrap">
                    <Gamepad2 className="size-3.5" />
                    MISSION {String(currentStepIdx + 1).padStart(2, "0")} / {String(steps.length).padStart(2, "0")}
                  </span>
                  <span className="inline-flex items-center gap-1 rounded-md bg-amber-500/15 border border-amber-400/30 px-2 py-0.5 font-mono text-xs font-semibold text-amber-300 whitespace-nowrap">
                    <Zap className="size-3 fill-current" />
                    +{xp} XP
                  </span>
                </div>

                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setSoundOn(!soundOn)}
                    title={soundOn ? "Mute audio cues" : "Enable sound cues"}
                    className="p-1 rounded-lg text-white/50 hover:text-white hover:bg-white/10 transition-colors"
                  >
                    {soundOn ? <Volume2 className="size-4" /> : <VolumeX className="size-4 text-white/30" />}
                  </button>
                  <button
                    type="button"
                    onClick={handleSkip}
                    title="Skip full tutorial"
                    className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-xs text-white/60 hover:text-rose-300 hover:bg-rose-500/10 transition-colors"
                  >
                    <X className="size-3.5" />
                    Skip
                  </button>
                </div>
              </div>

              {/* Mission Body */}
              <div className="mt-3.5">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[11px] font-bold tracking-widest text-sky-400 uppercase">
                    [{step.tag}]
                  </span>
                  {userName && (
                    <span className="text-[11px] text-white/40">
                      Operated by {userName}
                    </span>
                  )}
                </div>

                <h2 className="text-xl font-bold tracking-tight text-white mt-1 flex items-center gap-2">
                  {step.title}
                </h2>

                <p className="mt-2 text-sm text-slate-300/90 leading-relaxed whitespace-pre-line font-normal">
                  {step.description}
                </p>

                {/* Action Prompt Pill */}
                <div className="mt-3.5 flex items-center gap-2 rounded-xl bg-sky-950/40 border border-sky-400/20 px-3 py-2 text-xs text-sky-200">
                  <Sparkles className="size-3.5 shrink-0 text-amber-300 animate-pulse" />
                  <span className="font-medium">{step.actionPrompt}</span>
                </div>
              </div>

              {/* Progress Track */}
              <div className="mt-4 pt-3 border-t border-white/10 flex items-center justify-between gap-3">
                <div className="flex items-center gap-1">
                  {steps.map((s, idx) => (
                    <button
                      key={s.id}
                      type="button"
                      onClick={() => setCurrentStepIdx(idx)}
                      className={cn(
                        "h-1.5 rounded-full transition-all duration-300 cursor-pointer",
                        idx === currentStepIdx
                          ? "w-6 bg-sky-400 shadow-sm shadow-sky-400"
                          : idx < currentStepIdx
                          ? "w-2.5 bg-sky-600"
                          : "w-2.5 bg-white/15 hover:bg-white/30"
                      )}
                      aria-label={`Go to step ${idx + 1}: ${s.title}`}
                    />
                  ))}
                </div>

                {/* Navigation Buttons */}
                <div className="flex items-center gap-2">
                  {currentStepIdx > 0 && (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={handlePrev}
                      className="h-8 border-white/15 bg-white/[0.04] text-xs text-white/80 hover:bg-white/10 hover:text-white"
                    >
                      <ChevronLeft className="size-3.5" />
                      Prev
                    </Button>
                  )}
                  <Button
                    type="button"
                    size="sm"
                    onClick={handleNext}
                    className="h-8 rounded-lg bg-gradient-to-r from-sky-400 to-indigo-500 font-semibold text-slate-950 text-xs hover:from-sky-300 hover:to-indigo-400 shadow-lg shadow-sky-500/25 px-3.5"
                  >
                    {currentStepIdx === steps.length - 1 ? (
                      <>
                        Complete Mission
                        <CheckCircle2 className="size-3.5 ml-1" />
                      </>
                    ) : (
                      <>
                        Next Target
                        <ChevronRight className="size-3.5 ml-0.5" />
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>
        ) : (
          /* Victory / Level Up Screen */
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-auto"
          >
            <div className="relative w-full max-w-md rounded-3xl border border-sky-400/40 bg-slate-950/95 p-6 sm:p-8 text-center text-white shadow-2xl shadow-sky-950/90 backdrop-blur-2xl ring-1 ring-white/10 overflow-hidden">
              {/* Background celebration glow */}
              <div className="absolute -top-24 left-1/2 -translate-x-1/2 size-48 rounded-full bg-sky-500/20 blur-3xl pointer-events-none" />
              <div className="absolute -bottom-24 left-1/2 -translate-x-1/2 size-48 rounded-full bg-indigo-500/20 blur-3xl pointer-events-none" />

              <div className="relative z-10 flex flex-col items-center">
                <div className="relative flex items-center justify-center size-20 rounded-2xl bg-gradient-to-br from-amber-400 to-amber-600 shadow-xl shadow-amber-500/30 mb-4 animate-bounce" style={{ animationDuration: "2s" }}>
                  <Trophy className="size-10 text-slate-950" />
                  <span className="absolute -top-1.5 -right-1.5 size-6 rounded-full bg-sky-400 text-slate-950 font-mono text-[10px] font-black flex items-center justify-center shadow-md">
                    LV2
                  </span>
                </div>

                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/15 border border-sky-400/30 font-mono text-xs font-semibold text-sky-300 uppercase tracking-widest mb-2">
                  <Sparkles className="size-3.5 fill-current" />
                  Training Completed
                </div>

                <h3 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
                  {userName ? `Congratulations, ${userName}!` : "Mission Accomplished!"}
                </h3>
                <p className="mt-2 text-sm text-slate-300 leading-relaxed">
                  You know your way around StutterClear. Press Start (or Space) whenever you're ready to talk.
                </p>

                {/* Gamified Stat Pill */}
                <div className="mt-5 grid grid-cols-3 gap-2 w-full rounded-2xl border border-white/10 bg-white/[0.04] p-3 text-center">
                  <div>
                    <div className="text-[11px] font-medium text-white/50">XP Gained</div>
                    <div className="text-lg font-bold text-amber-300 font-mono">+{steps.length * 50} XP</div>
                  </div>
                  <div>
                    <div className="text-[11px] font-medium text-white/50">Rank</div>
                    <div className="text-lg font-bold text-sky-300 font-mono">Pilot</div>
                  </div>
                  <div>
                    <div className="text-[11px] font-medium text-white/50">Missions</div>
                    <div className="text-lg font-bold text-emerald-300 font-mono">{steps.length}/{steps.length}</div>
                  </div>
                </div>

                {/* Buttons */}
                <div className="mt-6 flex flex-col sm:flex-row items-center gap-3 w-full">
                  <Button
                    type="button"
                    onClick={handleFinish}
                    className="w-full h-10 rounded-xl bg-gradient-to-r from-sky-400 to-indigo-500 font-semibold text-slate-950 hover:from-sky-300 hover:to-indigo-400 shadow-lg shadow-sky-500/25"
                  >
                    Start Speaking Live
                    <ArrowRight className="size-4 ml-1.5" />
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleRestart}
                    className="w-full sm:w-auto h-10 rounded-xl border-white/15 bg-white/[0.04] text-xs text-white/70 hover:bg-white/10 hover:text-white"
                  >
                    <RotateCcw className="size-3.5 mr-1" />
                    Replay
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </AnimatePresence>
  )
}
