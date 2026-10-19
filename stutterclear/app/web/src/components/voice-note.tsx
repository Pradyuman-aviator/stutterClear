import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { LoaderCircle, Mic, Square } from "lucide-react"
import { IsoBadge, IsoNote } from "@/components/iso-icon"
import { GlassCard } from "@/components/glass-card"
import { createNote, getNote, type NoteStatus } from "@/lib/api"
import { openMic } from "@/lib/browser-mic"
import { SPRING } from "@/lib/motion"
import { toWav16k } from "@/lib/wav"
import { cn } from "@/lib/utils"

const MAX_S = 180
const STEP: Record<NoteStatus["status"], string> = {
  queued: "Waiting for the models…",
  listening: "Finding the stutters…",
  transcribing: "Writing down every word…",
  done: "Done",
  error: "",
}
const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`

type Phase =
  | { kind: "idle" }
  | { kind: "recording"; started: number }
  | { kind: "preparing" }
  | { kind: "cleaning"; step: NoteStatus["status"] }
  | { kind: "error"; message: string }

// Record or upload a message; it is cleaned offline (teacher + word-aligned cuts) and opens in the
// player above. Nothing is saved: the server keeps it in memory for an hour.
export function VoiceNoteCard({
  enabled,
  busy,
  onReady,
}: {
  enabled: boolean
  busy: boolean // a live session owns the mic
  onReady: (id: string, seconds: number) => void
}) {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" })
  const [elapsed, setElapsed] = useState(0)
  const recorder = useRef<MediaRecorder | null>(null)
  const ring = useRef<HTMLSpanElement | null>(null)
  const stopMeter = useRef<() => void>(() => undefined)
  const fileInput = useRef<HTMLInputElement>(null)

  useEffect(() => () => stopMeter.current(), [])

  const clean = async (blob: Blob) => {
    setPhase({ kind: "preparing" })
    try {
      const { wav, seconds } = await toWav16k(blob)
      if (seconds < 1) throw new Error("That recording is shorter than a second.")
      if (seconds > MAX_S) throw new Error(`Voice notes can be up to ${MAX_S / 60} minutes long.`)
      const { id } = await createNote(wav)
      setPhase({ kind: "cleaning", step: "queued" })
      for (;;) {
        await new Promise((r) => setTimeout(r, 600))
        const s = await getNote(id)
        if (s.status === "error") throw new Error(s.error ?? "Cleaning failed.")
        if (s.status === "done") {
          setPhase({ kind: "idle" })
          onReady(id, s.duration_s)
          return
        }
        setPhase({ kind: "cleaning", step: s.status })
      }
    } catch (e) {
      setPhase({ kind: "error", message: (e as Error).message })
    }
  }

  const record = async () => {
    let stream: MediaStream
    try {
      stream = await openMic()
    } catch (e) {
      setPhase({ kind: "error", message: (e as Error).message })
      return
    }
    const chunks: Blob[] = []
    const rec = new MediaRecorder(stream)
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
    rec.onstop = () => {
      stopMeter.current()
      stream.getTracks().forEach((t) => t.stop())
      clean(new Blob(chunks, { type: rec.mimeType }))
    }
    // Level ring: swells with the voice while recording.
    const ctx = new AudioContext()
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 512
    ctx.createMediaStreamSource(stream).connect(analyser)
    const data = new Float32Array(analyser.fftSize)
    const started = performance.now()
    let raf = 0
    const tick = () => {
      analyser.getFloatTimeDomainData(data)
      let peak = 0
      for (const v of data) peak = Math.max(peak, Math.abs(v))
      const level = Math.min(1, Math.max(0, (20 * Math.log10(peak + 1e-6) + 55) / 50))
      if (ring.current) ring.current.style.transform = `scale(${1 + level * 0.45})`
      const s = (performance.now() - started) / 1000
      setElapsed(s)
      if (s >= MAX_S && rec.state === "recording") rec.stop()
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    stopMeter.current = () => {
      cancelAnimationFrame(raf)
      ctx.close().catch(() => undefined)
      stopMeter.current = () => undefined
    }
    recorder.current = rec
    rec.start(250)
    setPhase({ kind: "recording", started })
  }

  const stop = () => recorder.current?.state === "recording" && recorder.current.stop()

  const working = phase.kind === "preparing" || phase.kind === "cleaning"
  const recording = phase.kind === "recording"
  const disabled = !enabled || busy || working

  return (
    <GlassCard className="flex flex-1 flex-col gap-4 p-5 sm:p-6">
      <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-white/85">
        <IsoBadge>
          <IsoNote />
        </IsoBadge>
        Clean a voice note
      </h2>
      <p className="text-xs leading-relaxed text-white/50">
        Record or upload a message. Repeated and broken-off words are removed between words, in your own voice,
        and nothing is saved.
      </p>
      <div className="flex items-center gap-4">
        <motion.button
          type="button"
          onClick={recording ? stop : record}
          disabled={disabled && !recording}
          aria-label={recording ? "Stop recording" : "Record a voice note"}
          whileTap={disabled && !recording ? undefined : { scale: 0.98 }}
          transition={SPRING.press}
          className={cn(
            "relative grid size-14 shrink-0 cursor-pointer place-items-center rounded-full outline-none focus-visible:ring-2 focus-visible:ring-sky-300/60 disabled:cursor-not-allowed disabled:opacity-40",
            recording ? "bg-rose-500/85 text-white" : "bg-white/[0.07] text-rose-300 hover:bg-white/[0.11]"
          )}
        >
          <span
            ref={ring}
            aria-hidden
            className={cn(
              "absolute inset-0 rounded-full border-2 border-rose-300/50 transition-transform duration-75",
              !recording && "opacity-0"
            )}
          />
          {recording ? <Square className="size-5 fill-current" /> : <Mic className="size-6" />}
        </motion.button>
        <div className="min-w-0 flex-1 text-sm">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={phase.kind + (phase.kind === "cleaning" ? phase.step : "")}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2 }}
            >
              {phase.kind === "recording" ? (
                <span className="text-rose-200 tabular-nums">
                  Recording… {clock(elapsed)} <span className="text-white/40">/ {clock(MAX_S)}</span>
                </span>
              ) : working ? (
                <span className="inline-flex items-center gap-2 text-white/70">
                  <LoaderCircle className="size-4 animate-spin text-sky-300" />
                  {phase.kind === "preparing" ? "Preparing the audio…" : STEP[phase.step]}
                </span>
              ) : phase.kind === "error" ? (
                <span className="text-rose-300">{phase.message}</span>
              ) : !enabled ? (
                <span className="text-white/45">Needs the offline model, which is not downloaded.</span>
              ) : busy ? (
                <span className="text-white/45">Stop the live session to record a note.</span>
              ) : (
                <span className="text-white/60">Tap to record, up to 3 minutes.</span>
              )}
            </motion.div>
          </AnimatePresence>
          <button
            type="button"
            onClick={() => fileInput.current?.click()}
            disabled={disabled || recording}
            className="mt-1 cursor-pointer text-xs text-sky-300/80 underline-offset-4 transition-colors hover:text-sky-200 hover:underline disabled:cursor-not-allowed disabled:opacity-40"
          >
            or upload a file
          </button>
          <input
            ref={fileInput}
            type="file"
            accept="audio/*,.wav,.mp3,.m4a,.ogg,.webm,.flac"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              e.target.value = ""
              if (f) clean(f)
            }}
          />
        </div>
      </div>
    </GlassCard>
  )
}
