import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { LoaderCircle, Mic, RotateCcw, Square } from "lucide-react"
import { GpuNote } from "@/components/gpu-note"
import { IsoBadge, IsoNote } from "@/components/iso-icon"
import { GlassCard } from "@/components/glass-card"
import { blocked as gpuBlocked, queuedText, type Gpu } from "@/hooks/use-gpu"
import { clock } from "@/hooks/use-now"
import { createNote, getNote, transient, type ApiError, type NoteStatus } from "@/lib/api"
import { openMic } from "@/lib/browser-mic"
import { SPRING } from "@/lib/motion"
import { toWav16k } from "@/lib/wav"
import { cn } from "@/lib/utils"

const MAX_S = 180
const STEP: Record<string, string> = {
  listening: "Finding the stutters…",
  transcribing: "Writing down every word…",
}
const FORGOTTEN = "The server restarted and forgot this recording (nothing is kept on disk)."
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

type Phase =
  | { kind: "idle" }
  | { kind: "recording"; started: number }
  | { kind: "preparing" }
  | { kind: "cleaning"; step: string; slow?: boolean; ahead?: number | null }
  | { kind: "error"; message: string; again: boolean } // again: sending the kept audio once more may work

// Record or upload a message; it is cleaned offline (teacher + word-aligned cuts) and opens in the
// player above. Nothing is saved: the server keeps it in memory for an hour. With a GPU worker
// elsewhere (`gpu`), the GPU is woken as soon as the recording is sure to come (the mic allowed, a
// file that passed its checks), so it wakes while the visitor speaks.
export function VoiceNoteCard({
  enabled,
  busy,
  onReady,
  title = "Clean a voice note",
  blurb = "Record or upload a message. Repeated and broken-off words are removed between words, in your own voice, and nothing is saved.",
  offText = "Needs the offline model, which is not downloaded.",
  gpu,
  feature = "voice-note",
  onGoto,
}: {
  enabled: boolean
  busy: boolean // a live session owns the mic
  onReady: (id: string, seconds: number) => void
  title?: string
  blurb?: string
  offText?: string // why voice notes are off here
  gpu?: Gpu // the GPU worker that cleans notes on this server, if it has one
  feature?: "voice-note" | "dictate"
  onGoto?: (tab: "calls" | "recordings") => void
}) {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" })
  const [elapsed, setElapsed] = useState(0)
  const recorder = useRef<MediaRecorder | null>(null)
  const stream = useRef<MediaStream | null>(null)
  const ring = useRef<HTMLSpanElement | null>(null)
  const stopMeter = useRef<() => void>(() => undefined)
  const fileInput = useRef<HTMLInputElement>(null)
  const last = useRef<Blob | null>(null) // the last recording or file, for Try again
  const alive = useRef(true)

  // Unmounting ends a recording without sending it and lets go of the mic.
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
      stopMeter.current()
      if (recorder.current?.state === "recording") recorder.current.stop()
      stream.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  // Follow the note until it is done: every 2 s while it waits for a GPU elsewhere, faster while it
  // is being cleaned. Up to five failed polls in a row (a network blip, a restart) are waited out.
  const follow = async (id: string) => {
    let waiting = true
    for (let misses = 0; ; ) {
      await sleep(misses ? Math.min(8000, 1000 * 2 ** misses) : waiting && gpu ? 2000 : 600)
      if (!alive.current) return
      let s: NoteStatus
      try {
        s = await getNote(id)
        misses = 0
      } catch (e) {
        if ((e as ApiError).status === 404) throw new Error(FORGOTTEN)
        if (transient(e) && ++misses < 5) continue
        throw e
      }
      if (!alive.current) return
      if (s.status === "error") throw new Error(s.error ?? "Cleaning failed.")
      if (s.status === "done") {
        setPhase({ kind: "idle" })
        onReady(id, s.duration_s)
        return
      }
      waiting = s.status === "queued"
      setPhase({ kind: "cleaning", step: s.status, slow: s.slow, ahead: s.ahead })
    }
  }

  const clean = async (blob: Blob) => {
    last.current = blob
    setPhase({ kind: "preparing" })
    let wav: Blob
    try {
      const r = await toWav16k(blob, MAX_S) // a recording that ran to the limit ends a moment past it: its first 3 minutes
      if (r.seconds < 1) throw new Error("That recording is shorter than a second.")
      if (r.seconds > MAX_S + 1) throw new Error(`Voice notes can be up to ${MAX_S / 60} minutes long.`)
      wav = r.wav
    } catch (e) {
      setPhase({ kind: "error", message: (e as Error).message, again: false })
      return
    }
    gpu?.wake(feature, { quiet: true }) // it passed its checks: the GPU wakes while it uploads
    try {
      const { id } = await createNote(wav)
      setPhase({ kind: "cleaning", step: "queued" })
      await follow(id)
    } catch (e) {
      if (alive.current) setPhase({ kind: "error", message: (e as Error).message, again: (e as ApiError).status !== 415 })
    }
  }

  const record = async () => {
    let mic: MediaStream
    try {
      mic = await openMic()
    } catch (e) {
      setPhase({ kind: "error", message: (e as Error).message, again: false })
      return
    }
    if (!alive.current) {
      mic.getTracks().forEach((t) => t.stop())
      return
    }
    gpu?.wake(feature, { quiet: true }) // the mic is allowed: a recording is coming, so wake the GPU now
    stream.current = mic
    const chunks: Blob[] = []
    const rec = new MediaRecorder(mic)
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
    rec.onstop = () => {
      stopMeter.current()
      mic.getTracks().forEach((t) => t.stop())
      stream.current = null
      if (alive.current) clean(new Blob(chunks, { type: rec.mimeType }))
    }
    // Level ring: swells with the voice while recording.
    const ctx = new AudioContext()
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 512
    ctx.createMediaStreamSource(mic).connect(analyser)
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

  const st = gpu?.status ?? null
  const paused = gpuBlocked(st) // the GPU rests or is offline: nothing could clean it now
  const working = phase.kind === "preparing" || phase.kind === "cleaning"
  const recording = phase.kind === "recording"
  const disabled = !enabled || busy || working || paused
  const queued = phase.kind === "cleaning" && phase.step === "queued"

  return (
    <GlassCard className="flex flex-1 flex-col gap-4 p-5 sm:p-6">
      <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-white/85">
        <IsoBadge>
          <IsoNote />
        </IsoBadge>
        {title}
      </h2>
      <p className="text-xs leading-relaxed text-white/50">{blurb}</p>
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
                  <LoaderCircle className="size-4 shrink-0 animate-spin text-sky-300" />
                  {phase.kind === "preparing"
                    ? "Preparing the audio…"
                    : phase.step === "queued"
                      ? queuedText(st, phase.ahead, !!gpu)
                      : phase.slow && phase.step === "transcribing"
                        ? "Writing down every word (on the CPU this time, so it takes longer)…"
                        : STEP[phase.step] ?? "Working…"}
                </span>
              ) : phase.kind === "error" ? (
                <span className="text-rose-300">
                  {phase.message}
                  {phase.again && !paused && !busy && (
                    <button
                      type="button"
                      onClick={() => last.current && clean(last.current)}
                      className="ml-2 inline-flex cursor-pointer items-center gap-1 text-xs text-sky-300/90 underline-offset-4 hover:text-sky-200 hover:underline"
                    >
                      <RotateCcw className="size-3" /> Try again
                    </button>
                  )}
                </span>
              ) : !enabled ? (
                <span className="text-white/45">{offText}</span>
              ) : busy ? (
                <span className="text-white/45">Stop the live session to record a note.</span>
              ) : paused ? (
                <span className="text-white/45">Paused until the GPU is back (see below).</span>
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
      {gpu && enabled && (
        <GpuNote
          gpu={gpu}
          feature={feature}
          job={queued}
          onWake={() => gpu.wake("button")}
          onGoto={recording ? undefined : onGoto}
        />
      )}
    </GlassCard>
  )
}
