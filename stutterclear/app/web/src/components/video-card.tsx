import { useEffect, useRef, useState } from "react"
import { Clapperboard, Download, LoaderCircle, RotateCcw } from "lucide-react"

import { GlassCard } from "@/components/glass-card"
import { GpuNote } from "@/components/gpu-note"
import { TranscriptView } from "@/components/transcript-view"
import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { blocked as gpuBlocked, queuedText, type Gpu } from "@/hooks/use-gpu"
import { getVideo, transient, uploadVideo, type ApiError, type GpuStatus, type VideoStatus } from "@/lib/api"

const MAX_S = 180
const STEP: Record<string, string> = {
  queued: "Waiting for the models…",
  sound: "Reading the sound…",
  cleaning: "Finding the stutters…",
  cutting: "Cutting the video…",
}
const FORGOTTEN = "The server restarted and forgot this video (nothing is kept on disk)."
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

type Phase =
  | { kind: "idle" }
  | { kind: "uploading"; pct: number }
  | { kind: "working"; status: VideoStatus }
  | { kind: "done"; status: VideoStatus }
  | { kind: "error"; message: string; again: boolean } // again: sending the kept file once more may work

// The video's length from its own header, read by this browser (nothing is uploaded), or null when
// the browser cannot read the format: then the server checks it.
function videoSeconds(file: File) {
  return new Promise<number | null>((resolve) => {
    const url = URL.createObjectURL(file)
    const v = document.createElement("video")
    let settled = false
    const done = (s: number | null) => {
      if (settled) return
      settled = true
      URL.revokeObjectURL(url)
      v.removeAttribute("src")
      resolve(s !== null && Number.isFinite(s) ? s : null)
    }
    v.preload = "metadata"
    v.muted = true
    v.onloadedmetadata = () => done(v.duration)
    v.onerror = () => done(null)
    setTimeout(() => done(null), 8000)
    v.src = url
  })
}

// Where the job is, in words: its soundtrack goes through the voice-note engines (on the GPU worker
// when the server has one), then ffmpeg cuts the picture.
function stepText(s: VideoStatus, gpu: GpuStatus | null, remote: boolean) {
  if (s.status !== "cleaning") return STEP[s.status] ?? "Working…"
  if (s.note === "queued") return queuedText(gpu, s.ahead, remote)
  if (s.note === "transcribing") return s.slow ? "Writing down every word (on the CPU this time, so it takes longer)…" : "Writing down every word…"
  return s.slow && !s.note ? "Finding the stutters (on the CPU this time, so it takes longer)…" : STEP.cleaning
}

// Clean my video: the same video back with the stutters cut from picture and sound together.
export function VideoCard({
  busy,
  gpu,
  maxMb = 300,
  onGoto,
}: {
  busy: boolean
  gpu?: Gpu // the GPU worker that cleans the soundtrack on this server, if it has one
  maxMb?: number // the server's size limit
  onGoto?: (tab: "calls" | "recordings") => void
}) {
  const [engine, setEngine] = useState<"natural" | "thorough">("natural")
  const [phase, setPhase] = useState<Phase>({ kind: "idle" })
  const input = useRef<HTMLInputElement>(null)
  const last = useRef<File | null>(null) // the last video chosen, for Try again
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  // Every 2 s while the soundtrack waits for a GPU elsewhere, 800 ms otherwise; up to five failed
  // polls in a row (a network blip, a restart) are waited out.
  const follow = async (id: string) => {
    let waiting = true
    for (let misses = 0; ; ) {
      await sleep(misses ? Math.min(8000, 1000 * 2 ** misses) : waiting && gpu ? 2000 : 800)
      if (!alive.current) return
      let s: VideoStatus
      try {
        s = await getVideo(id)
        misses = 0
      } catch (e) {
        if ((e as ApiError).status === 404) throw new Error(FORGOTTEN)
        if (transient(e) && ++misses < 5) continue
        throw e
      }
      if (!alive.current) return
      if (s.status === "error") throw new Error(s.error ?? "Cleaning the video failed.")
      if (s.status === "done") {
        setPhase({ kind: "done", status: s })
        return
      }
      waiting = s.status === "queued" || (s.status === "cleaning" && s.note === "queued")
      setPhase({ kind: "working", status: s })
    }
  }

  const clean = async (file: File) => {
    last.current = file
    const tooLarge = `That video is too large (${maxMb} MB at most).`
    if (file.size > maxMb * 1e6 || !file.size) {
      setPhase({ kind: "error", message: file.size ? tooLarge : "That file is empty. Pick the video again.", again: false })
      return
    }
    setPhase({ kind: "uploading", pct: 0 })
    const seconds = await videoSeconds(file)
    if (seconds !== null && seconds > MAX_S + 1) {
      // (a "3:00" video a moment over is fine: the server keeps its first three minutes)
      setPhase({ kind: "error", message: "Videos can be up to 3 minutes long. Trim it first.", again: false })
      return
    }
    gpu?.wake("video", { quiet: true }) // it passed its checks: the GPU wakes while it uploads
    try {
      const { id } = await uploadVideo(file, engine, (pct) => alive.current && setPhase({ kind: "uploading", pct }))
      await follow(id)
    } catch (e) {
      const status = (e as ApiError).status
      const message = status === 413 ? tooLarge : (e as Error).message
      if (alive.current) setPhase({ kind: "error", message, again: status !== 400 && status !== 413 })
    }
  }

  const st = gpu?.status ?? null
  const paused = gpuBlocked(st)
  const working = phase.kind === "uploading" || phase.kind === "working"
  const done = phase.kind === "done" ? phase.status : null
  const queued =
    phase.kind === "working" && (phase.status.status === "queued" || (phase.status.status === "cleaning" && phase.status.note === "queued"))

  return (
    <GlassCard className="flex flex-col gap-4 p-5 sm:p-6 md:col-span-2">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold tracking-wide text-white/85">
          <Clapperboard className="size-4 text-sky-300" /> Clean my video
        </h2>
        <ToggleGroup aria-label="Engine" value={[engine]} onValueChange={(v) => v[0] && setEngine(v[0] as "natural" | "thorough")} variant="outline" spacing={1}>
          <ToggleGroupItem value="natural">Natural</ToggleGroupItem>
          <ToggleGroupItem value="thorough">Thorough</ToggleGroupItem>
        </ToggleGroup>
      </div>
      <p className="text-xs leading-relaxed text-white/50">
        A talk, a post or a recorded answer, up to 3 minutes. Stutters are cut from the picture and the sound together, at
        word boundaries. Natural keeps fillers; Thorough removes them too. The video is deleted as soon as it is cut; the
        result is kept for an hour.
      </p>
      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={() => input.current?.click()} disabled={working || busy || paused}>
          {working ? <LoaderCircle data-icon="inline-start" className="animate-spin" /> : <Clapperboard data-icon="inline-start" />}
          {working ? "Working…" : "Choose a video"}
        </Button>
        <span className="min-w-0 text-sm text-white/60">
          {busy && !working && "Stop the live session first."}
          {!busy && !working && paused && phase.kind !== "error" && "Paused until the GPU is back (see below)."}
          {phase.kind === "uploading" && `Uploading… ${Math.round(phase.pct * 100)}%`}
          {phase.kind === "working" && stepText(phase.status, st, !!gpu)}
          {phase.kind === "error" && (
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
          )}
        </span>
        <input
          ref={input}
          type="file"
          accept="video/*,.mp4,.mov,.m4v,.webm,.mkv,.avi"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            e.target.value = ""
            if (f) clean(f)
          }}
        />
      </div>
      {gpu && <GpuNote gpu={gpu} feature="video" job={queued} onWake={() => gpu.wake("button")} onGoto={onGoto} />}
      {done && (
        <div className="grid gap-4 md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <video controls playsInline className="w-full rounded-2xl border border-white/8 bg-black" src={`/api/video/${done.id}/clean.mp4`} />
          <div className="flex flex-col gap-3">
            <div className="text-sm text-white/75">
              {done.duration_s?.toFixed(1)} s → <span className="text-white">{done.clean_s?.toFixed(1)} s</span> ·{" "}
              {done.cuts?.length ?? 0} {done.cuts?.length === 1 ? "cut" : "cuts"}
            </div>
            <Button variant="outline" size="sm" className="w-fit" render={<a href={`/api/video/${done.id}/clean.mp4?download=1`} />}>
              <Download data-icon="inline-start" /> Download the video
            </Button>
            {done.text && (
              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-white/45">Transcript: struck-through words were cut</span>
                <TranscriptView text={done.text} />
              </div>
            )}
          </div>
        </div>
      )}
    </GlassCard>
  )
}
