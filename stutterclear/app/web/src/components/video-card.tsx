import { useRef, useState } from "react"
import { Clapperboard, Download, LoaderCircle } from "lucide-react"

import { GlassCard } from "@/components/glass-card"
import { TranscriptView } from "@/components/transcript-view"
import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { getVideo, uploadVideo, type VideoStatus } from "@/lib/api"

const STEP: Record<string, string> = {
  queued: "Waiting for the models…",
  sound: "Reading the sound…",
  cleaning: "Finding the stutters…",
  cutting: "Cutting the video…",
}

type Phase =
  | { kind: "idle" }
  | { kind: "uploading"; pct: number }
  | { kind: "working"; status: VideoStatus }
  | { kind: "done"; status: VideoStatus }
  | { kind: "error"; message: string }

// Clean my video: the same video back with the stutters cut from picture and sound together.
export function VideoCard({ busy }: { busy: boolean }) {
  const [engine, setEngine] = useState<"natural" | "thorough">("natural")
  const [phase, setPhase] = useState<Phase>({ kind: "idle" })
  const input = useRef<HTMLInputElement>(null)

  const clean = async (file: File) => {
    if (file.size > 300_000_000) {
      setPhase({ kind: "error", message: "That video is too large (300 MB at most)." })
      return
    }
    try {
      setPhase({ kind: "uploading", pct: 0 })
      const { id } = await uploadVideo(file, engine, (pct) => setPhase({ kind: "uploading", pct }))
      for (;;) {
        await new Promise((r) => setTimeout(r, 800))
        const s = await getVideo(id)
        if (s.status === "error") throw new Error(s.error ?? "Cleaning the video failed.")
        if (s.status === "done") {
          setPhase({ kind: "done", status: s })
          return
        }
        setPhase({ kind: "working", status: s })
      }
    } catch (e) {
      setPhase({ kind: "error", message: (e as Error).message })
    }
  }

  const working = phase.kind === "uploading" || phase.kind === "working"
  const done = phase.kind === "done" ? phase.status : null

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
        <Button onClick={() => input.current?.click()} disabled={working || busy}>
          {working ? <LoaderCircle data-icon="inline-start" className="animate-spin" /> : <Clapperboard data-icon="inline-start" />}
          {working ? "Working…" : "Choose a video"}
        </Button>
        <span className="text-sm text-white/60">
          {busy && !working && "Stop the live session first."}
          {phase.kind === "uploading" && `Uploading… ${Math.round(phase.pct * 100)}%`}
          {phase.kind === "working" &&
            (phase.status.slow && phase.status.status === "cleaning"
              ? "Finding the stutters (on the CPU: the GPU is busy, so this takes longer)…"
              : STEP[phase.status.status] ?? "Working…")}
          {phase.kind === "error" && <span className="text-rose-300">{phase.message}</span>}
        </span>
        <input
          ref={input}
          type="file"
          accept="video/*,.mp4,.mov,.webm,.mkv"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            e.target.value = ""
            if (f) clean(f)
          }}
        />
      </div>
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
