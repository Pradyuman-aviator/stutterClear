import { useEffect, useState } from "react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getFeedback, sendFeedback, type FeedbackSummary, type Span } from "@/lib/api"
import { keepLocal, localSummary, reviewOff, setReviewOff } from "@/lib/feedback"

// The server reads a missed stutter's time on the original recording's clock. The cleaned track is
// shorter by every cut before that point (and, from the live engine, starts its delay later).
function originalTime(t: number, cuts: Span[], hear: "clean" | "original", lagS: number) {
  if (hear === "original") return t
  let o = Math.max(0, t - lagS)
  for (const c of [...cuts].sort((a, b) => a.start - b.start)) {
    if (c.start > o) break
    o += c.end - c.start
  }
  return o
}

export function CutReview({
  id,
  cuts,
  at,
  hear,
  lagS,
  browserStore,
}: {
  id: string
  cuts: Span[]
  at: number // playback position of the track being heard
  hear: "clean" | "original"
  lagS: number
  browserStore: boolean
}) {
  const [done, setDone] = useState<Record<number, string>>({})
  const [summary, setSummary] = useState<FeedbackSummary | null>(null)
  const [off, setOff] = useState(reviewOff)

  useEffect(() => {
    let live = true
    const load = browserStore ? localSummary() : getFeedback().catch(() => null)
    load.then((s) => live && setSummary(s))
    return () => {
      live = false
    }
  }, [browserStore])

  if (!id.startsWith("note-") || !cuts.length) return null

  if (off) {
    return (
      <div className="flex w-full items-center gap-2 text-xs text-muted-foreground">
        Auto mode: cuts are no longer reviewed.
        <Button size="xs" variant="ghost" onClick={() => { setReviewOff(false); setOff(false) }}>
          Review again
        </Button>
      </div>
    )
  }

  const send = async (body: Record<string, unknown>, mark?: () => void) => {
    try {
      const r = await sendFeedback(body)
      if (!r.stored || browserStore) {
        keepLocal(r.entry)
        setSummary(await localSummary())
      } else if (r.multipliers) {
        setSummary(r as FeedbackSummary)
      }
      mark?.()
    } catch (e) {
      toast.error((e as Error).message)
    }
  }

  const cut = (c: Span) => ({ start: c.start, end: c.end, kind: c.kind, words: c.word ?? "" })

  return (
    <div className="flex w-full flex-col gap-2">
      <div className="text-xs font-medium text-muted-foreground">
        Review each cut. StutterClear adjusts to your answers{browserStore ? " (kept in this browser)" : ""}.
      </div>
      {summary?.auto_ready && (
        <div className="flex flex-wrap items-center gap-2 rounded-xl border border-emerald-300/20 bg-emerald-300/5 px-3 py-2 text-xs text-emerald-100/90">
          You agreed with {Math.round((summary.agreement ?? 0) * 100)}% of your last {summary.recent} reviews.
          <Button size="xs" variant="outline" onClick={() => { setReviewOff(true); setOff(true) }}>
            Stop asking
          </Button>
        </div>
      )}
      {cuts.filter((c) => c.kind !== "pause").map((c, i) => (
        <div key={`${c.start}-${c.end}`} className="flex flex-wrap items-center gap-2">
          <Badge variant="secondary">{c.kind}</Badge>
          <span className="text-xs text-muted-foreground">
            {c.word || "cut"} · {c.start.toFixed(1)}s
          </span>
          <Button size="xs" variant="outline" disabled={!!done[i]} onClick={() => send({ source: "note", id, verdict: "accept", cut: cut(c) }, () => setDone((d) => ({ ...d, [i]: "accept" })))}>
            Good cut
          </Button>
          <Button size="xs" variant="ghost" disabled={!!done[i]} onClick={() => send({ source: "note", id, verdict: "reject", cut: cut(c) }, () => setDone((d) => ({ ...d, [i]: "reject" })))}>
            Undo cut
          </Button>
        </div>
      ))}
      <Button size="xs" variant="outline" onClick={() => send({ source: "note", id, missed_at_s: Math.round(originalTime(at, cuts, hear, lagS) * 100) / 100 })}>
        This moment was a stutter
      </Button>
    </div>
  )
}
