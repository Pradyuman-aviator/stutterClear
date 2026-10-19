import { useState } from "react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { sendFeedback, type Span } from "@/lib/api"

const KEY = "sc.feedback"

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

function keepLocal(entry: Record<string, unknown>) {
  try {
    const cur = JSON.parse(localStorage.getItem(KEY) || "[]") as unknown[]
    cur.push(entry)
    localStorage.setItem(KEY, JSON.stringify(cur.slice(-2000)))
  } catch {
    // private window: the review is not remembered
  }
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
  if (!id.startsWith("note-") || !cuts.length) return null

  const send = async (body: Record<string, unknown>, mark?: () => void) => {
    try {
      const r = await sendFeedback(body)
      if (!r.stored || browserStore) keepLocal(r.entry)
      mark?.()
    } catch (e) {
      toast.error((e as Error).message)
    }
  }

  return (
    <div className="flex w-full flex-col gap-2">
      <div className="text-xs font-medium text-muted-foreground">Review each cut</div>
      {cuts.map((c, i) => (
        <div key={`${c.start}-${c.end}`} className="flex flex-wrap items-center gap-2">
          <Badge variant="secondary">{c.kind}</Badge>
          <span className="text-xs text-muted-foreground">
            {c.word || "cut"} · {c.start.toFixed(1)}s
          </span>
          <Button size="xs" variant="outline" disabled={!!done[i]} onClick={() => send({ source: "note", id, verdict: "accept", cut: { start: c.start, end: c.end, kind: c.kind, words: c.word ?? "" } }, () => setDone((d) => ({ ...d, [i]: "accept" })))}>
            Good cut
          </Button>
          <Button size="xs" variant="ghost" disabled={!!done[i]} onClick={() => send({ source: "note", id, verdict: "reject", cut: { start: c.start, end: c.end, kind: c.kind, words: c.word ?? "" } }, () => setDone((d) => ({ ...d, [i]: "reject" })))}>
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
