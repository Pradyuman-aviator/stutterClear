import { useState } from "react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { sendFeedback, type Span } from "@/lib/api"

export function CutReview({ id, cuts }: { id: string; cuts: Span[] }) {
  const [done, setDone] = useState<Record<number, string>>({})
  if (!id.startsWith("note-") || !cuts.length) return null

  const mark = async (i: number, verdict: "accept" | "reject", cut: Span) => {
    try {
      await sendFeedback({ source: "note", id, verdict, cut })
      setDone((d) => ({ ...d, [i]: verdict }))
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
          <Button size="xs" variant="outline" disabled={!!done[i]} onClick={() => mark(i, "accept", c)}>
            Keep
          </Button>
          <Button size="xs" variant="ghost" disabled={!!done[i]} onClick={() => mark(i, "reject", c)}>
            Undo
          </Button>
        </div>
      ))}
    </div>
  )
}
