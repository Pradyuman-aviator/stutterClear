import type { Transcript } from "@/lib/api"
import { cn } from "@/lib/utils"

const WHY: Record<string, string> = {
  repeat: "repeated word",
  cutoff: "broken-off start",
  filler: "filler",
}

// What was said, word by word: words a cut removed are struck through, fillers a cut kept are dim.
export function TranscriptView({ text, className }: { text: Transcript; className?: string }) {
  if (!text.words.length) return null
  return (
    <p className={cn("text-sm leading-relaxed text-white/75", className)}>
      {text.words.map((w, i) => (
        <span
          key={i}
          title={w.removed ? `Removed: ${WHY[w.label] ?? "stutter"}` : undefined}
          className={cn(
            w.removed && "text-rose-300/80 line-through decoration-rose-300/70",
            !w.removed && w.label === "filler" && "text-white/35"
          )}
        >
          {w.w}{" "}
        </span>
      ))}
    </p>
  )
}
