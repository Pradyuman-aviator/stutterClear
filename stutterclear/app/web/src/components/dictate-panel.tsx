import { useState } from "react"
import { Check, Copy, LoaderCircle } from "lucide-react"
import { toast } from "sonner"

import { GlassCard } from "@/components/glass-card"
import { TranscriptView } from "@/components/transcript-view"
import { Button } from "@/components/ui/button"
import { VoiceNoteCard } from "@/components/voice-note"
import type { Gpu } from "@/hooks/use-gpu"
import { getDemo, type Transcript } from "@/lib/api"

type Result = { kind: "tidying" } | { kind: "done"; text: Transcript } | { kind: "error"; message: string }

// Dictate: speak, get the text without the stutters, ready to paste anywhere. The same offline
// engine as a voice note (Thorough: repeats, broken-off starts, fillers and repeated phrases go),
// read as text instead of audio.
export function DictatePanel({
  enabled,
  busy,
  gpu,
  onGoto,
}: {
  enabled: boolean
  busy: boolean
  gpu?: Gpu
  onGoto?: (tab: "calls" | "recordings") => void
}) {
  const [result, setResult] = useState<Result | null>(null)
  const [copied, setCopied] = useState(false)

  const ready = async (id: string) => {
    setResult({ kind: "tidying" })
    setCopied(false)
    try {
      const tl = await getDemo(id, "thorough", { captions: false }) // the text is enough: no timed captions
      if (!tl.text) throw new Error("No words were heard in that recording.")
      setResult({ kind: "done", text: tl.text })
    } catch (e) {
      setResult({ kind: "error", message: (e as Error).message })
    }
  }

  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      toast.error("The browser did not allow copying; select the text instead.")
    }
  }

  const removed = result?.kind === "done" ? result.text.words.filter((w) => w.removed).length : 0

  return (
    <div className="grid gap-4">
      <VoiceNoteCard
        enabled={enabled}
        busy={busy}
        title="Dictate"
        blurb="Speak, or upload a recording, and get the text without the stutters, ready to paste into an email, a chat or a document. Nothing is saved."
        onReady={(id) => ready(id)}
        gpu={gpu}
        feature="dictate"
        onGoto={onGoto}
      />
      {result && (
        <GlassCard className="flex flex-col gap-4 p-5 sm:p-6">
          {result.kind === "tidying" && (
            <span className="inline-flex items-center gap-2 text-sm text-white/70">
              <LoaderCircle className="size-4 animate-spin text-sky-300" /> Tidying the text…
            </span>
          )}
          {result.kind === "error" && <span className="text-sm text-rose-300">{result.message}</span>}
          {result.kind === "done" && (
            <>
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold text-white/85">Your text</h3>
                <Button size="sm" variant="outline" onClick={() => copy(result.text.clean)} disabled={!result.text.clean}>
                  {copied ? <Check data-icon="inline-start" /> : <Copy data-icon="inline-start" />}
                  {copied ? "Copied" : "Copy"}
                </Button>
              </div>
              <p className="text-lg leading-relaxed text-white select-text">
                {result.text.clean || <span className="text-white/45">No words were heard.</span>}
              </p>
              <div className="flex flex-col gap-1.5 border-t border-white/8 pt-3">
                <span className="text-xs text-white/45">
                  What you said{removed ? `: ${removed} ${removed === 1 ? "word" : "words"} removed` : ""}
                </span>
                <TranscriptView text={result.text} />
              </div>
            </>
          )}
        </GlassCard>
      )}
    </div>
  )
}
