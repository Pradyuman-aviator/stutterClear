import { useState } from "react"
import { toast } from "sonner"

import { InsightsPanel } from "@/components/insights-panel"
import { IsoBadge, IsoPractice } from "@/components/iso-icon"
import { Playback } from "@/components/playback"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import type { Caption } from "@/hooks/use-live"
import { interviewNext, interviewStart, interviewSummary, sayIt, type InterviewTurn, type PlaybackInfo } from "@/lib/api"
import type { Store } from "@/lib/store"

async function speak(text: string) {
  try {
    const r = await sayIt(text.slice(0, 200))
    const url = URL.createObjectURL(r.wav)
    const audio = new Audio(url)
    audio.onended = () => URL.revokeObjectURL(url)
    await audio.play()
  } catch {
    if (!window.speechSynthesis) return
    speechSynthesis.cancel()
    speechSynthesis.speak(new SpeechSynthesisUtterance(text))
  }
}

export function PracticePanel({
  running,
  caption,
  playback,
  playbackVersion,
  insights,
  onStart,
  onStop,
}: {
  running: boolean
  caption: Store<Caption | null>
  playback: PlaybackInfo | null
  playbackVersion: number
  insights: boolean
  onStart: () => void
  onStop: () => void
}) {
  const [role, setRole] = useState("")
  const [company, setCompany] = useState("")
  const [posting, setPosting] = useState("")
  const [card, setCard] = useState("")
  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState("")
  const [history, setHistory] = useState<InterviewTurn[]>([])
  const [points, setPoints] = useState<string[]>([])
  const [busy, setBusy] = useState(false)

  const start = async () => {
    setBusy(true)
    setPoints([])
    try {
      const r = await interviewStart({ role, company, posting })
      setCard(r.card)
      setQuestion(r.question)
      setHistory([])
      setAnswer("")
      void speak(r.question)
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const spoken = () => {
    if (!running && answer.trim()) return answer.trim()
    const c = caption.get()
    const heard = c ? [...c.lines, c.partial].join(" ").trim() : ""
    return heard || answer.trim()
  }

  const next = async () => {
    const turn = { question, answer: spoken() }
    const nextHistory = [...history, turn]
    setBusy(true)
    try {
      if (nextHistory.length >= 5) {
        const r = await interviewSummary({ role, company, history: nextHistory })
        setPoints(r.points)
        setHistory(nextHistory)
        setQuestion("")
        return
      }
      const r = await interviewNext({ role, company, card, history: nextHistory })
      setHistory(nextHistory)
      setQuestion(r.question)
      setAnswer("")
      void speak(r.question)
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card className="glass border-0 bg-transparent ring-0">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <IsoBadge><IsoPractice /></IsoBadge>
          Practice
        </CardTitle>
        <CardDescription>A practice interview about the role. It never comments on how you speak. The question is a synthetic voice.</CardDescription>
      </CardHeader>
      <CardContent>
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor="role">Role</FieldLabel>
            <Input id="role" value={role} onChange={(e) => setRole(e.target.value)} placeholder="Product designer" />
          </Field>
          <Field>
            <FieldLabel htmlFor="company">Company</FieldLabel>
            <Input id="company" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="Optional" />
          </Field>
          <Field>
            <FieldLabel htmlFor="posting">Job post</FieldLabel>
            <Textarea id="posting" value={posting} onChange={(e) => setPosting(e.target.value)} rows={3} placeholder="Optional" />
          </Field>
          {!question && points.length === 0 && (
            <Button type="button" disabled={busy || !role.trim()} onClick={start}>
              {busy ? "Starting…" : "Start the interview"}
            </Button>
          )}
          {question && <p className="text-base text-foreground">{question}</p>}
          {answer && <p className="text-sm text-muted-foreground">{answer}</p>}
          {playback && <Playback info={playback} version={playbackVersion} />}
          {playback && (
            <InsightsPanel
              key={playbackVersion}
              enabled={insights}
              build={() => ({ kinds: {}, duration_s: playback.original_s, removed_s: 0, words: [], transcript: spoken() })}
            />
          )}
          {points.length > 0 && (
            <ul className="flex flex-col gap-2 text-sm text-muted-foreground">
              {points.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          )}
        </FieldGroup>
      </CardContent>
      {question && (
        <CardFooter className="gap-2">
          <Button
            type="button"
            variant="secondary"
            onClick={() => {
              if (running) {
                onStop()
                setAnswer(spoken())
              } else onStart()
            }}
          >
            {running ? "Stop answering" : "Answer"}
          </Button>
          <Button type="button" disabled={busy || !spoken()} onClick={next}>
            {history.length >= 4 ? "Finish" : "Next question"}
          </Button>
        </CardFooter>
      )}
    </Card>
  )
}
