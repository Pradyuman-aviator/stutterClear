import { useState } from "react"
import { toast } from "sonner"

import { IsoBadge, IsoPractice } from "@/components/iso-icon"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { interviewNext, interviewStart, interviewSummary, type InterviewTurn } from "@/lib/api"

export function PracticePanel() {
  const [role, setRole] = useState("")
  const [company, setCompany] = useState("")
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
      const r = await interviewStart({ role, company })
      setCard(r.card)
      setQuestion(r.question)
      setHistory([])
      setAnswer("")
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const next = async () => {
    const turn = { question, answer: answer.trim() }
    const nextHistory = [...history, turn]
    setBusy(true)
    try {
      const r = await interviewNext({ role, company, card, history: nextHistory })
      setHistory(nextHistory)
      setQuestion(r.question)
      setAnswer("")
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const finish = async () => {
    const nextHistory = answer.trim() ? [...history, { question, answer: answer.trim() }] : history
    setBusy(true)
    try {
      const r = await interviewSummary({ role, company, history: nextHistory })
      setPoints(r.points)
      setHistory(nextHistory)
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
        <CardDescription>A practice interview about the role. It never comments on how you speak.</CardDescription>
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
          {!question && (
            <Button type="button" disabled={busy || !role.trim()} onClick={start}>
              {busy ? "Starting…" : "Start the interview"}
            </Button>
          )}
          {question && (
            <>
              <p className="text-base text-foreground">{question}</p>
              <Field>
                <FieldLabel htmlFor="answer">Your answer</FieldLabel>
                <Textarea id="answer" value={answer} onChange={(e) => setAnswer(e.target.value)} rows={5} />
              </Field>
            </>
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
          <Button type="button" variant="secondary" disabled={busy || !answer.trim()} onClick={next}>
            Next question
          </Button>
          <Button type="button" variant="outline" disabled={busy || history.length + (answer.trim() ? 1 : 0) === 0} onClick={finish}>
            Finish
          </Button>
        </CardFooter>
      )}
    </Card>
  )
}
