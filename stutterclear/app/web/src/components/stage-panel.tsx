import { useEffect, useState } from "react"
import { Volume2 } from "lucide-react"
import { toast } from "sonner"

import { IsoBadge, IsoStage } from "@/components/iso-icon"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Textarea } from "@/components/ui/textarea"
import type { LiveData } from "@/hooks/use-live"
import { sayIt } from "@/lib/api"
import { followScript, normaliseScript } from "@/lib/script-follow"
import type { Store } from "@/lib/store"
import { useStore } from "@/lib/store"

async function speak(text: string) {
  try {
    const r = await sayIt(text.slice(0, 200))
    const url = URL.createObjectURL(r.wav)
    const audio = new Audio(url)
    audio.onended = () => URL.revokeObjectURL(url)
    await audio.play()
    return r.label
  } catch {
    if (!window.speechSynthesis) throw new Error("Say it needs a voice key, and this browser cannot speak either.")
    await new Promise<void>((resolve, reject) => {
      const u = new SpeechSynthesisUtterance(text)
      u.onend = () => resolve()
      u.onerror = () => reject(new Error("The browser voice did not play."))
      speechSynthesis.speak(u)
    })
    return "This browser's voice"
  }
}

export function StagePanel({
  running,
  caption,
  data,
  onToggle,
}: {
  running: boolean
  caption: Store<{ lines: string[]; partial: string } | null>
  data: { current: LiveData }
  onToggle: () => void
}) {
  const [scriptText, setScriptText] = useState("")
  const [pointer, setPointer] = useState(0)
  const [prompt, setPrompt] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [label, setLabel] = useState("")
  const cap = useStore(caption)
  const words = normaliseScript(scriptText)

  useEffect(() => {
    if (!running || !cap) return
    const spoken = normaliseScript([...cap.lines, cap.partial].join(" "))
    const script = normaliseScript(scriptText)
    setPointer((p) => followScript(script, spoken, p))
  }, [cap, running, scriptText])

  useEffect(() => {
    if (!running) return
    const id = window.setInterval(() => {
      const script = normaliseScript(scriptText)
      const silent = data.current.heardSound && performance.now() - data.current.lastSoundAt > 700
      const blocked = performance.now() - data.current.lastBlockAt < 1200
      const mid = pointer > 0 && pointer < script.length && !/[.!?]$/.test(script[pointer - 1] ?? "")
      if ((blocked || silent) && mid) {
        const next = script.slice(pointer, pointer + 3).join(" ")
        setPrompt((cur) => (cur.join(" ") === next ? cur : script.slice(pointer, pointer + 3)))
      }
    }, 200)
    return () => window.clearInterval(id)
  }, [running, data, pointer, scriptText])

  const say = async (text: string) => {
    if (!text.trim()) return
    setBusy(true)
    try {
      setLabel(await speak(text.trim()))
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
          <IsoBadge><IsoStage /></IsoBadge>
          On stage
        </CardTitle>
        <CardDescription>Paste the speech, then start. The highlight follows what you say. Nothing is spoken until you press Say it, and that voice is synthetic.</CardDescription>
      </CardHeader>
      <CardContent>
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor="script">Script</FieldLabel>
            <Textarea id="script" value={scriptText} onChange={(e) => { setScriptText(e.target.value); setPointer(0) }} rows={8} placeholder="The lines you are about to say" />
          </Field>
          {words.length > 0 && (
            <p className="text-sm leading-relaxed text-white/80">
              {words.map((w, i) => (
                <span key={`${w}-${i}`} className={i === pointer ? "rounded bg-white/15 px-0.5 text-white" : i < pointer ? "text-white/35" : ""}>
                  {w}{" "}
                </span>
              ))}
            </p>
          )}
          {prompt.length > 0 && <p className="text-3xl font-medium text-white">{prompt.join(" ")}</p>}
          <div className="flex flex-wrap gap-2">
            <Button type="button" variant="secondary" onClick={onToggle}>{running ? "Stop" : "Start"}</Button>
            <Button type="button" variant="outline" disabled={!running || pointer >= words.length} onClick={() => setPrompt(words.slice(pointer, pointer + 3))}>Stuck</Button>
            <Button type="button" disabled={busy || prompt.length === 0} onClick={() => say(prompt.join(" "))}>
              <Volume2 data-icon="inline-start" />
              {busy ? "Speaking…" : "Say it"}
            </Button>
          </div>
          {label && (
            <Alert>
              <AlertTitle>{label}</AlertTitle>
              <AlertDescription>This voice is synthetic. It is not a recording of a person. Nothing plays until you press Say it.</AlertDescription>
            </Alert>
          )}
        </FieldGroup>
      </CardContent>
    </Card>
  )
}
